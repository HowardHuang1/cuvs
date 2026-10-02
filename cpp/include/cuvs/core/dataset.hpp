/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <raft/core/device_container_policy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_container_policy.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>   // get_device_for_address, copy_matrix
#include <raft/util/integer_utils.hpp>  // rounding up

#include <cuvs/core/export.hpp>
#include <raft/core/detail/macros.hpp>

#include <cuda_fp16.h>

#include <concepts>
#include <cstring>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>
#ifdef __cpp_lib_bitops
#include <bit>
#endif

namespace CUVS_EXPORT cuvs {
namespace neighbors {

/**
 * @brief Spec-based `dataset` / `dataset_view`.
 *
 * `dataset<T,IdxT,SpecT>` and `dataset_view<T,IdxT,SpecT>` are single generic templates that know
 * nothing about any particular kind of dataset. They hold exactly one payload (`data_type` /
 * `view_type`, chosen by the spec) and expose only what every dataset has: `n_rows()`, `dim()`,
 * `as_matrix_view()`, `as_dataset_view()` and `data()`. Each is a one-line forward to one of the
 * three spec functions `get_data_view()`, `get_n_rows()` and `get_dim()`. Anything else a kind
 * needs (e.g. VPQ codebooks, BBQ quantizers) is state and methods of that kind's payload type,
 * reached through `data()`; `dataset`/`dataset_view` never name or branch on it. `dataset` and
 * `dataset_view` are deliberately two independent, non-inheriting types (no shared_ptr, no
 * "sometimes owning" object): `dataset` holds the owning payload, `dataset_view` the
 * corresponding non-owning payload.
 */

template <typename T, typename IdxT, typename SpecT>
struct dataset;

template <typename T, typename IdxT, typename SpecT>
struct dataset_view;

namespace detail {

// Default owning/view accessors for public dataset aliases.
template <typename T>
using device_owning_accessor = raft::device_accessor<raft::device_container_policy<T>>;

template <typename T>
using host_owning_accessor = raft::host_accessor<raft::host_container_policy<T>>;

template <typename T>
using device_view_accessor = raft::device_accessor<cuda::std::default_accessor<const T>>;

template <typename T>
using host_view_accessor = raft::host_accessor<cuda::std::default_accessor<const T>>;

/** View accessor paired with an owning dataset accessor (same residency). */
template <typename DataT, typename Accessor>
using dataset_view_accessor_for_owning = std::conditional_t<Accessor::is_device_accessible,
                                                            device_view_accessor<DataT>,
                                                            host_view_accessor<DataT>>;

/** Owning accessor paired with a view accessor (same residency). */
template <typename DataT, typename Accessor>
using dataset_owning_accessor_for_view = std::conditional_t<Accessor::is_device_accessible,
                                                            device_owning_accessor<DataT>,
                                                            host_owning_accessor<DataT>>;

// Accessor here is already device_owning_accessor<DataT> / host_owning_accessor<DataT> at every
// call site -- exactly the container policy raft::device_mdarray/host_mdarray default to for
// element type DataT -- so pass it straight through instead of re-deriving a
// raft::device_matrix/host_matrix from scratch.
template <typename DataT, typename IdxT, typename Accessor>
using dense_owning_matrix =
  raft::mdarray<DataT, raft::matrix_extent<IdxT>, raft::row_major, Accessor>;

template <typename DataT, typename IdxT, typename Accessor>
using dense_view_matrix = raft::mdspan<const DataT,
                                       raft::matrix_extent<IdxT>,
                                       raft::row_major,
                                       dataset_view_accessor_for_owning<DataT, Accessor>>;

template <typename MathT, typename IdxT, typename Accessor>
using vpq_vq_book_matrix =
  raft::mdarray<MathT, raft::matrix_extent<uint32_t>, raft::row_major, Accessor>;

// VPQ codes are always uint8_t regardless of MathT, so retarget the owning accessor's element
// type instead of re-deriving a device/host matrix; residency is still driven by Accessor.
template <typename NewT, typename Accessor>
using owning_accessor_with_value_type = std::conditional_t<Accessor::is_device_accessible,
                                                           device_owning_accessor<NewT>,
                                                           host_owning_accessor<NewT>>;

template <typename IdxT, typename Accessor>
using vpq_data_matrix = raft::mdarray<uint8_t,
                                      raft::matrix_extent<IdxT>,
                                      raft::row_major,
                                      owning_accessor_with_value_type<uint8_t, Accessor>>;

// -----------------------------------------------------------------------------
// empty
// -----------------------------------------------------------------------------

template <typename IdxT>
struct empty_dataset_storage {
  uint32_t suggested_dim{};
  empty_dataset_storage() noexcept = default;
  explicit empty_dataset_storage(uint32_t dim) noexcept : suggested_dim(dim) {}
  [[nodiscard]] auto n_rows() const noexcept -> IdxT { return 0; }
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return suggested_dim; }
};

// -----------------------------------------------------------------------------
// dense row-major (logical dim may differ from row pitch; shared by padded & standard)
// -----------------------------------------------------------------------------

/**
 * Dense row-major owning storage shared by padded and standard dataset specs. Publicly inherits
 * from MatrixT (a `raft::mdarray`) so `view()`/`data_handle()`/`extent()` etc. are reused as-is
 * rather than hand-forwarded; `logical_dim_` is the only state this struct adds.
 *
 * Template parameters:
 * - MatrixT: owning matrix type that stores the payload (host/device matrix).
 * - ViewT: non-owning row-major view type returned by `view()`.
 * - DataT: scalar element type of the dataset payload.
 * - IdxT: index type used for row counts (`n_rows()` return type).
 */
template <typename MatrixT, typename ViewT, typename DataT, typename IdxT>
struct dense_row_major_dataset_owning_storage : public MatrixT {
  uint32_t logical_dim_;

  // MatrixT (mdarray) also has its own stride(size_t); pull it back into scope since declaring
  // our own no-arg stride() below would otherwise hide it entirely (C++ name hiding).
  using MatrixT::stride;

  dense_row_major_dataset_owning_storage(MatrixT&& data, uint32_t logical_dim) noexcept
    : MatrixT{std::move(data)}, logical_dim_{logical_dim}
  {
  }

  [[nodiscard]] auto n_rows() const noexcept -> IdxT { return this->extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return logical_dim_; }
  [[nodiscard]] auto stride() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(this->extent(1));
  }
  // view() and data_handle() are inherited directly from MatrixT (raft::mdarray); no hand-written
  // forwarding needed since MatrixT::view() const already returns exactly ViewT.
};

template <typename ViewT, typename DataT, typename IdxT>
struct dense_row_major_dataset_view_storage : public ViewT {
  uint32_t logical_dim_;

  // ViewT (mdspan) also has its own stride(size_t); pull it back into scope since declaring our
  // own no-arg stride() below would otherwise hide it entirely (C++ name hiding), and the body of
  // that stride() itself needs to call the inherited one.
  using ViewT::stride;

  dense_row_major_dataset_view_storage() noexcept = default;

  explicit dense_row_major_dataset_view_storage(ViewT v) noexcept
    : ViewT(v), logical_dim_(static_cast<uint32_t>(v.extent(1)))
  {
  }

  dense_row_major_dataset_view_storage(ViewT v, uint32_t logical_dim) noexcept
    : ViewT(v), logical_dim_(logical_dim)
  {
  }

  [[nodiscard]] auto n_rows() const noexcept -> IdxT { return this->extent(0); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return logical_dim_; }
  [[nodiscard]] auto stride() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(ViewT::stride(0) > 0 ? ViewT::stride(0) : this->extent(1));
  }
};

/** Spec-side implementation shared by `padded_dataset_spec`/`standard_dataset_spec`; those two
 * stay distinct top-level types (identical bodies) purely so classification traits can tell them
 * apart -- exactly mirroring today's `padded_dataset_container`/`standard_dataset_container`,
 * which are likewise two differently-named tags over one shared storage implementation. */
template <typename ContainerPolicy>
struct dense_dataset_spec_impl {
  template <typename T, typename IdxT>
  struct apply {
    using value_type = std::remove_cv_t<T>;
    using index_type = std::remove_cv_t<IdxT>;
    using MatrixT    = dense_owning_matrix<T, IdxT, ContainerPolicy>;
    using ViewT      = dense_view_matrix<T, IdxT, ContainerPolicy>;
    using data_type  = dense_row_major_dataset_owning_storage<MatrixT, ViewT, T, IdxT>;
    using view_type  = dense_row_major_dataset_view_storage<ViewT, T, IdxT>;

    [[nodiscard]] static auto get_data_view(data_type const& data) noexcept -> view_type
    {
      return view_type(data.view(), data.dim());
    }
    template <typename AnyStorage>
    [[nodiscard]] static auto get_n_rows(AnyStorage const& data) noexcept -> index_type
    {
      return data.n_rows();
    }
    template <typename AnyStorage>
    [[nodiscard]] static auto get_dim(AnyStorage const& data) noexcept -> uint32_t
    {
      return data.dim();
    }
  };
};

// -----------------------------------------------------------------------------
// vpq payloads: everything VPQ-specific lives here, not in dataset/dataset_view.
// -----------------------------------------------------------------------------

/** Read-only helpers derived from the codebook shapes; shared by the owning and view payloads.
 * `Derived` provides `vq_code_book`, `pq_code_book` and the codes' `extent(r)`. */
template <typename Derived>
struct vpq_codebook_helpers {
  /** Logical dimension: it comes from the VQ codebook, not from the encoded rows (row padding
   * makes the encoded-row width ambiguous as a dimension). */
  [[nodiscard]] auto dim() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(self().vq_code_book.extent(1));
  }
  [[nodiscard]] auto vq_n_centers() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(self().vq_code_book.extent(0));
  }
  [[nodiscard]] auto pq_n_centers() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(self().pq_code_book.extent(0));
  }
  [[nodiscard]] auto pq_len() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(self().pq_code_book.extent(1));
  }
  [[nodiscard]] auto pq_bits() const noexcept -> uint32_t
  {
    auto pq_width = pq_n_centers();
#ifdef __cpp_lib_bitops
    return std::countr_zero(pq_width);
#else
    uint32_t bits = 0;
    while (pq_width > 1) {
      bits++;
      pq_width >>= 1;
    }
    return bits;
#endif
  }
  [[nodiscard]] auto pq_dim() const noexcept -> uint32_t
  {
    return raft::div_rounding_up_unsafe(dim(), pq_len());
  }
  [[nodiscard]] auto encoded_row_length() const noexcept -> uint32_t
  {
    return static_cast<uint32_t>(self().extent(1));
  }

 private:
  [[nodiscard]] auto self() const noexcept -> Derived const&
  {
    return static_cast<Derived const&>(*this);
  }
};

/** Owning VPQ payload: the encoded rows (it *is* the `uint8_t` codes mdarray) plus the VQ and PQ
 * codebooks. `Accessor` drives both codebook and code residency. */
template <typename MathT, typename IdxT, typename Accessor>
struct vpq_owning_storage : public vpq_data_matrix<IdxT, Accessor>,
                            public vpq_codebook_helpers<vpq_owning_storage<MathT, IdxT, Accessor>> {
  using codes_type   = vpq_data_matrix<IdxT, Accessor>;
  using vq_book_type = vpq_vq_book_matrix<MathT, IdxT, Accessor>;
  using pq_book_type = vpq_vq_book_matrix<MathT, IdxT, Accessor>;

  vq_book_type vq_code_book;
  pq_book_type pq_code_book;

  vpq_owning_storage(codes_type&& codes, vq_book_type&& vq_codes, pq_book_type&& pq_codes) noexcept
    : codes_type{std::move(codes)},
      vq_code_book{std::move(vq_codes)},
      pq_code_book{std::move(pq_codes)}
  {
  }
};

/** Non-owning VPQ payload: a view of the encoded rows plus views of the VQ and PQ codebooks. */
template <typename MathT, typename IdxT, typename Accessor>
struct vpq_view_storage : public raft::mdspan<const uint8_t,
                                              raft::matrix_extent<IdxT>,
                                              raft::row_major,
                                              dataset_view_accessor_for_owning<uint8_t, Accessor>>,
                          public vpq_codebook_helpers<vpq_view_storage<MathT, IdxT, Accessor>> {
  using codes_view_type = raft::mdspan<const uint8_t,
                                       raft::matrix_extent<IdxT>,
                                       raft::row_major,
                                       dataset_view_accessor_for_owning<uint8_t, Accessor>>;
  using vq_book_view_type =
    typename vpq_owning_storage<MathT, IdxT, Accessor>::vq_book_type::const_view_type;
  using pq_book_view_type =
    typename vpq_owning_storage<MathT, IdxT, Accessor>::pq_book_type::const_view_type;

  vq_book_view_type vq_code_book;
  pq_book_view_type pq_code_book;

  vpq_view_storage() noexcept = default;
  vpq_view_storage(codes_view_type codes,
                   vq_book_view_type vq_codes,
                   pq_book_view_type pq_codes) noexcept
    : codes_view_type(codes), vq_code_book(vq_codes), pq_code_book(pq_codes)
  {
  }
};

}  // namespace detail

// -----------------------------------------------------------------------------
// Public specs -- the only place per-kind logic lives.
// -----------------------------------------------------------------------------

template <typename Accessor>
struct empty_dataset_spec {
  using accessor_type = Accessor;

  template <typename T, typename IdxT>
  struct apply {
    using value_type = std::remove_cv_t<T>;
    using index_type = std::remove_cv_t<IdxT>;
    using data_type  = detail::empty_dataset_storage<IdxT>;
    using view_type  = detail::empty_dataset_storage<IdxT>;

    [[nodiscard]] static auto get_data_view(data_type const& data) noexcept -> view_type
    {
      return data;
    }
    [[nodiscard]] static auto get_n_rows(data_type const& data) noexcept -> index_type
    {
      return static_cast<index_type>(data.n_rows());
    }
    [[nodiscard]] static auto get_dim(data_type const& data) noexcept -> uint32_t
    {
      return data.dim();
    }
  };
};

template <typename ContainerPolicy>
struct padded_dataset_spec {
  using accessor_type = ContainerPolicy;
  template <typename T, typename IdxT>
  struct apply : detail::dense_dataset_spec_impl<ContainerPolicy>::template apply<T, IdxT> {};
};

template <typename ContainerPolicy>
struct standard_dataset_spec {
  using accessor_type = ContainerPolicy;
  template <typename T, typename IdxT>
  struct apply : detail::dense_dataset_spec_impl<ContainerPolicy>::template apply<T, IdxT> {};
};

/** `Accessor` drives both codebook and code residency, mirroring today's
 * single-`Accessor`-per-VPQ-dataset design. The payload (`detail::vpq_owning_storage` /
 * `detail::vpq_view_storage`) holds the encoded rows and the VQ/PQ codebooks. */
template <typename MathT, typename Accessor>
struct vpq_dataset_spec {
  using accessor_type = Accessor;

  template <typename T, typename IdxT>
  struct apply {
    using value_type = std::remove_cv_t<T>;
    using index_type = std::remove_cv_t<IdxT>;
    using math_type  = MathT;

    using data_type = detail::vpq_owning_storage<MathT, IdxT, Accessor>;
    using view_type = detail::vpq_view_storage<MathT, IdxT, Accessor>;

    [[nodiscard]] static auto get_data_view(data_type const& data) noexcept -> view_type
    {
      return view_type(data.view(), data.vq_code_book.view(), data.pq_code_book.view());
    }
    template <typename AnyStorage>
    [[nodiscard]] static auto get_n_rows(AnyStorage const& data) noexcept -> index_type
    {
      return static_cast<index_type>(data.extent(0));
    }
    template <typename AnyStorage>
    [[nodiscard]] static auto get_dim(AnyStorage const& data) noexcept -> uint32_t
    {
      return data.dim();
    }
  };
};

// -----------------------------------------------------------------------------
// dataset / dataset_view
// -----------------------------------------------------------------------------

/** Owning dataset: value-held payload (no shared_ptr -- exclusive ownership). Every member is a
 * one-line forward to `spec_type::get_*` or to the payload; all per-kind state and logic lives in
 * the spec's `data_type`, never inside this struct. */
template <typename T, typename IdxT, typename SpecT>
struct dataset {
  using spec_type  = typename SpecT::template apply<T, IdxT>;
  using value_type = typename spec_type::value_type;
  using index_type = typename spec_type::index_type;
  using data_type  = typename spec_type::data_type;

  // Forward constructor args straight to data_type's own constructor (e.g. (MatrixT&&, uint32_t
  // logical_dim) for dense, (uint32_t dim) for empty, (codes&&, vq&&, pq&&) for VPQ).
  template <typename... Args>
  explicit dataset(Args&&... args)
    requires(std::is_constructible_v<data_type, Args...>)
    : data_(std::forward<Args>(args)...)
  {
  }

  [[nodiscard]] auto n_rows() const noexcept -> index_type { return spec_type::get_n_rows(data_); }
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return spec_type::get_dim(data_); }
  /** The spec-defined non-owning view of the payload (for dense and VPQ it is an mdspan
   * derivative). */
  [[nodiscard]] auto as_matrix_view() const noexcept { return spec_type::get_data_view(data_); }
  [[nodiscard]] auto as_dataset_view() const noexcept -> dataset_view<T, IdxT, SpecT>
  {
    return dataset_view<T, IdxT, SpecT>(as_matrix_view());
  }

  /** The owning payload; kind-specific state and methods are reached through it. */
  [[nodiscard]] auto data() const noexcept -> data_type const& { return data_; }
  [[nodiscard]] auto data() noexcept -> data_type& { return data_; }

 private:
  data_type data_;
};

/** Non-owning dataset view: holds only the view-shaped payload. Deliberately not derived from
 * `dataset` -- a view type holds "all view state" with no inheritance and no shared ownership tying
 * it to the owning type. Reuses the same `get_n_rows`/`get_dim` spec functions as `dataset`, fed
 * the view payload instead of the owning one. */
template <typename T, typename IdxT, typename SpecT>
struct dataset_view {
  using spec_type  = typename SpecT::template apply<T, IdxT>;
  using value_type = typename spec_type::value_type;
  using index_type = typename spec_type::index_type;
  using view_type  = typename spec_type::view_type;

  dataset_view() noexcept = default;

  // Already-constructed view payload -- the shape `as_dataset_view()` always constructs with, for
  // every kind. Not a template, so it's preferred over the forwarding constructor below whenever
  // both could apply.
  explicit dataset_view(view_type data_view) noexcept : data_view_{data_view} {}

  // Forward raw constructor args straight to view_type's own constructor (e.g. (ViewT, uint32_t
  // logical_dim) for dense, (uint32_t dim) for empty) -- preserves today's direct-construction
  // call sites (e.g. `device_padded_dataset_view<T,IdxT>(raw_mdspan, dim)`) unchanged.
  template <typename... Args>
  explicit dataset_view(Args&&... args)
    requires(std::is_constructible_v<view_type, Args...>)
    : data_view_(std::forward<Args>(args)...)
  {
  }

  [[nodiscard]] auto n_rows() const noexcept -> index_type
  {
    return spec_type::get_n_rows(data_view_);
  }
  [[nodiscard]] auto dim() const noexcept -> uint32_t { return spec_type::get_dim(data_view_); }
  [[nodiscard]] auto as_matrix_view() const noexcept -> view_type { return data_view_; }

  /** The view payload; kind-specific state and methods are reached through it. */
  [[nodiscard]] auto data() const noexcept -> view_type const& { return data_view_; }
  [[nodiscard]] auto data() noexcept -> view_type& { return data_view_; }

 private:
  view_type data_view_{};
};

/**
 * @brief Aliases for concrete `dataset` / `dataset_view` layouts.
 */
template <typename IdxT>
using device_empty_dataset =
  dataset<void, IdxT, empty_dataset_spec<detail::device_view_accessor<char>>>;

template <typename IdxT>
using device_empty_dataset_view =
  dataset_view<void, IdxT, empty_dataset_spec<detail::device_view_accessor<char>>>;

template <typename IdxT>
using host_empty_dataset =
  dataset<void, IdxT, empty_dataset_spec<detail::host_view_accessor<char>>>;

template <typename IdxT>
using host_empty_dataset_view =
  dataset_view<void, IdxT, empty_dataset_spec<detail::host_view_accessor<char>>>;

template <typename DataT, typename IdxT>
using device_padded_dataset =
  dataset<DataT, IdxT, padded_dataset_spec<detail::device_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using device_padded_dataset_view =
  dataset_view<DataT, IdxT, padded_dataset_spec<detail::device_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using host_padded_dataset =
  dataset<DataT, IdxT, padded_dataset_spec<detail::host_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using host_padded_dataset_view =
  dataset_view<DataT, IdxT, padded_dataset_spec<detail::host_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using device_standard_dataset =
  dataset<DataT, IdxT, standard_dataset_spec<detail::device_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using device_standard_dataset_view =
  dataset_view<DataT, IdxT, standard_dataset_spec<detail::device_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using host_standard_dataset =
  dataset<DataT, IdxT, standard_dataset_spec<detail::host_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using host_standard_dataset_view =
  dataset_view<DataT, IdxT, standard_dataset_spec<detail::host_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using device_vpq_dataset =
  dataset<DataT, IdxT, vpq_dataset_spec<DataT, detail::device_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using device_vpq_dataset_view =
  dataset_view<DataT, IdxT, vpq_dataset_spec<DataT, detail::device_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using host_vpq_dataset =
  dataset<DataT, IdxT, vpq_dataset_spec<DataT, detail::host_owning_accessor<DataT>>>;

template <typename DataT, typename IdxT>
using host_vpq_dataset_view =
  dataset_view<DataT, IdxT, vpq_dataset_spec<DataT, detail::host_owning_accessor<DataT>>>;

// Maps a dataset view type to its owning (allocating) dataset counterpart. Trivial and total under
// the Spec design: the owning type for `dataset_view<T,IdxT,SpecT>` is always
// `dataset<T,IdxT,SpecT>`
// -- no per-kind specialization table needed (unlike the old Container-tagged design).
template <typename DatasetViewT>
struct owning_dataset_for_view;

template <typename T, typename IdxT, typename SpecT>
struct owning_dataset_for_view<dataset_view<T, IdxT, SpecT>> {
  using type = dataset<T, IdxT, SpecT>;
};

template <typename DatasetViewT>
using owning_dataset_for_view_t = typename owning_dataset_for_view<DatasetViewT>::type;

// -----------------------------------------------------------------------------
// Spec-kind classification (all derived from SpecT; dataset/dataset_view never branch on kind).
// -----------------------------------------------------------------------------

template <typename SpecT>
struct is_empty_spec : std::false_type {};
template <typename Accessor>
struct is_empty_spec<empty_dataset_spec<Accessor>> : std::true_type {};
template <typename SpecT>
inline constexpr bool is_empty_spec_v = is_empty_spec<SpecT>::value;

template <typename SpecT>
struct is_padded_spec : std::false_type {};
template <typename ContainerPolicy>
struct is_padded_spec<padded_dataset_spec<ContainerPolicy>> : std::true_type {};
template <typename SpecT>
inline constexpr bool is_padded_spec_v = is_padded_spec<SpecT>::value;

template <typename SpecT>
struct is_standard_spec : std::false_type {};
template <typename ContainerPolicy>
struct is_standard_spec<standard_dataset_spec<ContainerPolicy>> : std::true_type {};
template <typename SpecT>
inline constexpr bool is_standard_spec_v = is_standard_spec<SpecT>::value;

template <typename SpecT>
struct is_vpq_spec : std::false_type {};
template <typename MathT, typename Accessor>
struct is_vpq_spec<vpq_dataset_spec<MathT, Accessor>> : std::true_type {};
template <typename SpecT>
inline constexpr bool is_vpq_spec_v = is_vpq_spec<SpecT>::value;

template <typename SpecT>
struct vpq_spec_math_type {};
template <typename MathT, typename Accessor>
struct vpq_spec_math_type<vpq_dataset_spec<MathT, Accessor>> {
  using type = MathT;
};
template <typename SpecT>
using vpq_spec_math_type_t = typename vpq_spec_math_type<SpecT>::type;

/** Owning-side kind traits (mirror today's `is_padded_dataset_v`/`is_standard_dataset_v`/
 * `is_vpq_dataset_v`, used for SFINAE overload selection in factory.cuh/compute_distance_vpq.hpp).
 */
template <typename DatasetT>
struct is_padded_dataset : std::false_type {};
template <typename T, typename IdxT, typename SpecT>
struct is_padded_dataset<dataset<T, IdxT, SpecT>> : std::bool_constant<is_padded_spec_v<SpecT>> {};
template <typename T, typename IdxT, typename SpecT>
struct is_padded_dataset<dataset_view<T, IdxT, SpecT>>
  : std::bool_constant<is_padded_spec_v<SpecT>> {};
template <typename DatasetT>
inline constexpr bool is_padded_dataset_v = is_padded_dataset<DatasetT>::value;

template <typename DatasetT>
struct is_standard_dataset : std::false_type {};
template <typename T, typename IdxT, typename SpecT>
struct is_standard_dataset<dataset<T, IdxT, SpecT>>
  : std::bool_constant<is_standard_spec_v<SpecT>> {};
template <typename T, typename IdxT, typename SpecT>
struct is_standard_dataset<dataset_view<T, IdxT, SpecT>>
  : std::bool_constant<is_standard_spec_v<SpecT>> {};
template <typename DatasetT>
inline constexpr bool is_standard_dataset_v = is_standard_dataset<DatasetT>::value;

template <typename DatasetT>
struct is_vpq_dataset : std::false_type {};
template <typename T, typename IdxT, typename SpecT>
struct is_vpq_dataset<dataset<T, IdxT, SpecT>> : std::bool_constant<is_vpq_spec_v<SpecT>> {};
template <typename DatasetT>
inline constexpr bool is_vpq_dataset_v = is_vpq_dataset<DatasetT>::value;

// -----------------------------------------------------------------------------
// Dataset view compile-time classification (replaces runtime std::variant dispatch).
// -----------------------------------------------------------------------------

/** Any non-owning dataset view exposing row count and logical dimension. */
template <typename V, typename IdxT = int64_t>
concept ann_dataset_view = requires(V const& v) {
  { v.n_rows() } -> std::convertible_to<IdxT>;
  { v.dim() } -> std::convertible_to<uint32_t>;
};

enum class dataset_view_kind {
  // TODO(removal): Remove `unknown` once all deprecated host_matrix_view / device_matrix_view /
  // mdspan overloads are deleted. It exists solely so that overload resolution on the deprecated
  // build(host_matrix_view) / build(device_matrix_view) shims does not cause a hard error when
  // the compiler evaluates is_host/device_dataset_view_v for a plain mdspan type.
  unknown,
  empty,
  padded,
  standard,
  vpq_f16,
  vpq_f32,
  bbq,
};

template <typename V>
using dataset_view_type_t = std::remove_cvref_t<V>;

/** Primary template returns `unknown` so traits safely return `false` for non-dataset-view types.
 */
template <typename V>
struct dataset_view_kind_of {
  static constexpr dataset_view_kind value = dataset_view_kind::unknown;
};

template <typename T, typename IdxT, typename SpecT>
struct dataset_view_kind_of<dataset_view<T, IdxT, SpecT>> {
  static constexpr dataset_view_kind value = []() constexpr {
    if constexpr (is_empty_spec_v<SpecT>) {
      return dataset_view_kind::empty;
    } else if constexpr (is_padded_spec_v<SpecT>) {
      return dataset_view_kind::padded;
    } else if constexpr (is_standard_spec_v<SpecT>) {
      return dataset_view_kind::standard;
    } else if constexpr (is_vpq_spec_v<SpecT>) {
      static_assert(std::is_same_v<vpq_spec_math_type_t<SpecT>, half> ||
                      std::is_same_v<vpq_spec_math_type_t<SpecT>, float>,
                    "VPQ dataset_view_kind_of expects MathT to be half or float");
      return std::is_same_v<vpq_spec_math_type_t<SpecT>, half> ? dataset_view_kind::vpq_f16
                                                               : dataset_view_kind::vpq_f32;
    } else {
      return dataset_view_kind::unknown;
    }
  }();
};

/** True when the dataset view accessor is device-accessible. */
template <typename V>
struct dataset_view_is_device_accessible : std::false_type {};

template <typename T, typename IdxT, typename SpecT>
struct dataset_view_is_device_accessible<dataset_view<T, IdxT, SpecT>>
  : std::bool_constant<SpecT::accessor_type::is_device_accessible> {};

template <typename V>
inline constexpr bool dataset_view_is_device_accessible_v =
  dataset_view_is_device_accessible<dataset_view_type_t<V>>::value;

template <typename V>
inline constexpr dataset_view_kind dataset_view_kind_v =
  dataset_view_kind_of<dataset_view_type_t<V>>::value;

template <typename V>
inline constexpr bool is_device_empty_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::empty && dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_host_empty_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::empty && !dataset_view_is_device_accessible_v<V>;

/** True for any empty dataset view (device or host). */
template <typename V>
inline constexpr bool is_empty_dataset_view_v =
  is_device_empty_dataset_view_v<V> || is_host_empty_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_padded_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::padded && dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_host_padded_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::padded && !dataset_view_is_device_accessible_v<V>;

/** True for either `device_padded_dataset_view` or `host_padded_dataset_view`. */
template <typename V>
inline constexpr bool is_padded_dataset_view_v =
  is_device_padded_dataset_view_v<V> || is_host_padded_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_standard_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::standard && dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_host_standard_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::standard && !dataset_view_is_device_accessible_v<V>;

/** True for either `device_standard_dataset_view` or `host_standard_dataset_view`. */
template <typename V>
inline constexpr bool is_standard_dataset_view_v =
  is_device_standard_dataset_view_v<V> || is_host_standard_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_vpq_f16_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::vpq_f16 && dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_host_vpq_f16_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::vpq_f16 && !dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_vpq_f16_dataset_view_v =
  is_device_vpq_f16_dataset_view_v<V> || is_host_vpq_f16_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_vpq_f32_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::vpq_f32 && dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_host_vpq_f32_dataset_view_v =
  dataset_view_kind_v<V> == dataset_view_kind::vpq_f32 && !dataset_view_is_device_accessible_v<V>;

template <typename V>
inline constexpr bool is_vpq_f32_dataset_view_v =
  is_device_vpq_f32_dataset_view_v<V> || is_host_vpq_f32_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_device_vpq_dataset_view_v =
  is_device_vpq_f16_dataset_view_v<V> || is_device_vpq_f32_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_host_vpq_dataset_view_v =
  is_host_vpq_f16_dataset_view_v<V> || is_host_vpq_f32_dataset_view_v<V>;

template <typename V>
inline constexpr bool is_vpq_dataset_view_v =
  is_device_vpq_dataset_view_v<V> || is_host_vpq_dataset_view_v<V>;

/** True for any device-resident dataset view. */
template <typename V>
inline constexpr bool is_device_dataset_view_v =
  dataset_view_kind_v<V> != dataset_view_kind::unknown && dataset_view_is_device_accessible_v<V>;

/** True for any host-resident dataset view. */
template <typename V>
inline constexpr bool is_host_dataset_view_v =
  dataset_view_kind_v<V> != dataset_view_kind::unknown && !dataset_view_is_device_accessible_v<V>;

/**
 * True when a host view `H` and device view `D` represent the same storage kind and differ
 * only in residency (host vs. device). Used by host/device conversion helpers.
 */
template <typename HostViewT, typename DeviceViewT>
inline constexpr bool compatible_host_device_dataset_views_v =
  is_host_dataset_view_v<HostViewT> && is_device_dataset_view_v<DeviceViewT> &&
  (dataset_view_kind_v<HostViewT> == dataset_view_kind_v<DeviceViewT>);

/**
 * Generic accessor retargeting while preserving the dataset tag/layout and value/index types:
 * `dataset<T, IdxT, SpecT<..., OldAccessor>>      -> dataset<T, IdxT, SpecT<..., NewAccessor>>`
 * `dataset_view<T, IdxT, SpecT<..., OldAccessor>> -> dataset_view<T, IdxT, SpecT<...,
 * NewAccessor>>`
 */
template <typename DatasetLikeT, typename NewAccessor>
struct with_accessor;

template <typename T, typename IdxT, typename NewAccessor>
struct with_accessor<dataset<T, IdxT, empty_dataset_spec<NewAccessor>>, NewAccessor> {
  using type = dataset<T, IdxT, empty_dataset_spec<NewAccessor>>;
};

template <typename T, typename IdxT, typename OldAccessor, typename NewAccessor>
struct with_accessor<dataset<T, IdxT, padded_dataset_spec<OldAccessor>>, NewAccessor> {
  using type = dataset<T, IdxT, padded_dataset_spec<NewAccessor>>;
};

template <typename T, typename IdxT, typename OldAccessor, typename NewAccessor>
struct with_accessor<dataset<T, IdxT, standard_dataset_spec<OldAccessor>>, NewAccessor> {
  using type = dataset<T, IdxT, standard_dataset_spec<NewAccessor>>;
};

template <typename T, typename IdxT, typename MathT, typename OldAccessor, typename NewAccessor>
struct with_accessor<dataset<T, IdxT, vpq_dataset_spec<MathT, OldAccessor>>, NewAccessor> {
  using type = dataset<T, IdxT, vpq_dataset_spec<MathT, NewAccessor>>;
};

template <typename T, typename IdxT, typename OldAccessor, typename NewAccessor>
struct with_accessor<dataset_view<T, IdxT, empty_dataset_spec<OldAccessor>>, NewAccessor> {
  using type = dataset_view<T, IdxT, empty_dataset_spec<NewAccessor>>;
};

template <typename T, typename IdxT, typename OldAccessor, typename NewAccessor>
struct with_accessor<dataset_view<T, IdxT, padded_dataset_spec<OldAccessor>>, NewAccessor> {
  using type = dataset_view<T, IdxT, padded_dataset_spec<NewAccessor>>;
};

template <typename T, typename IdxT, typename OldAccessor, typename NewAccessor>
struct with_accessor<dataset_view<T, IdxT, standard_dataset_spec<OldAccessor>>, NewAccessor> {
  using type = dataset_view<T, IdxT, standard_dataset_spec<NewAccessor>>;
};

template <typename T, typename IdxT, typename MathT, typename OldAccessor, typename NewAccessor>
struct with_accessor<dataset_view<T, IdxT, vpq_dataset_spec<MathT, OldAccessor>>, NewAccessor> {
  using type = dataset_view<T, IdxT, vpq_dataset_spec<MathT, NewAccessor>>;
};

template <typename DatasetLikeT, typename NewAccessor>
using with_accessor_t =
  typename with_accessor<dataset_view_type_t<DatasetLikeT>, NewAccessor>::type;

/** Map any host accessor to its device counterpart (same payload policy). */
template <typename Accessor>
struct to_device_accessor {
  using type = Accessor;
};

template <typename T>
struct to_device_accessor<detail::host_view_accessor<T>> {
  using type = detail::device_view_accessor<T>;
};

template <typename T>
struct to_device_accessor<detail::host_owning_accessor<T>> {
  using type = detail::device_owning_accessor<T>;
};

template <typename Accessor>
using to_device_accessor_t = typename to_device_accessor<Accessor>::type;

/** Maps a host dataset view type to its device-resident counterpart. */
template <typename HostViewT>
struct device_counterpart;

template <typename T, typename IdxT, typename SpecT>
struct device_counterpart<dataset_view<T, IdxT, SpecT>> {
  using type = with_accessor_t<dataset_view<T, IdxT, SpecT>,
                               to_device_accessor_t<typename SpecT::accessor_type>>;
};

template <typename HostViewT>
using device_counterpart_t = typename device_counterpart<dataset_view_type_t<HostViewT>>::type;

/** True for device padded or standard views accepted by dense graph build (VPQ excluded). */
template <typename V>
inline constexpr bool is_dense_row_major_device_dataset_view_v =
  is_device_padded_dataset_view_v<V> || is_device_standard_dataset_view_v<V>;

/** True for host or device padded/standard views (dense graph build; VPQ excluded). */
template <typename V>
inline constexpr bool is_dense_row_major_dataset_view_v =
  is_padded_dataset_view_v<V> || is_standard_dataset_view_v<V>;

/** Element type `T` for `cagra::build(res, params, dataset_view)` (deduced, not a template arg).
 * Trivial under the Spec design: every `dataset_view<T,IdxT,SpecT>` already carries `T` directly.
 */
template <typename V>
using cagra_view_element_type_t = typename dataset_view_type_t<V>::value_type;

// -----------------------------------------------------------------------------
// CAGRA row width in elements (same for make_device_padded_dataset* and index layout checks).
// -----------------------------------------------------------------------------

/**
 * @brief Required row width in elements for CAGRA: minimum leading dimension (LDA) per row for the
 *        default per-row byte alignment (16 bytes, combined with `sizeof` element type), given
 *        `logical_columns` feature columns.
 */
[[nodiscard]] inline uint32_t cagra_required_row_width(uint32_t logical_columns,
                                                       std::size_t sizeof_value,
                                                       uint32_t align_bytes = 16)
{
  return static_cast<uint32_t>(
    raft::round_up_safe<std::size_t>(static_cast<std::size_t>(logical_columns) * sizeof_value,
                                     std::lcm(align_bytes, static_cast<uint32_t>(sizeof_value))) /
    sizeof_value);
}

template <typename ValueT>
[[nodiscard]] inline uint32_t cagra_required_row_width(uint32_t logical_columns,
                                                       uint32_t align_bytes = 16)
{
  return cagra_required_row_width(logical_columns, sizeof(ValueT), align_bytes);
}

/** Actual row width in elements (leading dimension) of a 2D row-major matrix view. */
template <typename T, typename I, typename L>
[[nodiscard]] inline uint32_t matrix_actual_row_width(raft::device_matrix_view<T, I, L> m)
{
  return m.stride(0) > 0 ? static_cast<uint32_t>(m.stride(0)) : static_cast<uint32_t>(m.extent(1));
}

template <typename T, typename I, typename L>
[[nodiscard]] inline uint32_t matrix_actual_row_width(raft::host_matrix_view<T, I, L> m)
{
  return m.stride(0) > 0 ? static_cast<uint32_t>(m.stride(0)) : static_cast<uint32_t>(m.extent(1));
}

/**
 * @brief True if the matrix's row width in elements matches `cagra_required_row_width` for
 *        `m.extent(1)` and element type `T` (CAGRA row layout is satisfied for this view).
 */
template <typename T, typename I, typename L>
[[nodiscard]] inline bool matrix_row_width_matches_cagra_required(
  raft::device_matrix_view<T, I, L> m, uint32_t align_bytes = 16)
{
  using value_type = std::remove_const_t<T>;
  const uint32_t need =
    cagra_required_row_width<value_type>(static_cast<uint32_t>(m.extent(1)), align_bytes);
  return matrix_actual_row_width(m) == need;
}

template <typename T, typename I, typename L>
[[nodiscard]] inline bool matrix_row_width_matches_cagra_required(raft::host_matrix_view<T, I, L> m,
                                                                  uint32_t align_bytes = 16)
{
  using value_type = std::remove_const_t<T>;
  const uint32_t need =
    cagra_required_row_width<value_type>(static_cast<uint32_t>(m.extent(1)), align_bytes);
  return matrix_actual_row_width(m) == need;
}

namespace detail {

template <typename SrcT>
[[nodiscard]] inline uint32_t mdspan_row_stride_elements(SrcT const& src)
{
  return src.stride(0) > 0 ? static_cast<uint32_t>(src.stride(0))
                           : static_cast<uint32_t>(src.extent(1));
}

template <typename ValueT, typename SrcT>
[[nodiscard]] inline ValueT* expect_device_accessible_data_handle(SrcT const& src,
                                                                  char const* error_msg)
{
  cudaPointerAttributes ptr_attrs;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&ptr_attrs, src.data_handle()));
  // `devicePointer` is relative to the *current* device: it is null for an allocation owned by
  // another device without peer access, even though that allocation is perfectly usable once the
  // caller switches to the owning device (as the multi-GPU paths do). Accept device and managed
  // allocations on their own merit and only consult `devicePointer` for host memory, which needs a
  // mapping to be reachable at all.
  if (ptr_attrs.type == cudaMemoryTypeDevice || ptr_attrs.type == cudaMemoryTypeManaged) {
    return const_cast<ValueT*>(src.data_handle());
  }
  auto* device_ptr = reinterpret_cast<ValueT*>(ptr_attrs.devicePointer);
  RAFT_EXPECTS(device_ptr != nullptr, "%s", error_msg);
  return device_ptr;
}

template <typename ValueT, typename IndexT, typename ViewT, typename SrcT>
[[nodiscard]] inline ViewT make_device_dense_row_major_view_from_src(SrcT const& src,
                                                                     uint32_t logical_dim)
{
  auto* device_ptr = expect_device_accessible_data_handle<ValueT>(
    src, "make_device_*_dataset_view: source must be device-accessible.");
  auto v = raft::make_device_matrix_view(
    device_ptr, src.extent(0), static_cast<IndexT>(mdspan_row_stride_elements(src)));
  return ViewT(v, logical_dim);
}

template <typename ValueT, typename IndexT, typename ViewT, typename SrcT>
[[nodiscard]] inline ViewT make_host_dense_row_major_view_from_src(SrcT const& src,
                                                                   uint32_t logical_dim)
{
  RAFT_EXPECTS(raft::get_device_for_address(src.data_handle()) == -1,
               "make_host_*_dataset_view: source must be host-accessible.");
  auto v = raft::make_host_matrix_view(const_cast<ValueT*>(src.data_handle()),
                                       src.extent(0),
                                       static_cast<IndexT>(mdspan_row_stride_elements(src)));
  return ViewT(v, logical_dim);
}

template <typename DatasetT, typename ValueT, typename IndexT, typename SrcT>
auto make_device_dense_row_major_dataset_from_src(raft::resources const& res,
                                                  SrcT const& src,
                                                  uint32_t logical_dim,
                                                  uint32_t target_stride,
                                                  char const* view_factory_name)
  -> std::unique_ptr<DatasetT>
{
  uint32_t const src_stride = mdspan_row_stride_elements(src);
  RAFT_EXPECTS(logical_dim <= target_stride,
               "logical dim (%u) must not exceed row stride (%u).",
               static_cast<unsigned>(logical_dim),
               static_cast<unsigned>(target_stride));
  RAFT_EXPECTS(static_cast<uint32_t>(src.extent(1)) <= target_stride,
               "Source row length must not exceed required stride.");
  cudaPointerAttributes ptr_attrs;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&ptr_attrs, src.data_handle()));
  bool const device_src =
    (ptr_attrs.type == cudaMemoryTypeDevice) || (ptr_attrs.type == cudaMemoryTypeManaged);
  if (device_src && src_stride == target_stride) {
    RAFT_EXPECTS(false,
                 "source is device and stride is already correct. "
                 "Use %s() to get a view instead.",
                 view_factory_name);
  }
  auto out_array = raft::make_device_matrix<ValueT, IndexT>(res, src.extent(0), target_stride);
  RAFT_CUDA_TRY(cudaMemsetAsync(out_array.data_handle(),
                                0,
                                out_array.size() * sizeof(ValueT),
                                raft::resource::get_cuda_stream(res).get()));
  raft::copy_matrix(out_array.data_handle(),
                    target_stride,
                    src.data_handle(),
                    src_stride,
                    logical_dim,
                    src.extent(0),
                    raft::resource::get_cuda_stream(res));
  return std::make_unique<DatasetT>(std::move(out_array), logical_dim);
}

template <typename DatasetT, typename ValueT, typename IndexT, typename SrcT>
auto make_host_dense_row_major_dataset_from_src(raft::resources const& res,
                                                SrcT const& src,
                                                uint32_t logical_dim,
                                                uint32_t target_stride,
                                                char const* view_factory_name)
  -> std::unique_ptr<DatasetT>
{
  uint32_t const src_stride = mdspan_row_stride_elements(src);
  constexpr bool device_src = SrcT::accessor_type::is_device_accessible;
  RAFT_EXPECTS(logical_dim <= target_stride,
               "logical dim (%u) must not exceed row stride (%u).",
               static_cast<unsigned>(logical_dim),
               static_cast<unsigned>(target_stride));
  if (!device_src && src_stride == target_stride) {
    RAFT_EXPECTS(false,
                 "source stride is already correct. Use %s() to get a view instead.",
                 view_factory_name);
  }
  RAFT_EXPECTS(static_cast<uint32_t>(src.extent(1)) <= target_stride,
               "Source row length must not exceed required stride.");
  auto out_array = raft::make_host_matrix<ValueT, IndexT>(src.extent(0), target_stride);
  std::memset(out_array.data_handle(), 0, out_array.size() * sizeof(ValueT));
  raft::copy_matrix(out_array.data_handle(),
                    target_stride,
                    src.data_handle(),
                    src_stride,
                    logical_dim,
                    src.extent(0),
                    raft::resource::get_cuda_stream(res));
  if (device_src) { raft::resource::sync_stream(res); }
  return std::make_unique<DatasetT>(std::move(out_array), logical_dim);
}

}  // namespace detail

template <typename SrcT>
auto make_device_padded_dataset_view(const raft::resources& res,
                                     SrcT const& src,
                                     uint32_t align_bytes = 16)
  -> device_padded_dataset_view<typename SrcT::value_type, typename SrcT::index_type>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  uint32_t required_stride =
    cagra_required_row_width<value_type>(static_cast<uint32_t>(src.extent(1)), align_bytes);
  RAFT_EXPECTS(
    detail::mdspan_row_stride_elements(src) == required_stride,
    "make_device_padded_dataset_view: stride is incorrect (required stride for alignment). "
    "Use make_device_padded_dataset() to get an owning padded copy.");
  return detail::make_device_dense_row_major_view_from_src<
    value_type,
    index_type,
    device_padded_dataset_view<value_type, index_type>>(src, static_cast<uint32_t>(src.extent(1)));
}

template <typename SrcT>
auto make_device_padded_dataset(const raft::resources& res,
                                SrcT const& src,
                                uint32_t align_bytes = 16)
  -> std::unique_ptr<device_padded_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using value_type               = typename SrcT::value_type;
  using index_type               = typename SrcT::index_type;
  uint32_t const logical_dim     = static_cast<uint32_t>(src.extent(1));
  uint32_t const required_stride = cagra_required_row_width<value_type>(logical_dim, align_bytes);
  return detail::make_device_dense_row_major_dataset_from_src<
    device_padded_dataset<value_type, index_type>,
    value_type,
    index_type>(res, src, logical_dim, required_stride, "make_device_padded_dataset_view");
}

template <typename SrcT>
auto make_host_padded_dataset_view(SrcT const& src, uint32_t align_bytes = 16)
  -> host_padded_dataset_view<typename SrcT::value_type, typename SrcT::index_type>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  uint32_t required_stride =
    cagra_required_row_width<value_type>(static_cast<uint32_t>(src.extent(1)), align_bytes);
  RAFT_EXPECTS(
    detail::mdspan_row_stride_elements(src) == required_stride,
    "make_host_padded_dataset_view: stride is incorrect (required stride for alignment). "
    "Use make_host_padded_dataset() to get an owning padded copy.");
  return detail::make_host_dense_row_major_view_from_src<
    value_type,
    index_type,
    host_padded_dataset_view<value_type, index_type>>(src, static_cast<uint32_t>(src.extent(1)));
}

template <typename SrcT>
auto make_host_padded_dataset(const raft::resources& res,
                              SrcT const& src,
                              uint32_t align_bytes = 16)
  -> std::unique_ptr<host_padded_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using value_type               = typename SrcT::value_type;
  using index_type               = typename SrcT::index_type;
  uint32_t const logical_dim     = static_cast<uint32_t>(src.extent(1));
  uint32_t const required_stride = cagra_required_row_width<value_type>(logical_dim, align_bytes);
  return detail::make_host_dense_row_major_dataset_from_src<
    host_padded_dataset<value_type, index_type>,
    value_type,
    index_type>(res, src, logical_dim, required_stride, "make_host_padded_dataset_view");
}

template <typename SrcT>
auto make_device_standard_dataset_view(SrcT const& src)
  -> device_standard_dataset_view<typename SrcT::value_type, typename SrcT::index_type>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  return detail::make_device_dense_row_major_view_from_src<
    value_type,
    index_type,
    device_standard_dataset_view<value_type, index_type>>(src,
                                                          static_cast<uint32_t>(src.extent(1)));
}

/**
 * @brief Create an owning device standard dataset with explicit row layout.
 *
 * Internal use only: the sole call site today is
 * `cuvs::neighbors::detail::deserialize_standard()` in `dataset_serialize.hpp`, which must pass
 * wire-format `(logical_dim, stride)` because the deserialized host buffer is tight `[n_rows x
 * dim]` while the on-disk stride may be larger. Do not call from user code; prefer
 * `make_device_standard_dataset_view()` when wrapping existing correctly-strided storage.
 */
template <typename SrcT>
auto make_device_standard_dataset(const raft::resources& res,
                                  SrcT const& src,
                                  uint32_t logical_dim,
                                  uint32_t target_stride)
  -> std::unique_ptr<device_standard_dataset<typename SrcT::value_type, typename SrcT::index_type>>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  return detail::make_device_dense_row_major_dataset_from_src<
    device_standard_dataset<value_type, index_type>,
    value_type,
    index_type>(res, src, logical_dim, target_stride, "make_device_standard_dataset_view");
}

template <typename SrcT>
auto make_host_standard_dataset_view(SrcT const& src)
  -> host_standard_dataset_view<typename SrcT::value_type, typename SrcT::index_type>
{
  using value_type = typename SrcT::value_type;
  using index_type = typename SrcT::index_type;
  return detail::make_host_dense_row_major_view_from_src<
    value_type,
    index_type,
    host_standard_dataset_view<value_type, index_type>>(src, static_cast<uint32_t>(src.extent(1)));
}

}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
