---
slug: api-reference/cpp-api-core-dataset
---

# Dataset

_Source header: `cuvs/core/dataset.hpp`_

## Types

<a id="neighbors-dataset"></a>
### neighbors::dataset

Spec-based `dataset` / `dataset_view`.

`dataset&lt;T,IdxT,SpecT&gt;` and `dataset_view&lt;T,IdxT,SpecT&gt;` are single generic templates that know nothing about any particular kind of dataset. They hold exactly one payload (`data_type` / `view_type`, chosen by the spec) and expose only what every dataset has: `n_rows()`, `dim()`, `as_matrix_view()`, `as_dataset_view()` and `data()`. Each is a one-line forward to one of the three spec functions `get_data_view()`, `get_n_rows()` and `get_dim()`. Anything else a kind needs (e.g. VPQ codebooks, BBQ quantizers) is state and methods of that kind's payload type, reached through `data()`; `dataset`/`dataset_view` never name or branch on it. `dataset` and `dataset_view` are deliberately two independent, non-inheriting types (no shared_ptr, no "sometimes owning" object): `dataset` holds the owning payload, `dataset_view` the corresponding non-owning payload.

```cpp
template <typename T, typename IdxT, typename SpecT>
struct dataset;
```

<a id="neighbors-vpq-dataset-spec"></a>
### neighbors::vpq_dataset_spec

`Accessor` drives both codebook and code residency, mirroring today's

single-`Accessor`-per-VPQ-dataset design. The payload (`detail::vpq_owning_storage` / `detail::vpq_view_storage`) holds the encoded rows and the VQ/PQ codebooks.

```cpp
template <typename MathT, typename Accessor>
struct vpq_dataset_spec;
```

<a id="neighbors-dataset"></a>
### neighbors::dataset

Owning dataset: value-held payload (no shared_ptr -- exclusive ownership). Every member is a

one-line forward to `spec_type::get_*` or to the payload; all per-kind state and logic lives in the spec's `data_type`, never inside this struct.

```cpp
template <typename T, typename IdxT, typename SpecT>
struct dataset;
```

<a id="neighbors-dataset-view"></a>
### neighbors::dataset_view

Non-owning dataset view: holds only the view-shaped payload. Deliberately not derived from

`dataset` -- a view type holds "all view state" with no inheritance and no shared ownership tying it to the owning type. Reuses the same `get_n_rows`/`get_dim` spec functions as `dataset`, fed the view payload instead of the owning one.

```cpp
template <typename T, typename IdxT, typename SpecT>
struct dataset_view;
```

<a id="neighbors-is-padded-dataset"></a>
### neighbors::is_padded_dataset

Owning-side kind traits (mirror today's `is_padded_dataset_v`/`is_standard_dataset_v`/

`is_vpq_dataset_v`, used for SFINAE overload selection in factory.cuh/compute_distance_vpq.hpp).

```cpp
template <typename DatasetT>
struct is_padded_dataset;
```

<a id="neighbors-dataset-view-kind-of"></a>
### neighbors::dataset_view_kind_of

Primary template returns `unknown` so traits safely return `false` for non-dataset-view types.

```cpp
template <typename V>
struct dataset_view_kind_of {
  static constexpr dataset_view_kind value;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `value` | `static constexpr dataset_view_kind` |  |

<a id="neighbors-dataset-view-is-device-accessible"></a>
### neighbors::dataset_view_is_device_accessible

True when the dataset view accessor is device-accessible.

```cpp
template <typename V>
struct dataset_view_is_device_accessible;
```

<a id="neighbors-with-accessor"></a>
### neighbors::with_accessor

Generic accessor retargeting while preserving the dataset tag/layout and value/index types:

`dataset&lt;T, IdxT, SpecT&lt;..., OldAccessor&gt;&gt;      -&gt; dataset&lt;T, IdxT, SpecT&lt;..., NewAccessor&gt;&gt;` `dataset_view&lt;T, IdxT, SpecT&lt;..., OldAccessor&gt;&gt; -&gt; dataset_view&lt;T, IdxT, SpecT&lt;..., NewAccessor&gt;&gt;`

```cpp
template <typename DatasetLikeT, typename NewAccessor>
struct with_accessor;
```

<a id="neighbors-to-device-accessor"></a>
### neighbors::to_device_accessor

Map any host accessor to its device counterpart (same payload policy).

```cpp
template <typename Accessor>
struct to_device_accessor;
```

<a id="neighbors-device-counterpart"></a>
### neighbors::device_counterpart

Maps a host dataset view type to its device-resident counterpart.

```cpp
template <typename HostViewT>
struct device_counterpart;
```
