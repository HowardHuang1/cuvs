/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/common.h>

#include <dlpack/dlpack.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generic dataset layout kind for C API dataset handles.
 */
typedef enum {
  CUVS_DATASET_LAYOUT_STANDARD = 0,
  CUVS_DATASET_LAYOUT_PADDED   = 1
} cuvsDatasetLayout_t;

/**
 * @brief Memory space holding a C API dataset handle's data.
 */
typedef enum {
  CUVS_DATASET_MEM_TYPE_HOST   = 0,
  CUVS_DATASET_MEM_TYPE_DEVICE = 1
} cuvsDatasetMemType_t;

/**
 * @brief Owning padded dataset handle.
 *
 * `addr` points to C++ owning dataset storage managed by the C API. The layout is implied by
 * the handle type; `mem_type` identifies the memory space.
 */
typedef struct {
  uintptr_t addr;
  void (*destroy_addr)(void*);
  DLDataType dtype;
  cuvsDatasetMemType_t mem_type;
} cuvsDatasetPadded;
typedef cuvsDatasetPadded* cuvsDatasetPadded_t;

/**
 * @brief Owning standard dataset handle.
 *
 * `addr` points to C++ owning dataset storage managed by the C API. The layout is implied by
 * the handle type; `mem_type` identifies the memory space.
 */
typedef struct {
  uintptr_t addr;
  void (*destroy_addr)(void*);
  DLDataType dtype;
  cuvsDatasetMemType_t mem_type;
} cuvsDatasetStandard;
typedef cuvsDatasetStandard* cuvsDatasetStandard_t;

/**
 * @brief Non-owning dataset view handle.
 *
 * `addr` points to C API-owned metadata that references caller-provided tensor memory. The
 * `mem_type` and `layout` fields identify the concrete dataset view type.
 */
typedef struct {
  uintptr_t addr;
  void (*destroy_addr)(void*);
  DLDataType dtype;
  cuvsDatasetMemType_t mem_type;
  cuvsDatasetLayout_t layout;
} cuvsDatasetView;
typedef cuvsDatasetView* cuvsDatasetView_t;

/**
 * @brief Typed aliases for padded and standard non-owning dataset views.
 */
typedef cuvsDatasetView cuvsDatasetPaddedView;
typedef cuvsDatasetPaddedView* cuvsDatasetPaddedView_t;
typedef cuvsDatasetView cuvsDatasetStandardView;
typedef cuvsDatasetStandardView* cuvsDatasetStandardView_t;

/**
 * @brief Generic storage kind for operation-specific dataset storage.
 */
typedef enum {
  CUVS_DATASET_STORAGE_KIND_EXTENDED = 0,
  CUVS_DATASET_STORAGE_KIND_MERGED   = 1
} cuvsDatasetStorageKind_t;

/**
 * @brief Generic opaque storage handle for dataset-backed operation outputs.
 *
 * Used by operations like CAGRA extend/merge where storage shape is identical
 * but semantics differ by operation kind.
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
  cuvsDatasetStorageKind_t kind;
} cuvsDatasetStorage;
typedef cuvsDatasetStorage* cuvsDatasetStorage_t;

typedef struct cuvsCagraIndex* cuvsCagraIndex_t;

CUVS_EXPORT cuvsError_t cuvsDatasetMakeDevicePadded(cuvsResources_t res,
                                                    DLManagedTensor* dataset,
                                                    cuvsDatasetPadded_t* padded_dataset);

CUVS_EXPORT cuvsError_t cuvsDatasetMakeHostPadded(cuvsResources_t res,
                                                  DLManagedTensor* dataset,
                                                  cuvsDatasetPadded_t* padded_dataset);

CUVS_EXPORT cuvsError_t cuvsDatasetMakeDevicePaddedView(cuvsResources_t res,
                                                        DLManagedTensor* dataset,
                                                        cuvsDatasetPaddedView_t* padded_dataset);

/**
 * @brief Create a non-owning device padded view handle from an owning device padded dataset.
 *
 * This is useful when APIs require a padded view handle (e.g. attach-for-search), while callers
 * keep ownership in a padded dataset handle created by `cuvsDatasetMakeDevicePadded`.
 *
 * @param[in] padded_dataset owning device padded dataset handle
 * @param[out] padded_view output padded view handle
 */
CUVS_EXPORT cuvsError_t cuvsDatasetMakeViewFromOwningPadded(
  cuvsDatasetPadded_t padded_dataset, cuvsDatasetPaddedView_t* padded_view);

CUVS_EXPORT cuvsError_t cuvsDatasetMakeHostPaddedView(cuvsResources_t res,
                                                      DLManagedTensor* dataset,
                                                      cuvsDatasetPaddedView_t* padded_dataset);

CUVS_EXPORT cuvsError_t cuvsDatasetPaddedDestroy(cuvsDatasetPadded_t padded_dataset);
CUVS_EXPORT cuvsError_t cuvsDatasetStandardDestroy(cuvsDatasetStandard_t standard_dataset);
CUVS_EXPORT cuvsError_t cuvsDatasetPaddedViewDestroy(cuvsDatasetPaddedView_t padded_dataset);

CUVS_EXPORT cuvsError_t cuvsDatasetMakeDeviceStandardView(cuvsResources_t res,
                                                          DLManagedTensor* dataset,
                                                          cuvsDatasetStandardView_t* standard_dataset);

CUVS_EXPORT cuvsError_t cuvsDatasetMakeHostStandardView(cuvsResources_t res,
                                                        DLManagedTensor* dataset,
                                                        cuvsDatasetStandardView_t* standard_dataset);

CUVS_EXPORT cuvsError_t cuvsDatasetStandardViewDestroy(cuvsDatasetStandardView_t standard_dataset);

CUVS_EXPORT cuvsError_t cuvsMakeMergedStorage(cuvsResources_t res,
                                              cuvsCagraIndex_t* indices,
                                              size_t num_indices,
                                              cuvsFilter filter,
                                              cuvsDatasetStorage_t* merged_storage);

CUVS_EXPORT cuvsError_t cuvsDatasetStorageDestroy(cuvsDatasetStorage_t dataset_storage);

#ifdef __cplusplus
}
#endif
