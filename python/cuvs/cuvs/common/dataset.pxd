#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libc.stdint cimport uintptr_t
from libcpp cimport bool

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor


cdef extern from "cuvs/core/dataset.h" nogil:
    ctypedef enum cuvsDatasetLayout_t:
        CUVS_DATASET_LAYOUT_STANDARD
        CUVS_DATASET_LAYOUT_PADDED

    ctypedef enum cuvsDatasetMemType_t:
        CUVS_DATASET_MEM_TYPE_HOST
        CUVS_DATASET_MEM_TYPE_DEVICE

    ctypedef struct cuvsDataset:
        uintptr_t addr
        void (*destroy_addr)(void*)
        DLDataType dtype
        cuvsDatasetMemType_t mem_type
        cuvsDatasetLayout_t layout
        bool is_owning
    ctypedef cuvsDataset* cuvsDataset_t

    cuvsError_t cuvsDatasetCreate(cuvsDataset_t* dataset)

    cuvsError_t cuvsDatasetMakePadded(cuvsResources_t res,
                                      DLManagedTensor* dataset,
                                      cuvsDatasetMemType_t target_mem_type,
                                      cuvsDataset_t* padded_dataset)

    cuvsError_t cuvsDatasetMakePaddedView(cuvsResources_t res,
                                          DLManagedTensor* dataset,
                                          cuvsDataset_t* padded_dataset)

    cuvsError_t cuvsDatasetMakeStandardView(cuvsResources_t res,
                                            DLManagedTensor* dataset,
                                            cuvsDataset_t* standard_dataset)

    cuvsError_t cuvsDatasetDestroy(cuvsDataset_t dataset)


cdef class Dataset:
    cdef cuvsDataset_t dataset


cdef Dataset make_device_padded_dataset_handle(
    cuvsResources_t res,
    DLManagedTensor* dataset_dlpack)
