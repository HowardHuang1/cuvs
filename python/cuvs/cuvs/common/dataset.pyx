#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

import numpy as np

from libcpp cimport bool as cbool

cimport cuvs.common.cydlpack
from cuvs.common cimport cydlpack
from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t

from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible

from cuvs.common.exceptions import check_cuvs, get_last_error_text
from cuvs.common.resources import auto_sync_resources


cdef class Dataset:
    """Wrapper around a ``cuvsDataset`` handle."""

    def __cinit__(self):
        self.dataset = NULL
        self._source = None

    def __dealloc__(self):
        if self.dataset != NULL:
            check_cuvs(cuvsDatasetDestroy(self.dataset))

    @property
    def memory_type(self):
        cdef cuvsDatasetMemType_t mem_type
        if self.dataset == NULL:
            return None
        check_cuvs(cuvsDatasetGetMemType(self.dataset, &mem_type))
        if mem_type == CUVS_DATASET_MEM_TYPE_DEVICE:
            return "device"
        return "host"

    @property
    def layout(self):
        cdef cuvsDatasetLayout_t layout
        if self.dataset == NULL:
            return None
        check_cuvs(cuvsDatasetGetLayout(self.dataset, &layout))
        if layout == CUVS_DATASET_LAYOUT_PADDED:
            return "padded"
        return "standard"

    @property
    def is_owning(self):
        cdef cbool owning
        if self.dataset == NULL:
            return None
        check_cuvs(cuvsDatasetGetIsOwning(self.dataset, &owning))
        return owning != 0

    @property
    def dtype(self):
        cdef DLDataType dtype
        if self.dataset == NULL:
            return None
        check_cuvs(cuvsDatasetGetDtype(self.dataset, &dtype))
        return (dtype.code, dtype.bits, dtype.lanes)


cdef Dataset make_device_padded_dataset_handle(
        cuvsResources_t res,
        cydlpack.DLManagedTensor* dataset_dlpack):
    cdef Dataset padded = Dataset()
    cdef cuvsError_t status = cuvsDatasetMakePadded(
        res,
        dataset_dlpack,
        CUVS_DATASET_MEM_TYPE_DEVICE,
        &padded.dataset
    )
    if status == cuvsError_t.CUVS_SUCCESS:
        return padded
    err = get_last_error_text() or ""
    if "stride is already correct" not in err:
        check_cuvs(status)
    check_cuvs(cuvsDatasetMakePaddedView(
        res,
        dataset_dlpack,
        &padded.dataset
    ))
    return padded


def _check_dataset_array(dataset_ai):
    if dataset_ai.dtype not in (np.dtype('float32'),
                                np.dtype('float16'),
                                np.dtype('byte'),
                                np.dtype('ubyte')):
        raise TypeError("dtype %s not supported" % dataset_ai.dtype)
    if not dataset_ai.c_contiguous:
        raise ValueError("Row major input is expected")


@auto_sync_resources
def make_device_padded_dataset(dataset, resources=None):
    """Create a device-padded dataset from an array."""
    dataset_ai = wrap_array(dataset)
    _check_dataset_array(dataset_ai)
    cdef cydlpack.DLManagedTensor* dataset_dlpack = \
        cydlpack.dlpack_c(dataset_ai)
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    cdef Dataset padded
    with cuda_interruptible():
        padded = make_device_padded_dataset_handle(res, dataset_dlpack)
    if not padded.is_owning:
        padded._source = dataset
    return padded
