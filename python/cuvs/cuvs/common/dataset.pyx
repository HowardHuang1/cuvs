#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

import numpy as np

cimport cuvs.common.cydlpack

from cuvs.common cimport cydlpack
from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.exceptions import check_cuvs, get_last_error_text
from cuvs.common.resources import auto_sync_resources

from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible


cdef class Dataset:
    """Wrapper around a ``cuvsDataset`` handle."""

    def __cinit__(self):
        self.dataset = NULL

    def __dealloc__(self):
        if self.dataset != NULL:
            check_cuvs(cuvsDatasetDestroy(self.dataset))

    @property
    def memory_type(self):
        if self.dataset == NULL:
            return None
        if self.dataset.mem_type == CUVS_DATASET_MEM_TYPE_DEVICE:
            return "device"
        return "host"

    @property
    def layout(self):
        if self.dataset == NULL:
            return None
        if self.dataset.layout == CUVS_DATASET_LAYOUT_PADDED:
            return "padded"
        return "standard"

    @property
    def is_owning(self):
        if self.dataset == NULL:
            return None
        return self.dataset.is_owning != 0

    @property
    def dtype(self):
        if self.dataset == NULL:
            return None
        return (self.dataset.dtype.code,
                self.dataset.dtype.bits,
                self.dataset.dtype.lanes)


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
    with cuda_interruptible():
        return make_device_padded_dataset_handle(res, dataset_dlpack)
