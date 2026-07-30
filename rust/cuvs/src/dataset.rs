/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::marker::PhantomData;

use crate::dlpack::{AsDlTensor, DeviceTypeExt};
use crate::error::check_cuvs;
use crate::ffi_utils::{init_handle, report_drop_failure};
use crate::neighbors::cagra::CagraError;
use crate::resources::Resources;

type Result<T> = std::result::Result<T, CagraError>;

/// Host/device residency and row layout of a [`DatasetView`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DatasetKind {
    /// Device-resident rows with CAGRA's required padded width.
    DevicePadded,
    /// Device-resident rows with a standard, unpadded width.
    DeviceStandard,
    /// Host-resident rows with CAGRA's required padded width.
    HostPadded,
    /// Host-resident rows with a standard, unpadded width.
    HostStandard,
}

impl DatasetKind {
    fn from_ffi(mem_type: ffi::cuvsDatasetMemType_t, layout: ffi::cuvsDatasetLayout_t) -> Self {
        match (mem_type, layout) {
            (
                ffi::cuvsDatasetMemType_t::CUVS_DATASET_MEM_TYPE_DEVICE,
                ffi::cuvsDatasetLayout_t::CUVS_DATASET_LAYOUT_PADDED,
            ) => Self::DevicePadded,
            (
                ffi::cuvsDatasetMemType_t::CUVS_DATASET_MEM_TYPE_DEVICE,
                ffi::cuvsDatasetLayout_t::CUVS_DATASET_LAYOUT_STANDARD,
            ) => Self::DeviceStandard,
            (
                ffi::cuvsDatasetMemType_t::CUVS_DATASET_MEM_TYPE_HOST,
                ffi::cuvsDatasetLayout_t::CUVS_DATASET_LAYOUT_PADDED,
            ) => Self::HostPadded,
            (
                ffi::cuvsDatasetMemType_t::CUVS_DATASET_MEM_TYPE_HOST,
                ffi::cuvsDatasetLayout_t::CUVS_DATASET_LAYOUT_STANDARD,
            ) => Self::HostStandard,
        }
    }
}

/// A non-owning CAGRA dataset view.
///
/// The view records the storage's residency and layout while borrowing its
/// backing tensor for `'a`. Constructing a view allocates only native metadata;
/// it never copies vector storage.
#[derive(Debug)]
pub struct DatasetView<'a> {
    handle: ffi::cuvsDataset_t,
    kind: DatasetKind,
    destroy_handle: bool,
    _dataset: PhantomData<&'a ()>,
}

impl<'a> DatasetView<'a> {
    /// Borrow a tensor as the host/device and padded/standard view CAGRA
    /// derives from its DLPack metadata.
    pub fn new<T>(res: &Resources, dataset: &'a T) -> Result<Self>
    where
        T: AsDlTensor + ?Sized,
    {
        let dataset = dataset.as_dl_tensor()?;
        let mut dataset_c = dataset.to_c();
        unsafe {
            let mut mem_type = std::mem::MaybeUninit::<ffi::cuvsDatasetMemType_t>::uninit();
            let mut layout = std::mem::MaybeUninit::<ffi::cuvsDatasetLayout_t>::uninit();
            check_cuvs(ffi::cuvsCagraGetDatasetMemTypeAndLayout(
                dataset_c.as_mut_ptr(),
                mem_type.as_mut_ptr(),
                layout.as_mut_ptr(),
            ))?;
            let kind = DatasetKind::from_ffi(mem_type.assume_init(), layout.assume_init());

            // Unified factories infer host vs device from the tensor; only
            // layout selects padded vs standard.
            let handle = init_handle(|out| match kind {
                DatasetKind::DevicePadded | DatasetKind::HostPadded => {
                    ffi::cuvsDatasetMakePaddedView(res.handle(), dataset_c.as_mut_ptr(), out)
                }
                DatasetKind::DeviceStandard | DatasetKind::HostStandard => {
                    ffi::cuvsDatasetMakeStandardView(res.handle(), dataset_c.as_mut_ptr(), out)
                }
            })?;
            Ok(Self { handle, kind, destroy_handle: true, _dataset: PhantomData })
        }
    }

    /// Return this view's immutable residency/layout classification.
    pub fn kind(&self) -> DatasetKind {
        self.kind
    }

    pub(crate) fn raw(&self) -> ffi::cuvsDataset_t {
        self.handle
    }
}

impl Drop for DatasetView<'_> {
    fn drop(&mut self) {
        if self.destroy_handle {
            if let Err(e) = check_cuvs(unsafe { ffi::cuvsDatasetDestroy(self.handle) }) {
                report_drop_failure("dataset view", &e);
            }
        }
    }
}

/// Storage owned by the caller, padded to CAGRA's required row width.
///
/// Construction performs an explicit allocation and copy. Memory residency is
/// inferred from the source tensor; use [`DatasetView::new`] when its existing
/// layout is already suitable.
#[derive(Debug)]
pub struct PaddedDataset {
    handle: ffi::cuvsDataset_t,
    kind: DatasetKind,
}

impl PaddedDataset {
    /// Copy a tensor into freshly allocated, CAGRA-padded storage.
    pub fn new<T>(res: &Resources, dataset: &T) -> Result<Self>
    where
        T: AsDlTensor + ?Sized,
    {
        let dataset = dataset.as_dl_tensor()?;
        let mut dataset_c = dataset.to_c();
        let device_type = dataset_c.inner.dl_tensor.device.device_type;
        let kind = if device_type.is_device_compatible() {
            DatasetKind::DevicePadded
        } else {
            DatasetKind::HostPadded
        };
        let target_mem_type = match kind {
            DatasetKind::DevicePadded => ffi::cuvsDatasetMemType_t::CUVS_DATASET_MEM_TYPE_DEVICE,
            DatasetKind::HostPadded => ffi::cuvsDatasetMemType_t::CUVS_DATASET_MEM_TYPE_HOST,
            DatasetKind::DeviceStandard | DatasetKind::HostStandard => unreachable!(),
        };
        unsafe {
            let handle = init_handle(|out| {
                ffi::cuvsDatasetMakePadded(
                    res.handle(),
                    dataset_c.as_mut_ptr(),
                    target_mem_type,
                    out,
                )
            })?;
            Ok(Self { handle, kind })
        }
    }

    /// Borrow this allocation as a padded view.
    pub fn as_view(&self) -> Result<DatasetView<'_>> {
        Ok(DatasetView {
            handle: self.handle,
            kind: self.kind,
            destroy_handle: false,
            _dataset: PhantomData,
        })
    }
}

impl Drop for PaddedDataset {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsDatasetDestroy(self.handle) }) {
            report_drop_failure("padded dataset", &e);
        }
    }
}

/// Owning dataset storage returned by CAGRA deserialization.
///
/// The allocation preserves the serialized host/device residency and
/// standard/padded row layout. CAGRA keeps only a non-owning view, so this
/// owner must remain alive while the deserialized index uses it.
#[derive(Debug)]
pub struct Dataset {
    handle: ffi::cuvsDataset_t,
    kind: DatasetKind,
}

impl Dataset {
    pub(crate) fn from_raw(handle: ffi::cuvsDataset_t) -> Self {
        debug_assert!(!handle.is_null());
        let kind = unsafe { DatasetKind::from_ffi((*handle).mem_type, (*handle).layout) };
        Self { handle, kind }
    }

    /// Return this allocation's immutable residency/layout classification.
    pub fn kind(&self) -> DatasetKind {
        self.kind
    }
}

impl Drop for Dataset {
    fn drop(&mut self) {
        if let Err(e) = check_cuvs(unsafe { ffi::cuvsDatasetDestroy(self.handle) }) {
            report_drop_failure("deserialized dataset", &e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_dataset_accepts_host_storage() {
        let res = Resources::new().unwrap();
        let dataset = ndarray::Array::<f32, _>::zeros((256, 15));

        let owner = PaddedDataset::new(&res, &*dataset).expect("host padded dataset");
        let view = owner.as_view().expect("host padded dataset view");
        assert_eq!(view.kind(), DatasetKind::HostPadded);
    }
}
