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
    fn from_residency_and_padded(is_device: bool, is_padded: bool) -> Self {
        match (is_device, is_padded) {
            (true, true) => Self::DevicePadded,
            (true, false) => Self::DeviceStandard,
            (false, true) => Self::HostPadded,
            (false, false) => Self::HostStandard,
        }
    }

    fn from_handle_fields(mem_type: ffi::cuvsDatasetMemType_t, layout: ffi::cuvsDatasetLayout_t) -> Self {
        Self::from_residency_and_padded(
            mem_type == ffi::cuvsDatasetMemType_t::CUVS_DATASET_MEM_TYPE_DEVICE,
            layout == ffi::cuvsDatasetLayout_t::CUVS_DATASET_LAYOUT_PADDED,
        )
    }
}

/// Matches C++ `cagra_required_row_width` (16-byte default alignment).
fn cagra_required_row_width(logical_columns: u32, sizeof_value: usize) -> u32 {
    let align_bytes: usize = 16;
    let lcm = lcm(align_bytes, sizeof_value);
    let bytes = (logical_columns as usize) * sizeof_value;
    let rounded = ((bytes + lcm - 1) / lcm) * lcm;
    (rounded / sizeof_value) as u32
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn lcm(a: usize, b: usize) -> usize {
    a / gcd(a, b) * b
}

fn dlpack_element_size(dtype: &ffi::DLDataType) -> Option<usize> {
    match (dtype.code, dtype.bits) {
        (code, 32) if code == ffi::DLDataTypeCode::kDLFloat as u8 => Some(4),
        (code, 16) if code == ffi::DLDataTypeCode::kDLFloat as u8 => Some(2),
        (code, 8)
            if code == ffi::DLDataTypeCode::kDLInt as u8
                || code == ffi::DLDataTypeCode::kDLUInt as u8 =>
        {
            Some(1)
        }
        _ => None,
    }
}

fn is_cagra_padded_layout(tensor: &ffi::DLTensor) -> Result<bool> {
    if tensor.ndim != 2 {
        return Err(CagraError::Validation(
            "CAGRA datasets must be 2-D".to_string(),
        ));
    }
    let sizeof_value = dlpack_element_size(&tensor.dtype).ok_or_else(|| {
        CagraError::Validation(
            "unsupported dataset dtype for CAGRA layout check".to_string(),
        )
    })?;
    let logical_columns = unsafe { *tensor.shape.add(1) } as u32;
    let actual_row_width = if tensor.strides.is_null() {
        logical_columns
    } else {
        unsafe { *tensor.strides } as u32
    };
    Ok(actual_row_width == cagra_required_row_width(logical_columns, sizeof_value))
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
    /// Borrow a tensor as the host/device and padded/standard view matching its
    /// DLPack shape/strides (CAGRA row-width rule).
    pub fn new<T>(res: &Resources, dataset: &'a T) -> Result<Self>
    where
        T: AsDlTensor + ?Sized,
    {
        let dataset = dataset.as_dl_tensor()?;
        let mut dataset_c = dataset.to_c();
        unsafe {
            let is_device = dataset_c.inner.dl_tensor.device.device_type.is_device_compatible();
            let is_padded = is_cagra_padded_layout(&dataset_c.inner.dl_tensor)?;
            let kind = DatasetKind::from_residency_and_padded(is_device, is_padded);

            let handle = init_handle(|out| {
                if is_padded {
                    ffi::cuvsDatasetMakePaddedView(res.handle(), dataset_c.as_mut_ptr(), out)
                } else {
                    ffi::cuvsDatasetMakeStandardView(res.handle(), dataset_c.as_mut_ptr(), out)
                }
            })?;
            Ok(Self {
                handle,
                kind,
                destroy_handle: true,
                _dataset: PhantomData,
            })
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
        let kind = unsafe { DatasetKind::from_handle_fields((*handle).mem_type, (*handle).layout) };
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
