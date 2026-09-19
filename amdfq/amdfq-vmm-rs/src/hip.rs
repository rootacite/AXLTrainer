/* The HIP types and codes this crate names, spelled out here so the object builds with no ROCm
 * headers present (the same approach as the retired C VMM tree's amdfq_vmm.h, whose copies these match):
 *   hip/hip_runtime_api.h — the VMM entry points `real.rs` resolves, plus hipMemAllocationProp,
 *                           hipMemLocation and hipMemAccessDesc
 *   hip/driver_types.h    — hipMemGenericAllocationHandle_t
 *
 * The struct layouts are the runtime's, not ours: they are passed by pointer into hipMemCreate and
 * hipMemSetAccess, so field order and widths have to stay exactly as ROCm declares them. */

use std::ffi::c_void;

pub(crate) type HipError = i32;

pub(crate) const HIP_SUCCESS: HipError = 0;
/* hipErrorNotFound, per ROCm 10.0.0's hip/hip_runtime_api.h. */
pub(crate) const HIP_ERROR_NOT_FOUND: HipError = 500;

/* hipMemGenericAllocationHandle_t: an opaque handle to a physical allocation. Wrapped instead of left
 * a bare pointer so a record can live in the registry, which the compiler holds to Send + Sync
 * (DESIGN.md D9); the handle names a device object, not thread-local memory. */
#[derive(Clone, Copy)]
pub(crate) struct Handle(*mut c_void);

unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

impl Handle {
    pub(crate) fn from_raw(raw: *mut c_void) -> Self {
        Handle(raw)
    }

    pub(crate) fn as_raw(self) -> *mut c_void {
        self.0
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct MemLocation {
    pub(crate) type_: i32,
    pub(crate) id: i32,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct AllocFlags {
    pub(crate) compression_type: u8,
    pub(crate) gpu_direct_rdma_capable: u8,
    pub(crate) usage: u16,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct AllocationProp {
    pub(crate) type_: i32,
    pub(crate) requested_handle_types: i32,
    pub(crate) location: MemLocation,
    pub(crate) win32_handle_meta_data: *mut c_void,
    pub(crate) alloc_flags: AllocFlags,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct AccessDesc {
    pub(crate) location: MemLocation,
    pub(crate) flags: i32,
}

pub(crate) const MEM_ALLOCATION_TYPE_PINNED: i32 = 0x1;
pub(crate) const MEM_HANDLE_TYPE_NONE: i32 = 0x0;
pub(crate) const MEM_LOCATION_DEVICE: i32 = 1;
pub(crate) const MEM_LOCATION_HOST: i32 = 2;
pub(crate) const MEM_ACCESS_PROT_READWRITE: i32 = 3;
pub(crate) const MEM_GRANULARITY_MINIMUM: i32 = 0x0;
pub(crate) const MEM_GRANULARITY_RECOMMENDED: i32 = 0x1;
