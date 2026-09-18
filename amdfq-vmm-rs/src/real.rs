/* Resolving the real implementations — in one place. Nothing outside this module names a HIP symbol
 * or calls dlsym (DESIGN.md D8). Each symbol is cached on its own, so a runtime that lacks only some
 * of them still gets the rest. */

use crate::hip::{AccessDesc, AllocationProp, HipError};
use std::ffi::{CStr, c_char, c_void};
use std::mem::{size_of, transmute_copy};
use std::sync::LazyLock;

/* The two allocation gates (../amdfq.md §7.1). */
pub(crate) type HipMallocFn = unsafe extern "C" fn(ptr: *mut *mut c_void, size: usize) -> HipError;
pub(crate) type HipFreeFn = unsafe extern "C" fn(ptr: *mut c_void) -> HipError;

/* The VMM entry points the peralloc route builds on (../amdfq.md §12). */
pub(crate) type HipGetDeviceFn = unsafe extern "C" fn(device: *mut i32) -> HipError;
pub(crate) type HipMemGetAllocationGranularityFn = unsafe extern "C" fn(
    granularity: *mut usize,
    prop: *const AllocationProp,
    option: i32,
) -> HipError;
pub(crate) type HipMemAddressReserveFn = unsafe extern "C" fn(
    ptr: *mut *mut c_void,
    size: usize,
    alignment: usize,
    addr: *mut c_void,
    flags: u64,
) -> HipError;
pub(crate) type HipMemAddressFreeFn =
    unsafe extern "C" fn(ptr: *mut c_void, size: usize) -> HipError;
pub(crate) type HipMemCreateFn = unsafe extern "C" fn(
    handle: *mut *mut c_void,
    size: usize,
    prop: *const AllocationProp,
    flags: u64,
) -> HipError;
pub(crate) type HipMemMapFn = unsafe extern "C" fn(
    ptr: *mut c_void,
    size: usize,
    offset: usize,
    handle: *mut c_void,
    flags: u64,
) -> HipError;
pub(crate) type HipMemUnmapFn = unsafe extern "C" fn(ptr: *mut c_void, size: usize) -> HipError;
pub(crate) type HipMemReleaseFn = unsafe extern "C" fn(handle: *mut c_void) -> HipError;
pub(crate) type HipMemSetAccessFn = unsafe extern "C" fn(
    ptr: *mut c_void,
    size: usize,
    desc: *const AccessDesc,
    count: usize,
) -> HipError;
/* hipGetLastError: the runtime keeps a per-thread sticky error that every other call's failure sets
 * and that the caller's own error check reads. One of the route's calls is allowed to fail (asking
 * for the host access of a range the device refuses), so that state is consumed after it. */
pub(crate) type HipGetLastErrorFn = unsafe extern "C" fn() -> HipError;

/* hipDeviceSynchronize: the runtime's own free path waits for the device before it hands memory
 * back (ihipFree), and the route's teardown does not go through that path, so it asks for the same
 * wait itself. */
pub(crate) type HipDeviceSynchronizeFn = unsafe extern "C" fn() -> HipError;

/* RTLD_NEXT, as glibc spells it: the search starts after this object, so it can never hand back the
 * interposer itself. */
const RTLD_NEXT: *mut c_void = -1isize as *mut c_void;

unsafe extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

/* The only dlsym in this crate. A symbol this runtime does not export resolves to None, and the
 * caller then either forwards the call or leaves the route switched off. */
fn resolve<F>(name: &CStr) -> Option<F> {
    const {
        assert!(
            size_of::<F>() == size_of::<*mut c_void>(),
            "resolved symbols are function pointers"
        )
    };
    let symbol = unsafe { dlsym(RTLD_NEXT, name.as_ptr()) };
    if symbol.is_null() {
        None
    } else {
        Some(unsafe { transmute_copy(&symbol) })
    }
}

pub(crate) static HIP_MALLOC: LazyLock<Option<HipMallocFn>> =
    LazyLock::new(|| resolve(c"hipMalloc"));
pub(crate) static HIP_FREE: LazyLock<Option<HipFreeFn>> = LazyLock::new(|| resolve(c"hipFree"));

pub(crate) static HIP_GET_DEVICE: LazyLock<Option<HipGetDeviceFn>> =
    LazyLock::new(|| resolve(c"hipGetDevice"));
pub(crate) static HIP_MEM_GET_ALLOCATION_GRANULARITY: LazyLock<
    Option<HipMemGetAllocationGranularityFn>,
> = LazyLock::new(|| resolve(c"hipMemGetAllocationGranularity"));
pub(crate) static HIP_MEM_ADDRESS_RESERVE: LazyLock<Option<HipMemAddressReserveFn>> =
    LazyLock::new(|| resolve(c"hipMemAddressReserve"));
pub(crate) static HIP_MEM_ADDRESS_FREE: LazyLock<Option<HipMemAddressFreeFn>> =
    LazyLock::new(|| resolve(c"hipMemAddressFree"));
pub(crate) static HIP_MEM_CREATE: LazyLock<Option<HipMemCreateFn>> =
    LazyLock::new(|| resolve(c"hipMemCreate"));
pub(crate) static HIP_MEM_MAP: LazyLock<Option<HipMemMapFn>> =
    LazyLock::new(|| resolve(c"hipMemMap"));
pub(crate) static HIP_MEM_UNMAP: LazyLock<Option<HipMemUnmapFn>> =
    LazyLock::new(|| resolve(c"hipMemUnmap"));
pub(crate) static HIP_MEM_RELEASE: LazyLock<Option<HipMemReleaseFn>> =
    LazyLock::new(|| resolve(c"hipMemRelease"));
pub(crate) static HIP_MEM_SET_ACCESS: LazyLock<Option<HipMemSetAccessFn>> =
    LazyLock::new(|| resolve(c"hipMemSetAccess"));
pub(crate) static HIP_GET_LAST_ERROR: LazyLock<Option<HipGetLastErrorFn>> =
    LazyLock::new(|| resolve(c"hipGetLastError"));
pub(crate) static HIP_DEVICE_SYNCHRONIZE: LazyLock<Option<HipDeviceSynchronizeFn>> =
    LazyLock::new(|| resolve(c"hipDeviceSynchronize"));
