/* The C symbols, and the whole of this object's behaviour: serve the request if the peralloc route
 * can (peralloc.rs), otherwise forward it unchanged, then record and log what happened.
 *
 * hipMalloc and hipFree are the allocation path of a training run in this configuration
 * (../amdfq.md §7.1): the HSA calls behind them, and everything the other gates do, allocate nothing
 * themselves. A second gate joins them by writing one function here plus one resolver in `real.rs`.
 *
 * Neither hook locks anything: what comes in concurrently goes out concurrently (DESIGN.md D2). */

use crate::hip::{HIP_ERROR_NOT_FOUND, HIP_SUCCESS, HipError};
use crate::logging;
use crate::peralloc;
use crate::real;
use crate::registry::{self, Address, HookData, Origin};
use std::ffi::c_void;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> HipError {
    logging::init();
    let Some(real) = *real::HIP_MALLOC else {
        return HIP_ERROR_NOT_FOUND;
    };

    if let Some(record) = peralloc::serve(size) {
        unsafe { *ptr = record.address.as_ptr() };
        if let Origin::Extent(extent) = &record.origin {
            log::info!(
                "hipMalloc(size={size}) -> ret=0 served va={} block={} extent={} pad={} device={}",
                record.address,
                extent.block,
                extent.total,
                extent.pad.unwrap_or(0),
                extent.device,
            );
        }
        if let Some(replaced) = registry::insert(record) {
            log::warn!(
                "duplicate {}, previous size={}",
                replaced.address,
                replaced.size
            );
        }
        return HIP_SUCCESS;
    }

    let ret = unsafe { real(ptr, size) };
    if ret == HIP_SUCCESS {
        let address = Address::from_ptr(unsafe { *ptr });
        if let Some(replaced) = registry::insert(HookData {
            address,
            size,
            origin: Origin::Runtime,
        }) {
            log::warn!(
                "duplicate {}, previous size={}",
                replaced.address,
                replaced.size
            );
        }
        log::info!("hipMalloc(size={size}) -> ret={ret} forwarded va={address}");
    } else {
        log::info!("hipMalloc(size={size}) -> ret={ret} forwarded");
    }
    ret
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hipFree(ptr: *mut c_void) -> HipError {
    logging::init();
    let Some(real) = *real::HIP_FREE else {
        return HIP_ERROR_NOT_FOUND;
    };

    let address = Address::from_ptr(ptr);
    match registry::remove(address) {
        Some(HookData {
            address,
            size,
            origin: Origin::Extent(extent),
        }) => {
            let outcome = peralloc::release(address, extent);
            log::info!(
                "hipFree(ptr={address}) -> {outcome} size={size} device={}",
                extent.device
            );
            HIP_SUCCESS
        }
        Some(HookData {
            address,
            size,
            origin: Origin::Runtime,
        }) => {
            let ret = unsafe { real(ptr) };
            log::info!("hipFree(ptr={address}) -> ret={ret} forwarded size={size}");
            ret
        }
        None => {
            let ret = unsafe { real(ptr) };
            log::warn!("hipFree(ptr={address}) -> ret={ret} untracked");
            ret
        }
    }
}
