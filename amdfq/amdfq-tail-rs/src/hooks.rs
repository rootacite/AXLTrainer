/* The C symbols, and the whole of this object's behaviour: forward hipMalloc unchanged, then let
 * the tail guard map a page behind the block when the runtime backs nothing there (tail.rs).
 *
 * hipMalloc, hipFree and hipHostMalloc are the allocation path of a training run in this
 * configuration: the HSA calls behind them, and everything the other gates
 * do, allocate nothing themselves. A second gate joins them by writing one function here plus one
 * resolver in `real.rs`.
 *
 * Neither hook locks anything: what comes in concurrently goes out concurrently (DESIGN.md D2). */

use crate::hip::{HIP_ERROR_NOT_FOUND, HIP_SUCCESS, HipError};
use crate::logging;
use crate::real;
use crate::registry::{self, Address, Origin};
use crate::tail::{self, AfterMalloc, KeepKind};
use std::ffi::c_void;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> HipError {
    logging::init();
    let Some(real) = *real::HIP_MALLOC else {
        return HIP_ERROR_NOT_FOUND;
    };

    let ret = unsafe { real(ptr, size) };
    if ret != HIP_SUCCESS || ptr.is_null() {
        log::info!("hipMalloc(size={size}) -> ret={ret}");
        return ret;
    }
    let address = Address::from_ptr(unsafe { *ptr });
    if address.is_null() || size == 0 {
        log::info!("hipMalloc(size={size}) -> ret={ret} va={address}");
        return ret;
    }

    match tail::after_malloc(address, size) {
        AfterMalloc::Keep { record, kind } => {
            let va = record.address;
            let block = record.block();
            if let Some(replaced) = registry::insert(record) {
                log::warn!(
                    "duplicate {}, previous size={}",
                    replaced.address,
                    replaced.size
                );
            }
            match kind {
                KeepKind::Guarded => log::info!(
                    "hipMalloc(size={size}) -> ret=0 va={va} block={block} guarded"
                ),
                KeepKind::Backed => log::info!(
                    "hipMalloc(size={size}) -> ret=0 va={va} block={block} backed"
                ),
                KeepKind::Unaligned => log::info!(
                    "hipMalloc(size={size}) -> ret=0 va={va} block={block} unaligned"
                ),
                KeepKind::Unguarded => log::info!(
                    "hipMalloc(size={size}) -> ret=0 va={va} block={block} unguarded"
                ),
                KeepKind::Off => {
                    log::info!("hipMalloc(size={size}) -> ret=0 va={va} block={block} off")
                }
            }
            HIP_SUCCESS
        }
        AfterMalloc::Replaced { record, old } => {
            let va = record.address;
            let granted = record.size;
            let block = record.block();
            unsafe { *ptr = va.as_ptr() };
            if let Some(replaced) = registry::insert(record) {
                log::warn!(
                    "duplicate {}, previous size={}",
                    replaced.address,
                    replaced.size
                );
            }
            log::info!(
                "hipMalloc(size={size}) -> ret=0 va={va} block={block} padded from={old} granted={granted}"
            );
            HIP_SUCCESS
        }
        AfterMalloc::Failed { error } => {
            unsafe { *ptr = std::ptr::null_mut() };
            log::info!("hipMalloc(size={size}) -> ret={error} padded-failed from={address}");
            error
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hipFree(ptr: *mut c_void) -> HipError {
    logging::init();
    let Some(real) = *real::HIP_FREE else {
        return HIP_ERROR_NOT_FOUND;
    };

    let address = Address::from_ptr(ptr);
    match registry::remove(address) {
        Some(record) => {
            let size = record.size;
            let device = record.device();
            if let Origin::Guarded {
                page, granule, ..
            } = record.origin
            {
                tail::unprotect(page, granule, device);
                let ret = unsafe { real(ptr) };
                let re = tail::reprotect(address);
                match re {
                    Some(pred) => log::info!(
                        "hipFree(ptr={address}) -> ret={ret} released page={page} size={size} re-guarded pred={pred}"
                    ),
                    None => log::info!(
                        "hipFree(ptr={address}) -> ret={ret} released page={page} size={size}"
                    ),
                }
                ret
            } else {
                let ret = unsafe { real(ptr) };
                let re = if ret == HIP_SUCCESS {
                    tail::reprotect(address)
                } else {
                    None
                };
                match re {
                    Some(pred) => log::info!(
                        "hipFree(ptr={address}) -> ret={ret} size={size} re-guarded pred={pred}"
                    ),
                    None => log::info!("hipFree(ptr={address}) -> ret={ret} size={size}"),
                }
                ret
            }
        }
        None => {
            let ret = unsafe { real(ptr) };
            if ptr.is_null() {
                log::info!("hipFree(ptr={address}) -> ret={ret}");
            } else {
                log::warn!("hipFree(ptr={address}) -> ret={ret} untracked");
            }
            ret
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hipHostMalloc(
    ptr: *mut *mut c_void,
    size: usize,
    flags: u32,
) -> HipError {
    logging::init();
    let Some(real) = *real::HIP_HOST_MALLOC else {
        return HIP_ERROR_NOT_FOUND;
    };
    let ret = unsafe { real(ptr, size, flags) };
    let va = if ptr.is_null() {
        Address::from_ptr(std::ptr::null_mut())
    } else {
        Address::from_ptr(unsafe { *ptr })
    };
    log::info!("hipHostMalloc(size={size} flags={flags:#x}) -> ret={ret} va={va}");
    ret
}
