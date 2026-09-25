/* The C symbols, and the whole of this object's behaviour: serve the request if the peralloc route
 * can (peralloc.rs), otherwise forward it unchanged, then record and log what happened.
 *
 * hipMalloc and hipFree are the allocation path of a training run in this configuration: the HSA
 * calls behind them, and everything the other gates do, allocate nothing
 * themselves. hipMemGetInfo is hooked so the VRAM reserve (driver remaining minus the floor) is
 * visible to the caching allocator.
 * A new gate joins them by writing one function here plus one resolver in `real.rs`.
 *
 * Neither hook locks anything: what comes in concurrently goes out concurrently (DESIGN.md D2). */

use crate::hip::{HIP_ERROR_NOT_FOUND, HIP_ERROR_OUT_OF_MEMORY, HIP_SUCCESS, HipError};
use crate::logging;
use crate::peralloc;
use crate::pool;
use crate::real;
use crate::registry::{self, Address, HookData, Origin};
use std::ffi::c_void;
use std::ptr;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> HipError {
    logging::init();
    let Some(real) = *real::HIP_MALLOC else {
        return HIP_ERROR_NOT_FOUND;
    };

    match peralloc::serve(size) {
        peralloc::Serve::Extent(record) => {
            unsafe { *ptr = record.address.as_ptr() };
            match &record.origin {
                Origin::Extent(extent) => log::info!(
                    "hipMalloc(size={size}) -> ret=0 served va={} block={} extent={} pad={} device={}",
                    record.address,
                    extent.block,
                    extent.total,
                    extent.pad.unwrap_or(0),
                    extent.device,
                ),
                Origin::Pooled(pooled) => log::info!(
                    "hipMalloc(size={size}) -> ret=0 pooled va={} pool={} offset={} len={}",
                    record.address,
                    pooled.pool,
                    pooled.offset,
                    pooled.len,
                ),
                /* A record the runtime owns is never built here, so this is unreachable; saying so
                 * is cheaper than a silent wrong line if that ever changes. */
                Origin::Runtime => log::warn!(
                    "hipMalloc(size={size}) -> ret=0 served va={} has an origin the route does not build",
                    record.address,
                ),
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
        peralloc::Serve::Oom => {
            if !ptr.is_null() {
                unsafe { *ptr = ptr::null_mut() };
            }
            log::warn!("hipMalloc(size={size}) -> ret={HIP_ERROR_OUT_OF_MEMORY} reserved");
            return HIP_ERROR_OUT_OF_MEMORY;
        }
        peralloc::Serve::Forward => {}
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
pub unsafe extern "C" fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> HipError {
    logging::init();
    let Some(real) = *real::HIP_MEM_GET_INFO else {
        return HIP_ERROR_NOT_FOUND;
    };
    let ret = unsafe { real(free, total) };
    if ret == HIP_SUCCESS {
        peralloc::shade_mem_info(free, total);
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
            origin: Origin::Pooled(pooled),
        }) => {
            /* The range goes back to its pool, and the pool itself only when this was its last live
             * block (pool.rs). The line keeps the `released size=` shape the self-check counts, and
             * names the pool after it so a pooled free is told apart from a solo teardown. */
            let outcome = pool::release(address, pooled);
            log::info!(
                "hipFree(ptr={address}) -> {outcome} size={size} pool={} offset={} len={}",
                pooled.pool,
                pooled.offset,
                pooled.len
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
