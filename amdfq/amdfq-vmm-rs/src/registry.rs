/* The allocation state: one map, keyed by the block's start address (DESIGN.md D1). Its RwLock is
 * the crate's only lock, and it is held only across the map operation — never across a call into the
 * runtime, never across a log line (DESIGN.md D3). */

use crate::hip::Handle;
use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt;
use std::sync::{LazyLock, RwLock};

/* The block's start address: what hipMalloc returned to the caller, which is also the pointer the
 * caller will free — and therefore this registry's key. */
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Address(usize);

impl Address {
    pub(crate) fn from_ptr(ptr: *mut c_void) -> Self {
        Address(ptr as usize)
    }

    pub(crate) fn as_ptr(self) -> *mut c_void {
        self.0 as *mut c_void
    }

    pub(crate) fn as_usize(self) -> usize {
        self.0
    }

    /* `bytes` further on, to reach the pad granule behind a block. */
    pub(crate) fn offset(self, bytes: usize) -> Self {
        Address(self.0 + bytes)
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#x}", self.0)
    }
}

/* One record per allocation this process made while the hook was loaded. Everything a free has to
 * undo belongs to the record (DESIGN.md D1); nothing about an allocation lives in a global. */
pub(crate) struct HookData {
    pub(crate) address: Address,
    /* What the caller asked for, unchanged — the value that says which layout this block got. */
    pub(crate) size: usize,
    pub(crate) origin: Origin,
}

/* Which side owns the block, and therefore what a free has to do with it. */
pub(crate) enum Origin {
    /* The runtime's own memory: hipFree gets the caller's pointer back unchanged. */
    Runtime,
    /* An extent this crate reserved (peralloc.rs): unmapped and the handle released here, without
     * the runtime ever seeing the pointer. The VA goes back too unless AMDFQ_VA_NEVER_REUSE is on. */
    Extent(Extent),
}

/* What a free needs to undo one served allocation. */
#[derive(Clone, Copy)]
pub(crate) struct Extent {
    /* Bytes mapped from `handle`: the request rounded up to the allocation granularity. */
    pub(crate) block: usize,
    /* Bytes reserved at the record's address: `block` plus one pad granule. Given back to the driver
     * with the handle, unless AMDFQ_VA_NEVER_REUSE keeps the span (DESIGN.md D10). */
    pub(crate) total: usize,
    /* The block's allocation handle. */
    pub(crate) handle: Handle,
    /* The shared pad granule mapped behind the block, when that mapping succeeded. */
    pub(crate) pad: Option<usize>,
    /* The device the block was made on. The VMM calls a free needs act on the current device, so the
     * record carries it and the free switches back before tearing anything down (DESIGN.md D11). */
    pub(crate) device: i32,
}

static REGISTRY: LazyLock<RwLock<HashMap<Address, HookData>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/* Returns the record displaced at that address: a duplicate key means hipMalloc handed out an address
 * that was still live, which the caller reports. */
pub(crate) fn insert(data: HookData) -> Option<HookData> {
    REGISTRY
        .write()
        .unwrap_or_else(|poison| poison.into_inner())
        .insert(data.address, data)
}

/* Takes the record out of the map; the caller reads it outside the lock. */
pub(crate) fn remove(address: Address) -> Option<HookData> {
    REGISTRY
        .write()
        .unwrap_or_else(|poison| poison.into_inner())
        .remove(&address)
}
