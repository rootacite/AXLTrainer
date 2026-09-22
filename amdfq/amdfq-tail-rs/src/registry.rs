/* The allocation state: one map keyed by the block's start address, plus an index keyed by where
 * that block ends (DESIGN.md D1). The RwLock is the crate's only lock, and it is held only across the
 * map operation — never across a call into the runtime, never across a log line (DESIGN.md D3). */

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

    pub(crate) fn offset(self, bytes: usize) -> Self {
        Address(self.0 + bytes)
    }

    pub(crate) fn is_null(self) -> bool {
        self.0 == 0
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
    /* What the caller holds: the request, or the padded size when the end page could not be taken. */
    pub(crate) size: usize,
    pub(crate) origin: Origin,
}

impl HookData {
    pub(crate) fn block(&self) -> usize {
        match self.origin {
            Origin::Runtime { block, .. } | Origin::Guarded { block, .. } => block,
        }
    }

    pub(crate) fn end(&self) -> Address {
        self.address.offset(self.block())
    }

    pub(crate) fn device(&self) -> i32 {
        match self.origin {
            Origin::Runtime { device, .. } | Origin::Guarded { device, .. } => device,
        }
    }
}

/* Which side owns the bytes past the block, and therefore what a free has to do with them. */
#[derive(Clone, Copy)]
pub(crate) enum Origin {
    /* The runtime's own memory, nothing of ours behind it: already backed, unaligned, padded, or
     * the guard is off. hipFree gets the caller's pointer back unchanged. */
    Runtime { block: usize, device: i32 },
    /* A page this crate reserved at `page` (the first granule-aligned page start at or after the
     * block's end — equal to the end when that is already aligned). Unmapped and given back here;
     * the runtime never sees that page. */
    Guarded {
        block: usize,
        page: Address,
        granule: usize,
        device: i32,
    },
}

struct Registry {
    by_start: HashMap<Address, HookData>,
    /* end address -> start: the predecessor a free has to re-guard. */
    by_end: HashMap<Address, Address>,
}

static REGISTRY: LazyLock<RwLock<Registry>> = LazyLock::new(|| {
    RwLock::new(Registry {
        by_start: HashMap::new(),
        by_end: HashMap::new(),
    })
});

fn write() -> std::sync::RwLockWriteGuard<'static, Registry> {
    REGISTRY.write().unwrap_or_else(|poison| poison.into_inner())
}

fn read() -> std::sync::RwLockReadGuard<'static, Registry> {
    REGISTRY.read().unwrap_or_else(|poison| poison.into_inner())
}

/* Returns the record displaced at that address: a duplicate key means hipMalloc handed out an address
 * that was still live, which the caller reports. */
pub(crate) fn insert(data: HookData) -> Option<HookData> {
    let end = data.end();
    let start = data.address;
    let mut reg = write();
    let old = reg.by_start.insert(start, data);
    if let Some(ref previous) = old {
        reg.by_end.remove(&previous.end());
    }
    reg.by_end.insert(end, start);
    old
}

/* Takes the record out of both maps; the caller reads it outside the lock. */
pub(crate) fn remove(address: Address) -> Option<HookData> {
    let mut reg = write();
    let data = reg.by_start.remove(&address)?;
    reg.by_end.remove(&data.end());
    Some(data)
}

/* A still-live allocation that ends at `end` and does not already have a page of ours behind it.
 * Copied out so the caller can ask the runtime whether that page is free, then map it, without
 * holding this lock (D3). */
#[derive(Clone, Copy)]
pub(crate) struct Predecessor {
    pub start: Address,
    pub device: i32,
}

pub(crate) fn unguarded_ending_at(end: Address) -> Option<Predecessor> {
    let reg = read();
    let start = *reg.by_end.get(&end)?;
    let data = reg.by_start.get(&start)?;
    match data.origin {
        Origin::Runtime { device, .. } => Some(Predecessor { start, device }),
        Origin::Guarded { .. } => None,
    }
}

/* Installs a guard on a predecessor that is still live and still unguarded. False means the caller
 * mapped a page nobody wants, and has to give it back. */
pub(crate) fn attach_guard(
    start: Address,
    page: Address,
    granule: usize,
    device: i32,
) -> bool {
    let mut reg = write();
    let Some(data) = reg.by_start.get_mut(&start) else {
        return false;
    };
    match data.origin {
        Origin::Runtime { block, device: rec_device } if rec_device == device => {
            data.origin = Origin::Guarded {
                block,
                page,
                granule,
                device,
            };
            true
        }
        _ => false,
    }
}
