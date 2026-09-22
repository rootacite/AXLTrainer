/* The tail guard: hipMalloc still allocates, and this route only adds what the pad was buying — a
 * mapped page immediately past the end of an allocation, so a kernel that over-reads its operand
 * lands in memory that exists.
 *
 * After hipMalloc returns:
 *
 *   1. take the block's extent (hipMemGetAddressRange, rounded up to the granule);
 *   2. the guard page starts at the first granule-aligned address at or after that end — when the
 *      end is already aligned this is the end itself; when it is not, the next granule;
 *   3. if that page is already backed, leave it;
 *   4. otherwise reserve that page (hint must land exactly) and map one shared handle into it;
 *   5. if the hint missed, give the block back and take one with 16 bytes added — §10.1's pad, for
 *      that block alone.
 *
 * hipFree unmaps the block's own page, then — after the real free — asks the same question about the
 * page the freed block was occupying, so a neighbour that just lost its protection gets a new one.
 *
 * The shared handle is one per device, created once, released never (D7, D11). The grant a page gets
 * is its own device, every device that can reach this one, and the host. A mapping's reach comes only
 * from hipMemSetAccess.
 *
 * The serve path locks nothing: concurrent hipMalloc calls run their own reserve/map sequences at the
 * same time (DESIGN.md D2). The only lock is the registry's, and it is held only while the maps
 * change (DESIGN.md D3). */

use crate::hip::{
    AccessDesc, AllocationProp, HIP_ERROR_NOT_FOUND, HIP_SUCCESS, Handle, HipError,
    MEM_ACCESS_PROT_READWRITE, MEM_ALLOCATION_TYPE_PINNED, MEM_GRANULARITY_MINIMUM,
    MEM_GRANULARITY_RECOMMENDED, MEM_HANDLE_TYPE_NONE, MEM_LOCATION_DEVICE, MEM_LOCATION_HOST,
    MemLocation,
};
use crate::real;
use crate::registry::{self, Address, HookData, Origin};
use std::ffi::c_void;
use std::ptr;
use std::sync::{LazyLock, OnceLock};

/* §10.1's pad, used only when the end page cannot be taken. */
const TAIL_PAD: usize = 16;

#[derive(Clone, Copy)]
struct Entries {
    get_device: real::HipGetDeviceFn,
    set_device: real::HipSetDeviceFn,
    get_granularity: real::HipMemGetAllocationGranularityFn,
    reserve: real::HipMemAddressReserveFn,
    address_free: real::HipMemAddressFreeFn,
    create: real::HipMemCreateFn,
    map: real::HipMemMapFn,
    unmap: real::HipMemUnmapFn,
    set_access: real::HipMemSetAccessFn,
    get_range: real::HipMemGetAddressRangeFn,
    get_last_error: real::HipGetLastErrorFn,
    device_count: Option<real::HipGetDeviceCountFn>,
    can_access_peer: Option<real::HipDeviceCanAccessPeerFn>,
}

impl Entries {
    fn resolve() -> Option<Self> {
        Some(Entries {
            get_device: (*real::HIP_GET_DEVICE)?,
            set_device: (*real::HIP_SET_DEVICE)?,
            get_granularity: (*real::HIP_MEM_GET_ALLOCATION_GRANULARITY)?,
            reserve: (*real::HIP_MEM_ADDRESS_RESERVE)?,
            address_free: (*real::HIP_MEM_ADDRESS_FREE)?,
            create: (*real::HIP_MEM_CREATE)?,
            map: (*real::HIP_MEM_MAP)?,
            unmap: (*real::HIP_MEM_UNMAP)?,
            set_access: (*real::HIP_MEM_SET_ACCESS)?,
            get_range: (*real::HIP_MEM_GET_ADDRESS_RANGE)?,
            get_last_error: (*real::HIP_GET_LAST_ERROR)?,
            device_count: *real::HIP_GET_DEVICE_COUNT,
            can_access_peer: *real::HIP_DEVICE_CAN_ACCESS_PEER,
        })
    }
}

struct Device {
    id: i32,
    granule: usize,
    /* The one physical object on this device that every guard page is a mapping of: created once,
     * released never. */
    pad: Handle,
    peers: Vec<i32>,
}

static ENTRIES: LazyLock<Option<Entries>> = LazyLock::new(Entries::resolve);

const MAX_DEVICES: usize = 32;

static DEVICES: [OnceLock<Option<Device>>; MAX_DEVICES] = [const { OnceLock::new() }; MAX_DEVICES];

static ENABLED: LazyLock<bool> = LazyLock::new(|| {
    match std::env::var("AMDFQ_TAIL") {
        Ok(value) if value == "0" => {
            log::info!("tail: disabled by AMDFQ_TAIL=0");
            false
        }
        _ => true,
    }
});

fn device_state(entries: &Entries, id: i32) -> Option<&'static Device> {
    let index = usize::try_from(id).ok()?;
    let Some(slot) = DEVICES.get(index) else {
        log::warn!("device ordinal {id} is past the {MAX_DEVICES}-slot device table, not guarding");
        return None;
    };
    slot.get_or_init(|| build_device(entries, id)).as_ref()
}

fn build_device(entries: &Entries, id: i32) -> Option<Device> {
    let prop = prop(id);
    let mut minimum = 0;
    if unsafe { (entries.get_granularity)(&mut minimum, &prop, MEM_GRANULARITY_MINIMUM) }
        != HIP_SUCCESS
        || minimum == 0
    {
        log::warn!("device {id}: hipMemGetAllocationGranularity failed, not guarding");
        return None;
    }
    let mut recommended = 0;
    if unsafe { (entries.get_granularity)(&mut recommended, &prop, MEM_GRANULARITY_RECOMMENDED) }
        != HIP_SUCCESS
        || recommended < minimum
    {
        recommended = minimum;
    }
    let mut raw: *mut c_void = ptr::null_mut();
    if unsafe { (entries.create)(&mut raw, recommended, &prop, 0) } != HIP_SUCCESS {
        log::warn!("device {id}: hipMemCreate({recommended}) failed, not guarding");
        return None;
    }
    let peers = peers_of(entries, id);
    log::info!("device {id}: granule={recommended} peers={peers:?}");
    Some(Device {
        id,
        granule: recommended,
        pad: Handle::from_raw(raw),
        peers,
    })
}

fn peers_of(entries: &Entries, id: i32) -> Vec<i32> {
    let (Some(device_count), Some(can_access_peer)) =
        (entries.device_count, entries.can_access_peer)
    else {
        log::warn!(
            "hipGetDeviceCount/hipDeviceCanAccessPeer not resolvable: guard pages get no peer grant"
        );
        return Vec::new();
    };
    let mut count = 0;
    if unsafe { device_count(&mut count) } != HIP_SUCCESS {
        clear_error(entries);
        return Vec::new();
    }
    let mut peers = Vec::new();
    for peer in 0..count {
        if peer == id {
            continue;
        }
        if peers.len() == MAX_DEVICES - 1 {
            log::warn!(
                "device {id} can reach more devices than the descriptor set holds ({}), the rest get no grant",
                MAX_DEVICES - 1
            );
            break;
        }
        let mut can = 0;
        if unsafe { can_access_peer(&mut can, id, peer) } != HIP_SUCCESS {
            clear_error(entries);
            continue;
        }
        if can != 0 {
            peers.push(peer);
        }
    }
    peers
}

fn current_device(entries: &Entries) -> Option<i32> {
    let mut id = 0;
    if unsafe { (entries.get_device)(&mut id) } != HIP_SUCCESS {
        clear_error(entries);
        return None;
    }
    Some(id)
}

fn prop(device_id: i32) -> AllocationProp {
    AllocationProp {
        type_: MEM_ALLOCATION_TYPE_PINNED,
        requested_handle_types: MEM_HANDLE_TYPE_NONE,
        location: MemLocation {
            type_: MEM_LOCATION_DEVICE,
            id: device_id,
        },
        ..AllocationProp::default()
    }
}

fn clear_error(entries: &Entries) {
    unsafe { (entries.get_last_error)() };
}

fn switch_device(entries: &Entries, device: i32, what: &str, address: Address) {
    let ret = unsafe { (entries.set_device)(device) };
    if ret != HIP_SUCCESS {
        log::warn!("hipSetDevice({device}) -> {ret}: {what} of va={address} runs on whatever device is current");
        clear_error(entries);
    }
}

/* What the access grant for one range came to. */
#[derive(Clone, Copy, PartialEq, Eq)]
enum Access {
    Granted,
    DeviceOnly,
    None,
}

fn device_desc(id: i32) -> AccessDesc {
    AccessDesc {
        location: MemLocation {
            type_: MEM_LOCATION_DEVICE,
            id,
        },
        flags: MEM_ACCESS_PROT_READWRITE,
    }
}

fn set_access(entries: &Entries, dev: &Device, address: Address, size: usize) -> Access {
    let mut desc = [AccessDesc::default(); MAX_DEVICES + 2];
    let mut count = 0;
    desc[count] = device_desc(dev.id);
    count += 1;
    for peer in &dev.peers {
        desc[count] = device_desc(*peer);
        count += 1;
    }
    desc[count] = AccessDesc {
        location: MemLocation {
            type_: MEM_LOCATION_HOST,
            id: 0,
        },
        flags: MEM_ACCESS_PROT_READWRITE,
    };
    count += 1;

    let both = unsafe { (entries.set_access)(address.as_ptr(), size, desc.as_ptr(), count) };
    if both == HIP_SUCCESS {
        return Access::Granted;
    }
    clear_error(entries);
    let device = unsafe { (entries.set_access)(address.as_ptr(), size, desc.as_ptr(), 1) };
    if device == HIP_SUCCESS {
        log::warn!(
            "set_access(va={address} size={size}): device+{} peer(s)+host -> {both}, device-only -> {device}: host and peer reads of this range fault",
            dev.peers.len()
        );
        return Access::DeviceOnly;
    }
    clear_error(entries);
    log::warn!(
        "set_access(va={address} size={size}): device+{} peer(s)+host -> {both}, device-only -> {device}: no access granted",
        dev.peers.len()
    );
    Access::None
}

/* First granule-aligned page start at or after `end`. Equal to `end` when `end` is already
 * aligned; otherwise the next granule boundary. Independent of how the block was allocated. */
fn first_page_at_or_after(end: Address, granule: usize) -> Option<Address> {
    if granule < 2 || !granule.is_power_of_two() {
        return None;
    }
    let end_u = end.as_ptr() as usize;
    let mask = granule - 1;
    if end_u & mask == 0 {
        return Some(end);
    }
    let next = end_u.checked_add(granule - (end_u & mask))?;
    Some(Address::from_ptr(next as *mut c_void))
}

fn address_backed(entries: &Entries, addr: Address) -> bool {
    let mut base: *mut c_void = ptr::null_mut();
    let mut size = 0usize;
    if unsafe { (entries.get_range)(&mut base, &mut size, addr.as_ptr()) } != HIP_SUCCESS {
        clear_error(entries);
        return false;
    }
    let base_u = base as usize;
    let addr_u = addr.as_ptr() as usize;
    base_u <= addr_u && addr_u.saturating_add(TAIL_PAD) <= base_u.saturating_add(size)
}

fn extent_of(entries: &Entries, ptr: Address, size: usize, granule: usize) -> usize {
    let mut block = size.div_ceil(granule).saturating_mul(granule);
    let mut base: *mut c_void = ptr::null_mut();
    let mut range = 0usize;
    if unsafe { (entries.get_range)(&mut base, &mut range, ptr.as_ptr()) } == HIP_SUCCESS {
        if Address::from_ptr(base) == ptr && range > block {
            block = range.div_ceil(granule).saturating_mul(granule);
        }
    } else {
        clear_error(entries);
    }
    block
}

/* Reserves `addr` exactly and maps the device's shared handle into it. A hint that lands elsewhere
 * is given back: the page is held by something we cannot move. */
fn protect_page(entries: &Entries, dev: &Device, addr: Address) -> Option<Address> {
    let mut va: *mut c_void = ptr::null_mut();
    let reserved = unsafe {
        (entries.reserve)(&mut va, dev.granule, dev.granule, addr.as_ptr(), 0)
    };
    if reserved != HIP_SUCCESS {
        clear_error(entries);
        log::warn!(
            "hipMemAddressReserve(hint={addr}, {}) -> {reserved}",
            dev.granule
        );
        return None;
    }
    let got = Address::from_ptr(va);
    if got != addr {
        log::warn!("{addr} not takeable (reserve gave {got}); falling back to pad");
        let freed = unsafe { (entries.address_free)(va, dev.granule) };
        if freed != HIP_SUCCESS {
            clear_error(entries);
            log::warn!("hipMemAddressFree(va={got}) -> {freed}");
        }
        return None;
    }
    let mapped = unsafe { (entries.map)(va, dev.granule, 0, dev.pad.as_raw(), 0) };
    if mapped != HIP_SUCCESS {
        clear_error(entries);
        log::warn!("hipMemMap(va={addr}, {}) -> {mapped}", dev.granule);
        let freed = unsafe { (entries.address_free)(va, dev.granule) };
        if freed != HIP_SUCCESS {
            clear_error(entries);
            log::warn!("hipMemAddressFree(va={addr}) -> {freed}");
        }
        return None;
    }
    if set_access(entries, dev, addr, dev.granule) == Access::None {
        let unmapped = unsafe { (entries.unmap)(va, dev.granule) };
        if unmapped != HIP_SUCCESS {
            clear_error(entries);
            log::warn!("hipMemUnmap(va={addr}) -> {unmapped}");
        }
        let freed = unsafe { (entries.address_free)(va, dev.granule) };
        if freed != HIP_SUCCESS {
            clear_error(entries);
            log::warn!("hipMemAddressFree(va={addr}) -> {freed}");
        }
        return None;
    }
    Some(addr)
}

pub(crate) fn unprotect(page: Address, granule: usize, device: i32) {
    let Some(entries) = (*ENTRIES).as_ref() else {
        log::warn!("unprotect va={page}: no runtime entry points to undo a guard with");
        return;
    };
    switch_device(entries, device, "unprotect", page);
    let unmapped = unsafe { (entries.unmap)(page.as_ptr(), granule) };
    if unmapped != HIP_SUCCESS {
        clear_error(entries);
        log::warn!("hipMemUnmap(va={page}) -> {unmapped}");
    }
    let freed = unsafe { (entries.address_free)(page.as_ptr(), granule) };
    if freed != HIP_SUCCESS {
        clear_error(entries);
        log::warn!("hipMemAddressFree(va={page}) -> {freed}");
    }
}

/* What became of a successful hipMalloc after the route looked at it. */
pub(crate) enum AfterMalloc {
    Keep {
        record: HookData,
        kind: KeepKind,
    },
    Replaced {
        record: HookData,
        old: Address,
    },
    Failed {
        error: HipError,
    },
}

#[derive(Clone, Copy)]
pub(crate) enum KeepKind {
    Off,
    Guarded,
    Backed,
    Unaligned,
    Unguarded,
}

fn runtime_record(address: Address, size: usize, block: usize, device: i32) -> HookData {
    HookData {
        address,
        size,
        origin: Origin::Runtime { block, device },
    }
}

/* Looks at a pointer the real hipMalloc just handed back, and either maps a page behind it, leaves
 * it, or replaces it with a padded block. Called with no lock held. */
pub(crate) fn after_malloc(address: Address, size: usize) -> AfterMalloc {
    if !*ENABLED {
        return AfterMalloc::Keep {
            record: runtime_record(address, size, size, 0),
            kind: KeepKind::Off,
        };
    }
    let Some(entries) = (*ENTRIES).as_ref() else {
        log::warn!("tail: VMM entry points missing from this runtime; allocations go unguarded");
        return AfterMalloc::Keep {
            record: runtime_record(address, size, size, 0),
            kind: KeepKind::Off,
        };
    };
    let Some(id) = current_device(entries) else {
        return AfterMalloc::Keep {
            record: runtime_record(address, size, size, 0),
            kind: KeepKind::Off,
        };
    };
    let Some(dev) = device_state(entries, id) else {
        return AfterMalloc::Keep {
            record: runtime_record(address, size, size, id),
            kind: KeepKind::Off,
        };
    };

    let block = extent_of(entries, address, size, dev.granule);
    let end = address.offset(block);
    let Some(page_at) = first_page_at_or_after(end, dev.granule) else {
        return AfterMalloc::Keep {
            record: runtime_record(address, size, block, id),
            kind: KeepKind::Unaligned,
        };
    };

    if address_backed(entries, page_at) {
        return AfterMalloc::Keep {
            record: runtime_record(address, size, block, id),
            kind: KeepKind::Backed,
        };
    }
    if let Some(page) = protect_page(entries, dev, page_at) {
        return AfterMalloc::Keep {
            record: HookData {
                address,
                size,
                origin: Origin::Guarded {
                    block,
                    page,
                    granule: dev.granule,
                    device: id,
                },
            },
            kind: KeepKind::Guarded,
        };
    }

    /* The page behind this block is held by something we cannot move. Give it back and take a padded
     * one: the +16 bytes of §10.1 make the runtime reserve a granule more than the request. */
    let padded = size.saturating_add(TAIL_PAD);
    let Some(real_free) = *real::HIP_FREE else {
        return AfterMalloc::Keep {
            record: runtime_record(address, size, block, id),
            kind: KeepKind::Unguarded,
        };
    };
    let freed = unsafe { real_free(address.as_ptr()) };
    if freed != HIP_SUCCESS {
        clear_error(entries);
        log::warn!(
            "{address} could not be given back for a padded block (rc={freed}); it stays unguarded"
        );
        return AfterMalloc::Keep {
            record: runtime_record(address, size, block, id),
            kind: KeepKind::Unguarded,
        };
    }

    let Some(real_malloc) = *real::HIP_MALLOC else {
        return AfterMalloc::Failed {
            error: HIP_ERROR_NOT_FOUND,
        };
    };
    let mut fresh: *mut c_void = ptr::null_mut();
    let rc = unsafe { real_malloc(&mut fresh, padded) };
    if rc != HIP_SUCCESS || fresh.is_null() {
        clear_error(entries);
        log::warn!(
            "{address} (block {block} B) was given back and the padded hipMalloc({padded}) failed: rc={rc} ptr={fresh:?}"
        );
        return AfterMalloc::Failed { error: rc };
    }
    let fresh_addr = Address::from_ptr(fresh);
    let fresh_block = extent_of(entries, fresh_addr, padded, dev.granule);
    AfterMalloc::Replaced {
        record: runtime_record(fresh_addr, padded, fresh_block, id),
        old: address,
    }
}

/* After a successful hipFree of `freed`, map a guard at that address if a still-live predecessor
 * ends there and nothing else backs the page. Returns the predecessor when a page was attached. */
pub(crate) fn reprotect(freed: Address) -> Option<Address> {
    if !*ENABLED {
        return None;
    }
    let Some(entries) = (*ENTRIES).as_ref() else {
        return None;
    };
    let Some(pred) = registry::unguarded_ending_at(freed) else {
        return None;
    };
    let Some(dev) = device_state(entries, pred.device) else {
        return None;
    };
    let Some(page_at) = first_page_at_or_after(freed, dev.granule) else {
        return None;
    };
    if address_backed(entries, page_at) {
        return None;
    }
    switch_device(entries, pred.device, "reprotect", page_at);
    let Some(page) = protect_page(entries, dev, page_at) else {
        return None;
    };
    if registry::attach_guard(pred.start, page, dev.granule, pred.device) {
        Some(pred.start)
    } else {
        unprotect(page, dev.granule, pred.device);
        None
    }
}
