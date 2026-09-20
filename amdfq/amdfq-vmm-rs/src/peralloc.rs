/* The peralloc route (../doc/amdfq.md §12), ported from the retired C VMM tree (amdfq-vmm/amdfq_vmm.c): hipMalloc's requests are
 * served from address ranges this crate reserves and populates itself, so the slack behind a block
 * costs *one* physical granule for the whole process instead of one granule of VRAM per live
 * allocation — which is what padding every request costs, once the runtime rounds it up.
 *
 * The layout of one served allocation:
 *
 *     va                    va + block                     va + block + pad
 *     |---- block ----|---- pad (shared handle) ----|
 *
 * `block` is the request rounded up to the mapping granularity, mapped from a handle created for it;
 * `pad` is a mapping of one handle created once per process, so a kernel reading past the end of its
 * operand (../../conclusions/bf16-kernel-overrun.md) lands in memory that exists. Measured 2026-09-17
 * (the retired C VMM tree (amdfq-vmm/vmm_probe.c)): one handle maps at many addresses at once, 1000 mappings of one pad
 * handle cost the physical 2 MiB once, and a read across the boundary returns rather than faulting.
 *
 * One hipMemAddressReserve per allocation. After that range has been mapped, it is never given
 * back: a free unmaps and releases the handle, and leaves the VA reserved so the process cannot
 * map any address in the span again. hipMemAddressFree is only for a reserve that never became a
 * mapping (create/map failed). No arena base, no bump pointer and no free list.
 *
 * State is per device, not per process: whichever device a gate runs on has its own granularity, its
 * own shared pad granule and its own peer set, built on that device's first use (D11). The device is
 * read at the gate, never remembered, and every served block records the device it was made on, so a
 * free can switch back to it — the VMM calls act on the current device.
 *
 * The grant a block gets is its own device, every device that can reach this one, and the host. A
 * mapping's reach comes only from hipMemSetAccess, and hipDeviceEnablePeerAccess adds nothing to a
 * mapping it did not make, so the peers are asked for here; without them a kernel, a peer copy or a
 * collective on another card would meet a page with no permission where a hipMalloc'd block is
 * reachable.
 *
 * A request is served only when every step of it succeeded, and the steps that can end a request
 * with a warning instead of a failure say so: a block whose device+host grant did not land is
 * unmapped (VA kept) and forwarded rather than handed out — device-only and ungranted ranges fault
 * the reader — and a pad that could not be mapped costs this allocation its protection.
 *
 * A free of a served extent is a teardown: the runtime never sees the pointer (DESIGN.md D10).
 *
 * The serve path locks nothing: concurrent hipMalloc calls run their own reserve/create/map sequences
 * at the same time, and that is the runtime's business to get right (DESIGN.md D2). */

use crate::hip::{
    AccessDesc, AllocationProp, HIP_SUCCESS, Handle, HipError, MEM_ACCESS_PROT_READWRITE,
    MEM_ALLOCATION_TYPE_PINNED, MEM_GRANULARITY_MINIMUM, MEM_GRANULARITY_RECOMMENDED,
    MEM_HANDLE_TYPE_NONE, MEM_LOCATION_DEVICE, MEM_LOCATION_HOST, MemLocation,
};
use crate::real;
use crate::registry::{Address, Extent, HookData, Origin};
use std::collections::BTreeMap;
use std::ffi::{c_void, OsString};
use std::fmt;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{LazyLock, OnceLock, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/* The runtime entry points the route needs, resolved as a set: the route stays off unless all of
 * them are there, rather than failing halfway through a request. */
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
    release: real::HipMemReleaseFn,
    set_access: real::HipMemSetAccessFn,
    get_last_error: real::HipGetLastErrorFn,
    device_synchronize: real::HipDeviceSynchronizeFn,
    /* Optional: without them the route still serves, it just cannot grant peers (D11). */
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
            release: (*real::HIP_MEM_RELEASE)?,
            set_access: (*real::HIP_MEM_SET_ACCESS)?,
            get_last_error: (*real::HIP_GET_LAST_ERROR)?,
            device_synchronize: (*real::HIP_DEVICE_SYNCHRONIZE)?,
            device_count: *real::HIP_GET_DEVICE_COUNT,
            can_access_peer: *real::HIP_DEVICE_CAN_ACCESS_PEER,
        })
    }
}

/* What the route knows about one device: its allocation granularity, the shared pad granule mapped
 * behind that device's blocks, and the devices that may reach them. */
struct Device {
    id: i32,
    /* Block rounding and pad size, both the recommended granularity — the same magnitude the
     * runtime's own pool charges, so rounding costs nothing hipMalloc would not have cost anyway
     * (measured: a 1/2/4/8/32/44 MiB request all cost the request). */
    granule: usize,
    /* The one physical object on this device that every served block's pad is a mapping of: created
     * once, released never. */
    pad: Handle,
    /* Devices that can read and write this device's memory (hipDeviceCanAccessPeer). */
    peers: Vec<i32>,
}

/* The entry points the route needs, resolved as a set: the route stays off unless all of them are
 * there, rather than failing halfway through a request. A failure is remembered (DESIGN.md D7). */
static ENTRIES: LazyLock<Option<Entries>> = LazyLock::new(Entries::resolve);

/* One slot per device ordinal, built on that device's first hipMalloc. The list of globals is closed
 * (D5), so a device ordinal past the end of this table is forwarded rather than remembered. */
const MAX_DEVICES: usize = 32;

static DEVICES: [OnceLock<Option<Device>>; MAX_DEVICES] = [const { OnceLock::new() }; MAX_DEVICES];

/* Spans that have been mapped at least once: start -> reserved length. A later reserve that
 * overlaps one of these is not mapped (DESIGN.md D10). Held only across the map operation. */
static EVER_MAPPED: LazyLock<RwLock<BTreeMap<usize, usize>>> =
    LazyLock::new(|| RwLock::new(BTreeMap::new()));

/* True if `[address, address + total)` overlaps a span that has already been mapped. */
fn range_taken(address: Address, total: usize) -> bool {
    let start = address.as_usize();
    let end = start.saturating_add(total);
    let mapped = EVER_MAPPED
        .read()
        .unwrap_or_else(|poison| poison.into_inner());
    if let Some((&prev, &len)) = mapped.range(..=start).next_back() {
        if start < prev.saturating_add(len) && prev < end {
            return true;
        }
    }
    if let Some((&next, _)) = mapped.range(start.saturating_add(1)..).next() {
        if next < end {
            return true;
        }
    }
    false
}

/* Records that this reserved span has been mapped, so it is never mapped again. */
fn remember_mapped(address: Address, total: usize) {
    let mut mapped = EVER_MAPPED
        .write()
        .unwrap_or_else(|poison| poison.into_inner());
    mapped.insert(address.as_usize(), total);
    let used = mapped.values().fold(0u64, |acc, &len| acc.saturating_add(len as u64));
    let spans = mapped.len();
    drop(mapped);
    publish_va_status(used, spans);
}

/* Ranko reads `<stem>.<pid>.json`. Stem is `AMDFQ_VA_STATUS` if set, otherwise the same
 * runtime dir as trainer/control.py (`AXL_RUNTIME_DIR` / `$XDG_RUNTIME_DIR/axltrainer` /
 * `/tmp/axltrainer-$UID`) plus `amdfq_vmm_va`. Throttled. */
const VA_STATUS_INTERVAL_MS: u64 = 1000;
static VA_LAST_WRITE_MS: AtomicU64 = AtomicU64::new(0);
static VA_WRITE_WARNED: AtomicBool = AtomicBool::new(false);

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn process_uid() -> Option<u32> {
    let text = fs::read_to_string("/proc/self/status").ok()?;
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("Uid:") else {
            continue;
        };
        return rest.split_whitespace().next()?.parse().ok();
    }
    None
}

fn va_status_stem() -> Option<OsString> {
    let explicit = std::env::var_os("AMDFQ_VA_STATUS");
    if let Some(stem) = explicit {
        if !stem.is_empty() {
            return Some(stem);
        }
    }
    let mut dir = if let Some(axl) = std::env::var_os("AXL_RUNTIME_DIR") {
        if axl.is_empty() {
            return None;
        }
        PathBuf::from(axl)
    } else if let Some(xdg) = std::env::var_os("XDG_RUNTIME_DIR") {
        if xdg.is_empty() {
            return None;
        }
        let mut path = PathBuf::from(xdg);
        path.push("axltrainer");
        path
    } else {
        PathBuf::from(format!("/tmp/axltrainer-{}", process_uid()?))
    };
    if fs::create_dir_all(&dir).is_err() {
        return None;
    }
    dir.push("amdfq_vmm_va");
    Some(dir.into_os_string())
}

fn publish_va_status(used_bytes: u64, spans: usize) {
    let Some(stem) = va_status_stem() else {
        return;
    };
    let now = now_ms();
    let last = VA_LAST_WRITE_MS.load(Ordering::Relaxed);
    if last != 0 && now.saturating_sub(last) < VA_STATUS_INTERVAL_MS {
        return;
    }
    VA_LAST_WRITE_MS.store(now, Ordering::Relaxed);

    let pid = std::process::id();
    let mut dest = stem;
    dest.push(format!(".{pid}.json"));
    let mut tmp = dest.clone();
    tmp.push(".tmp");
    let ts = (now as f64) / 1000.0;
    let body = format!(
        "{{\"pid\":{pid},\"used_bytes\":{used_bytes},\"spans\":{spans},\"ts\":{ts:.3}}}\n"
    );
    let write_ok = (|| {
        let mut file = fs::File::create(&tmp)?;
        file.write_all(body.as_bytes())?;
        file.sync_all()?;
        fs::rename(&tmp, &dest)
    })();
    if let Err(err) = write_ok {
        let _ = fs::remove_file(&tmp);
        if !VA_WRITE_WARNED.swap(true, Ordering::Relaxed) {
            log::warn!(
                "amdfq va status write {} failed: {err}",
                PathBuf::from(&dest).display()
            );
        }
    }
}

/* The state of one device, built on its first use and remembered even when the build failed. */
fn device_state(entries: &Entries, id: i32) -> Option<&'static Device> {
    let index = usize::try_from(id).ok()?;
    let Some(slot) = DEVICES.get(index) else {
        log::warn!("device ordinal {id} is past the {MAX_DEVICES}-slot device table, forwarding");
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
        return None;
    }
    /* Once per device per process: what this device's blocks are rounded to, and which other devices
     * their mappings are granted to (D11). */
    let peers = peers_of(entries, id);
    log::info!("device {id}: granule={recommended} peers={peers:?}");
    Some(Device {
        id,
        granule: recommended,
        pad: Handle::from_raw(raw),
        peers,
    })
}

/* The devices that may reach memory on `id`. Capability, not hipDeviceEnablePeerAccess: that gate adds
 * nothing to a mapping this route made, and asking for capability covers peers the app turns on later.
 * The query is the one the current device can make — this device reaching the other — and is taken as
 * symmetric, which is what AMD's topology gives. */
fn peers_of(entries: &Entries, id: i32) -> Vec<i32> {
    let (Some(device_count), Some(can_access_peer)) =
        (entries.device_count, entries.can_access_peer)
    else {
        log::warn!(
            "hipGetDeviceCount/hipDeviceCanAccessPeer not resolvable: served blocks get no peer grant"
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
        /* The grant is written into a fixed-size descriptor set: this device, one entry per peer, and
         * the host. */
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

/* The device the calling thread is on, read at the gate rather than remembered (D11). A failure here
 * leaves the runtime's sticky error set, and it is consumed before forwarding. */
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

/* Serves one hipMalloc request, or returns None to have the hook forward it unchanged. */
pub(crate) fn serve(size: usize) -> Option<HookData> {
    let entries = (*ENTRIES).as_ref()?;
    if size == 0 {
        return None;
    }
    /* The device this request belongs to is the one the caller is on, read here (D11). */
    let id = current_device(entries)?;
    /* A device whose state cannot be built serves nothing: its pad granule would be missing (the C
     * version disabled the route the same way). */
    let dev = device_state(entries, id)?;
    /* The block rounds up to the granularity and the pad is one granule behind it, so the request has
     * to leave room for both. */
    if size > usize::MAX - 2 * dev.granule {
        return None;
    }

    let block = size.div_ceil(dev.granule) * dev.granule;
    let total = block + dev.granule;

    let address = match reserve(entries, total, dev.granule) {
        Ok(address) => address,
        Err(ret) => {
            log::warn!(
                "hipMalloc(size={size}) hipMemAddressReserve(total={total}) -> {ret}, forwarding"
            );
            return None;
        }
    };
    /* A span that has already been mapped stays reserved; mapping it again is the reuse this
     * route refuses. The overlapping reserve is also left in place — freeing it could punch a
     * hole in the older span. */
    if range_taken(address, total) {
        log::warn!(
            "hipMalloc(size={size}) hipMemAddressReserve(total={total}) -> va={address} overlaps a mapped span, forwarding"
        );
        return None;
    }

    let handle = match create(entries, block, dev.id) {
        Ok(handle) => handle,
        Err(ret) => {
            log::warn!("hipMalloc(size={size}) hipMemCreate(block={block}) -> {ret}, forwarding");
            warn_if_failed(
                "hipMemAddressFree",
                address,
                free_address(entries, address, total),
            );
            return None;
        }
    };
    let mapped = unsafe { (entries.map)(address.as_ptr(), block, 0, handle.as_raw(), 0) };
    if mapped != HIP_SUCCESS {
        log::warn!(
            "hipMalloc(size={size}) hipMemMap(va={address} block={block}) -> {mapped}, forwarding"
        );
        warn_if_failed("hipMemRelease", address, unsafe {
            (entries.release)(handle.as_raw())
        });
        warn_if_failed(
            "hipMemAddressFree",
            address,
            free_address(entries, address, total),
        );
        return None;
    }
    /* The range has been mapped: it is abandoned as a VA even if this request is not served. */
    remember_mapped(address, total);
    /* A block whose grant did not land is not served: with device-only the host read faults, with no
     * grant the device read does, and the runtime never sees this pointer either way (DESIGN.md D10),
     * so the mapping and handle are undone here and the VA stays reserved. */
    if set_access(entries, dev, address, block, "block") != Access::Granted {
        log::warn!(
            "hipMalloc(size={size}) not served: va={address} block={block} has no device+host grant"
        );
        warn_if_failed("hipMemUnmap", address, unsafe {
            (entries.unmap)(address.as_ptr(), block)
        });
        warn_if_failed("hipMemRelease", address, unsafe {
            (entries.release)(handle.as_raw())
        });
        return None;
    }

    /* The block is valid and usable without the slack behind it, so a pad mapping that fails costs
     * this allocation its protection and nothing else. */
    let pad_behind = address.offset(block);
    let mapped_pad =
        unsafe { (entries.map)(pad_behind.as_ptr(), dev.granule, 0, dev.pad.as_raw(), 0) };
    let pad_size = if mapped_pad == HIP_SUCCESS {
        if set_access(entries, dev, pad_behind, dev.granule, "pad") != Access::Granted {
            log::warn!(
                "hipMalloc(size={size}) pad at {pad_behind} has no device+host grant: the slack behind this block is unprotected"
            );
        }
        Some(dev.granule)
    } else {
        log::warn!(
            "hipMalloc(size={size}) hipMemMap(pad va={pad_behind}) -> {mapped_pad}: the slack behind this block is unmapped"
        );
        None
    };

    Some(HookData {
        address,
        size,
        origin: Origin::Extent(Extent {
            block,
            total,
            handle,
            pad: pad_size,
            device: dev.id,
        }),
    })
}

/* What became of a free the registry had an extent for. */
#[derive(Clone, Copy)]
pub(crate) enum Outcome {
    /* Unmapped and given back here; the runtime never saw the pointer. */
    Released,
    /* Some step of the teardown failed. The record is out of the map either way (DESIGN.md D6), so
     * the warnings it left are the only record that this extent was not fully given back. */
    Incomplete,
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Outcome::Released => write!(f, "released"),
            Outcome::Incomplete => write!(f, "released with failures, see the warnings"),
        }
    }
}

/* Gives one served extent back. Only called for a record whose origin is an extent, so the pointer
 * is one of ours: the runtime must not see it. */
pub(crate) fn release(address: Address, extent: Extent) -> Outcome {
    let Some(entries) = (*ENTRIES).as_ref() else {
        /* Without the entry points nothing was ever served, so this cannot be reached; the extent is
         * not handed to hipFree either way, and saying it was given back would be a lie. */
        log::warn!("hipFree(va={address}): no runtime entry points to undo an extent with");
        return Outcome::Incomplete;
    };

    if teardown(entries, address, extent) {
        Outcome::Incomplete
    } else {
        Outcome::Released
    }
}

/* Takes one extent apart for real, and reports whether a step of that failed. */
fn teardown(entries: &Entries, address: Address, extent: Extent) -> bool {
    /* The block belongs to the device it was made on, and every call below acts on the current device,
     * which the caller may have moved away from (D11). */
    let switched = unsafe { (entries.set_device)(extent.device) };
    if switched != HIP_SUCCESS {
        log::warn!(
            "hipSetDevice({}) -> {switched}: the teardown of va={address} runs on whatever device is current",
            extent.device
        );
        clear_error(entries);
    }

    /* hipFree is not a bare teardown: ihipFree waits for the device (SyncAllStreams, before the
     * external/SVM free) when the memory is not pool-owned, so work still reading the block has
     * finished by the time the mapping goes away. The route bypasses that path and takes its blocks
     * apart itself, so it owes the same wait — there is no public "sync all streams", the device-wide
     * sync is the equivalent. Kept for that parity, not as a fix: the NaN and the hang survive it
     * (../doc/hook-free-path-nan-vs-oom.md §6). */
    let synced = unsafe { (entries.device_synchronize)() };

    if synced != HIP_SUCCESS {
        /* A wait that failed reports a fault the workload already had pending; the teardown still
         * has to happen, and a failure left sticky here would be read as the next call's own. */
        clear_error(entries);
    }

    /* Unmap and release the handle; the reserved VA is left in place so this span cannot be
     * mapped again (DESIGN.md D10). A step that fails is a warning — the record is already out
     * of the registry (D6). */
    let mut incomplete = false;
    let unmapped = unsafe { (entries.unmap)(address.as_ptr(), extent.block) };
    incomplete |= unmapped != HIP_SUCCESS;
    warn_if_failed("hipMemUnmap(block)", address, unmapped);

    if let Some(pad) = extent.pad {
        let pad_behind = address.offset(extent.block);
        let unmapped_pad = unsafe { (entries.unmap)(pad_behind.as_ptr(), pad) };
        incomplete |= unmapped_pad != HIP_SUCCESS;
        warn_if_failed("hipMemUnmap(pad)", pad_behind, unmapped_pad);
    }
    let released = unsafe { (entries.release)(extent.handle.as_raw()) };
    incomplete |= released != HIP_SUCCESS;
    warn_if_failed("hipMemRelease", address, released);

    incomplete
}

/* Both of these hand the runtime's own return code back: a request the route cannot set up is
 * forwarded, and why it was forwarded is the only thing that makes that visible. */
fn reserve(entries: &Entries, total: usize, granule: usize) -> Result<Address, HipError> {
    let mut raw: *mut c_void = ptr::null_mut();
    let ret = unsafe { (entries.reserve)(&mut raw, total, granule, ptr::null_mut(), 0) };
    if ret != HIP_SUCCESS {
        return Err(ret);
    }
    Ok(Address::from_ptr(raw))
}

fn create(entries: &Entries, block: usize, device_id: i32) -> Result<Handle, HipError> {
    let mut raw: *mut c_void = ptr::null_mut();
    let ret = unsafe { (entries.create)(&mut raw, block, &prop(device_id), 0) };
    if ret != HIP_SUCCESS {
        return Err(ret);
    }
    Ok(Handle::from_raw(raw))
}

fn free_address(entries: &Entries, address: Address, total: usize) -> HipError {
    unsafe { (entries.address_free)(address.as_ptr(), total) }
}

/* A step of a teardown that returned an error, if it did. None of them is supposed to fail, so a
 * failure is a warning and not a reason to keep the caller waiting; the record leaves the map either
 * way (DESIGN.md D6), and the warning is the only trace it leaves. */
fn warn_if_failed(what: &str, address: Address, ret: HipError) {
    if ret != HIP_SUCCESS {
        log::warn!("{what}(va={address}) -> {ret}");
    }
}

/* What the access grant for one range came to. */
#[derive(Clone, Copy, PartialEq, Eq)]
enum Access {
    /* Every location that needs it may read and write the range: the block's device, its peers, the
     * host. What a served block needs. */
    Granted,
    /* The device alone. Its own kernels are fine; the host read torch's `.item()` does, and any peer,
     * meet a page with no permission. */
    DeviceOnly,
    /* Neither grant landed: nothing may touch the range. */
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

/* More than one location, because hipMalloc's memory is readable by the CPU at the same address and a
 * device-only mapping is not: measured 2026-09-17 by reading a mapped block from host code
 * (the retired C VMM tree (amdfq-vmm/vmm_probe.c) `hostaccess`), after a training run died in
 * at::native::_local_scalar_dense_cuda doing exactly that read — torch's `.item()` reads a device
 * pointer from the host on this stack. A range with an unmapped gap in it is rejected outright, so
 * this is done per region.
 *
 * The peer descriptors are the part a mapping does not get for free (D11): hipDeviceEnablePeerAccess
 * grants nothing to a mapping this route made, so a device that can reach this one is written into
 * the set here. What came of the grant is reported rather than swallowed: the second call asks for the
 * device alone, which says whether the range is unusable or only unreachable from outside the device. */
fn set_access(
    entries: &Entries,
    dev: &Device,
    address: Address,
    size: usize,
    what: &str,
) -> Access {
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
            "set_access({what} va={address} size={size}): device+{} peer(s)+host -> {both}, device-only -> {device}: host and peer reads of this range fault",
            dev.peers.len()
        );
        return Access::DeviceOnly;
    }
    clear_error(entries);
    log::warn!(
        "set_access({what} va={address} size={size}): device+{} peer(s)+host -> {both}, device-only -> {device}: no access granted",
        dev.peers.len()
    );
    Access::None
}

/* Consume the runtime's sticky error state after a call of ours that was allowed to fail. Nothing can
 * set that state back, so a failure the caller had left pending before entering our hook is lost with
 * it; that is the price of asking the runtime a question whose answer is an error. */
fn clear_error(entries: &Entries) {
    unsafe { (entries.get_last_error)() };
}
