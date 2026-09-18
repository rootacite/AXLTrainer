/* The peralloc route (../amdfq.md §12), ported from ../amdfq-vmm/amdfq_vmm.c: hipMalloc's requests are
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
 * operand (../conclusions/bf16-kernel-overrun.md) lands in memory that exists. Measured 2026-09-17
 * (../amdfq-vmm/vmm_probe.c): one handle maps at many addresses at once, 1000 mappings of one pad
 * handle cost the physical 2 MiB once, and a read across the boundary returns rather than faulting.
 *
 * One hipMemAddressReserve per allocation, given back by hipMemAddressFree when the pointer is freed:
 * no arena base, no bump pointer and no free list.
 *
 * State is per process, and whichever process first calls hipMalloc builds it and owns it. A process
 * that inherited another's state through fork touches none of it: it serves nothing, and it releases
 * nothing, because the handle belongs to the parent, which is still using it. Ownership is the pid
 * recorded when the state was built, so a process that forked before its parent ever allocated builds
 * a state of its own and serves out of that.
 *
 * Nothing here locks: concurrent hipMalloc calls run their own reserve/create/map sequences at the
 * same time, and that is the runtime's business to get right (DESIGN.md D2). */

use crate::hip::{
    AccessDesc, AllocationProp, HIP_SUCCESS, Handle, MEM_ACCESS_PROT_READWRITE,
    MEM_ALLOCATION_TYPE_PINNED, MEM_GRANULARITY_MINIMUM, MEM_GRANULARITY_RECOMMENDED,
    MEM_HANDLE_TYPE_NONE, MEM_LOCATION_DEVICE, MEM_LOCATION_HOST, MemLocation,
};
use crate::real;
use crate::registry::{Address, Extent, HookData, Origin};
use std::ffi::c_void;
use std::fmt;
use std::ptr;
use std::sync::LazyLock;

/* The runtime entry points the route needs, resolved as a set: the route stays off unless all of
 * them are there, rather than failing halfway through a request. */
#[derive(Clone, Copy)]
struct Entries {
    get_device: real::HipGetDeviceFn,
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
}

impl Entries {
    fn resolve() -> Option<Self> {
        Some(Entries {
            get_device: (*real::HIP_GET_DEVICE)?,
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
        })
    }
}

/* What the route knows once it is running. */
#[derive(Clone, Copy)]
struct Setup {
    entries: Entries,
    device_id: i32,
    /* Block rounding and pad size, both the recommended granularity — the same magnitude the
     * runtime's own pool charges, so rounding costs nothing hipMalloc would not have cost anyway
     * (measured: a 1/2/4/8/32/44 MiB request all cost the request). */
    granule: usize,
    /* The pid that built this state; a fork child finds its parent's here. */
    owner: u32,
}

/* Built on the first hipMalloc this process makes, when the real entry points are resolvable. A
 * failure is remembered, and later calls forward without asking the runtime again (DESIGN.md D7). */
static SETUP: LazyLock<Option<Setup>> = LazyLock::new(build);

/* The one physical object every served block's pad is a mapping of: created once, released never. */
static PAD: LazyLock<Option<Handle>> = LazyLock::new(build_pad);

fn build() -> Option<Setup> {
    let entries = Entries::resolve()?;
    let mut device_id = 0;
    if unsafe { (entries.get_device)(&mut device_id) } != HIP_SUCCESS {
        return None;
    }
    let prop = prop(device_id);
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
    Some(Setup {
        entries,
        device_id,
        granule: recommended,
        owner: std::process::id(),
    })
}

fn build_pad() -> Option<Handle> {
    let setup = (*SETUP).as_ref()?;
    let mut raw: *mut c_void = ptr::null_mut();
    if unsafe { (setup.entries.create)(&mut raw, setup.granule, &prop(setup.device_id), 0) }
        != HIP_SUCCESS
    {
        return None;
    }
    Some(Handle::from_raw(raw))
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
    let setup = (*SETUP).as_ref()?;
    /* A fork child never touches state it inherited. */
    if setup.owner != std::process::id() {
        return None;
    }
    if size == 0 {
        return None;
    }
    /* The block rounds up to the granularity and the pad is one granule behind it, so the request has
     * to leave room for both. */
    if size > usize::MAX - 2 * setup.granule {
        return None;
    }
    /* A block without its pad behind it is not worth serving: a process whose pad could not be
     * created forwards everything (the C version disabled the route the same way). */
    let pad = (*PAD)?;

    let block = size.div_ceil(setup.granule) * setup.granule;
    let total = block + setup.granule;

    let address = reserve(&setup.entries, total, setup.granule)?;

    let Some(handle) = create(&setup.entries, block, setup.device_id) else {
        free_address(&setup.entries, address, total);
        return None;
    };
    if unsafe { (setup.entries.map)(address.as_ptr(), block, 0, handle.as_raw(), 0) } != HIP_SUCCESS
    {
        unsafe { (setup.entries.release)(handle.as_raw()) };
        free_address(&setup.entries, address, total);
        return None;
    }
    set_access(setup, address, block);

    /* The block is valid and usable without the slack behind it, so a pad mapping that fails costs
     * this allocation its protection and nothing else. */
    let pad_behind = address.offset(block);
    let pad_size =
        if unsafe { (setup.entries.map)(pad_behind.as_ptr(), setup.granule, 0, pad.as_raw(), 0) }
            == HIP_SUCCESS
        {
            set_access(setup, pad_behind, setup.granule);
            Some(setup.granule)
        } else {
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
        }),
    })
}

/* What became of a free the registry had an extent for. */
#[derive(Clone, Copy)]
pub(crate) enum Outcome {
    /* Unmapped and given back here; the runtime never saw the pointer. */
    Released,
    /* Inherited through fork: left alone, because the parent is still using that handle. */
    Inherited,
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Outcome::Released => write!(f, "released"),
            Outcome::Inherited => write!(f, "inherited, left to the parent"),
        }
    }
}

/* Gives one served extent back. Only called for a record whose origin is an extent, so the pointer is
 * one of ours: the runtime must not see it either way. */
pub(crate) fn release(address: Address, extent: Extent) -> Outcome {
    let Some(setup) = (*SETUP).as_ref() else {
        /* Nothing to undo the mapping with — better to leave it than to hand a pointer the runtime
         * never allocated to hipFree. */
        return Outcome::Inherited;
    };
    if setup.owner != std::process::id() {
        return Outcome::Inherited;
    }

    /* hipFree is not a bare teardown: ihipFree waits for the device (SyncAllStreams, before the
     * external/SVM free) when the memory is not pool-owned, so work still reading the block has
     * finished by the time the mapping goes away. The route bypasses that path and takes its blocks
     * apart itself, so it owes the same wait — there is no public "sync all streams", the device-wide
     * sync is the equivalent. Kept for that parity, not as a fix: the NaN and the hang survive it
     * (../conclusions/hook-free-path-nan-vs-oom.md §6). */
    let synced = unsafe { (setup.entries.device_synchronize)() };

    if synced != HIP_SUCCESS {
        /* A wait that failed reports a fault the workload already had pending; the teardown still
         * has to happen, and a failure left sticky here would be read as the next call's own. */
        clear_error(setup);
    }

    unsafe { (setup.entries.unmap)(address.as_ptr(), extent.block) };
    if let Some(pad) = extent.pad {
        unsafe { (setup.entries.unmap)(address.offset(extent.block).as_ptr(), pad) };
    }
    unsafe { (setup.entries.release)(extent.handle.as_raw()) };
    free_address(&setup.entries, address, extent.total);
    Outcome::Released
}

fn reserve(entries: &Entries, total: usize, granule: usize) -> Option<Address> {
    let mut raw: *mut c_void = ptr::null_mut();
    if unsafe { (entries.reserve)(&mut raw, total, granule, ptr::null_mut(), 0) } != HIP_SUCCESS {
        return None;
    }
    Some(Address::from_ptr(raw))
}

fn create(entries: &Entries, block: usize, device_id: i32) -> Option<Handle> {
    let mut raw: *mut c_void = ptr::null_mut();
    if unsafe { (entries.create)(&mut raw, block, &prop(device_id), 0) } != HIP_SUCCESS {
        return None;
    }
    Some(Handle::from_raw(raw))
}

fn free_address(entries: &Entries, address: Address, total: usize) {
    unsafe { (entries.address_free)(address.as_ptr(), total) };
}

/* Two locations, because hipMalloc's memory is readable by the CPU at the same address and a
 * device-only mapping is not: measured 2026-09-17 by reading a mapped block from host code
 * (../amdfq-vmm/vmm_probe.c `hostaccess`), after a training run died in
 * at::native::_local_scalar_dense_cuda doing exactly that read — torch's `.item()` reads a device
 * pointer from the host on this stack. A range with an unmapped gap in it is rejected outright, so
 * this is done per region. */
fn set_access(setup: &Setup, address: Address, size: usize) {
    let desc = [
        AccessDesc {
            location: MemLocation {
                type_: MEM_LOCATION_DEVICE,
                id: setup.device_id,
            },
            flags: MEM_ACCESS_PROT_READWRITE,
        },
        AccessDesc {
            location: MemLocation {
                type_: MEM_LOCATION_HOST,
                id: 0,
            },
            flags: MEM_ACCESS_PROT_READWRITE,
        },
    ];
    if unsafe { (setup.entries.set_access)(address.as_ptr(), size, desc.as_ptr(), 2) }
        == HIP_SUCCESS
    {
        return;
    }
    clear_error(setup);
    if unsafe { (setup.entries.set_access)(address.as_ptr(), size, desc.as_ptr(), 1) }
        == HIP_SUCCESS
    {
        return;
    }
    clear_error(setup);
}

/* Consume the runtime's sticky error state after a call of ours that was allowed to fail. Nothing can
 * set that state back, so a failure the caller had left pending before entering our hook is lost with
 * it; that is the price of asking the runtime a question whose answer is an error. */
fn clear_error(setup: &Setup) {
    unsafe { (setup.entries.get_last_error)() };
}
