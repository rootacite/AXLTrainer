/* The pool route: one `hipMemCreate` per pool instead of one per `hipMalloc`.
 *
 * A pool is one `PoolSize` handle, mapped once at one reserved span, with one shared pad granule
 * mapped behind that span — the three steps the solo route (peralloc.rs) runs per request, run once
 * for the whole pool. A request the knob's rule admits (`size <= PoolSize / 2`, so two such blocks
 * always share one pool) is then carved out by bookkeeping alone: no `hipMemCreate`, no
 * `hipMemAddressReserve`, no `hipMemMap`, no `hipMemSetAccess`. More than one pool may exist at once
 * (a new one is built when no existing pool has a range left), and a pool whose last block is freed
 * gives its handle and its VA back, so its VRAM is held only while something uses it.
 *
 * Layout, one size up from the solo route's:
 *
 *     va                     va + PoolSize                  va + PoolSize + granule
 *     |---- PoolSize (one mapping of the pool handle) ----|---- tail pad (shared handle) ----|
 *     |-- block 1 --|-- block 2 --| ... |-- live blocks are neighbours inside the pool --|
 *
 * The tail pad is what the solo route puts behind every block; inside a pool, everything up to the
 * pool's end is mapped anyway, so one granule behind the whole pool covers every block in it.
 *
 * What is *not* free about this, and why the knob is off by default and bigger is not better
 * (DESIGN.md D12): a pool is committed VRAM no other client can get until its last block is freed,
 * and the blocks sharing a pool are neighbours — an over-read still lands in mapped memory, but an
 * over-*write* past a block's end can reach another live block where a solo allocation would have hit
 * the throwaway pad behind it.
 *
 * Two globals (DESIGN.md D5): the parsed knob and the table of live pools. The table's lock covers
 * crate bookkeeping only — never a call into the runtime, never a log line (D3). Building a pool
 * happens with the lock released, and a pool enters the table only once every step of its
 * construction succeeded; a pool that could not be built completely is undone and the request that
 * asked for it goes down the solo route instead, so "the pool is unusable" degrades to today's
 * behaviour rather than to a new failure mode. */

use crate::hip::{HIP_SUCCESS, Handle};
use crate::peralloc::{self, Access, Device, Entries, Outcome};
use crate::registry::{Address, Extent, HookData, Origin, Pooled};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, RwLock, RwLockWriteGuard};

/* The knob's range, in bytes: the pool size a config value is clamped into. */
pub(crate) const MIN_POOL_SIZE: usize = 16 << 20;
pub(crate) const MAX_POOL_SIZE: usize = 512 << 20;

/* The requested pool size in bytes, or 0 for off. `AMDFQ_POOL_SIZE` is a byte count, like
 * `AMDFQ_VRAM_RESERVE`; the config key behind it (`amdfq_pool_mib`) is MiB. */
static POOL_SIZE: LazyLock<usize> = LazyLock::new(requested_pool_size);

static SIZE_UNAVAILABLE_WARNED: AtomicBool = AtomicBool::new(false);

fn requested_pool_size() -> usize {
    match std::env::var("AMDFQ_POOL_SIZE") {
        Ok(raw) => parse_pool_size(Some(raw.as_str())),
        Err(std::env::VarError::NotPresent) => parse_pool_size(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            log::warn!("AMDFQ_POOL_SIZE is not UTF-8, pool off");
            0
        }
    }
}

/* Pure, so the clamp and the "off" cases are testable without a runtime. An out-of-range value is
 * clamped rather than refused: the floor is what must not be crossed silently, and the ceiling is
 * the size that costs a 16 GB card real headroom. */
fn parse_pool_size(raw: Option<&str>) -> usize {
    let Some(raw) = raw else {
        log::info!("pool off (AMDFQ_POOL_SIZE unset)");
        return 0;
    };
    match raw.trim().parse::<usize>() {
        Ok(0) => {
            log::info!("pool off");
            0
        }
        Ok(bytes) if (MIN_POOL_SIZE..=MAX_POOL_SIZE).contains(&bytes) => {
            log::info!("pool size={bytes} bytes requested");
            bytes
        }
        Ok(bytes) => {
            let clamped = bytes.clamp(MIN_POOL_SIZE, MAX_POOL_SIZE);
            log::warn!(
                "AMDFQ_POOL_SIZE={bytes} is outside [{MIN_POOL_SIZE}, {MAX_POOL_SIZE}], using {clamped}"
            );
            clamped
        }
        Err(_) => {
            log::warn!("AMDFQ_POOL_SIZE={raw:?} is not an integer byte count, pool off");
            0
        }
    }
}

/* The pool size for one device: an even number of that device's allocation granules, so `PoolSize/2`
 * is a whole number of granules and every request the rule admits fits an empty pool exactly. */
pub(crate) fn size_for(granule: usize) -> Option<usize> {
    size_for_request(*POOL_SIZE, granule)
}

fn size_for_request(requested: usize, granule: usize) -> Option<usize> {
    if requested == 0 || granule == 0 {
        return None;
    }
    let granules = (requested / granule) & !1;
    if granules == 0 {
        /* One granule is bigger than the whole pool: nothing could ever be carved (the granule is
         * read per device, so this is a per-device condition, but it does not change per call). */
        if !SIZE_UNAVAILABLE_WARNED.swap(true, Ordering::Relaxed) {
            log::warn!("pool unusable: a granule of {granule} does not fit in {requested} bytes");
        }
        return None;
    }
    Some(granules * granule)
}

/* Where a block goes inside one pool: the ranges that are free right now, and nothing else. Taking
 * and returning is arithmetic on this map — no runtime call, no ordering guarantee needed beyond
 * "never hand the same byte out twice", which the map enforces by construction. Pure, so the layout
 * is unit-tested without a GPU. */
struct Layout {
    size: usize,
    /* offset -> length, non-empty, non-overlapping, never adjacent (they are coalesced). */
    free: BTreeMap<usize, usize>,
}

impl Layout {
    fn new(size: usize) -> Self {
        let mut free = BTreeMap::new();
        free.insert(0, size);
        Layout { size, free }
    }

    /* First fit: the lowest-offset range that is big enough, then the untouched space a pool has
     * left. A carve never spans two ranges (a block has to stay one contiguous mapping). */
    fn take(&mut self, len: usize) -> Option<usize> {
        if len == 0 || len > self.size {
            return None;
        }
        /* BTreeMap's iterator yields `(&offset, &length)`, so the two patterns are on the values. */
        let (offset, available) = self
            .free
            .iter()
            .find(|(_, available)| **available >= len)
            .map(|(&offset, &available)| (offset, available))?;
        self.free.remove(&offset);
        if available > len {
            self.free.insert(offset + len, available - len);
        }
        Some(offset)
    }

    /* Hands a range back and coalesces it with whatever it now touches. */
    fn give_back(&mut self, offset: usize, len: usize) {
        if len == 0 || offset.checked_add(len).is_none_or(|end| end > self.size) {
            return;
        }
        let mut start = offset;
        let mut end = offset + len;
        if let Some((&previous, &previous_len)) = self.free.range(..start).next_back() {
            if previous + previous_len == start {
                self.free.remove(&previous);
                start = previous;
            }
        }
        if let Some((&next, &next_len)) = self.free.range(end..).next() {
            if next == end {
                self.free.remove(&next);
                end = next + next_len;
            }
        }
        self.free.insert(start, end - start);
    }
}

/* One pool that is mapped right now. Its `handle` is released (and its VA given back, unless
 * AMDFQ_VA_NEVER_REUSE keeps it) the moment `live` reaches zero. */
struct Pool {
    id: u64,
    device: i32,
    /* Block starts: this address is what a carved block's record points at (plus `offset`). */
    address: Address,
    /* Bytes reserved at `address`: `size` + one granule, the tail pad included. */
    total: usize,
    /* PoolSize: the bytes mapped from `handle`, and the span blocks are carved out of. */
    size: usize,
    granule: usize,
    handle: Handle,
    /* The tail pad granule's length; it is mapped by construction, so this is not an Option. */
    pad: usize,
    layout: Layout,
    /* Blocks carved out of this pool that the upper layer has not freed yet. */
    live: usize,
}

struct Table {
    /* Monotonic and never reused: a registry record that names a pool can only name the one it was
     * carved from, even after that pool has been given back. */
    next_id: u64,
    pools: BTreeMap<u64, Pool>,
}

static POOLS: LazyLock<RwLock<Table>> = LazyLock::new(|| {
    RwLock::new(Table {
        next_id: 0,
        pools: BTreeMap::new(),
    })
});

/* The crate's second lock (registry.rs holds the first). Held only across the table and the layouts
 * (DESIGN.md D3) — every runtime call below is made with it released. */
fn lock() -> RwLockWriteGuard<'static, Table> {
    POOLS.write().unwrap_or_else(|poison| poison.into_inner())
}

/* Serves one hipMalloc request out of a pool, or asks the caller to run the solo route (None: the
 * knob is off, the request is above `PoolSize/2`, or the pool a request needed could not be built). */
pub(crate) fn serve(entries: &Entries, dev: &Device, size: usize) -> Option<HookData> {
    let pool_size = size_for(dev.granule)?;
    if size == 0 || size > pool_size / 2 {
        return None;
    }
    /* The carve is rounded up the way the solo route rounds a block, so a pooled block costs the
     * same bytes a solo one would. `PoolSize` is an even number of granules, so `pool_size / 2` is a
     * whole number of them and this can never exceed what an empty pool holds. */
    let len = size.div_ceil(dev.granule) * dev.granule;

    /* An existing pool first: pure bookkeeping, nothing committed, no runtime call at all. */
    if let Some(carved) = carve(dev.id, len) {
        return Some(record(size, carved));
    }

    /* A new pool commits `pool_size` of physical memory, so the reserve floor is asked about that
     * and not about `len`; a refusal sends this request to the solo route, where it is judged on its
     * own size — which is exactly what the knob being off would do. */
    if peralloc::vram_reserve_blocks(dev.id, pool_size) {
        log::warn!(
            "pool refused: creating pool_size={pool_size} would leave less than the reserve"
        );
        return None;
    }

    /* Physical first, then the VA, then the mappings: the same order as the solo route, so a
     * physical OOM never leaves a reserved span behind. */
    let handle = match peralloc::create(entries, pool_size, dev.id) {
        Ok(handle) => handle,
        Err(ret) => {
            log::warn!(
                "pool failed: hipMemCreate(pool_size={pool_size}) -> {ret}, staying on one block per request"
            );
            return None;
        }
    };
    let Some(total) = pool_size.checked_add(dev.granule) else {
        peralloc::release_warned(entries, handle, "pool");
        return None;
    };
    /* A pool held for a device that is past the device table cannot be tracked, but this one is. */
    let address = match peralloc::reserve(entries, total, dev.granule) {
        Ok(address) => address,
        Err(ret) => {
            log::warn!("pool failed: hipMemAddressReserve(total={total}) -> {ret}");
            peralloc::release_warned(entries, handle, "pool");
            return None;
        }
    };
    if peralloc::range_taken(address, total) {
        log::warn!("pool failed: reserve va={address} overlaps a mapped span");
        peralloc::release_warned(entries, handle, "pool");
        return None;
    }
    let mapped = peralloc::map(entries, address, pool_size, handle);
    if mapped != HIP_SUCCESS {
        log::warn!("pool failed: hipMemMap(va={address} size={pool_size}) -> {mapped}");
        peralloc::release_warned(entries, handle, "pool");
        peralloc::free_address_warned(entries, address, total);
        return None;
    }
    peralloc::remember_mapped(address, total);
    if peralloc::set_access(entries, dev, address, pool_size, "pool") != Access::Granted {
        log::warn!("pool failed: va={address} size={pool_size} has no device+host grant");
        /* Unmapped and released, but the span keeps its VA and stays in the mapped-span map, the way
         * the solo route's own grant failure does: this is not a teardown, so nothing may be served
         * inside the span afterwards. */
        peralloc::unmap_warned(entries, address, pool_size);
        peralloc::release_warned(entries, handle, "pool");
        return None;
    }
    /* The tail pad is what every block placed at the pool's end leans on, so it is part of the
     * construction rather than an optional extra the way it is for one solo block: without it this
     * pool would keep handing out a tail that faults on an over-read, over and over. */
    let pad_behind = address.offset(pool_size);
    let pad_mapped = peralloc::map(entries, pad_behind, dev.granule, dev.pad);
    if pad_mapped != HIP_SUCCESS {
        log::warn!(
            "pool failed: hipMemMap(tail pad va={pad_behind} size={}) -> {pad_mapped}",
            dev.granule
        );
        peralloc::unmap_warned(entries, address, pool_size);
        peralloc::release_warned(entries, handle, "pool");
        return None;
    }
    if peralloc::set_access(entries, dev, pad_behind, dev.granule, "pool pad") != Access::Granted {
        log::warn!("pool failed: tail pad va={pad_behind} has no device+host grant");
        peralloc::unmap_warned(entries, pad_behind, dev.granule);
        peralloc::unmap_warned(entries, address, pool_size);
        peralloc::release_warned(entries, handle, "pool");
        return None;
    }

    let mut table = lock();
    let id = table.next_id;
    table.next_id += 1;
    let mut layout = Layout::new(pool_size);
    let Some(offset) = layout.take(len) else {
        drop(table);
        /* Not reachable while `pool_size` is an even number of granules, but a cdylib aborts on a
         * panic, so the request falls back rather than trusting the arithmetic. This is the one
         * failure path that gives the span back as well: a mapping that never served anything and
         * cannot be published has no reason to hold its VA. */
        log::warn!("pool failed: a fresh pool of {pool_size} cannot hold one block of {len}");
        peralloc::unmap_warned(entries, pad_behind, dev.granule);
        peralloc::unmap_warned(entries, address, pool_size);
        peralloc::release_warned(entries, handle, "pool");
        peralloc::free_address_warned(entries, address, total);
        peralloc::forget_mapped(address);
        return None;
    };
    table.pools.insert(
        id,
        Pool {
            id,
            device: dev.id,
            address,
            total,
            size: pool_size,
            granule: dev.granule,
            handle,
            pad: dev.granule,
            layout,
            live: 1,
        },
    );
    drop(table);
    log::info!(
        "pool created id={id} device={} size={pool_size} granule={}",
        dev.id,
        dev.granule
    );
    Some(record(
        size,
        Carve {
            pool: id,
            offset,
            len,
            address: address.offset(offset),
        },
    ))
}

/* One range taken out of one pool. */
#[derive(Clone, Copy)]
struct Carve {
    pool: u64,
    offset: usize,
    len: usize,
    address: Address,
}

fn record(size: usize, carved: Carve) -> HookData {
    HookData {
        address: carved.address,
        size,
        origin: Origin::Pooled(Pooled {
            pool: carved.pool,
            offset: carved.offset,
            len: carved.len,
        }),
    }
}

/* A range out of a pool that already exists, on this device: the whole point of the route, and the
 * only user of `Layout::take`. `live` goes up before the lock is released, so a concurrent free
 * cannot see this pool as empty and tear it down under the block that is being handed out. */
fn carve(device: i32, len: usize) -> Option<Carve> {
    let mut table = lock();
    let mut found = None;
    for pool in table.pools.values_mut() {
        if pool.device != device {
            continue;
        }
        if let Some(offset) = pool.layout.take(len) {
            pool.live += 1;
            found = Some(Carve {
                pool: pool.id,
                offset,
                len,
                address: pool.address.offset(offset),
            });
            break;
        }
    }
    drop(table);
    found
}

/* Gives one carved block back. The pool's own object leaves the driver when its *last* block is
 * freed (`live == 0`) — the rule that keeps a pool from parking VRAM nobody is using: a pool lives
 * exactly as long as something it served does. */
pub(crate) fn release(address: Address, pooled: Pooled) -> Outcome {
    let emptied = {
        let mut table = lock();
        let Some(pool) = table.pools.get_mut(&pooled.pool) else {
            drop(table);
            log::warn!(
                "hipFree(va={address}): no pool {} left to give {} bytes back to",
                pooled.pool,
                pooled.len
            );
            return Outcome::Incomplete;
        };
        pool.layout.give_back(pooled.offset, pooled.len);
        pool.live = pool.live.saturating_sub(1);
        if pool.live == 0 {
            table.pools.remove(&pooled.pool)
        } else {
            None
        }
    };
    let Some(pool) = emptied else {
        return Outcome::Released;
    };

    log::info!(
        "pool released id={} device={} size={} granule={}",
        pool.id,
        pool.device,
        pool.size,
        pool.granule
    );
    let Some(entries) = peralloc::entries() else {
        /* Without the entry points nothing was ever served, so this cannot be reached; saying the
         * pool was given back would be a lie either way. */
        log::warn!("hipFree(va={address}): no runtime entry points to undo a pool with");
        return Outcome::Incomplete;
    };
    /* The pool's teardown is the solo teardown one size up: the whole mapped span plus the tail pad
     * behind it, on the pool's own device, with the wait the runtime's own free path does. */
    let extent = Extent {
        block: pool.size,
        total: pool.total,
        handle: pool.handle,
        pad: Some(pool.pad),
        device: pool.device,
    };
    if peralloc::teardown(entries, pool.address, extent) {
        Outcome::Incomplete
    } else {
        Outcome::Released
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /* The layout is the part of the pool that can be wrong without the GPU noticing, so it is the
     * part that gets tested here. */

    #[test]
    fn take_hands_out_the_lowest_offset_that_fits() {
        let mut layout = Layout::new(16);
        assert_eq!(layout.take(4), Some(0));
        assert_eq!(layout.take(4), Some(4));
        layout.give_back(0, 4);
        /* First fit, not "the end of the pool": the returned range is in front. */
        assert_eq!(layout.take(4), Some(0));
        assert_eq!(layout.take(8), Some(8));
    }

    #[test]
    fn take_splits_a_range_it_only_partly_uses() {
        let mut layout = Layout::new(64);
        assert_eq!(layout.take(32), Some(0));
        layout.give_back(0, 32);
        assert_eq!(layout.take(8), Some(0));
        /* 8 bytes are gone from the front of the returned range; the rest is still traceable. */
        assert_eq!(layout.take(24), Some(8));
        assert_eq!(layout.take(32), Some(32));
        assert_eq!(layout.take(1), None);
    }

    #[test]
    fn take_refuses_a_range_bigger_than_what_is_left() {
        let mut layout = Layout::new(16);
        assert_eq!(layout.take(16), Some(0));
        assert_eq!(layout.take(1), None);
        assert_eq!(layout.take(0), None);
    }

    #[test]
    fn a_returned_range_coalesces_with_both_neighbours() {
        let mut layout = Layout::new(96);
        assert_eq!(layout.take(32), Some(0));
        assert_eq!(layout.take(32), Some(32));
        assert_eq!(layout.take(32), Some(64));
        layout.give_back(32, 32);
        layout.give_back(0, 32);
        layout.give_back(64, 32);
        /* Three 32-byte returns and one 96-byte hole again. */
        assert_eq!(layout.free.len(), 1);
        assert_eq!(layout.take(96), Some(0));
        assert_eq!(layout.take(1), None);
    }

    #[test]
    fn give_back_ignores_a_range_that_does_not_fit_the_pool() {
        let mut layout = Layout::new(32);
        layout.give_back(0, 64);
        layout.give_back(48, 4);
        layout.give_back(0, 0);
        assert_eq!(layout.free.len(), 1);
        assert_eq!(layout.free.get(&0), Some(&32));
    }

    #[test]
    fn no_carve_is_ever_handed_out_twice() {
        let mut layout = Layout::new(1024);
        let mut live: Vec<(usize, usize)> = Vec::new();
        let mut seeded = 1usize;
        for round in 0..2000 {
            seeded = seeded
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let len = (seeded >> 33) % 64 + 1;
            if live.len() > 4 && (seeded >> 20) % 2 == 0 {
                let (offset, taken) = live.remove((seeded >> 40) as usize % live.len());
                layout.give_back(offset, taken);
                continue;
            }
            if let Some(offset) = layout.take(len) {
                assert!(
                    offset + len <= 1024,
                    "round {round}: {offset}+{len} runs past the pool"
                );
                for &(other, other_len) in &live {
                    assert!(
                        offset + len <= other || other + other_len <= offset,
                        "round {round}: {offset}+{len} overlaps {other}+{other_len}"
                    );
                }
                live.push((offset, len));
            }
        }
    }

    #[test]
    fn the_knob_is_parsed_clamped_and_off_the_way_the_config_says() {
        assert_eq!(parse_pool_size(None), 0);
        assert_eq!(parse_pool_size(Some("0")), 0);
        assert_eq!(parse_pool_size(Some("not a number")), 0);
        assert_eq!(parse_pool_size(Some("16777216")), MIN_POOL_SIZE);
        assert_eq!(parse_pool_size(Some("536870912")), MAX_POOL_SIZE);
        /* Out of range: clamped at both ends, the point being that one end is a floor and the other
         * the size that costs a 16 GB card headroom. */
        assert_eq!(parse_pool_size(Some("1048576")), MIN_POOL_SIZE);
        assert_eq!(parse_pool_size(Some("1073741824")), MAX_POOL_SIZE);
    }

    #[test]
    fn the_pool_size_is_an_even_number_of_granules_so_half_of_it_is_a_carve() {
        let granule = 2 << 20;
        assert_eq!(size_for_request(64 << 20, granule), Some(64 << 20));
        /* 5 granules -> 4, so PoolSize/2 = 2 granules is a whole carve. */
        assert_eq!(size_for_request(5 * granule, granule), Some(4 * granule));
        assert_eq!(size_for_request(64 << 20, 0), None);
        assert_eq!(size_for_request(0, granule), None);
        assert_eq!(size_for_request(granule, 4 * granule), None);
    }

    #[test]
    fn every_request_the_rule_admits_fits_an_empty_pool() {
        let granule = 2 << 20;
        for pool_size in [
            16 << 20,
            32 << 20,
            64 << 20,
            128 << 20,
            256 << 20,
            512 << 20,
        ] {
            let size = size_for_request(pool_size, granule).expect("in range");
            let mut layout = Layout::new(size);
            let half = size / 2;
            /* The largest request the rule admits, rounded the way serve() rounds it. */
            let len = half.div_ceil(granule) * granule;
            assert_eq!(layout.take(len), Some(0), "pool_size={pool_size} len={len}");
        }
    }
}
