/* High-frequency HIP VMM churn with a 256-slot byte-tag integrity check.
 *
 * Each live slot i holds size bytes, every byte equal to i. The process keeps at most 256
 * mappings. Once full, it tears one down (Unmap -> hipMemRelease -> hipMemAddressFree) and
 * immediately maps a replacement into that slot — the teardown that let the trainer reuse VA.
 * A tag mismatch is logged and counted; the run continues until --iters (HIP API errors still abort).
 *
 *   bash amdfq/vmm-cc/run.sh [--iters N] [--size B] [--check-every K]
 *                            [--never-reuse] [--hold-old-phys] [--no-sync-before-unmap]
 *
 * Do not LD_PRELOAD the hook; this talks to the VMM API itself.
 */
#include <hip/hip_runtime.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <time.h>
#include <unistd.h>

#include <unordered_map>
#include <vector>

namespace {

const int kLive = 256;
const int kThreads = 256;

#define CHECK(expr)                                                                 \
    do {                                                                            \
        hipError_t e_ = (expr);                                                     \
        if (e_ != hipSuccess) {                                                     \
            std::fprintf(stderr, "FAIL %s -> %s (%d)\n", #expr,                     \
                         hipGetErrorString(e_), static_cast<int>(e_));              \
            std::exit(1);                                                           \
        }                                                                           \
    } while (0)

volatile sig_atomic_t g_stop = 0;

void on_sigint(int) { g_stop = 1; }

__global__ void fill_tag(unsigned char* p, size_t n, unsigned char tag) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        p[i] = tag;
    }
}

__global__ void check_tag(const unsigned char* p, size_t n, unsigned char tag,
                          unsigned long long* mismatches, unsigned long long* first_off) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        if (p[i] != tag) {
            atomicAdd(mismatches, 1ull);
            atomicMin(first_off, static_cast<unsigned long long>(i));
        }
    }
}

int launch_blocks(size_t n) {
    size_t blocks = (n + static_cast<size_t>(kThreads) - 1) / static_cast<size_t>(kThreads);
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > 65535) {
        blocks = 65535;
    }
    return static_cast<int>(blocks);
}

void gpu_fill(void* va, size_t n, unsigned char tag) {
    fill_tag<<<launch_blocks(n), kThreads>>>(static_cast<unsigned char*>(va), n, tag);
    CHECK(hipGetLastError());
}

struct CheckOut {
    unsigned long long mismatches;
    unsigned long long first_off;
};

struct Slot {
    void* va = nullptr;
    hipMemGenericAllocationHandle_t handle = nullptr;
    bool live = false;
};

struct Config {
    uint64_t iters = 1024;
    size_t size = 2ull << 20;
    uint64_t check_every = 1;
    bool never_reuse = false;
    bool hold_old_phys = false;
    bool no_sync_before_unmap = false;
};

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s [--iters N] [--size B] [--check-every K]\n"
                 "          [--never-reuse] [--hold-old-phys] [--no-sync-before-unmap]\n",
                 argv0);
}

Config parse_args(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            c.iters = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            c.size = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--check-every") == 0 && i + 1 < argc) {
            c.check_every = std::strtoull(argv[++i], nullptr, 10);
            if (c.check_every == 0) {
                c.check_every = 1;
            }
        } else if (std::strcmp(argv[i], "--never-reuse") == 0) {
            c.never_reuse = true;
        } else if (std::strcmp(argv[i], "--hold-old-phys") == 0) {
            c.hold_old_phys = true;
        } else if (std::strcmp(argv[i], "--no-sync-before-unmap") == 0) {
            c.no_sync_before_unmap = true;
        } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            usage(argv[0]);
            std::exit(2);
        }
    }
    if (c.size == 0) {
        std::fprintf(stderr, "FAIL --size must be > 0\n");
        std::exit(2);
    }
    return c;
}

double monotonic_s() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

hipMemAllocationProp device_prop() {
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = 0;
    return prop;
}

void set_access(void* va, size_t size) {
    hipMemAccessDesc desc[2]{};
    desc[0].location.type = hipMemLocationTypeDevice;
    desc[0].location.id = 0;
    desc[0].flags = hipMemAccessFlagsProtReadWrite;
    desc[1].location.type = hipMemLocationTypeHost;
    desc[1].location.id = 0;
    desc[1].flags = hipMemAccessFlagsProtReadWrite;
    hipError_t both = hipMemSetAccess(va, size, desc, 2);
    if (both == hipSuccess) {
        return;
    }
    (void)hipGetLastError();
    CHECK(hipMemSetAccess(va, size, desc, 1));
}

void dump_bad(int slot, void* va, size_t size, unsigned char expect, unsigned long long off,
              unsigned long long mismatches) {
    unsigned char got = 0;
    size_t nbytes = 16;
    if (off >= size) {
        off = 0;
    }
    if (off + nbytes > size) {
        nbytes = size - static_cast<size_t>(off);
    }
    std::vector<unsigned char> buf(nbytes);
    CHECK(hipMemcpy(&got, static_cast<unsigned char*>(va) + off, 1, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(buf.data(), static_cast<unsigned char*>(va) + off, nbytes,
                    hipMemcpyDeviceToHost));
    std::fprintf(stderr,
                 "FAIL slot=%d expect=0x%02x got=0x%02x off=%llu mismatches=%llu va=%p size=%zu "
                 "bytes:",
                 slot, expect, got, off, mismatches, va, size);
    for (size_t i = 0; i < nbytes; i++) {
        std::fprintf(stderr, " %02x", buf[i]);
    }
    std::fprintf(stderr, "\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);

    Config cfg = parse_args(argc, argv);
    signal(SIGINT, on_sigint);

    CHECK(hipSetDevice(0));
    hipDeviceProp_t dprop{};
    CHECK(hipGetDeviceProperties(&dprop, 0));

    hipMemAllocationProp prop = device_prop();
    size_t gran = 0;
    CHECK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum));
    size_t size = ((cfg.size + gran - 1) / gran) * gran;

    std::fprintf(stderr,
                 "device %s (%s) gran=%zu size=%zu live=%d iters=%" PRIu64
                 " check-every=%" PRIu64 " never-reuse=%d hold-old-phys=%d no-sync-before-unmap=%d\n",
                 dprop.name, dprop.gcnArchName, gran, size, kLive, cfg.iters, cfg.check_every,
                 cfg.never_reuse ? 1 : 0, cfg.hold_old_phys ? 1 : 0,
                 cfg.no_sync_before_unmap ? 1 : 0);

    Slot slots[kLive];
    int filled = 0;
    int victim = 0;

    CheckOut* d_out = nullptr;
    CHECK(hipMalloc(&d_out, sizeof(CheckOut) * static_cast<size_t>(kLive)));

    void* ghost = nullptr;
    if (cfg.hold_old_phys) {
        CHECK(hipMemAddressReserve(&ghost, size, gran, nullptr, 0));
    }

    std::unordered_map<uintptr_t, uint64_t> va_hits;
    uint64_t reuse_hits = 0;
    uint64_t checks = 0;
    uint64_t integrity_events = 0;
    uint64_t integrity_repeats = 0;
    bool dumped_bad[kLive] = {};
    uint64_t done = 0;
    double t0 = monotonic_s();
    double t_beat = t0;

    auto record_va = [&](void* va) {
        uintptr_t key = reinterpret_cast<uintptr_t>(va);
        uint64_t& n = va_hits[key];
        if (n > 0) {
            reuse_hits++;
        }
        n++;
    };

    auto ranges_overlap = [&](void* a, void* b) {
        auto aa = reinterpret_cast<uintptr_t>(a);
        auto bb = reinterpret_cast<uintptr_t>(b);
        return aa < bb + size && bb < aa + size;
    };

    auto map_new = [&](int slot) {
        hipMemGenericAllocationHandle_t handle = nullptr;
        CHECK(hipMemCreate(&handle, size, &prop, 0));
        void* va = nullptr;
        CHECK(hipMemAddressReserve(&va, size, gran, nullptr, 0));
        for (int i = 0; i < kLive; i++) {
            if (!slots[i].live) {
                continue;
            }
            if (ranges_overlap(va, slots[i].va)) {
                std::fprintf(stderr,
                             "FAIL Reserve returned va=%p overlapping live slot=%d va=%p size=%zu\n",
                             va, i, slots[i].va, size);
                std::exit(1);
            }
        }
        record_va(va);
        CHECK(hipMemMap(va, size, 0, handle, 0));
        set_access(va, size);
        slots[slot].va = va;
        slots[slot].handle = handle;
        slots[slot].live = true;
        dumped_bad[slot] = false;
    };

    auto teardown = [&](int slot, hipMemGenericAllocationHandle_t* kept) {
        if (!cfg.no_sync_before_unmap) {
            CHECK(hipDeviceSynchronize());
        }
        CHECK(hipMemUnmap(slots[slot].va, size));
        /* HIP how-to sample + CUDA VMM: Unmap, then Release the handle, then AddressFree.
         * --hold-old-phys keeps the handle so old backing can be remapped at the ghost VA. */
        if (cfg.hold_old_phys) {
            *kept = slots[slot].handle;
        } else {
            CHECK(hipMemRelease(slots[slot].handle));
            *kept = nullptr;
        }
        if (cfg.never_reuse) {
            /* VA stays reserved so this span cannot be mapped again. */
        } else {
            CHECK(hipMemAddressFree(slots[slot].va, size));
        }
        slots[slot].va = nullptr;
        slots[slot].handle = nullptr;
        slots[slot].live = false;
    };

    auto verify_one = [&](void* va, unsigned char tag, int slot_for_msg, const char* what) {
        CheckOut init{};
        init.mismatches = 0;
        init.first_off = ~0ull;
        CHECK(hipMemcpy(d_out, &init, sizeof(init), hipMemcpyHostToDevice));
        check_tag<<<launch_blocks(size), kThreads>>>(static_cast<unsigned char*>(va), size, tag,
                                                     &d_out->mismatches, &d_out->first_off);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        CheckOut out{};
        CHECK(hipMemcpy(&out, d_out, sizeof(out), hipMemcpyDeviceToHost));
        checks++;
        if (out.mismatches != 0) {
            integrity_events++;
            std::fprintf(stderr, "FAIL %s\n", what);
            dump_bad(slot_for_msg, va, size, tag, out.first_off, out.mismatches);
            std::fprintf(stderr, "     unique_va=%zu reuse_hits=%" PRIu64 " iters=%" PRIu64
                                 " integrity_events=%" PRIu64 " (continuing)\n",
                         va_hits.size(), reuse_hits, done, integrity_events);
        }
    };

    auto verify_all_live = [&]() {
        CheckOut init[kLive];
        for (int i = 0; i < kLive; i++) {
            init[i].mismatches = 0;
            init[i].first_off = ~0ull;
        }
        CHECK(hipMemcpy(d_out, init, sizeof(init), hipMemcpyHostToDevice));
        int nlive = 0;
        for (int i = 0; i < kLive; i++) {
            if (!slots[i].live) {
                continue;
            }
            nlive++;
            unsigned char tag = static_cast<unsigned char>(i);
            check_tag<<<launch_blocks(size), kThreads>>>(
                static_cast<unsigned char*>(slots[i].va), size, tag, &d_out[i].mismatches,
                &d_out[i].first_off);
            CHECK(hipGetLastError());
        }
        CHECK(hipDeviceSynchronize());
        CheckOut host[kLive];
        CHECK(hipMemcpy(host, d_out, sizeof(host), hipMemcpyDeviceToHost));
        checks++;
        for (int i = 0; i < kLive; i++) {
            if (!slots[i].live) {
                continue;
            }
            if (host[i].mismatches != 0) {
                if (dumped_bad[i]) {
                    integrity_repeats++;
                    continue;
                }
                dumped_bad[i] = true;
                integrity_events++;
                unsigned char tag = static_cast<unsigned char>(i);
                dump_bad(i, slots[i].va, size, tag, host[i].first_off, host[i].mismatches);
                auto it = va_hits.find(reinterpret_cast<uintptr_t>(slots[i].va));
                uint64_t hits = it == va_hits.end() ? 0 : it->second;
                std::fprintf(stderr,
                             "     va_hits_for_this=%" PRIu64 " unique_va=%zu reuse_hits=%" PRIu64
                             " iters=%" PRIu64 " live=%d integrity_events=%" PRIu64
                             " (continuing)\n",
                             hits, va_hits.size(), reuse_hits, done, nlive, integrity_events);
                unsigned char got_tag = 0;
                CHECK(hipMemcpy(&got_tag,
                                static_cast<unsigned char*>(slots[i].va) +
                                    (host[i].first_off < size ? host[i].first_off : 0),
                                1, hipMemcpyDeviceToHost));
                for (int j = 0; j < kLive; j++) {
                    if (slots[j].live && static_cast<unsigned char>(j) == got_tag) {
                        std::fprintf(stderr, "     slot whose tag equals got: slot=%d va=%p\n", j,
                                     slots[j].va);
                    }
                }
            }
        }
    };

    auto heartbeat = [&](bool force) {
        double now = monotonic_s();
        if (!force && now - t_beat < 1.0) {
            return;
        }
        t_beat = now;
        std::fprintf(stderr,
                     "hb iters=%" PRIu64 " live=%d unique_va=%zu reuse_hits=%" PRIu64
                     " checks=%" PRIu64 " integrity_events=%" PRIu64 " repeats=%" PRIu64
                     " dt=%.1fs\n",
                     done, filled, va_hits.size(), reuse_hits, checks, integrity_events,
                     integrity_repeats, now - t0);
    };

    for (done = 0; done < cfg.iters && !g_stop; done++) {
        hipMemGenericAllocationHandle_t old_handle = nullptr;
        unsigned char old_tag = 0;
        int slot;
        if (filled < kLive) {
            slot = filled++;
        } else {
            slot = victim;
            victim = (victim + 1) % kLive;
            old_tag = static_cast<unsigned char>(slot);
            teardown(slot, &old_handle);
            if (cfg.hold_old_phys) {
                CHECK(hipMemMap(ghost, size, 0, old_handle, 0));
                set_access(ghost, size);
            }
        }

        map_new(slot);

        unsigned char tag = static_cast<unsigned char>(slot);
        if (cfg.hold_old_phys && old_handle != nullptr) {
            /* A fill of the same tag into reused VA is invisible on old phys. Poison first. */
            unsigned char poison = static_cast<unsigned char>(tag ^ 0xff);
            gpu_fill(slots[slot].va, size, poison);
            CHECK(hipDeviceSynchronize());
            verify_one(ghost, old_tag, slot, "hold-old-phys: GPU fill of new VA landed on old phys");
            gpu_fill(slots[slot].va, size, tag);
            CHECK(hipDeviceSynchronize());
            CHECK(hipMemUnmap(ghost, size));
            CHECK(hipMemRelease(old_handle));
        } else {
            gpu_fill(slots[slot].va, size, tag);
            CHECK(hipDeviceSynchronize());
        }

        if ((done + 1) % cfg.check_every == 0) {
            verify_all_live();
        }
        heartbeat(false);
    }

    heartbeat(true);

    for (int i = 0; i < kLive; i++) {
        if (!slots[i].live) {
            continue;
        }
        CHECK(hipDeviceSynchronize());
        CHECK(hipMemUnmap(slots[i].va, size));
        CHECK(hipMemRelease(slots[i].handle));
        if (!cfg.never_reuse) {
            CHECK(hipMemAddressFree(slots[i].va, size));
        }
        slots[i].live = false;
    }
    if (ghost) {
        CHECK(hipMemAddressFree(ghost, size));
    }
    CHECK(hipFree(d_out));

    std::printf("%s iters=%" PRIu64 " unique_va=%zu reuse_hits=%" PRIu64 " checks=%" PRIu64
                " integrity_events=%" PRIu64 " repeats=%" PRIu64 " never-reuse=%d hold-old-phys=%d\n",
                integrity_events == 0 ? "PASS" : "FAIL", done, va_hits.size(), reuse_hits, checks,
                integrity_events, integrity_repeats, cfg.never_reuse ? 1 : 0,
                cfg.hold_old_phys ? 1 : 0);
    if (g_stop) {
        return 130;
    }
    return integrity_events == 0 ? 0 : 1;
}
