/* The early HIP touch: one `hipGetDeviceCount` from this object's load-time constructor, aimed at
 * ROCr's `AsyncEventsLoop` busy-spin.
 *
 * The bug: on this box's ROCR (1.21.0, the runtime inside both `_rocm_sdk_core` wheels this repo
 * runs — ROCm 7.14.1 and 10.0.0), any GPU work leaves `rocr::core::Runtime::AsyncEventsLoop`
 * spinning one CPU core for the rest of the process. Upstream has it as ROCm/TheRock#7051 and
 * ROCm/legacy-rocm-build#6634 (gfx1201, this box's architecture) and ComfyUI as
 * Comfy-Org/ComfyUI#16339; TheRock's writeup puts it on a registered async signal with no
 * `EopEvent` flipping the loop into a polling rescan that has no backoff. Measured here with
 * `import torch; torch.cuda.init(); torch.ones(8, 8, device='cuda')`: 99.92% of the hot thread's
 * samples in `AsyncEventsLoop`, and the process stays one core busy from then until it exits. No
 * ROCR switch in this build helps (`HSA_ENABLE_INTERRUPT=0`, `GPU_MAX_HW_QUEUES=1`,
 * `HSA_ENABLE_MWAITX=0` all still saturate the core).
 *
 * What does help is a warm-up, and *when* it runs is the whole trick. Whole-process CPU over a 5s
 * window once the workload has settled; 500 ticks is one core:
 *
 *     no HIP call at all ..................................... 605   spinning
 *     after torch's own preload of its 21 ROCm libraries ..... 0     clean
 *     after `import torch` ................................... 507   spinning
 *     after loading libtorch_cpu.so .......................... 508   spinning
 *     after `torch.cuda.init()` .............................. 605   spinning
 *     from an LD_PRELOAD .init_array constructor ............. 0     clean
 *
 * So the boundary is a *load*, not a call: the runtime has to be touched before torch's own
 * `libtorch_cpu.so` is mapped, and `import torch` maps it. A HIP gate cannot be early enough —
 * `import torch` makes no HIP call at all before that (`/dev/kfd` is untouched for the whole
 * import) — which is why this is a constructor and not a hook (DESIGN.md D7's one exception).
 *
 * The library has to be the one torch will use. A bare soname is not that: with the conda env
 * active, `libamdhip64.so.7` still resolves through ld.so.cache to the system ROCm under
 * /opt/rocm/core, and loading that one first kills `import torch` with
 * `libamd_comgr.so.3: undefined symbol ... version LLVM_23.0`. So the path is derived from the
 * interpreter that is actually running, and when it does not resolve this does nothing at all. */

use crate::logging;
use std::ffi::{CString, c_char};
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

/* The bundled runtime's own layout: every ROCm library the wheels carry lives here. */
const BUNDLED: &str = "site-packages/_rocm_sdk_core/lib";

/* Named by glibc as `void (*)(int, char **, char **)` when this object is loaded — for a preload,
 * before the program itself runs. The entry has to be an integer/pointer-sized static, and the
 * function must not be exported (or `test.sh`'s `exports == declared` would grow a name that is not
 * a HIP gate). */
#[used]
#[unsafe(link_section = ".init_array")]
static EARLY_TOUCH: unsafe extern "C" fn(i32, *mut *mut c_char, *mut *mut c_char) = run;

unsafe extern "C" fn run(_argc: i32, _argv: *mut *mut c_char, _envp: *mut *mut c_char) {
    logging::init();
    if std::env::var("AMDFQ_EARLY_HIP").is_ok_and(|value| value == "0") {
        return;
    }
    let touched = library().and_then(|path| {
        let name = CString::new(path.as_os_str().as_bytes()).ok()?;
        crate::real::early_touch(&name).map(|devices| (path, devices))
    });
    match touched {
        Some((path, devices)) => {
            log::info!("early touch -> {devices} device(s) via {}", path.display())
        }
        None => log::debug!("early touch -> no bundled ROCm runtime found, left alone"),
    }
}

/* `AMDFQ_HIP_LIB` names one file outright; otherwise the interpreter's own environment decides. */
fn library() -> Option<PathBuf> {
    if let Some(explicit) = std::env::var_os("AMDFQ_HIP_LIB") {
        let path = PathBuf::from(explicit);
        return path.is_file().then_some(path);
    }
    /* `/proc/self/exe` is `<prefix>/bin/python3.x`, so the prefix is two levels up — and that is
     * the interpreter whose site-packages the runtime will be imported from, stale environment
     * variables or not. */
    let exe = fs::read_link("/proc/self/exe").ok()?;
    bundled(exe.parent()?.parent()?)
}

fn bundled(prefix: &Path) -> Option<PathBuf> {
    let mut versions: Vec<PathBuf> = fs::read_dir(prefix.join("lib"))
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .is_some_and(|name| name.as_bytes().starts_with(b"python3."))
        })
        .collect();
    /* Newest first: a prefix can carry more than one interpreter. */
    versions.sort();
    versions.iter().rev().find_map(|lib| candidate(&lib.join(BUNDLED)))
}

fn candidate(directory: &Path) -> Option<PathBuf> {
    let mut matches: Vec<PathBuf> = fs::read_dir(directory)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .is_some_and(|name| name.as_bytes().starts_with(b"libamdhip64.so"))
        })
        .collect();
    matches.sort();
    matches.into_iter().next()
}
