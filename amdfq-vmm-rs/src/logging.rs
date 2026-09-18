/* The log sink: the `log` crate does the plumbing, this only decides where a record ends up.
 *
 * One record is one write(2) to stderr — no lock, and nothing held across a call into the runtime. If
 * the host process installed a logger of its own, set_logger fails and its sink keeps our records
 * (DESIGN.md D4); we do not take its filter away either, so the level knob below only applies to the
 * sink we install. */

use log::{LevelFilter, Log, Metadata, Record};
use std::ffi::{c_int, c_void};
use std::sync::LazyLock;

unsafe extern "C" {
    fn write(fd: c_int, buf: *const c_void, count: usize) -> isize;
    fn gettid() -> i32;
}

const STDERR: c_int = 2;

struct Stderr;

static LOGGER: Stderr = Stderr;

static INSTALL: LazyLock<()> = LazyLock::new(|| {
    let level = match std::env::var("AMDFQ_LOG_LEVEL").as_deref() {
        Ok("off") => LevelFilter::Off,
        Ok("error") => LevelFilter::Error,
        Ok("warn") => LevelFilter::Warn,
        Ok("debug") => LevelFilter::Debug,
        Ok("trace") => LevelFilter::Trace,
        _ => LevelFilter::Info,
    };
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(level);
    }
});

/* Called at the top of every hook: after the first call this is one atomic load. */
pub(crate) fn init() {
    LazyLock::force(&INSTALL);
}

impl Log for Stderr {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &Record<'_>) {
        let line = format!(
            "{} T={} {} {}\n",
            record.level(),
            unsafe { gettid() },
            record.target(),
            record.args()
        );
        unsafe { write(STDERR, line.as_ptr() as *const c_void, line.len()) };
    }

    fn flush(&self) {}
}
