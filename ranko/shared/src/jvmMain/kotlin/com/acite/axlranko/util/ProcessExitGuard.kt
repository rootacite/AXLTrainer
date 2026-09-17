package com.acite.axlranko.util

import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Guarantees the app dies once it has decided to quit.
 *
 * Closing the window finishes the Compose application, but a JVM that hangs on the way out never
 * gets that far: the process outlives its only window, still holding the GPU render nodes and the
 * `api.py` child it spawned, with nothing left on screen to close. Nothing inside a hung VM can
 * undo that, so this watcher runs outside it: armed while the JVM is still healthy, it kills the
 * process if it is still there once the grace has passed.
 */
object ProcessExitGuard {
    /** Long enough for an honest shutdown, short enough not to leave a windowless process around. */
    const val DEFAULT_GRACE_SECONDS = 10

    private val armed = AtomicBoolean(false)

    /** Register the watcher for every exit, not only the window's: a SIGTERM can hang just as well. */
    fun install() {
        Runtime.getRuntime().addShutdownHook(Thread { armOnce() })
    }

    /** Arm on the first quit request. Later calls — the shutdown hook of the same quit — do nothing. */
    fun armOnce(pid: Long = ProcessHandle.current().pid(), graceSeconds: Int = DEFAULT_GRACE_SECONDS): Boolean {
        if (!armed.compareAndSet(false, true)) return false
        return arm(pid, graceSeconds) != null
    }

    /** Starts the watcher, or returns null when `/proc` told us nothing or no shell could be run. */
    fun arm(pid: Long, graceSeconds: Int = DEFAULT_GRACE_SECONDS): Process? {
        val born = startTicks(pid) ?: return null
        return arm(pid, graceSeconds, born)
    }

    internal fun arm(pid: Long, graceSeconds: Int, born: String): Process? =
        runCatching {
            ProcessBuilder("sh", "-c", watcherScript(pid, graceSeconds, born))
                .redirectOutput(ProcessBuilder.Redirect.DISCARD)
                .redirectError(ProcessBuilder.Redirect.DISCARD)
                .start()
        }.getOrNull()

    /** Field 22 of `/proc/<pid>/stat`: the start time, which tells a reused PID from the original. */
    internal fun startTicks(pid: Long): String? = runCatching {
        File("/proc/$pid/stat").readText().substringAfterLast(')').trim().split(' ').getOrNull(19)
    }.getOrNull()

    /**
     * Exit as soon as the process is gone, so a clean quit leaves no watcher behind; kill it once the
     * grace has passed, unless that PID has been reused (`born` no longer matches).
     *
     * "Gone" includes a zombie: a process that has exited but whose parent has not waited on it keeps
     * its `/proc` entry, and waiting for that entry to disappear would never end.
     */
    internal fun watcherScript(pid: Long, graceSeconds: Int, born: String): String =
        "state() { sed -n 's/^[^)]*) *//p' /proc/$pid/stat 2>/dev/null; }\n" +
            "i=0\n" +
            "while [ \"\$i\" -lt $graceSeconds ]; do\n" +
            "  s=\$(state); [ -n \"\$s\" ] || exit 0; [ \"\${s%% *}\" != \"Z\" ] || exit 0\n" +
            "  sleep 1\n" +
            "  i=\$((i + 1))\n" +
            "done\n" +
            "s=\$(state); [ -n \"\$s\" ] || exit 0; [ \"\${s%% *}\" != \"Z\" ] || exit 0\n" +
            "[ \"\$(echo \"\$s\" | cut -d' ' -f20)\" = \"$born\" ] || exit 0\n" +
            "kill -9 $pid 2>/dev/null\n"
}
