package com.acite.axlranko.util

import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class ProcessExitGuardTest {

    private fun spawnSleeper(): Process =
        ProcessBuilder("sleep", "300").start().also { process ->
            // Wait for it to exist, so /proc/<pid>/stat can be read.
            assertTrue(ProcessExitGuard.startTicks(process.pid()) != null)
        }

    private fun isAlive(pid: Long): Boolean = ProcessHandle.of(pid).map { it.isAlive }.orElse(false)

    @Test
    fun startTicksReadsTheFieldTheKernelUsesForStartTime() {
        val mine = ProcessExitGuard.startTicks(ProcessHandle.current().pid())
        assertNotNull(mine)
        assertTrue(mine.toLong() > 0)
        assertNull(ProcessExitGuard.startTicks(-1))
    }

    @Test
    fun watcherScriptCarriesThePidTheGraceAndTheStartTime() {
        val script = ProcessExitGuard.watcherScript(4242, 7, "1234567")
        assertTrue("i\" -lt 7" in script, script)
        assertTrue("/proc/4242/stat" in script)
        assertTrue("\"1234567\"" in script)
    }

    @Test
    fun killsAProcessThatOutlivedTheGrace() {
        val sleeper = spawnSleeper()
        try {
            val watcher = ProcessExitGuard.arm(sleeper.pid(), graceSeconds = 1)
            assertNotNull(watcher)
            assertTrue(sleeper.waitFor(10, TimeUnit.SECONDS), "the watcher did not kill the process")
            assertFalse(isAlive(sleeper.pid()))
            assertTrue(watcher.waitFor(10, TimeUnit.SECONDS), "the watcher did not exit")
        } finally {
            sleeper.destroyForcibly()
        }
    }

    @Test
    fun leavesTheProcessAloneWhenThePidWasReused() {
        val sleeper = spawnSleeper()
        try {
            // A start time that is not this process's: the watcher must stop short of killing it.
            val watcher = ProcessExitGuard.arm(sleeper.pid(), graceSeconds = 1, born = "1")
            assertNotNull(watcher)
            assertTrue(watcher.waitFor(10, TimeUnit.SECONDS), "the watcher did not exit")
            assertTrue(isAlive(sleeper.pid()), "the watcher killed a process whose start time differed")
        } finally {
            sleeper.destroyForcibly()
        }
    }

    @Test
    fun exitsOnItsOwnOnceTheProcessIsGone() {
        // The real shape: the guard is armed while the process is still alive, and the process then
        // quits. It may still be sitting in /proc as a zombie — the JVM does not always wait on it
        // promptly — and "gone" has to include that, or the watcher would sleep out its whole grace
        // beside a process that is already dead.
        val quick = ProcessBuilder("sh", "-c", "sleep 1").start()
        val watcher = ProcessExitGuard.arm(quick.pid(), graceSeconds = 60)
        assertNotNull(watcher, "a live process can be watched")
        assertTrue(quick.waitFor(10, TimeUnit.SECONDS))
        assertTrue(watcher.waitFor(10, TimeUnit.SECONDS),
            "a clean quit must not leave a watcher sleeping out its whole grace")
    }

    @Test
    fun refusesToArmForAProcessThatIsAlreadyGone() {
        val quick = ProcessBuilder("true").start()
        quick.waitFor(10, TimeUnit.SECONDS)
        assertNull(ProcessExitGuard.arm(quick.pid(), graceSeconds = 1),
            "without a start time to guard with there is nothing to watch")
    }

    @Test
    fun armOnceOnlyArmsTheFirstTime() {
        // Never arm against this JVM: a watcher that fires here would kill the test run itself.
        val sleeper = spawnSleeper()
        try {
            assertTrue(ProcessExitGuard.armOnce(sleeper.pid(), graceSeconds = 1),
                "the first quit request must arm the watcher")
            assertFalse(ProcessExitGuard.armOnce(sleeper.pid(), graceSeconds = 1),
                "the shutdown hook must not arm a second watcher")
            assertFalse(ProcessExitGuard.armOnce(sleeper.pid(), graceSeconds = 1))
        } finally {
            sleeper.destroyForcibly()
        }
    }
}
