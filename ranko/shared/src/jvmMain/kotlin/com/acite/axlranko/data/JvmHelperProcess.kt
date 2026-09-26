package com.acite.axlranko.data

import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/**
 * Spawns `python -u api.py --websocket` when nothing is listening. Only the child
 * this process started is destroyed on JVM shutdown.
 */
internal object JvmHelperProcess {
    private val spawned = AtomicReference<Process?>(null)
    private val hook = AtomicBoolean(false)

    fun ensure(host: String, port: Int) {
        if (helperListening(host, port)) return
        synchronized(this) {
            val running = spawned.get()
            if (running != null && running.isAlive && helperListening(host, port)) return
            if (helperListening(host, port)) return
            start(host, port)
        }
    }

    private fun start(host: String, port: Int) {
        val root = TrainerRepo.findRoot()
            ?: error("Could not locate api.py. Run AxlRanko from the trainer repo, or set the working directory to the repo root.")
        val python = System.getenv("AXL_PYTHON")?.takeIf { it.isNotBlank() } ?: "python3"
        val script = File(root, "api.py")
        val builder = ProcessBuilder(
            python, "-u", script.absolutePath,
            "--websocket",
            "--host", host,
            "--port", port.toString(),
        )
            .directory(root)
            .redirectError(ProcessBuilder.Redirect.INHERIT)
            .redirectOutput(ProcessBuilder.Redirect.INHERIT)
        builder.environment()["PYTHONUNBUFFERED"] = "1"
        val started = try {
            builder.start()
        } catch (e: Exception) {
            throw IllegalStateException(
                "Failed to start $python ${script.absolutePath} --websocket. Set AXL_PYTHON to your interpreter. ${e.message}",
                e,
            )
        }
        spawned.getAndSet(started)?.let { old ->
            runCatching { old.destroy() }
        }
        if (hook.compareAndSet(false, true)) {
            Runtime.getRuntime().addShutdownHook(Thread { stopSpawned() })
        }
        val deadline = System.nanoTime() + 60_000_000_000L
        while (System.nanoTime() < deadline) {
            if (!started.isAlive) {
                throw IllegalStateException("Dashboard helper exited before opening the WebSocket")
            }
            if (helperListening(host, port)) return
            Thread.sleep(200)
        }
        throw IllegalStateException("Dashboard helper did not listen on ws://$host:$port")
    }

    fun stopSpawned() {
        val running = spawned.getAndSet(null) ?: return
        running.destroy()
        if (running.isAlive) {
            running.destroyForcibly()
        }
    }
}
