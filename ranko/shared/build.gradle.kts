import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.composeMultiplatform)
    alias(libs.plugins.composeCompiler)
    alias(libs.plugins.kotlinx.serialization)
    alias(libs.plugins.metro)
}

// Version and commit shown on Home. Git is read from this project dir, which sits inside the repo.
val generateRankoInfo by tasks.registering {
    val gitHash = providers.exec {
        commandLine("git", "rev-parse", "--short", "HEAD")
        workingDir(rootProject.projectDir)
        isIgnoreExitValue = true
    }.standardOutput.asText.map { it.trim().ifBlank { "dev" } }
    val gitVersion = providers.exec {
        commandLine("git", "describe", "--tags", "--abbrev=0")
        workingDir(rootProject.projectDir)
        isIgnoreExitValue = true
    }.standardOutput.asText.map { it.trim().ifBlank { "dev" } }
    val gitTagRefs = providers.exec {
        commandLine(
            "git",
            "for-each-ref",
            "--format=%(if)%(*objectname)%(then)%(*objectname)%(else)%(objectname)%(end)|%(refname:short)",
            "refs/tags",
        )
        workingDir(rootProject.projectDir)
        isIgnoreExitValue = true
    }.standardOutput.asText.map { it.trim() }
    val changelog = providers.exec {
        commandLine("git", "log", "-n", "40", "--pretty=format:%H|%h|%ad|%s", "--date=short")
        workingDir(rootProject.projectDir)
        isIgnoreExitValue = true
    }.standardOutput.asText.map { it.trim() }
    val outputDir = layout.buildDirectory.dir("generated/rankoInfo/kotlin")

    inputs.property("gitVersion", gitVersion)
    inputs.property("gitHash", gitHash)
    inputs.property("gitTagRefs", gitTagRefs)
    inputs.property("changelog", changelog)
    outputs.dir(outputDir)

    doLast {
        val dir = outputDir.get().asFile.resolve("com/acite/axlranko/generated")
        dir.mkdirs()
        fun String.kotlinString(): String = buildString {
            append('"')
            for (ch in this@kotlinString) {
                when (ch) {
                    '\\' -> append("\\\\")
                    '"' -> append("\\\"")
                    '$' -> append("\\\$")
                    else -> append(ch)
                }
            }
            append('"')
        }
        fun List<String>.kotlinList(): String =
            if (isEmpty()) "emptyList()"
            else "listOf(${joinToString { it.kotlinString() }})"

        val tagsByCommit = HashMap<String, MutableList<String>>()
        gitTagRefs.get().lineSequence().forEach { line ->
            val sep = line.indexOf('|')
            if (sep <= 0) return@forEach
            val commit = line.substring(0, sep).trim()
            val tag = line.substring(sep + 1).trim()
            if (commit.isEmpty() || tag.isEmpty()) return@forEach
            tagsByCommit.getOrPut(commit) { ArrayList() }.add(tag)
        }
        val entries = changelog.get().lineSequence()
            .map { it.trim() }
            .filter { it.isNotEmpty() && it.count { c -> c == '|' } >= 3 }
            .joinToString(",\n        ") { line ->
                val first = line.indexOf('|')
                val second = line.indexOf('|', first + 1)
                val third = line.indexOf('|', second + 1)
                val full = line.substring(0, first)
                val hash = line.substring(first + 1, second)
                val date = line.substring(second + 1, third)
                val subject = line.substring(third + 1)
                val tags = tagsByCommit[full].orEmpty().distinct()
                "ChangelogEntry(${hash.kotlinString()}, ${date.kotlinString()}, ${subject.kotlinString()}, ${tags.kotlinList()})"
            }
        val body = buildString {
            appendLine("package com.acite.axlranko.generated")
            appendLine()
            appendLine("import com.acite.axlranko.changelog.ChangelogEntry")
            appendLine()
            appendLine("object AppInfo {")
            appendLine("    const val version: String = ${gitVersion.get().kotlinString()}")
            appendLine("    const val gitHash: String = ${gitHash.get().kotlinString()}")
            appendLine("    val changelog: List<ChangelogEntry> = listOf(")
            if (entries.isNotEmpty()) {
                appendLine("        $entries")
            }
            appendLine("    )")
            appendLine("}")
        }
        dir.resolve("AppInfo.kt").writeText(body)
    }
}

kotlin {
    jvm()

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
    }

    sourceSets {
        commonMain {
            kotlin.srcDir(generateRankoInfo.map { it.outputs.files.singleFile })
        }
        commonMain.dependencies {
            api("dev.zacsweers.metro:metrox-viewmodel-compose:1.3.0")

            implementation(libs.compose.runtime)
            implementation(libs.compose.foundation)
            implementation(libs.compose.material3)
            implementation(libs.compose.ui)
            implementation(libs.compose.components.resources)
            implementation(libs.compose.uiToolingPreview)
            implementation(libs.androidx.lifecycle.viewmodelCompose)
            implementation(libs.androidx.lifecycle.runtimeCompose)
            implementation("org.jetbrains.compose.material:material-icons-extended:1.7.3")
            implementation("com.squareup.okio:okio:3.17.0")

            implementation(libs.ktoml.core)
            implementation(libs.kotlinx.datetime)
            implementation(libs.kotlinx.serialization.json)
            implementation(libs.coil.compose)
            implementation(libs.haze)
            implementation(libs.haze.blur)
        }

        jvmMain.dependencies {
            implementation(libs.kotlinx.datetime)
            implementation(libs.ktoml.file)
            implementation(libs.filekit.core)
            implementation(libs.coil.network.ktor)
        }

        wasmJsMain.dependencies {
            implementation("org.jetbrains.kotlinx:kotlinx-browser:0.3.0")
            implementation("com.squareup.okio:okio-fakefilesystem:3.17.0")
        }

        // MaskPaintInputTest composes a real ComposeWindow, which needs the
        // Skiko native runtime for this OS.
        jvmTest.dependencies {
            implementation(compose.desktop.currentOs)
        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }
    }
}