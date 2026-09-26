# Adding a Web target to Ranko (Compose Multiplatform)

A runbook for Ranko's desktop + browser Compose Multiplatform tree. `:shared` now declares `wasmJs { browser() }` and `:webApp` is the executable. Keep this file for Gradle shape, FilePicker, and mask notes.

**Shipped:** `./gradlew :webApp:wasmJsBrowserDevelopmentRun` paints `App()`. Desktop FileKit stays; wasm uses `fs_listdir`. Mask raster is common `ByteArray`; wasm input is DOM. Helper bind may be LAN with `--allow-ip`. No Kotlin/JS compatibility mode, no auth token, no TLS.

Recorded 2026-09-26 against Ranko as it stands: Kotlin **2.4.10**, Compose Multiplatform **1.12.0**, Metro **1.3.0**. Official pages cited below were current on that date.

## What "a Web target" means here

Compose Multiplatform for Web compiles the same `@Composable` UI to a canvas in the browser. The compiler backend JetBrains recommends for a shared UI is **Kotlin/Wasm** (`wasmJs`). Kotlin/JS (`js`) is the fallback for browsers that do not implement WasmGC. A project that wants both at once enables **compatibility mode**: modern browsers load Wasm, older ones load JS, from one distribution.

Sources:

- [Choose the right web target](https://kotlinlang.org/docs/multiplatform/choosing-web-target.html) — Wasm for shared Compose UI; JS for HTML-native UI or logic-only sharing.
- [Compose Multiplatform 1.9.0](https://blog.jetbrains.com/kotlin/2025/09/compose-multiplatform-1-9-0-compose-for-web-beta/) — Web target moved to Beta (2025-09). Ranko is already on Compose 1.12.
- [Create your Compose Multiplatform app](https://kotlinlang.org/docs/multiplatform/compose-multiplatform-create-first-app.html) — wizard layout and run tasks.
- [Get started with Kotlin/Wasm](https://kotlinlang.org/docs/wasm-get-started.html) — `wasmJsBrowserDevelopmentRun` / `wasmJsBrowserDistribution`.
- [Recommended project structure](https://kotlinlang.org/docs/multiplatform/multiplatform-project-recommended-structure.html) — extract a `webApp` entry module.

`ui.py` (Streamlit) is a separate, deprecated read-only viewer. A CMP Web target would be Ranko's own UI in the browser, and is a different product surface.

## Ranko's current Gradle shape

Ranko already matches the structure the KMP wizard emits for Desktop:

```
ranko/
├── settings.gradle.kts          # include(":desktopApp"), include(":shared")
├── desktopApp/                  # JVM entry: Window { App(...) }
│   └── src/main/kotlin/.../main.kt
└── shared/                      # kotlin { jvm() } only
    ├── src/commonMain/          # UI, ViewModels, IPC, config
    ├── src/jvmMain/             # PathUtils, MaskPaint, ProcessExitGuard
    └── src/jvmTest/
```

`shared/build.gradle.kts` declares a single target, `jvm()`. `desktopApp` is a Kotlin/JVM application that depends on `projects.shared` and `compose.desktop.currentOs`. That is the same split the [recommended structure](https://kotlinlang.org/docs/multiplatform/multiplatform-project-recommended-structure.html) uses for Desktop; the missing piece is a `webApp` module plus `wasmJs` (and optionally `js`) on `shared`.

The first 60 seconds of work is therefore Gradle and an empty `ComposeViewport`. Making Ranko *useful* in a browser is a later, larger pass: `commonMain` currently calls `java.io.File`, AWT, `Process`, and `java.util.prefs` as if it were JVM (see [What Ranko still assumes](#what-ranko-still-assumes)).

## Decision: which web backend

| Choice | When |
| --- | --- |
| `wasmJs` only | First implementation. Official recommendation for a shared Compose UI. All current major browsers ship WasmGC. |
| `wasmJs` + `js` | Ship a public site that must also run in older browsers. Enables `composeCompatibilityBrowserDistribution`. |
| `js` only | Sharing logic with an HTML/React UI. Ranko wants the Compose chrome, so this is the wrong default. |

Start with `wasmJs` on both `:shared` and `:webApp`. Add `js` only if compatibility mode is a product requirement.

## Step 1 — Gradle: give `shared` a Wasm library target

In `ranko/shared/build.gradle.kts`, next to `jvm()`:

```kotlin
import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

kotlin {
    jvm()

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        browser()
        // library: no binaries.executable()
    }

    sourceSets {
        commonMain.dependencies { /* unchanged */ }
        jvmMain.dependencies { /* unchanged */ }
        wasmJsMain.dependencies {
            // platform-only deps; keep commonMain free of JVM artifacts
        }
    }
}
```

A library target must **not** call `binaries.executable()`. That belongs on `:webApp`.

`ExperimentalWasmDsl` is still required on Kotlin 2.4 (see the Compose `imageviewer` sample's `webApp/build.gradle.kts`). If a later Kotlin release stabilises the DSL, drop the opt-in.

Sync Gradle. The first compile of `commonMain` for Wasm will fail on every `java.*` import; that is expected and is the cue to start Step 4. A smoke path that only compiles a stub `wasmJsMain` plus a tiny `commonMain` subset is a valid first PR.

## Step 2 — New `:webApp` entry module

This follows [Creating a web app module](https://kotlinlang.org/docs/multiplatform/multiplatform-project-recommended-structure.html#web-app) and the Compose `imageviewer` sample.

1. `include(":webApp")` in `ranko/settings.gradle.kts`.
2. `ranko/webApp/build.gradle.kts`:

```kotlin
import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.composeMultiplatform)
    alias(libs.plugins.composeCompiler)
    alias(libs.plugins.metro)
}

kotlin {
    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        outputModuleName.set("axlranko")
        browser {
            commonWebpackConfig {
                outputFileName = "axlranko.js"
            }
        }
        binaries.executable()
    }

    sourceSets {
        commonMain.dependencies {
            implementation(projects.shared)
            implementation(compose.runtime)
            implementation(compose.ui)
            implementation(libs.compose.components.resources)
        }
    }
}
```

Keep webpack config free of captured `project` / top-level `val`s. Ranko enables the Gradle configuration cache (`ranko/gradle.properties`); [KT-68614](https://youtrack.jetbrains.com/issue/KT-68614) fails the webpack task when a script-level `rootDirPath` is closed over. Put any path you need *inside* the `commonWebpackConfig { }` lambda, or skip the extra `devServer.static` block.

3. Root `ranko/build.gradle.kts` already has `kotlinMultiplatform` / `compose*` as `apply false`. No change unless a new plugin is added.
4. Apply the Metro plugin on `:webApp` as well, because `createGraph<AppGraph>()` is what `desktopApp` does in `main.kt`.

If `js` is added later, copy the `js { browser(); binaries.executable() }` block from the same sample and keep `outputModuleName` / `outputFileName` identical so compatibility mode can swap the payload.

## Step 3 — HTML host, CSS, Compose entry

Compose for Web draws onto an HTML canvas through `ComposeViewport`. `CanvasBasedWindow` is deprecated: it injected global CSS and made embedding awkward ([Setting up the viewport](https://kotlinlang.org/docs/multiplatform/compose-css-styles.html)).

`ranko/webApp/src/wasmJsMain/resources/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AxlRanko</title>
    <link rel="stylesheet" href="styles.css">
    <script src="axlranko.js"></script>
</head>
<body>
    <div id="webApp"></div>
</body>
</html>
```

`ranko/webApp/src/wasmJsMain/resources/styles.css` (required; without it the canvas does not fill the window):

```css
html, body {
    width: 100%;
    height: 100%;
    margin: 0;
    padding: 0;
    overflow: hidden;
}
#webApp {
    width: 100%;
    height: 100%;
}
```

`ranko/webApp/src/wasmJsMain/kotlin/com/acite/axlranko/main.kt`:

```kotlin
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.window.ComposeViewport
import com.acite.axlranko.App
import dev.zacsweers.metro.createGraph

@OptIn(ExperimentalComposeUiApi::class)
fun main() {
    val appGraph = createGraph<AppGraph>()
    ComposeViewport(viewportContainerId = "webApp") {
        App(appGraph.metroViewModelFactory, appGraph.appearanceRepository)
    }
}
```

`ProcessExitGuard` is a JVM hang-watchdog around window close; it has no browser equivalent. Leave it in `desktopApp` / `jvmMain`.

Compose resources (`composeResources/` fonts and the app icon) already live under `shared/src/commonMain`. The web target picks them up through `compose.components.resources`. If a hosted path is not `/`, set `compose.resources { }` / `configureWebResources { resourcePathMapping { ... } }` as in [Setup and configuration for multiplatform resources](https://www.jetbrains.com/help/kotlin-multiplatform-dev/compose-multiplatform-resources-setup.html).

## Step 4 — Make `commonMain` actually common

Adding the target is mechanical. Compiling Ranko for Wasm is the real job: `shared/src/commonMain` currently imports JVM types throughout. A Wasm compile of `commonMain` is the inventory; the grep below is the 2026-09-26 snapshot.

### Already on `expect` / `actual`

| API | common | jvmMain today | wasmJsMain later |
| --- | --- | --- | --- |
| `getPlatform()` | `Platform.kt` | `Platform.jvm.kt` | `Platform.wasmJs.kt` |
| `getAppExecutionPath` / load / save config | `ConfigImporter.kt` | `PathUtils.kt` | string/blob IO, or a remote config |
| `Modifier.maskPaintInput` | `MaskPaint.kt` | AWT global listener + `MouseInfo` sampler (`MaskPaint.jvm.kt`) | Compose pointer + wheel; no AWT |

### JVM types sitting in `commonMain` (must move or wrap)

| Area | Files (representative) | Why it fails on Wasm |
| --- | --- | --- |
| `java.io.File` / `java.nio.file` | `TrainerRepo`, `ConfigProfileStore`, `DatasetShuffle`, `FileCopy`, `FileDialogs`, `ImageHeaderSize`, `MaskSidecar`, `TagTranslations`, Images / Statistics / Utils / Dashboard ViewModels | No `java.io` on Kotlin/Wasm. okio `Path` / `FileSystem` (or FileKit `PlatformFile`) is the usual replacement. |
| `Process` / stdin-stdout | `TrainerIpcClient.kt` | A browser cannot spawn `api.py`. See [IPC](#ipc-the-process-model-does-not-travel). |
| `java.awt.*` | `Clipboard.kt`, `MaskCanvas.kt` (`BufferedImage`), `ImageScreen.kt` (`Cursor`), Dashboard / Statistics / Utils (`Cursor`) | AWT is JVM. Clipboard → `expect`; raster → Skia / Compose `ImageBitmap`; cursor → Compose `PointerIcon` (already used in places). |
| `java.util.prefs.Preferences` | `AppearanceRepository.kt` | `localStorage` (or `expect`) on web. |
| `coil-network-okhttp` | `shared/build.gradle.kts` `commonMain` | OkHttp is JVM. Web uses `coil-network-ktor3` + Ktor's JS engine. Keep OkHttp on `jvmMain`. |
| `ktoml-file` | `commonMain` + `PathUtils.kt` | [ktoml-file is JVM/Native](https://github.com/akuleshov7/ktoml). `ktoml-core` already supports `wasmJs`; Ranko already parses from a `String` in `ConfigImporter.parseConfig`. Drop `ktoml-file` from `commonMain`. |
| FileKit directory picker | `FileDialogs.kt` | FileKit 0.8.8 documents directory pick as missing on WASM/JS (file pick uses `<input type="file">`). Later FileKit rebuilds a virtual tree from `webkitdirectory`. Ranko is on **0.8.8**; a web port either upgrades FileKit or accepts file-only picks. |
| `java.util.concurrent.atomic.*` | `TrainerIpcClient` | Replace with Kotlin atomics / a mutex; Wasm has no `java.util.concurrent`. |

### Libraries that already claim Wasm

These can stay in `commonMain` *once their JVM-only siblings are moved*:

- Compose UI / Material 3 / resources — first-party web.
- kotlinx.serialization, kotlinx.coroutines, kotlinx.datetime, okio (core).
- ktoml-core (`wasmJs`).
- Coil 3 compose (with a Ktor network backend on web).
- Haze 2.x — [Wasm / JS listed](https://github.com/chrisbanes/haze).
- Metro — [runtime supports Wasm](https://zacsweers.github.io/metro/1.1.1/multiplatform/). If `jvmMain` and `wasmJsMain` contribute different bindings (a local `Process` IPC vs a WebSocket client), declare the `@DependencyGraph` in each platform source set; Metro requires the final graph to live in platform-specific code when contributions mix.

### Suggested split of the first implementation

1. **Compile the chrome.** Move `java.io.File` / AWT / prefs out of `commonMain` behind `expect` or okio. Stub dataset/config IO on Wasm so `App()` + nav + Utils form *render*.
2. **File pickers and Coil.** FileKit file pick + Coil Ktor. Directory pick and in-place dataset edits wait on a filesystem story (Origin Private File System, a user-selected directory handle, or "web is view-only").
3. **IPC.** See below. Dashboard, train controls, tagger, generate-sample all go through `TrainerIpcClient`.
4. **Mask painting.** A Wasm `actual` of `maskPaintInput` using Compose pointer APIs. The AWT path exists because Compose coalesces motion and hid the right button; a web actual has to re-solve that with pointer events, not AWT.
5. **Tests.** `jvmTest` stays. Add `wasmJsBrowserTest` only for the new actuals; AWT / `ComposeWindow` tests stay JVM.

Do this as its own sequence of PRs. A single "add wasmJs" commit that also rewrites IO will be unreviewable.

## IPC: the process model does not travel

Ranko's contract (AGENT.md §2, `doc/overview.md`):

```
Ranko (JVM / future wasm)  --WebSocket JSON-RPC-->  api.py  --spawns-->  trainer/main.py
```

The browser has no `ProcessBuilder`, no `setsid`, and no right to the machine's `/dev/kfd`. Three product shapes are available; pick one *before* writing a Wasm IPC client:

| Shape | What the browser talks to | Cost |
| --- | --- | --- |
| **View-only static site** | Nothing. Dashboard is a TensorBoard/sample browser of files the user dropped in. | Smallest. Training stays desktop. |
| **Local companion** | Ranko desktop already speaks JSON-RPC on `ws://127.0.0.1:18765`. A `webApp` connects to the same helper (user starts `python -u api.py` if Ranko is not running). | Same methods, no second protocol. Bind stays loopback. |
| **Remote trainer** | A real HTTP API. | Out of scope unless explicitly asked (AGENT.md §13). |

`TrainerIpcClient` should become an `expect class` or an interface with a JVM `Process` actual. The Wasm actual is empty until a shape is chosen. Do not silently add an HTTP server to `api.py` as a side effect of the Gradle target.

## Step 5 — Run, test, ship

From `ranko/` (cwd is the Gradle root, as today):

```bash
# Dev server; opens http://localhost:8080/ (port may differ if 8080 is taken)
./gradlew :webApp:wasmJsBrowserDevelopmentRun

# Production static files
./gradlew :webApp:wasmJsBrowserDistribution
# → webApp/build/dist/wasmJs/productionExecutable/
```

Desktop commands stay:

```bash
./gradlew :desktopApp:run
./gradlew :desktopApp:hotRun --auto
./gradlew :shared:jvmTest
```

Compatibility mode (only if `js` and `wasmJs` both exist):

```bash
./gradlew :webApp:composeCompatibilityBrowserDistribution
# → webApp/build/dist/composeWebCompatibility/productionExecutable/
```

(The wizard's path names `composeApp/...`; Ranko's module is `webApp`.)

Host the `productionExecutable` directory as a static site (GitHub Pages, nginx, …). Serve it with COOP/COEP headers if a future feature needs `SharedArrayBuffer`; the default Compose Wasm app does not require them.

Yarn/npm: the Kotlin/JS plugin writes `kotlin-js-store/`. The repo root `.gitignore` already ignores that directory. Decide at implementation time whether to commit the lockfile (reproducible CI) or keep ignoring it (current policy).

## Checklist (copy into the implementing PR)

- [ ] `:shared` declares `wasmJs { browser() }` as a library.
- [ ] `:webApp` is included, executable, `outputModuleName` matches `index.html`'s script.
- [ ] `ComposeViewport` + `#webApp` + full-viewport CSS.
- [ ] No `java.*` in `commonMain` (the Wasm compile is the proof).
- [ ] Coil network engine is per-target (OkHttp on JVM, Ktor on Wasm).
- [ ] `ktoml-file` is JVM-only; parse stays on `ktoml-core` strings.
- [ ] `TrainerIpcClient` is an `expect` / interface; Wasm does not spawn processes.
- [ ] `maskPaintInput` has a Wasm `actual`.
- [ ] Appearance prefs have a Wasm `actual`.
- [ ] Metro graph still builds on both JVM and Wasm.
- [ ] `./gradlew :desktopApp:run` and `:shared:jvmTest` still pass.
- [ ] `./gradlew :webApp:wasmJsBrowserDevelopmentRun` paints `App()`.
- [ ] Docs: this file, plus a sentence in `doc/dashboard.md` / `ranko/README.md` once it ships.
- [ ] AGENT.md §7 / §13 updated only when the port is real, not when the Gradle stub lands.

## What this runbook does not decide

Whether Ranko on the web is worth the IPC redesign, which of the three IPC shapes to take, and whether FileKit should be upgraded past 0.8.8. Those are product calls. The Gradle and source-set steps above stay the same regardless.
