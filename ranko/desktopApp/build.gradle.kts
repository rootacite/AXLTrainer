import org.jetbrains.compose.desktop.application.dsl.TargetFormat

plugins {
    alias(libs.plugins.kotlinJvm)
    alias(libs.plugins.composeMultiplatform)
    alias(libs.plugins.composeCompiler)
    alias(libs.plugins.metro)
}

dependencies {
    implementation(projects.shared)

    implementation(compose.desktop.currentOs)
    implementation(libs.compose.components.resources)
    implementation(libs.kotlinx.coroutinesSwing)

    implementation(libs.compose.uiToolingPreview)
}

compose.desktop {
    application {
        mainClass = "com.acite.axlranko.MainKt"

        nativeDistributions {
            targetFormats(TargetFormat.Dmg, TargetFormat.Msi, TargetFormat.Deb)
            packageName = "com.acite.axlranko"
            packageVersion = "2.2.1"
            linux {
                iconFile.set(project.file("icons/app_icon.png"))
            }
            windows {
                iconFile.set(project.file("icons/app_icon.ico"))
            }
            macOS {
                iconFile.set(project.file("icons/app_icon.icns"))
            }
        }
    }
}
