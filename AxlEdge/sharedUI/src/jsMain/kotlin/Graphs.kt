
// Copyright (C) 2025 Zac Sweers
// SPDX-License-Identifier: Apache-2.0

import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.DependencyGraph
import dev.zacsweers.metro.Provides
import dev.zacsweers.metrox.viewmodel.ViewModelGraph
import io.ktor.client.HttpClient
import io.ktor.client.engine.js.Js
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.serialization.kotlinx.json.json
import kotlinx.serialization.json.Json
import org.acitelight.axledge.AppGraphBase

@DependencyGraph(AppScope::class) interface AppGraphJs : AppGraphBase
{
    @Provides
    fun provideHttpClient(): HttpClient = HttpClient(Js).config {
        install(ContentNegotiation) {
            json(Json {
                ignoreUnknownKeys = true
                prettyPrint = true
                isLenient = true
            })
        }
        install(HttpTimeout)
    }
}