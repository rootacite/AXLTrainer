
// Copyright (C) 2025 Zac Sweers
// SPDX-License-Identifier: Apache-2.0

package org.acitelight.axledge

import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.DependencyGraph
import dev.zacsweers.metro.Provides
import dev.zacsweers.metrox.viewmodel.ViewModelGraph
import io.ktor.client.HttpClient
import io.ktor.client.engine.cio.CIO
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.serialization.kotlinx.json.json
import kotlinx.serialization.json.Json

@DependencyGraph(AppScope::class) interface AppGraphCio : AppGraphBase
{
    @Provides
    fun provideHttpClient(): HttpClient = HttpClient(CIO).config {
        install(ContentNegotiation) {
            json(Json {
                ignoreUnknownKeys = true
                prettyPrint = true
                isLenient = true
            })
        }
    }
}