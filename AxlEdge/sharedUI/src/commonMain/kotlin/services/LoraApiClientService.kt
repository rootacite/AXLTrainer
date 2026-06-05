package services

import dataModel.HealthzResponse
import dataModel.PromptOverrides
import dataModel.SamplesResponse
import dev.zacsweers.metro.Inject
import dev.zacsweers.metro.SingleIn
import dev.zacsweers.metro.AppScope

import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.request.*
import io.ktor.client.statement.*
import io.ktor.client.statement.readRawBytes
import io.ktor.http.*
import kotlinx.serialization.json.JsonObject


@Inject
@SingleIn(AppScope::class)
class LoraApiClient(
    private val httpClient: HttpClient
) {
    private val baseUrl: String = "http://192.168.4.4:8000"
    // ==========================================
    // Server Health
    // ==========================================

    suspend fun getHealthz(): HealthzResponse {
        return httpClient.get("$baseUrl/healthz").body()
    }

    // ==========================================
    // Dashboard & Samples Endpoints
    // ==========================================

    suspend fun getDashboardData(
        name: String? = null,
        startStep: Int? = null,
        endStep: Int? = null
    ): JsonObject {
        return httpClient.get("$baseUrl/api/dashboard") {
            name?.let { parameter("name", it) }
            startStep?.let { parameter("start_step", it) }
            endStep?.let { parameter("end_step", it) }
        }.body()
    }

    suspend fun listSamples(name: String? = null): SamplesResponse {
        return httpClient.get("$baseUrl/api/samples") {
            name?.let { parameter("name", it) }
        }.body()
    }

    suspend fun getSampleImage(filename: String, name: String? = null): ByteArray {
        val response: HttpResponse = httpClient.get("$baseUrl/api/samples/$filename") {
            name?.let { parameter("name", it) }
        }
        return response.readRawBytes()
    }

    // ==========================================
    // Generation Endpoints
    // ==========================================

    suspend fun generateQuick(
        overrides: PromptOverrides = PromptOverrides(),
        seed: Int? = null,
        steps: Int? = null,
        cfgScale: Float? = null,
        width: Int? = null,
        height: Int? = null
    ): ByteArray {
        val response: HttpResponse = httpClient.post("$baseUrl/api/quick") {
            contentType(ContentType.Application.Json)
            setBody(overrides)

            seed?.let { parameter("seed", it) }
            steps?.let { parameter("steps", it) }
            cfgScale?.let { parameter("cfg_scale", it) }
            width?.let { parameter("width", it) }
            height?.let { parameter("height", it) }
        }
        return response.readRawBytes()
    }

    suspend fun generateApi(
        overrides: PromptOverrides = PromptOverrides(),
        stages: Int = 3,
        seed: Int? = null,
        steps: Int? = null,
        cfgScale: Float? = null,
        width: Int? = null,
        height: Int? = null
    ): ByteArray {
        val response: HttpResponse = httpClient.post("$baseUrl/api/generate") {
            contentType(ContentType.Application.Json)
            setBody(overrides)

            parameter("stages", stages)
            seed?.let { parameter("seed", it) }
            steps?.let { parameter("steps", it) }
            cfgScale?.let { parameter("cfg_scale", it) }
            width?.let { parameter("width", it) }
            height?.let { parameter("height", it) }
        }
        return response.readRawBytes()
    }
}