package services

import dataModel.HealthzResponse
import dataModel.PromptOverrides
import dataModel.SamplesResponse
import dev.zacsweers.metro.Inject
import dev.zacsweers.metro.SingleIn
import dev.zacsweers.metro.AppScope

import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.plugins.timeout
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
        seed: Long? = null,
        steps: Int? = null,
        cfgScale: Float? = null,
        width: Int? = null,
        height: Int? = null
    ): ByteArray {
        val response: HttpResponse = httpClient.post("$baseUrl/api/quick") {
            timeout {
                requestTimeoutMillis = 300000L
                socketTimeoutMillis = 300000L
            }
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
        seed: Long? = null,
        steps: Int? = null,
        cfgScale: Float? = null,
        width: Int? = null,
        height: Int? = null
    ): ByteArray {
        val response: HttpResponse = httpClient.post("$baseUrl/api/generate") {
            timeout {
                requestTimeoutMillis = 300000L
                socketTimeoutMillis = 300000L
            }
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
    suspend fun getTrainConfig(): JsonObject {
        return httpClient.get("$baseUrl/api/config/train").body()
    }

    /**
     * Fetches the current generation configuration from the server.
     *
     * The returned JsonObject contains the following configuration state:
     * - DEVICE (String): The compute device (e.g., "cuda" or "cpu").
     * - TORCH_DTYPE (String): String representation of the torch data type (e.g., "torch.bfloat16").
     * - BASE_MODEL_PATH (String): Absolute path or repo ID for the base model.
     * - LORA_PATH (String): Absolute path to the LoRA model file.
     * - LORA_SCALE (Float): Strength/scale of the LoRA model.
     * - WIDTH (Int): Default width of the generated image.
     * - HEIGHT (Int): Default height of the generated image.
     * - STEPS (Int): Default number of inference steps.
     * - CFG_SCALE (Float): Default classifier-free guidance scale.
     * - SEED (Long): The default seed used for generation (e.g., 8576160563625674040).
     * - POSITIVE_PROMPT (String): The default positive prompt used for generation.
     * - NEGATIVE_PROMPT (String): The default negative prompt used for generation.
     * - REFINEMENT_PASSES (JsonArray): A list of objects defining detailer passes. Each object contains:
     * - name (String): Target area (e.g., "face", "hand").
     * - model (String): Path to the bounding box detection model.
     * - denoise (Float): Denoising strength for the inpainting pass.
     * - guide_size (Int): Base resolution for the detailer crop.
     * - OUTPUT_FILENAME_PREFIX (String): Prefix path for saved images.
     * - REALESRGAN_MODEL_PATH (String): Path to the RealESRGAN upscaler model.
     * - max_token_length (Int): Maximum CLIP token length limit.
     * - clip_skip (Int): Number of CLIP layers to skip during encoding.
     *
     * @return JsonObject containing the dynamic key-value pairs of the generation config.
     */
    suspend fun getGenerateConfig(): JsonObject {
        return httpClient.get("$baseUrl/api/config/generate").body()
    }
}