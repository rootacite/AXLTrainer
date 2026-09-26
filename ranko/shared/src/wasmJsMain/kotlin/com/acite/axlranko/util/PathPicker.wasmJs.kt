package com.acite.axlranko.util

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import com.acite.axlranko.data.FsEntry
import com.acite.axlranko.data.FsRoot
import com.acite.axlranko.data.TrainerIpcClient
import com.acite.axlranko.ui.components.CapsuleButton
import com.acite.axlranko.ui.components.PorcelainCard
import com.acite.axlranko.ui.components.rankoFieldColors
import com.acite.axlranko.ui.theme.rankoColors
import dev.zacsweers.metro.AppScope
import dev.zacsweers.metro.ContributesBinding
import dev.zacsweers.metro.Inject
import dev.zacsweers.metro.SingleIn
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

enum class FilePickerMode { Directory, File, Save }

data class FilePickerRequest(
    val mode: FilePickerMode,
    val title: String,
    val current: String,
    val extensions: List<String>?,
    val suggestedName: String,
    val deferred: CompletableDeferred<String?>,
)

@SingleIn(AppScope::class)
@ContributesBinding(AppScope::class)
@Inject
class WasmPathPicker(
    private val ipc: TrainerIpcClient,
) : PathPicker {
    private val _request = MutableStateFlow<FilePickerRequest?>(null)
    val request: StateFlow<FilePickerRequest?> = _request
    val client: TrainerIpcClient get() = ipc

    override suspend fun pickDirectory(title: String, current: String): String? =
        prompt(FilePickerMode.Directory, title, current, null, "")

    override suspend fun pickFile(title: String, current: String, extensions: List<String>?): String? =
        prompt(FilePickerMode.File, title, current, extensions, "")

    override suspend fun saveFile(suggestedName: String, current: String): String? =
        prompt(FilePickerMode.Save, "Save as", current, listOf("safetensors"), suggestedName)

    private suspend fun prompt(
        mode: FilePickerMode,
        title: String,
        current: String,
        extensions: List<String>?,
        suggestedName: String,
    ): String? {
        val deferred = CompletableDeferred<String?>()
        _request.value = FilePickerRequest(mode, title, current, extensions, suggestedName, deferred)
        return try {
            deferred.await()
        } finally {
            _request.value = null
        }
    }
}

@Composable
actual fun InstallPathPickerHost(picker: PathPicker) {
    val wasm = picker as? WasmPathPicker ?: return
    val request by wasm.request.collectAsState()
    val active = request ?: return
    FilePickerDialog(
        ipc = wasm.client,
        request = active,
        onDismiss = { active.deferred.complete(null) },
        onConfirm = { active.deferred.complete(it) },
    )
}

@Composable
private fun FilePickerDialog(
    ipc: TrainerIpcClient,
    request: FilePickerRequest,
    onDismiss: () -> Unit,
    onConfirm: (String) -> Unit,
) {
    val colors = rankoColors
    val scope = rememberCoroutineScope()
    var path by remember { mutableStateOf(request.current.ifBlank { "/" }) }
    var parent by remember { mutableStateOf<String?>(null) }
    var entries by remember { mutableStateOf<List<FsEntry>>(emptyList()) }
    var roots by remember { mutableStateOf<List<FsRoot>>(emptyList()) }
    var selectedFile by remember { mutableStateOf<String?>(null) }
    var filename by remember { mutableStateOf(request.suggestedName.ifBlank { "model.safetensors" }) }
    var error by remember { mutableStateOf<String?>(null) }

    fun load(target: String) {
        scope.launch {
            try {
                val listed = ipc.fsListdir(target)
                path = listed.path
                parent = listed.parent
                entries = listed.entries
                selectedFile = null
                error = null
            } catch (e: Exception) {
                error = e.message
            }
        }
    }

    LaunchedEffect(request) {
        roots = runCatching { ipc.fsRoots().roots }.getOrDefault(emptyList())
        val start = request.current.ifBlank { roots.firstOrNull()?.path ?: "/" }
        load(start)
    }

    val visible = entries.filter { entry ->
        if (entry.isDir) true
        else if (request.mode == FilePickerMode.Directory) false
        else {
            val ext = request.extensions
            if (ext.isNullOrEmpty()) true
            else ext.any { entry.name.endsWith(".$it", ignoreCase = true) }
        }
    }

    Dialog(onDismissRequest = onDismiss) {
        PorcelainCard(title = request.title, modifier = Modifier.fillMaxWidth()) {
            if (roots.isNotEmpty()) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    roots.forEach { root ->
                        CapsuleButton(
                            text = root.name,
                            onClick = { load(root.path) },
                            compact = true,
                        )
                    }
                }
            }
            Text(
                text = path,
                style = MaterialTheme.typography.bodySmall,
                color = colors.textDim,
            )
            if (error != null) {
                Text(text = error!!, color = colors.qualityRed, style = MaterialTheme.typography.bodySmall)
            }
            Box(Modifier.fillMaxWidth().heightIn(min = 240.dp, max = 360.dp)) {
                LazyColumn {
                    if (parent != null) {
                        item {
                            Text(
                                text = "..",
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clickable { load(parent!!) }
                                    .padding(vertical = 6.dp),
                                color = colors.text,
                            )
                        }
                    }
                    items(visible, key = { it.path }) { entry ->
                        val label = if (entry.isDir) "${entry.name}/" else entry.name
                        val selected = selectedFile == entry.path
                        Text(
                            text = label,
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable {
                                    if (entry.isDir) load(entry.path)
                                    else selectedFile = entry.path
                                }
                                .padding(vertical = 6.dp),
                            color = if (selected) colors.accentPink else colors.text,
                        )
                    }
                }
            }
            if (request.mode == FilePickerMode.Save) {
                OutlinedTextField(
                    value = filename,
                    onValueChange = { filename = it },
                    label = { Text("File name") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                    colors = rankoFieldColors(),
                )
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End),
            ) {
                CapsuleButton(text = "Cancel", onClick = onDismiss, compact = true)
                val confirmEnabled = when (request.mode) {
                    FilePickerMode.Directory -> path.isNotBlank()
                    FilePickerMode.File -> selectedFile != null
                    FilePickerMode.Save -> filename.isNotBlank() && path.isNotBlank()
                }
                CapsuleButton(
                    text = "Select",
                    onClick = {
                        val chosen = when (request.mode) {
                            FilePickerMode.Directory -> path
                            FilePickerMode.File -> selectedFile ?: return@CapsuleButton
                            FilePickerMode.Save -> {
                                val name = if (filename.endsWith(".safetensors")) filename else "$filename.safetensors"
                                "$path/$name"
                            }
                        }
                        onConfirm(chosen)
                    },
                    enabled = confirmEnabled,
                    compact = true,
                    emphasized = true,
                )
            }
        }
    }
}
