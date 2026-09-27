package com.acite.axlranko.pages.components.automation

import com.acite.axlranko.prompt.PromptLang

/**
 * Chrome text of the Automation page. The wizard's own wording ("角色词", "总表情", …) comes from the
 * ported `tools/gen_prompts.py` string table; this covers the page around it, which that table never
 * had. A key with no entry falls back to the key itself.
 */
private val UI_TEXT: Map<String, Pair<String, String>> = mapOf(
    "language" to ("界面语言" to "Interface language"),
    "section_prompts" to ("向导、档案与生成结果" to "Wizard, profiles and the prompts they generate"),
    "section_comfy" to ("连接服务、选择工作流，把提示词批量送进去出图" to "Connect, pick a workflow, run the prompt batch"),
    "section_gallery" to ("浏览生成结果：任务、缩略图、单张保存" to "Browse the results: jobs, thumbnails, save one"),
    "placeholder_comfy_1" to (
        "连接服务、上传 API 格式工作流、把提示词批量送进去出图。" to
            "Connect to the server, upload an API-format workflow, and send the prompt batch through it."
        ),
    "placeholder_comfy_2" to (
        "这一步（服务端 automation_* 方法与 RPC 执行器）在下一个阶段落地。" to
            "This section (the automation_* methods and the RPC runner) lands in the next step."
        ),
    "placeholder_gallery_1" to (
        "任务列表、缩略图网格、单张保存与失败重试。" to
            "Job list, thumbnail grid, save one image, retry the failures."
        ),
    "placeholder_gallery_2" to (
        "随 ComfyUI 区一起在下一个阶段落地。" to "Lands together with the ComfyUI section."
        ),
    "matrix" to ("提示词矩阵" to "Prompt matrix"),
    "reload" to ("重新载入" to "Reload"),
    "refresh" to ("刷新" to "Refresh"),
    "load" to ("载入" to "Load"),
    "delete" to ("删除" to "Delete"),
    "wizard" to ("向导" to "Wizard"),
    "manifest" to ("总清单" to "List"),
    "mode_count" to ("模式 {mode} · 数量 {count}" to "mode {mode} · count {count}"),
    "results" to ("生成结果" to "Generated prompts"),
    "results_hint" to ("{n} 条 · seed {seed}" to "{n} prompts · seed {seed}"),
    "generate" to ("生成" to "Generate"),
    "copy_all" to ("复制全部" to "Copy all"),
    "download" to ("下载 .txt" to "Download .txt"),
    "send_to_batch" to ("用作批量输入" to "Send to batch"),
    "clear" to ("清空" to "Clear"),
    "results_empty" to (
        "还没有生成；在向导末页或总清单点「生成」" to "Nothing generated yet: press Generate"
        ),
    "batch_queue" to (
        "批量输入队列：{n} 条（在 ComfyUI 区使用）" to "Batch queue: {n} prompts (used in the ComfyUI section)"
        ),
    "profiles_empty" to (
        "还没有档案；在总清单里「存为档案」即可创建" to "No profile yet; save one from the list"
        ),
    "active_profile" to ("当前档案：" to "Active: "),
    "done" to ("完成" to "Done"),
    "save" to ("保存" to "Save"),
    "cancel" to ("取消" to "Cancel"),
    "back" to ("上一步" to "Back"),
    "next" to ("下一步" to "Next"),
    "exposure_empty" to (
        "没勾选时按模式的默认暴露抽" to "With nothing ticked the mode's default exposure is drawn"
        ),
    "pool_any" to ("从整池均匀抽" to "Drawn from the whole pool"),
    "no_entries" to ("矩阵里没有可用条目" to "No matrix entry applies"),
    "matrix_missing" to ("矩阵未载入" to "The matrix is not loaded"),
    "seed_hint" to ("种子（留空 = 随机）" to "Seed (blank = random)"),
    "face_candidates" to (
        "每条从勾选项里抽一个" to "One ticked tag per prompt"
        ),
    "ratio_split" to ("阴道 {v}% · 肛 {a}%" to "vaginal {v}% · anal {a}%"),
    "copied_one" to ("已复制 1 条" to "Copied 1 prompt"),
    "copied_many" to ("已复制 {n} 条" to "Copied {n} prompts"),
    "saved_to" to ("已保存 {path}" to "Saved {path}"),
    "sent_to_batch" to ("已交给批量生成：{n} 条" to "Sent {n} prompts to the batch queue"),
    "profile_deleted" to ("已删除档案 {name}" to "Deleted profile {name}"),
    "need_name" to ("请输入档案名称" to "Enter a profile name"),
    "save_failed" to ("保存失败" to "Save failed"),
    "load_failed" to ("载入档案失败" to "Could not load the profile"),
    "delete_failed" to ("删除档案失败" to "Could not delete the profile"),
    "matrix_failed" to ("读取矩阵失败" to "Could not read the matrix"),
    "profiles_failed" to ("读取档案失败" to "Could not read the profiles"),
    "matrix_unavailable" to ("矩阵不可用" to "The matrix is not available"),
    "generate_failed" to ("生成失败" to "Generation failed"),
    "settings_saved" to ("设置已保存" to "Settings saved"),
    "workflow_uploaded" to ("已上传工作流" to "Workflow uploaded"),
    "workflow_valid" to ("工作流校验通过" to "Workflow checks out"),
    "prompt_set_saved" to ("已保存 prompt 集" to "Prompt set saved"),
    "job_started" to ("已启动任务" to "Job started"),
    "job_deleted" to ("已删除任务" to "Job deleted"),
    "no_prompts" to ("没有可用的提示词" to "No prompts to generate"),
    "server" to ("服务" to "Server"),
    "server_hint" to (
        "留空则自动发现：扫描本机监听端口，只认 ComfyUI" to
            "Leave blank to auto-discover: loopback listeners are probed, only a ComfyUI answers"
        ),
    "discover" to ("重新探测" to "Probe again"),
    "discovering" to ("探测中…" to "Probing…"),
    "connected" to ("已连接" to "Connected"),
    "not_found" to ("未发现监听中的 ComfyUI" to "No ComfyUI found listening"),
    "queue" to ("队列" to "Queue"),
    "queue_line" to ("运行 {running} · 等待 {pending}" to "running {running} · pending {pending}"),
    "workflow" to ("工作流" to "Workflow"),
    "upload_workflow" to ("上传 JSON…" to "Upload JSON…"),
    "save_settings" to ("保存设置" to "Save settings"),
    "validate" to ("校验" to "Check"),
    "nodes_line" to (
        "{nodes} 个节点 · SaveImage ×{save} · batch_size ×{batch}" to
            "{nodes} nodes · SaveImage ×{save} · batch_size ×{batch}"
        ),
    "model_check_ok" to ("模型全部就位" to "every model resolves"),
    "model_check_skipped" to ("未连接，跳过模型预检" to "not connected, model pre-check skipped"),
    "missing_models" to ("缺少模型" to "Missing models"),
    "positive_node" to ("正提示词节点" to "Positive-prompt node"),
    "positive_node_guessed" to ("（自动推断）" to "(guessed)"),
    "no_workflows" to ("还没有上传工作流" to "No workflow uploaded yet"),
    "batch" to ("批量生成" to "Batch"),
    "prompt_source" to ("提示词来源" to "Prompt source"),
    "source_results" to ("当前生成结果 {n} 条" to "current results ({n})"),
    "source_set" to ("已存 prompt 集" to "saved prompt set"),
    "source_manual" to ("手输" to "type them"),
    "manual_hint" to ("一行一条" to "one prompt per line"),
    "images_per_prompt" to ("每 prompt 张数" to "images per prompt"),
    "poll_interval" to ("轮询间隔" to "poll interval"),
    "output_dir" to ("输出目录" to "Output folder"),
    "start_job" to ("开始生成" to "Start"),
    "cancel_job" to ("取消" to "Cancel"),
    "save_prompt_set" to ("存为 prompt 集" to "Save as prompt set"),
    "set_name" to ("名称" to "Name"),
    "log" to ("日志" to "Log"),
    "jobs" to ("任务" to "Jobs"),
    "filter_all" to ("全部" to "All"),
    "filter_running" to ("运行中" to "Running"),
    "filter_done" to ("完成" to "Done"),
    "filter_failed" to ("失败" to "Failed"),
    "filter_cancelled" to ("已取消" to "Cancelled"),
    "search" to ("搜索" to "Search"),
    "no_jobs" to ("还没有任务" to "No jobs yet"),
    "job_counts" to (
        "{done}/{total} · {images} 张 · {elapsed}" to "{done}/{total} · {images} images · {elapsed}"
        ),
    "retry_failed" to ("重试失败" to "Retry failed"),
    "open_folder" to ("打开目录" to "Open folder"),
    "thumb_size" to ("缩略图" to "Thumbnails"),
    "save_image" to ("保存这张…" to "Save this image…"),
    "save_records" to ("保存记录 .txt" to "Save records .txt"),
    "copy_prompt" to ("复制 prompt" to "Copy prompt"),
    "job_actions" to ("任务操作" to "Job actions"),
    "prompt_state_pending" to ("待处理" to "pending"),
    "prompt_state_running" to ("生成中" to "running"),
    "prompt_state_done" to ("完成" to "done"),
    "prompt_state_error" to ("失败" to "failed"),
    "no_images_yet" to ("这个任务还没有图片" to "This job has no images yet"),
    "upload_failed_hint" to (
        "选择 ComfyUI 里 Save (API Format) 导出的 JSON" to
            "Pick the JSON ComfyUI writes with Save (API Format)"
        ),
    "confirm_delete_job" to (
        "删除这个任务和它的图片？" to "Delete this job and its images?"
        ),
    "no_prompt_sets" to ("还没有 prompt 集" to "No prompt set saved yet"),
    "browse" to ("浏览…" to "Browse…"),
    "yes" to ("确定" to "Yes"),
    "no" to ("取消" to "No"),
)


fun uiText(lang: PromptLang, key: String): String {
    val entry = UI_TEXT[key] ?: return key
    return if (lang == PromptLang.Chinese) entry.first else entry.second
}
