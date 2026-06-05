package utils

fun String.appendQueryParams(params: Map<String, String?>): String {
    val filtered = params.filterValues { it != null }
    if (filtered.isEmpty()) return this
    val query = filtered.entries.joinToString("&") { "${it.key}=${it.value}" }
    return if (this.contains("?")) "$this&$query" else "$this?$query"
}