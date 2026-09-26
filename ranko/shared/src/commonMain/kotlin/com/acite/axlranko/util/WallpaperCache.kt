package com.acite.axlranko.util

internal object WallpaperCache {
    private var key: String = ""
    private var model: Any? = null

    fun remember(path: String, decode: (String) -> Any?): Any? {
        if (path.isEmpty()) {
            key = ""
            model = null
            return null
        }
        if (path == key) return model
        model = decode(path)
        key = path
        return model
    }

    fun prime(path: String, value: Any?) {
        key = path
        model = value
    }
}
