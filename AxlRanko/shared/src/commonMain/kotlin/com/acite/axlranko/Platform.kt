package com.acite.axlranko

interface Platform {
    val name: String
}

expect fun getPlatform(): Platform