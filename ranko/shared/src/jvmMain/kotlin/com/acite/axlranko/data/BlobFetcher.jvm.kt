package com.acite.axlranko.data

import okio.FileSystem

actual fun blobImageFileSystem(): FileSystem = FileSystem.SYSTEM
