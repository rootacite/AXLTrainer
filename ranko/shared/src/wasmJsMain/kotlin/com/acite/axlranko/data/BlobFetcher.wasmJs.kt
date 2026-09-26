package com.acite.axlranko.data

import okio.FileSystem
import okio.fakefilesystem.FakeFileSystem

actual fun blobImageFileSystem(): FileSystem = FakeFileSystem()
