package com.acite.axlranko.data

import coil3.ImageLoader
import coil3.decode.DataSource
import coil3.decode.ImageSource
import coil3.fetch.FetchResult
import coil3.fetch.Fetcher
import coil3.fetch.SourceFetchResult
import coil3.request.Options
import okio.Buffer
import okio.FileSystem

expect fun blobImageFileSystem(): FileSystem

class BlobFetcher(
    private val ref: BlobRef,
    private val store: BlobStore,
) : Fetcher {
    override suspend fun fetch(): FetchResult {
        val bytes = store.get(ref)
        val buffer = Buffer().write(bytes)
        return SourceFetchResult(
            source = ImageSource(buffer, blobImageFileSystem()),
            mimeType = when (ref.format) {
                "png" -> "image/png"
                "webp" -> "image/webp"
                else -> "image/jpeg"
            },
            dataSource = DataSource.NETWORK,
        )
    }

    class Factory(private val store: BlobStore) : Fetcher.Factory<BlobRef> {
        override fun create(data: BlobRef, options: Options, imageLoader: ImageLoader): Fetcher =
            BlobFetcher(data, store)
    }
}
