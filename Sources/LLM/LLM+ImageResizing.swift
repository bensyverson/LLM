//
//  LLM+ImageResizing.swift
//  LLM
//
//  Client-side image resizing for multimodal content.
//

import Foundation

// MARK: - Resize Cache

extension LLM {
    /// Cache key for resized images.
    struct ResizeCacheKey: Hashable {
        let dataHash: Int
        let targetWidth: Int
        let targetHeight: Int
    }
}

// MARK: - Resize Logic

public extension LLM {
    /// Resizes images in a conversation so that no image's long edge exceeds `maxLongEdge`.
    ///
    /// Uses a `TaskGroup` to resize images in parallel. Results are cached so that
    /// the same image is not resized twice.
    ///
    /// - Parameters:
    ///   - conversation: The conversation containing images to resize.
    ///   - maxLongEdge: The maximum long-edge dimension in pixels.
    ///   - resizer: A closure that resizes image data. Receives `(data, mediaType, targetSize)`.
    /// - Returns: A new conversation with oversized images replaced by resized versions.
    func resizingImages(
        in conversation: Conversation,
        maxLongEdge: Int,
        using resizer: @escaping @Sendable (Data, String, CGSize) async throws -> Data,
    ) async throws -> Conversation {
        // Collect all image parts that might need resizing
        typealias ImageRef = (msgIndex: Int, partIndex: Int, data: Data, mediaType: String)
        var imageRefs: [ImageRef] = []

        for (mi, msg) in conversation.messages.enumerated() {
            for (pi, part) in msg.content.enumerated() {
                if case let .image(data, mediaType, _, _) = part {
                    imageRefs.append((mi, pi, data, mediaType))
                }
            }
        }

        guard !imageRefs.isEmpty else { return conversation }

        let targetSize = CGSize(width: maxLongEdge, height: maxLongEdge)

        // Resize in parallel, using cache
        let resizedResults: [(Int, Int, Data)] = try await withThrowingTaskGroup(of: (Int, Int, Data)?.self) { group in
            for ref in imageRefs {
                group.addTask { [self] in
                    let cacheKey = ResizeCacheKey(
                        dataHash: ref.data.hashValue,
                        targetWidth: maxLongEdge,
                        targetHeight: maxLongEdge,
                    )

                    // Check cache
                    if let cached = await getCachedResize(for: cacheKey) {
                        return (ref.msgIndex, ref.partIndex, cached)
                    }

                    // Check if resize is needed by looking at dimensions
                    if let dims = Self.imageDimensions(from: ref.data), max(dims.width, dims.height) <= maxLongEdge {
                        return nil // No resize needed
                    }

                    let resized = try await resizer(ref.data, ref.mediaType, targetSize)
                    await cacheResize(resized, for: cacheKey)
                    return (ref.msgIndex, ref.partIndex, resized)
                }
            }

            var results: [(Int, Int, Data)] = []
            for try await result in group {
                if let r = result { results.append(r) }
            }
            return results
        }

        // Apply resized data back to conversation
        var result = conversation
        for (mi, pi, resizedData) in resizedResults {
            if case let .image(_, mediaType, filename, description) = result.messages[mi].content[pi] {
                result.messages[mi].content[pi] = .image(data: resizedData, mediaType: mediaType, filename: filename, description: description)
            }
        }

        return result
    }

    /// Reads image dimensions from data without fully decoding.
    static func imageDimensions(from data: Data) -> (width: Int, height: Int)? {
        #if canImport(CoreGraphics) && canImport(ImageIO)
            guard let source = CGImageSourceCreateWithData(data as CFData, nil),
                  let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any],
                  let width = properties[kCGImagePropertyPixelWidth as String] as? Int,
                  let height = properties[kCGImagePropertyPixelHeight as String] as? Int
            else {
                return nil
            }
            return (width, height)
        #else
            return nil
        #endif
    }
}

// MARK: - Cache Management

extension LLM {
    func getCachedResize(for key: ResizeCacheKey) -> Data? {
        resizeCache[key]
    }

    func cacheResize(_ data: Data, for key: ResizeCacheKey) {
        // FIFO eviction when cache is full
        if resizeCache.count >= resizeCacheMaxSize {
            if let firstKey = resizeCacheOrder.first {
                resizeCache.removeValue(forKey: firstKey)
                resizeCacheOrder.removeFirst()
            }
        }
        resizeCache[key] = data
        resizeCacheOrder.append(key)
    }
}

// MARK: - CoreGraphics Default Resizer

#if canImport(CoreGraphics) && canImport(ImageIO)
    import CoreGraphics
    import ImageIO
    import UniformTypeIdentifiers

    public extension LLM {
        /// Default image resizer using CoreGraphics.
        ///
        /// Scales the image so its long edge fits within `targetSize`, preserving aspect ratio.
        /// Returns JPEG data for JPEG inputs, PNG for everything else.
        static let coreGraphicsResizer: @Sendable (Data, String, CGSize) async throws -> Data = { data, mediaType, targetSize in
            guard let source = CGImageSourceCreateWithData(data as CFData, nil),
                  let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
            else {
                return data
            }

            let originalWidth = CGFloat(image.width)
            let originalHeight = CGFloat(image.height)
            let maxDimension = max(originalWidth, originalHeight)
            let targetMax = max(targetSize.width, targetSize.height)

            guard maxDimension > targetMax else { return data }

            let scale = targetMax / maxDimension
            let newWidth = Int(originalWidth * scale)
            let newHeight = Int(originalHeight * scale)

            guard let colorSpace = image.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB),
                  let context = CGContext(
                      data: nil,
                      width: newWidth,
                      height: newHeight,
                      bitsPerComponent: 8,
                      bytesPerRow: 0,
                      space: colorSpace,
                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue,
                  )
            else {
                return data
            }

            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: newWidth, height: newHeight))

            guard let resizedImage = context.makeImage() else { return data }

            let isJPEG = mediaType == "image/jpeg"
            let destType = isJPEG ? "public.jpeg" as CFString : "public.png" as CFString
            let mutableData = NSMutableData()
            guard let dest = CGImageDestinationCreateWithData(mutableData, destType, 1, nil) else { return data }

            let options: [CFString: Any] = isJPEG ? [kCGImageDestinationLossyCompressionQuality: 0.85] : [:]
            CGImageDestinationAddImage(dest, resizedImage, options as CFDictionary)
            guard CGImageDestinationFinalize(dest) else { return data }

            return mutableData as Data
        }
    }
#endif
