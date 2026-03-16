//
//  ContentPart.swift
//  LLM
//
//  Multimodal content parts for chat messages.
//

import Foundation

public extension LLM.OpenAICompatibleAPI {
    /// A piece of content within a chat message.
    ///
    /// Messages can contain text, images, PDFs, or (in future) audio/video.
    /// When a conversation is sent to a provider, each part is encoded in the
    /// provider's native format automatically.
    enum ContentPart: Friendly {
        /// Text content.
        case text(String)
        /// Image with raw data, MIME type, optional filename, and optional description.
        case image(data: Data, mediaType: String, filename: String? = nil, description: String? = nil)
        /// PDF document with raw data and optional title.
        case pdf(data: Data, title: String? = nil)
        /// Audio content (support forthcoming).
        @available(*, message: "Audio support is forthcoming")
        case audio(data: Data, mediaType: String)
        /// Video content (support forthcoming).
        @available(*, message: "Video support is forthcoming")
        case video(data: Data, mediaType: String)

        /// The text content if this is a `.text` part, `nil` otherwise.
        public var textContent: String? {
            if case let .text(s) = self { return s }
            return nil
        }

        /// Whether this part contains media (image, PDF, audio, or video).
        public var isMedia: Bool {
            switch self {
            case .text: return false
            default: return true
            }
        }

        /// The filename for images or title for PDFs, `nil` for other types.
        public var filename: String? {
            switch self {
            case let .image(_, _, filename, _): return filename
            case let .pdf(_, title): return title
            default: return nil
            }
        }
    }
}

// MARK: - URL Convenience Initializers

public extension LLM.OpenAICompatibleAPI.ContentPart {
    /// Creates an image content part from a URL.
    ///
    /// Local file URLs are loaded into `Data`. Remote URLs are loaded eagerly.
    /// Media type is inferred from the file extension if not provided.
    ///
    /// - Parameters:
    ///   - url: The URL of the image file.
    ///   - mediaType: The MIME type (e.g. `"image/jpeg"`). Inferred from extension if `nil`.
    ///   - filename: An optional filename. Defaults to the URL's last path component for file URLs.
    ///   - description: An optional text description of the image.
    static func image(url: URL, mediaType: String? = nil, filename: String? = nil, description: String? = nil) throws -> Self {
        let data = try Data(contentsOf: url)
        let resolvedMediaType = mediaType ?? Self.mediaType(forExtension: url.pathExtension)
        let resolvedFilename = filename ?? (url.isFileURL ? url.lastPathComponent : nil)
        guard let mt = resolvedMediaType else {
            throw LLM.OpenAICompatibleAPI.ContentPartError.unknownMediaType(url.pathExtension)
        }
        return .image(data: data, mediaType: mt, filename: resolvedFilename, description: description)
    }

    /// Creates a PDF content part from a URL.
    ///
    /// - Parameters:
    ///   - url: The URL of the PDF file.
    ///   - title: An optional title. Defaults to the URL's last path component for file URLs.
    static func pdf(url: URL, title: String? = nil) throws -> Self {
        let data = try Data(contentsOf: url)
        let resolvedTitle = title ?? (url.isFileURL ? url.lastPathComponent : nil)
        return .pdf(data: data, title: resolvedTitle)
    }
}

// MARK: - Data Convenience Initializer

public extension LLM.OpenAICompatibleAPI.ContentPart {
    /// Creates an image content part from raw data, inferring the media type from magic bytes.
    ///
    /// Supports JPEG, PNG, GIF, and WebP formats.
    ///
    /// - Parameters:
    ///   - data: The raw image data.
    ///   - filename: An optional filename.
    ///   - description: An optional text description.
    /// - Throws: ``LLM.OpenAICompatibleAPI.ContentPartError/unknownMediaType(_:)`` if the format cannot be determined.
    static func image(data: Data, filename: String? = nil, description: String? = nil) throws -> Self {
        guard let mediaType = inferMediaType(from: data) else {
            throw LLM.OpenAICompatibleAPI.ContentPartError.unknownMediaType("(data)")
        }
        return .image(data: data, mediaType: mediaType, filename: filename, description: description)
    }
}

// MARK: - Media Type Inference

extension LLM.OpenAICompatibleAPI.ContentPart {
    /// Infers a media type from raw data by reading magic bytes.
    static func inferMediaType(from data: Data) -> String? {
        guard data.count >= 4 else { return nil }
        let bytes = [UInt8](data.prefix(12))

        // JPEG: FF D8
        if bytes[0] == 0xFF, bytes[1] == 0xD8 {
            return "image/jpeg"
        }
        // PNG: 89 50 4E 47
        if bytes[0] == 0x89, bytes[1] == 0x50, bytes[2] == 0x4E, bytes[3] == 0x47 {
            return "image/png"
        }
        // GIF: 47 49 46
        if bytes[0] == 0x47, bytes[1] == 0x49, bytes[2] == 0x46 {
            return "image/gif"
        }
        // WebP: 52 49 46 46 ... 57 45 42 50
        if data.count >= 12, bytes[0] == 0x52, bytes[1] == 0x49, bytes[2] == 0x46, bytes[3] == 0x46,
           bytes[8] == 0x57, bytes[9] == 0x45, bytes[10] == 0x42, bytes[11] == 0x50
        {
            return "image/webp"
        }
        return nil
    }

    /// Maps a file extension to a MIME type.
    static func mediaType(forExtension ext: String) -> String? {
        switch ext.lowercased() {
        case "jpg", "jpeg": return "image/jpeg"
        case "png": return "image/png"
        case "gif": return "image/gif"
        case "webp": return "image/webp"
        case "pdf": return "application/pdf"
        default: return nil
        }
    }
}

// MARK: - Errors

public extension LLM.OpenAICompatibleAPI {
    /// Errors that can occur when creating content parts.
    enum ContentPartError: Error {
        /// The media type could not be determined from the file extension or data header.
        case unknownMediaType(String)
    }
}
