//
//  LLM.swift
//
//
//  Created by Ben Syverson on 2024-01-14.
//

import Foundation
#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

/// Convenience typealias for types that are fully value-semantic and serializable.
typealias Friendly = Codable & Equatable & Hashable & Sendable

/// The main entry point for interacting with language model APIs.
///
/// `LLM` is an actor that manages a provider connection, rate limiters, and a URL session.
/// Create an instance with a ``Provider``, then call chat or streaming methods.
///
/// ```swift
/// let llm = LLM(provider: .openAI(apiKey: "sk-..."))
/// let response: String = try await llm.chat(configuration: config)
/// ```
public actor LLM {
    /// Errors that can occur during LLM API calls.
    public enum LLMError: Error {
        /// The response contained no text content.
        case noText
        /// The response could not be parsed into text or tool calls.
        case parseResponse(LLM.OpenAICompatibleAPI.ChatCompletionResponse)
    }

    /// The provider this instance sends requests to.
    public var provider: LLM.Provider = .lmStudio

    /// The rate limiter used for chat completion requests.
    let chatRateLimiter: LLM.RateLimiter
    /// The rate limiter used for embedding requests.
    let embeddingRateLimiter: LLM.RateLimiter

    /// Resizes image data to fit within a target size.
    ///
    /// Parameters: `(imageData, mediaType, targetSize) -> resizedData`.
    /// On Apple platforms this defaults to a CoreGraphics implementation.
    /// On Linux this is `nil` by default — set it to enable image resizing.
    public var imageResizer: (@Sendable (Data, String, CGSize) async throws -> Data)?

    /// Generates a text description of an image for non-vision model fallback.
    ///
    /// Parameters: `(imageData, mediaType) -> description`.
    /// `nil` by default. When set, ``strippingMedia(_:using:)`` calls this
    /// for images that lack a description.
    public var imageDescriber: (@Sendable (Data, String) async throws -> String)?

    /// Cache for resized images, keyed by hash of original data + target size.
    var resizeCache: [ResizeCacheKey: Data] = [:]
    /// FIFO order for cache eviction.
    var resizeCacheOrder: [ResizeCacheKey] = []
    /// Maximum entries in the resize cache before FIFO eviction.
    let resizeCacheMaxSize: Int = 50

    private let session: URLSession = {
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 120
        configuration.timeoutIntervalForResource = 120
        return URLSession(configuration: configuration)
    }()

    /// Creates an LLM instance configured for the given provider.
    ///
    /// - Parameters:
    ///   - provider: The LLM service provider to use.
    ///   - chatLimiter: Optional custom rate limiter for chat requests. If `nil`, uses the provider's default.
    ///   - embeddingLimiter: Optional custom rate limiter for embedding requests. If `nil`, uses the provider's default.
    ///   - timeout: Request timeout in seconds.
    public init(
        provider: LLM.Provider = .lmStudio,
        chatLimiter: LLM.RateLimiter? = nil,
        embeddingLimiter: LLM.RateLimiter? = nil,
        timeout: TimeInterval = 120
    ) {
        self.provider = provider
        chatRateLimiter = chatLimiter ?? provider.chatLimiter
        embeddingRateLimiter = embeddingLimiter ?? provider.embeddingLimiter
        session.configuration.timeoutIntervalForRequest = timeout
        session.configuration.timeoutIntervalForResource = timeout
        #if canImport(CoreGraphics) && canImport(ImageIO)
            imageResizer = Self.coreGraphicsResizer
        #endif
    }
}
