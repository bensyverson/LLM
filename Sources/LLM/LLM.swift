//
//  LLM.swift
//
//
//  Created by Ben Syverson on 2024-01-14.
//

import Foundation

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
    }
}
