//
//  LLM.swift
//
//
//  Created by Ben Syverson on 2024-01-14.
//

import Foundation

typealias Friendly = Codable & Equatable & Hashable & Sendable

public actor LLM {
    public enum LLMError: Error {
        case noText
        case parseResponse(LLM.OpenAICompatibleAPI.ChatCompletionResponse)
    }

    public var provider: LLM.Provider = .lmStudio

    let chatRateLimiter: LLM.RateLimiter
    let embeddingRateLimiter: LLM.RateLimiter

    private let session: URLSession = {
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 120
        configuration.timeoutIntervalForResource = 120
        return URLSession(configuration: configuration)
    }()

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
