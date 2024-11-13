//
//  LLM.swift
//
//
//  Created by Ben Syverson on 2024-01-14.
//

import Foundation

typealias Friendly = Codable & Hashable & Equatable & Sendable

public actor LLM {
	public enum LLMError: Error {
		case parseResponse(LLM.OpenAICompatibleAPI.ChatCompletionResponse)
	}

	public var provider: LLM.Provider = .lmStudio

	internal let chatRateLimiter: LLM.RateLimiter
	internal let embeddingRateLimiter: LLM.RateLimiter

	public init(
		provider: LLM.Provider = .lmStudio,
		chatLimiter: LLM.RateLimiter? = nil,
		embeddingLimiter: LLM.RateLimiter? = nil
	) {
		self.provider = provider
		chatRateLimiter = chatLimiter ?? provider.chatLimiter
		embeddingRateLimiter = embeddingLimiter ?? provider.embeddingLimiter
	}
}
