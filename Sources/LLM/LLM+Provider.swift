//
//  File.swift
//  LLM
//
//  Created by Ben Syverson on 2024-11-13.
//

import Foundation

public extension LLM {
	enum Provider: Friendly {
		/// This will send requests to OpenAI's API (https://api.openai.com/)
		case openAI(apiKey: String)
		/// This will send requests to Anthropic's API
		case anthropic(apiKey: String)
		/// This sends requests to localhost:1234, the default port for LM Studio
		case lmStudio
		/// This sends requests to localhost:port
		case localhost(port: Int)
		/// Pass in the root URL to any OpenAI-compatible API (not including /v1 and beyond)
		case other(URL, apiKey: String?)

		public var chatLimiter: RateLimiter {
			switch self {
			case .openAI(apiKey: _):
				return RateLimiter(maxRequests: 8000, maxTokens: 1_500_000, interval: 60) // Tier 3
			default:
				return RateLimiter(maxRequests: 5, interval: 1.0) // Tier 1 Anthropic
			}
		}

		public var embeddingLimiter: RateLimiter {
			switch self {
			case .openAI(apiKey: _):
				return RateLimiter(maxRequests: 4000, maxTokens: 30_000_000, interval: 60) // Tier 2
			default:
				return RateLimiter(maxRequests: 10000, interval: 0.1)
			}
		}
	}

	var providerApi: OpenAICompatibleAPI {
		switch provider {
		case .openAI(let apiKey):
			return OpenAICompatibleAPI.openAI(apiKey: apiKey)
		case .anthropic(let apiKey):
			return OpenAICompatibleAPI.anthropic(apiKey: apiKey)
		case .lmStudio:
			return OpenAICompatibleAPI.localhost(port: 1234)
		case .other(let url, apiKey: let apiKey):
			if let apiKey {
				return OpenAICompatibleAPI(
					baseURL: url,
					authMethod: .bearer(apiKey: apiKey)
				)
			} else {
				return OpenAICompatibleAPI(baseURL: url)
			}
		case .localhost(let port):
			return OpenAICompatibleAPI.localhost(port: port)
		}
	}
}
