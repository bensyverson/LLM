//
//  LLM+Provider.swift
//  LLM
//
//  Created by Ben Syverson on 2024-11-13.
//

import Foundation

public extension LLM {
    /// Identifies the LLM service provider and holds its credentials.
    ///
    /// Each case supplies the base URL and authentication method for the provider.
    /// Rate limiters start with conservative defaults and adapt automatically
    /// as the library reads rate limit headers from API responses.
    enum Provider: Friendly {
        /// Sends requests to OpenAI's API (`https://api.openai.com/`).
        case openAI(apiKey: String)
        /// Sends requests to Anthropic's API (`https://api.anthropic.com/`).
        case anthropic(apiKey: String, baseURL: URL? = nil)
        /// Sends requests to Mistral's API (`https://api.mistral.ai/`).
        case mistral(apiKey: String)
        /// Sends requests to `localhost:1234`, the default port for LM Studio.
        case lmStudio
        /// Sends requests to `localhost` on the given port.
        case localhost(port: Int)
        /// Sends requests to any OpenAI-compatible API at the given URL (not including `/v1`).
        case other(URL, apiKey: String?)

        /// A conservative chat rate limiter that adapts after the first API response.
        ///
        /// Cloud providers start with safe defaults (50 requests, 40K tokens per minute).
        /// Local providers are effectively unlimited.
        public var chatLimiter: RateLimiter {
            if isLocal {
                return RateLimiter(maxRequests: 100_000, maxTokens: 100_000_000, interval: 60)
            }
            return RateLimiter(maxRequests: 50, maxTokens: 40000, interval: 60)
        }

        /// A conservative embedding rate limiter that adapts after the first API response.
        ///
        /// Cloud providers start with safe defaults. Local providers are effectively unlimited.
        public var embeddingLimiter: RateLimiter {
            if isLocal {
                return RateLimiter(maxRequests: 100_000, maxTokens: 100_000_000, interval: 60)
            }
            return RateLimiter(maxRequests: 50, maxTokens: 1_000_000, interval: 60)
        }

        /// Whether this provider targets a local server (no rate limiting needed).
        public var isLocal: Bool {
            switch self {
            case .lmStudio, .localhost:
                true
            default:
                false
            }
        }
    }

    /// The ``OpenAICompatibleAPI`` instance configured for this LLM's provider.
    var providerApi: OpenAICompatibleAPI {
        switch provider {
        case let .openAI(apiKey):
            OpenAICompatibleAPI.openAI(apiKey: apiKey)
        case let .anthropic(apiKey, baseURL):
            OpenAICompatibleAPI.anthropic(apiKey: apiKey, baseURL: baseURL)
        case let .mistral(apiKey):
            OpenAICompatibleAPI.mistral(apiKey: apiKey)
        case .lmStudio:
            OpenAICompatibleAPI.localhost(port: 1234)
        case let .other(url, apiKey: apiKey):
            if let apiKey {
                OpenAICompatibleAPI(
                    baseURL: url,
                    authMethod: .bearer(apiKey: apiKey),
                )
            } else {
                OpenAICompatibleAPI(baseURL: url)
            }
        case let .localhost(port):
            OpenAICompatibleAPI.localhost(port: port)
        }
    }
}
