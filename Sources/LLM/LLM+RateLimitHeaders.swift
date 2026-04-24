//
//  LLM+RateLimitHeaders.swift
//
//  Created by Claude on 2026-02-25.
//

import Foundation
#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

public extension LLM {
    /// Rate limit information extracted from provider HTTP response headers.
    ///
    /// Both OpenAI and Anthropic return rate limit details on every API response.
    /// This struct normalizes both formats into a single representation.
    struct RateLimitInfo: Friendly {
        /// Maximum requests allowed in the current window.
        public var requestLimit: Int?
        /// Maximum tokens allowed in the current window.
        public var tokenLimit: Int?
        /// Requests remaining in the current window.
        public var requestsRemaining: Int?
        /// Tokens remaining in the current window.
        public var tokensRemaining: Int?

        /// Parses rate limit information from an HTTP response's headers.
        ///
        /// Handles both header formats:
        /// - **Anthropic:** `anthropic-ratelimit-requests-limit`, `anthropic-ratelimit-tokens-limit`, etc.
        /// - **OpenAI:** `x-ratelimit-limit-requests`, `x-ratelimit-limit-tokens`, etc.
        ///
        /// - Parameters:
        ///   - response: The HTTP response containing rate limit headers.
        ///   - provider: The provider that sent the response, used to select the correct header format.
        /// - Returns: Parsed rate limit info, or `nil` if no recognized headers are present.
        public static func parse(
            from response: HTTPURLResponse,
            provider: Provider,
        ) -> RateLimitInfo? {
            func intHeader(_ key: String) -> Int? {
                guard let value = response.value(forHTTPHeaderField: key) else { return nil }
                return Int(value)
            }

            if provider.isAnthropic {
                let reqLimit = intHeader("anthropic-ratelimit-requests-limit")
                let tokLimit = intHeader("anthropic-ratelimit-tokens-limit")
                let reqRemaining = intHeader("anthropic-ratelimit-requests-remaining")
                let tokRemaining = intHeader("anthropic-ratelimit-tokens-remaining")

                guard reqLimit != nil || tokLimit != nil else { return nil }

                return RateLimitInfo(
                    requestLimit: reqLimit,
                    tokenLimit: tokLimit,
                    requestsRemaining: reqRemaining,
                    tokensRemaining: tokRemaining,
                )
            } else {
                let reqLimit = intHeader("x-ratelimit-limit-requests")
                let tokLimit = intHeader("x-ratelimit-limit-tokens")
                let reqRemaining = intHeader("x-ratelimit-remaining-requests")
                let tokRemaining = intHeader("x-ratelimit-remaining-tokens")

                guard reqLimit != nil || tokLimit != nil else { return nil }

                return RateLimitInfo(
                    requestLimit: reqLimit,
                    tokenLimit: tokLimit,
                    requestsRemaining: reqRemaining,
                    tokensRemaining: tokRemaining,
                )
            }
        }

        /// Parses rate limit information using a header lookup closure.
        ///
        /// This overload accepts a closure instead of an `HTTPURLResponse`,
        /// enabling use with HTTP client libraries that expose headers differently
        /// (e.g. AsyncHTTPClient on Linux).
        ///
        /// - Parameters:
        ///   - headerLookup: A closure that returns the value for a given header name, or `nil`.
        ///   - provider: The provider that sent the response, used to select the correct header format.
        /// - Returns: Parsed rate limit info, or `nil` if no recognized headers are present.
        public static func parse(
            headerLookup: (String) -> String?,
            provider: Provider,
        ) -> RateLimitInfo? {
            func intHeader(_ key: String) -> Int? {
                guard let value = headerLookup(key) else { return nil }
                return Int(value)
            }

            if provider.isAnthropic {
                let reqLimit = intHeader("anthropic-ratelimit-requests-limit")
                let tokLimit = intHeader("anthropic-ratelimit-tokens-limit")
                let reqRemaining = intHeader("anthropic-ratelimit-requests-remaining")
                let tokRemaining = intHeader("anthropic-ratelimit-tokens-remaining")

                guard reqLimit != nil || tokLimit != nil else { return nil }

                return RateLimitInfo(
                    requestLimit: reqLimit,
                    tokenLimit: tokLimit,
                    requestsRemaining: reqRemaining,
                    tokensRemaining: tokRemaining,
                )
            } else {
                let reqLimit = intHeader("x-ratelimit-limit-requests")
                let tokLimit = intHeader("x-ratelimit-limit-tokens")
                let reqRemaining = intHeader("x-ratelimit-remaining-requests")
                let tokRemaining = intHeader("x-ratelimit-remaining-tokens")

                guard reqLimit != nil || tokLimit != nil else { return nil }

                return RateLimitInfo(
                    requestLimit: reqLimit,
                    tokenLimit: tokLimit,
                    requestsRemaining: reqRemaining,
                    tokensRemaining: tokRemaining,
                )
            }
        }
    }
}
