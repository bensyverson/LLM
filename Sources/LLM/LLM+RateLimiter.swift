//
//  LLM+RateLimiter.swift
//
//  Created by Ben Syverson on 2024-06-17.
//

import Foundation

public extension LLM {
    /// Manages request and token rate limits for API calls.
    ///
    /// `RateLimiter` tracks requests and tokens within a sliding time window, sleeping
    /// when limits are exceeded. It starts with conservative defaults and adapts
    /// automatically when ``updateLimits(maxRequests:maxTokens:)`` is called with
    /// values parsed from provider response headers.
    actor RateLimiter {
        private var requestCount = -1
        private var tokenCount = 0
        private(set) var maxRequests: Int
        private(set) var maxTokens: Int
        private let interval: TimeInterval
        private let minDelay: TimeInterval
        private var lastResetDate: Date

        /// Creates a rate limiter with the given ceilings.
        ///
        /// - Parameters:
        ///   - maxRequests: Maximum number of requests allowed per interval.
        ///   - maxTokens: Maximum number of tokens allowed per interval.
        ///   - interval: The time window in seconds over which limits are enforced.
        ///   - minDelay: Minimum delay in seconds between consecutive requests.
        public init(
            maxRequests: Int = 50,
            maxTokens: Int = 40000,
            interval: TimeInterval = 60.0,
            minDelay: TimeInterval = 0.01,
        ) {
            self.maxRequests = maxRequests
            self.maxTokens = maxTokens
            self.interval = interval
            self.minDelay = minDelay
            lastResetDate = Date()
        }

        /// Updates the rate limit ceilings without resetting current counters.
        ///
        /// Call this with values parsed from provider response headers (e.g.
        /// `x-ratelimit-limit-requests`) to adapt the limiter to the user's
        /// actual API tier.
        ///
        /// - Parameters:
        ///   - maxRequests: New maximum requests per interval, if provided.
        ///   - maxTokens: New maximum tokens per interval, if provided.
        public func updateLimits(maxRequests: Int? = nil, maxTokens: Int? = nil) {
            if let maxRequests {
                self.maxRequests = maxRequests
            }
            if let maxTokens {
                self.maxTokens = maxTokens
            }
        }

        /// Acquires permission to make an API request consuming the given number of tokens.
        ///
        /// If the current window's request or token count has been exceeded, this method
        /// sleeps until the window resets, then retries. A minimum delay between requests
        /// is also enforced.
        ///
        /// - Parameter tokens: The estimated number of tokens this request will consume.
        /// - Throws: `CancellationError` if the task is cancelled while waiting.
        public func acquire(tokens: Int) async throws {
            let now = Date()
            if requestCount == -1 {
                lastResetDate = now
                requestCount = 0
                tokenCount = 0
            }
            let elapsed = now.timeIntervalSince(lastResetDate)
            if requestCount >= maxRequests || tokenCount >= maxTokens {
                if elapsed >= interval {
                    lastResetDate = now
                    requestCount = 0
                    tokenCount = 0
                } else {
                    let delay = interval - elapsed
                    try await Task.sleep(for: .seconds(delay))
                    try await acquire(tokens: tokens)
                    return
                }
            }
            requestCount += 1
            tokenCount += tokens
            let _: () = try await Task(priority: .userInitiated) {
                try await Task.sleep(for: .seconds(minDelay))
            }.value
        }
    }
}
