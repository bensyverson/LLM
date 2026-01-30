//
//  LLM+RateLimiter.swift
//
//  Created by Ben Syverson on 2024-06-17.
//

import Foundation

public extension LLM {
	actor RateLimiter {
		private var requestCount = -1
		private var tokenCount = 0
		private let maxRequests: Int
		private let maxTokens: Int
		private let interval: TimeInterval
		private let minDelay: TimeInterval
		private var lastResetDate: Date

		public init(
			maxRequests: Int = 2,
			maxTokens: Int = 10000,
			interval: TimeInterval = 1.0,
			minDelay: TimeInterval = 0.01
		) {
			self.maxRequests = maxRequests
			self.maxTokens = maxTokens
			self.interval = interval
			self.minDelay = minDelay
			self.lastResetDate = Date()
		}

		public func acquire(tokens: Int) async throws {
			let now = Date()
			if requestCount == -1 {
#if DEBUG_RATE
				print("🥇 First run. Resetting Date and request count.")
#endif
				lastResetDate = now
				requestCount = 0
				tokenCount = 0
			}
			let elapsed = now.timeIntervalSince(lastResetDate)
			if requestCount >= maxRequests || tokenCount >= maxTokens {
				if elapsed >= interval {
#if DEBUG_RATE
					print("⏰ \(elapsed)s elapsed; resetting Date and request count.")
#endif
					lastResetDate = now
					requestCount = 0
					tokenCount = 0
				} else {
#if DEBUG_RATE
					print("🛑 Rate limited (\(requestCount) vs \(maxRequests), \(elapsed)s elapsed) at \(Date().description)")
#endif
					let delay = interval - elapsed
					try await Task.sleep(for: .seconds(delay))
					try await acquire(tokens: tokens)
					return
				}
			}
#if DEBUG_RATE
			print("✅ Passing request (\(requestCount) vs \(maxRequests), \(elapsed)s elapsed) at \(Date().description)")
#endif
			requestCount += 1
			tokenCount += tokens
			let _: () = try await Task(priority: .userInitiated) {
				try await Task.sleep(for: .seconds(minDelay))
			}.value
//			try await Task.sleep(for: .seconds(minDelay))
		}
	}
}
