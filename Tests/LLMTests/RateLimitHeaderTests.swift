//
//  RateLimitHeaderTests.swift
//  LLMTests
//
//  Tests for RateLimitInfo parsing and RateLimiter.updateLimits
//

import Foundation
@testable import LLM
import Testing

// MARK: - RateLimitInfo Parsing: Anthropic Headers

@Test func rateLimitInfo_parsesAnthropicHeaders() throws {
    let url = try #require(URL(string: "https://api.anthropic.com/v1/messages"))
    let response = try #require(HTTPURLResponse(
        url: url,
        statusCode: 200,
        httpVersion: "HTTP/1.1",
        headerFields: [
            "anthropic-ratelimit-requests-limit": "1000",
            "anthropic-ratelimit-tokens-limit": "80000",
            "anthropic-ratelimit-requests-remaining": "999",
            "anthropic-ratelimit-tokens-remaining": "79500",
        ]
    ))

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let info = try #require(LLM.RateLimitInfo.parse(from: response, provider: provider))

    #expect(info.requestLimit == 1000)
    #expect(info.tokenLimit == 80000)
    #expect(info.requestsRemaining == 999)
    #expect(info.tokensRemaining == 79500)
}

@Test func rateLimitInfo_parsesAnthropicPartialHeaders() throws {
    let url = try #require(URL(string: "https://api.anthropic.com/v1/messages"))
    let response = try #require(HTTPURLResponse(
        url: url,
        statusCode: 200,
        httpVersion: "HTTP/1.1",
        headerFields: [
            "anthropic-ratelimit-requests-limit": "500",
        ]
    ))

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let info = try #require(LLM.RateLimitInfo.parse(from: response, provider: provider))

    #expect(info.requestLimit == 500)
    #expect(info.tokenLimit == nil)
    #expect(info.requestsRemaining == nil)
    #expect(info.tokensRemaining == nil)
}

@Test func rateLimitInfo_returnsNilForAnthropicWithNoHeaders() throws {
    let url = try #require(URL(string: "https://api.anthropic.com/v1/messages"))
    let response = try #require(HTTPURLResponse(
        url: url,
        statusCode: 200,
        httpVersion: "HTTP/1.1",
        headerFields: [:]
    ))

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let info = LLM.RateLimitInfo.parse(from: response, provider: provider)

    #expect(info == nil)
}

// MARK: - RateLimitInfo Parsing: OpenAI Headers

@Test func rateLimitInfo_parsesOpenAIHeaders() throws {
    let url = try #require(URL(string: "https://api.openai.com/v1/chat/completions"))
    let response = try #require(HTTPURLResponse(
        url: url,
        statusCode: 200,
        httpVersion: "HTTP/1.1",
        headerFields: [
            "x-ratelimit-limit-requests": "5000",
            "x-ratelimit-limit-tokens": "600000",
            "x-ratelimit-remaining-requests": "4999",
            "x-ratelimit-remaining-tokens": "599000",
        ]
    ))

    let provider = LLM.Provider.openAI(apiKey: "test")
    let info = try #require(LLM.RateLimitInfo.parse(from: response, provider: provider))

    #expect(info.requestLimit == 5000)
    #expect(info.tokenLimit == 600_000)
    #expect(info.requestsRemaining == 4999)
    #expect(info.tokensRemaining == 599_000)
}

@Test func rateLimitInfo_parsesOpenAIPartialHeaders() throws {
    let url = try #require(URL(string: "https://api.openai.com/v1/chat/completions"))
    let response = try #require(HTTPURLResponse(
        url: url,
        statusCode: 200,
        httpVersion: "HTTP/1.1",
        headerFields: [
            "x-ratelimit-limit-tokens": "300000",
        ]
    ))

    let provider = LLM.Provider.openAI(apiKey: "test")
    let info = try #require(LLM.RateLimitInfo.parse(from: response, provider: provider))

    #expect(info.requestLimit == nil)
    #expect(info.tokenLimit == 300_000)
}

@Test func rateLimitInfo_returnsNilForOpenAIWithNoHeaders() throws {
    let url = try #require(URL(string: "https://api.openai.com/v1/chat/completions"))
    let response = try #require(HTTPURLResponse(
        url: url,
        statusCode: 200,
        httpVersion: "HTTP/1.1",
        headerFields: [:]
    ))

    let provider = LLM.Provider.openAI(apiKey: "test")
    let info = LLM.RateLimitInfo.parse(from: response, provider: provider)

    #expect(info == nil)
}

// MARK: - RateLimitInfo Parsing: Non-numeric values

@Test func rateLimitInfo_ignoresNonNumericValues() throws {
    let url = try #require(URL(string: "https://api.openai.com/v1/chat/completions"))
    let response = try #require(HTTPURLResponse(
        url: url,
        statusCode: 200,
        httpVersion: "HTTP/1.1",
        headerFields: [
            "x-ratelimit-limit-requests": "not-a-number",
            "x-ratelimit-limit-tokens": "500000",
        ]
    ))

    let provider = LLM.Provider.openAI(apiKey: "test")
    let info = try #require(LLM.RateLimitInfo.parse(from: response, provider: provider))

    #expect(info.requestLimit == nil)
    #expect(info.tokenLimit == 500_000)
}

// MARK: - RateLimitInfo Codable Conformance

@Test func rateLimitInfo_roundTripsJSON() throws {
    let original = LLM.RateLimitInfo(
        requestLimit: 100,
        tokenLimit: 50000,
        requestsRemaining: 99,
        tokensRemaining: 49000
    )

    let data = try JSONEncoder().encode(original)
    let decoded = try JSONDecoder().decode(LLM.RateLimitInfo.self, from: data)

    #expect(decoded == original)
}

// MARK: - RateLimiter.updateLimits

@Test func rateLimiter_updateLimits_updatesMaxRequests() async {
    let limiter = LLM.RateLimiter(maxRequests: 10, maxTokens: 1000)

    await limiter.updateLimits(maxRequests: 500)

    let maxRequests = await limiter.maxRequests
    let maxTokens = await limiter.maxTokens
    #expect(maxRequests == 500)
    #expect(maxTokens == 1000) // unchanged
}

@Test func rateLimiter_updateLimits_updatesMaxTokens() async {
    let limiter = LLM.RateLimiter(maxRequests: 10, maxTokens: 1000)

    await limiter.updateLimits(maxTokens: 200_000)

    let maxRequests = await limiter.maxRequests
    let maxTokens = await limiter.maxTokens
    #expect(maxRequests == 10) // unchanged
    #expect(maxTokens == 200_000)
}

@Test func rateLimiter_updateLimits_updatesBoth() async {
    let limiter = LLM.RateLimiter(maxRequests: 10, maxTokens: 1000)

    await limiter.updateLimits(maxRequests: 8000, maxTokens: 1_500_000)

    let maxRequests = await limiter.maxRequests
    let maxTokens = await limiter.maxTokens
    #expect(maxRequests == 8000)
    #expect(maxTokens == 1_500_000)
}

@Test func rateLimiter_updateLimits_nilDoesNotChange() async {
    let limiter = LLM.RateLimiter(maxRequests: 42, maxTokens: 9999)

    await limiter.updateLimits()

    let maxRequests = await limiter.maxRequests
    let maxTokens = await limiter.maxTokens
    #expect(maxRequests == 42)
    #expect(maxTokens == 9999)
}

// MARK: - RateLimiter Default Values

@Test func rateLimiter_defaultsAreConservative() async {
    let limiter = LLM.RateLimiter()

    let maxRequests = await limiter.maxRequests
    let maxTokens = await limiter.maxTokens
    #expect(maxRequests == 50)
    #expect(maxTokens == 40000)
}

// MARK: - Provider Rate Limiter Defaults

@Test func provider_cloudLimiters_useConservativeDefaults() async {
    let openAI = LLM.Provider.openAI(apiKey: "test")
    let anthropic = LLM.Provider.anthropic(apiKey: "test")

    // Both cloud providers should use the same conservative defaults
    let openAIReqs = await openAI.chatLimiter.maxRequests
    let anthropicReqs = await anthropic.chatLimiter.maxRequests

    #expect(openAIReqs == 50)
    #expect(anthropicReqs == 50)
}

@Test func provider_localLimiters_arePermissive() async {
    let lmStudio = LLM.Provider.lmStudio
    let localhost = LLM.Provider.localhost(port: 8080)

    let lmReqs = await lmStudio.chatLimiter.maxRequests
    let localReqs = await localhost.chatLimiter.maxRequests

    #expect(lmReqs == 100_000)
    #expect(localReqs == 100_000)
}

@Test func provider_isLocal() {
    #expect(LLM.Provider.lmStudio.isLocal == true)
    #expect(LLM.Provider.localhost(port: 8080).isLocal == true)
    #expect(LLM.Provider.openAI(apiKey: "test").isLocal == false)
    #expect(LLM.Provider.anthropic(apiKey: "test").isLocal == false)
}
