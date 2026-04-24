//
//  ProviderTests.swift
//  LLMTests
//
//  Tests for Provider, SimpleProvider
//

import Foundation
@testable import LLM
import Testing

// MARK: - Provider.isOpenAI Tests

@Test func `provider is open AI true for open AI`() {
    let provider = LLM.Provider.openAI(apiKey: "test")
    #expect(provider.isOpenAI == true)
}

@Test func `provider is open AI false for others`() throws {
    #expect(LLM.Provider.anthropic(apiKey: "test").isOpenAI == false)
    #expect(LLM.Provider.lmStudio.isOpenAI == false)
    #expect(LLM.Provider.localhost(port: 8080).isOpenAI == false)
    #expect(try LLM.Provider.other(#require(URL(string: "https://example.com")), apiKey: nil).isOpenAI == false)
}

// MARK: - Provider.isAnthropic Tests

@Test func `provider is anthropic true for anthropic`() {
    let provider = LLM.Provider.anthropic(apiKey: "test")
    #expect(provider.isAnthropic == true)
}

@Test func `provider is anthropic false for others`() throws {
    #expect(LLM.Provider.openAI(apiKey: "test").isAnthropic == false)
    #expect(LLM.Provider.lmStudio.isAnthropic == false)
    #expect(LLM.Provider.localhost(port: 8080).isAnthropic == false)
    #expect(try LLM.Provider.other(#require(URL(string: "https://example.com")), apiKey: nil).isAnthropic == false)
}

// MARK: - Provider.simpleProvider Tests

@Test func `provider simple provider open AI`() {
    let provider = LLM.Provider.openAI(apiKey: "my-key")
    let simple = provider.simpleProvider

    if case .openAI = simple {
        // Correct
    } else {
        Issue.record("Expected .openAI simple provider")
    }
}

@Test func `provider simple provider anthropic`() {
    let provider = LLM.Provider.anthropic(apiKey: "my-key")
    let simple = provider.simpleProvider

    if case .anthropic = simple {
        // Correct
    } else {
        Issue.record("Expected .anthropic simple provider")
    }
}

@Test func `provider simple provider lm studio`() {
    let provider = LLM.Provider.lmStudio
    let simple = provider.simpleProvider

    if case .lmStudio = simple {
        // Correct
    } else {
        Issue.record("Expected .lmStudio simple provider")
    }
}

@Test func `provider simple provider localhost`() {
    let provider = LLM.Provider.localhost(port: 9000)
    let simple = provider.simpleProvider

    if case let .localhost(port) = simple {
        #expect(port == 9000)
    } else {
        Issue.record("Expected .localhost simple provider")
    }
}

@Test func `provider simple provider other`() throws {
    let url = try #require(URL(string: "https://my-api.example.com"))
    let provider = LLM.Provider.other(url, apiKey: "key")
    let simple = provider.simpleProvider

    if case let .other(simpleUrl) = simple {
        #expect(simpleUrl == url)
    } else {
        Issue.record("Expected .other simple provider")
    }
}

// MARK: - SimpleProvider.fullProvider Tests

@Test func `simple provider full provider open AI`() {
    let simple = LLM.SimpleProvider.openAI
    let full = simple.fullProvider(using: "my-api-key")

    if case let .openAI(apiKey) = full {
        #expect(apiKey == "my-api-key")
    } else {
        Issue.record("Expected .openAI provider")
    }
}

@Test func `simple provider full provider anthropic`() {
    let simple = LLM.SimpleProvider.anthropic(baseURL: nil)
    let full = simple.fullProvider(using: "my-api-key")

    if case let .anthropic(apiKey, _) = full {
        #expect(apiKey == "my-api-key")
    } else {
        Issue.record("Expected .anthropic provider")
    }
}

@Test func `simple provider full provider lm studio`() {
    let simple = LLM.SimpleProvider.lmStudio
    let full = simple.fullProvider(using: "ignored-key")

    if case .lmStudio = full {
        // Correct - lmStudio doesn't need an API key
    } else {
        Issue.record("Expected .lmStudio provider")
    }
}

@Test func `simple provider full provider localhost`() {
    let simple = LLM.SimpleProvider.localhost(port: 5000)
    let full = simple.fullProvider(using: "ignored-key")

    if case let .localhost(port) = full {
        #expect(port == 5000)
    } else {
        Issue.record("Expected .localhost provider")
    }
}

@Test func `simple provider full provider other`() throws {
    let url = try #require(URL(string: "https://custom.api.com"))
    let simple = LLM.SimpleProvider.other(url)
    let full = simple.fullProvider(using: "my-key")

    if case let .other(fullUrl, apiKey) = full {
        #expect(fullUrl == url)
        #expect(apiKey == "my-key")
    } else {
        Issue.record("Expected .other provider")
    }
}

// MARK: - Round-trip Tests (Provider -> SimpleProvider -> Provider)

@Test func `provider round trip open AI`() {
    let original = LLM.Provider.openAI(apiKey: "original-key")
    let simple = original.simpleProvider
    let restored = simple.fullProvider(using: "new-key")

    if case let .openAI(apiKey) = restored {
        #expect(apiKey == "new-key")
    } else {
        Issue.record("Expected .openAI provider")
    }
}

@Test func `provider round trip anthropic`() {
    let original = LLM.Provider.anthropic(apiKey: "original-key")
    let simple = original.simpleProvider
    let restored = simple.fullProvider(using: "new-key")

    if case let .anthropic(apiKey, _) = restored {
        #expect(apiKey == "new-key")
    } else {
        Issue.record("Expected .anthropic provider")
    }
}

@Test func `provider round trip localhost`() {
    let original = LLM.Provider.localhost(port: 7777)
    let simple = original.simpleProvider
    let restored = simple.fullProvider(using: "any-key")

    if case let .localhost(port) = restored {
        #expect(port == 7777)
    } else {
        Issue.record("Expected .localhost provider with correct port")
    }
}
