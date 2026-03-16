//
//  MistralTests.swift
//  LLMTests
//
//  Tests for Mistral provider support
//

import Foundation
@testable import LLM
import Testing

// MARK: - Provider.isMistral Tests

@Test func provider_isMistral_trueForMistral() {
    let provider = LLM.Provider.mistral(apiKey: "test")
    #expect(provider.isMistral == true)
}

@Test func provider_isMistral_falseForOthers() throws {
    #expect(LLM.Provider.openAI(apiKey: "test").isMistral == false)
    #expect(LLM.Provider.anthropic(apiKey: "test").isMistral == false)
    #expect(LLM.Provider.lmStudio.isMistral == false)
    #expect(LLM.Provider.localhost(port: 8080).isMistral == false)
    #expect(try LLM.Provider.other(#require(URL(string: "https://example.com")), apiKey: nil).isMistral == false)
}

// MARK: - Provider.simpleProvider Tests for Mistral

@Test func provider_simpleProvider_mistral() {
    let provider = LLM.Provider.mistral(apiKey: "my-key")
    let simple = provider.simpleProvider

    if case .mistral = simple {
        // Correct
    } else {
        Issue.record("Expected .mistral simple provider")
    }
}

// MARK: - SimpleProvider.fullProvider Tests for Mistral

@Test func simpleProvider_fullProvider_mistral() {
    let simple = LLM.SimpleProvider.mistral
    let full = simple.fullProvider(using: "my-api-key")

    if case let .mistral(apiKey) = full {
        #expect(apiKey == "my-api-key")
    } else {
        Issue.record("Expected .mistral provider")
    }
}

// MARK: - Round-trip Tests for Mistral

@Test func provider_roundTrip_mistral() {
    let original = LLM.Provider.mistral(apiKey: "original-key")
    let simple = original.simpleProvider
    let restored = simple.fullProvider(using: "new-key")

    if case let .mistral(apiKey) = restored {
        #expect(apiKey == "new-key")
    } else {
        Issue.record("Expected .mistral provider")
    }
}

// MARK: - Model Mapping Tests for Mistral

@Test func provider_model_mistral_fast() {
    let provider = LLM.Provider.mistral(apiKey: "test")
    let model = provider.model(type: .fast, inference: .direct)
    #expect(model == .mistralSmall)
}

@Test func provider_model_mistral_standard() {
    let provider = LLM.Provider.mistral(apiKey: "test")
    let model = provider.model(type: .standard, inference: .direct)
    #expect(model == .mistralMedium)
}

@Test func provider_model_mistral_flagship() {
    let provider = LLM.Provider.mistral(apiKey: "test")
    let model = provider.model(type: .flagship, inference: .direct)
    #expect(model == .mistralLarge)
}

// MARK: - API Client Tests for Mistral

@Test func api_client_mistral() {
    let api = LLM.OpenAICompatibleAPI.mistral(apiKey: "test-key")
    #expect(api.baseURL.absoluteString == "https://api.mistral.ai/")
    if case let .bearer(apiKey) = api.authenticationMethod {
        #expect(apiKey == "test-key")
    } else {
        Issue.record("Expected bearer authentication")
    }
    #expect(api.chatEndpoint == "v1/chat/completions")
}

// MARK: - Chat Completion Request Encoding Tests for Mistral

@Test func chatCompletion_encoding_mistral_parallelToolCalls() throws {
    let messages = [
        LLM.OpenAICompatibleAPI.ChatMessage(content: "Hello", role: .user),
    ]

    var completion = LLM.OpenAICompatibleAPI.ChatCompletion(
        model: .mistralSmall,
        messages: messages,
        tools: [
            LLM.OpenAICompatibleAPI.ToolDefinition(
                function: LLM.OpenAICompatibleAPI.FunctionDefinition(
                    name: "get_weather",
                    description: "Get weather information",
                    parameters: .object(properties: [:], required: [])
                )
            ),
        ],
        tool_choice: .auto,
        parallel_tool_calls: true
    )
    completion.useMistralFormat = true

    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let data = try encoder.encode(completion)
    let jsonString = String(data: data, encoding: .utf8)

    #expect(jsonString?.contains("\"parallel_tool_calls\":true") == true, "Should include parallel_tool_calls parameter")
    #expect(jsonString?.contains("\"model\":\"mistral-small-latest\"") == true, "Should include Mistral model")
}

// MARK: - Conversation Configuration Tests for Mistral

@Test func conversation_configuration_mistral_parallelToolCalls() {
    var config = LLM.ConversationConfiguration()
    config.parallelToolCalls = true

    #expect(config.parallelToolCalls == true)
}
