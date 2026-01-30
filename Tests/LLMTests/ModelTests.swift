//
//  ModelTests.swift
//  LLMTests
//
//  Tests for ModelName, ModelType, InferenceType, and Provider.model()
//

import Testing
import Foundation
@testable import LLM

// MARK: - ModelName Tests

@Test func modelNameIsGPT5_returnsTrue_forGPT5Models() {
    #expect(LLM.OpenAICompatibleAPI.ModelName.gpt52.isGPT5 == true)
    #expect(LLM.OpenAICompatibleAPI.ModelName.gpt5Mini.isGPT5 == true)
    #expect(LLM.OpenAICompatibleAPI.ModelName.gpt5Nano.isGPT5 == true)
}

@Test func modelNameIsGPT5_returnsFalse_forNonGPT5Models() {
    // OpenAI non-GPT5 models
    #expect(LLM.OpenAICompatibleAPI.ModelName.gpt4o.isGPT5 == false)
    #expect(LLM.OpenAICompatibleAPI.ModelName.gpt4oMini.isGPT5 == false)
    #expect(LLM.OpenAICompatibleAPI.ModelName.gpt41.isGPT5 == false)
    #expect(LLM.OpenAICompatibleAPI.ModelName.o3.isGPT5 == false)
    #expect(LLM.OpenAICompatibleAPI.ModelName.o1.isGPT5 == false)

    // Anthropic models
    #expect(LLM.OpenAICompatibleAPI.ModelName.claude45Opus.isGPT5 == false)
    #expect(LLM.OpenAICompatibleAPI.ModelName.claude45Sonnet.isGPT5 == false)
    #expect(LLM.OpenAICompatibleAPI.ModelName.claude45Haiku.isGPT5 == false)

    // Placeholder
    #expect(LLM.OpenAICompatibleAPI.ModelName.placeholder.isGPT5 == false)
}

@Test func modelNameCodable_roundTrip() throws {
    let models: [LLM.OpenAICompatibleAPI.ModelName] = [
        .gpt52, .gpt5Mini, .gpt4o, .claude45Opus, .o3, .placeholder
    ]

    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    for model in models {
        let data = try encoder.encode(model)
        let decoded = try decoder.decode(LLM.OpenAICompatibleAPI.ModelName.self, from: data)
        #expect(decoded == model)
    }
}

@Test func modelNameRawValues_matchAPIStrings() {
    #expect(LLM.OpenAICompatibleAPI.ModelName.gpt52.rawValue == "gpt-5.2")
    #expect(LLM.OpenAICompatibleAPI.ModelName.gpt5Mini.rawValue == "gpt-5-mini")
    #expect(LLM.OpenAICompatibleAPI.ModelName.gpt5Nano.rawValue == "gpt-5-nano")
    #expect(LLM.OpenAICompatibleAPI.ModelName.claude45Opus.rawValue == "claude-opus-4-5")
    #expect(LLM.OpenAICompatibleAPI.ModelName.claude45Sonnet.rawValue == "claude-sonnet-4-5")
    #expect(LLM.OpenAICompatibleAPI.ModelName.claude45Haiku.rawValue == "claude-haiku-4-5")
}

// MARK: - ModelType Tests

@Test func modelTypeRawValues() {
    #expect(LLM.ModelType.fast.rawValue == "fast")
    #expect(LLM.ModelType.flagship.rawValue == "flagship")
}

// MARK: - InferenceType Tests

@Test func inferenceTypeRawValues() {
    #expect(LLM.InferenceType.direct.rawValue == "direct")
    #expect(LLM.InferenceType.reasoning.rawValue == "reasoning")
}

// MARK: - Provider.model(type:inference:) Tests

@Test func providerModel_openAI_fast_direct() {
    let provider = LLM.Provider.openAI(apiKey: "test-key")
    let model = provider.model(type: .fast, inference: .direct)
    #expect(model == .gpt5Mini)
}

@Test func providerModel_openAI_flagship_direct() {
    let provider = LLM.Provider.openAI(apiKey: "test-key")
    let model = provider.model(type: .flagship, inference: .direct)
    #expect(model == .gpt52)
}

@Test func providerModel_openAI_fast_reasoning() {
    let provider = LLM.Provider.openAI(apiKey: "test-key")
    let model = provider.model(type: .fast, inference: .reasoning)
    #expect(model == .gpt5Mini)
}

@Test func providerModel_openAI_flagship_reasoning() {
    let provider = LLM.Provider.openAI(apiKey: "test-key")
    let model = provider.model(type: .flagship, inference: .reasoning)
    #expect(model == .gpt52)
}

@Test func providerModel_anthropic_fast_direct() {
    let provider = LLM.Provider.anthropic(apiKey: "test-key")
    let model = provider.model(type: .fast, inference: .direct)
    #expect(model == .claude45Haiku)
}

@Test func providerModel_anthropic_fast_reasoning() {
    let provider = LLM.Provider.anthropic(apiKey: "test-key")
    let model = provider.model(type: .fast, inference: .reasoning)
    #expect(model == .claude45Haiku)
}

@Test func providerModel_anthropic_flagship_direct() {
    let provider = LLM.Provider.anthropic(apiKey: "test-key")
    let model = provider.model(type: .flagship, inference: .direct)
    #expect(model == .claude45Opus)
}

@Test func providerModel_anthropic_flagship_reasoning() {
    let provider = LLM.Provider.anthropic(apiKey: "test-key")
    let model = provider.model(type: .flagship, inference: .reasoning)
    #expect(model == .claude45Opus)
}

@Test func providerModel_lmStudio_returnsPlaceholder() {
    let provider = LLM.Provider.lmStudio
    #expect(provider.model(type: .fast, inference: .direct) == .placeholder)
    #expect(provider.model(type: .flagship, inference: .direct) == .placeholder)
    #expect(provider.model(type: .fast, inference: .reasoning) == .placeholder)
    #expect(provider.model(type: .flagship, inference: .reasoning) == .placeholder)
}

@Test func providerModel_localhost_returnsPlaceholder() {
    let provider = LLM.Provider.localhost(port: 8080)
    #expect(provider.model(type: .fast, inference: .direct) == .placeholder)
    #expect(provider.model(type: .flagship, inference: .reasoning) == .placeholder)
}

@Test func providerModel_other_returnsPlaceholder() {
    let provider = LLM.Provider.other(URL(string: "https://example.com")!, apiKey: "key")
    #expect(provider.model(type: .fast, inference: .direct) == .placeholder)
    #expect(provider.model(type: .flagship, inference: .reasoning) == .placeholder)
}
