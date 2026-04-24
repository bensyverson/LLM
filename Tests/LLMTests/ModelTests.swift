//
//  ModelTests.swift
//  LLMTests
//
//  Tests for ModelName, ModelType, InferenceType, and Provider.model()
//

import Foundation
@testable import LLM
import Testing

// MARK: - ModelName Tests

@Test func `model name is GPT 5 returns true for GPT 5 models`() {
    #expect(LLM.OpenAICompatibleAPI.ModelName.gpt52.isGPT5 == true)
    #expect(LLM.OpenAICompatibleAPI.ModelName.gpt5Mini.isGPT5 == true)
    #expect(LLM.OpenAICompatibleAPI.ModelName.gpt5Nano.isGPT5 == true)
}

@Test func `model name is GPT 5 returns false for non GPT 5 models`() {
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

@Test func `model name codable round trip`() throws {
    let models: [LLM.OpenAICompatibleAPI.ModelName] = [
        .gpt52, .gpt5Mini, .gpt4o, .claude45Opus, .o3, .placeholder,
    ]

    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    for model in models {
        let data = try encoder.encode(model)
        let decoded = try decoder.decode(LLM.OpenAICompatibleAPI.ModelName.self, from: data)
        #expect(decoded == model)
    }
}

@Test func `model name raw values match API strings`() {
    #expect(LLM.OpenAICompatibleAPI.ModelName.gpt52.rawValue == "gpt-5.2")
    #expect(LLM.OpenAICompatibleAPI.ModelName.gpt5Mini.rawValue == "gpt-5-mini")
    #expect(LLM.OpenAICompatibleAPI.ModelName.gpt5Nano.rawValue == "gpt-5-nano")
    #expect(LLM.OpenAICompatibleAPI.ModelName.claude45Opus.rawValue == "claude-opus-4-5")
    #expect(LLM.OpenAICompatibleAPI.ModelName.claude45Sonnet.rawValue == "claude-sonnet-4-5")
    #expect(LLM.OpenAICompatibleAPI.ModelName.claude45Haiku.rawValue == "claude-haiku-4-5")
}

// MARK: - ModelType Tests

@Test func `model type raw values`() {
    #expect(LLM.ModelType.fast.rawValue == "fast")
    #expect(LLM.ModelType.flagship.rawValue == "flagship")
}

// MARK: - InferenceType Tests

@Test func `inference type raw values`() {
    #expect(LLM.InferenceType.direct.rawValue == "direct")
    #expect(LLM.InferenceType.reasoning.rawValue == "reasoning")
}

// MARK: - Provider.model(type:inference:) Tests

@Test func `provider model open AI fast direct`() {
    let provider = LLM.Provider.openAI(apiKey: "test-key")
    let model = provider.model(type: .fast, inference: .direct)
    #expect(model == .gpt54Nano)
}

@Test func `provider model open AI flagship direct`() {
    let provider = LLM.Provider.openAI(apiKey: "test-key")
    let model = provider.model(type: .flagship, inference: .direct)
    #expect(model == .gpt55)
}

@Test func `provider model open AI fast reasoning`() {
    let provider = LLM.Provider.openAI(apiKey: "test-key")
    let model = provider.model(type: .fast, inference: .reasoning)
    #expect(model == .gpt54Nano)
}

@Test func `provider model open AI flagship reasoning`() {
    let provider = LLM.Provider.openAI(apiKey: "test-key")
    let model = provider.model(type: .flagship, inference: .reasoning)
    #expect(model == .gpt55)
}

@Test func `provider model anthropic fast direct`() {
    let provider = LLM.Provider.anthropic(apiKey: "test-key")
    let model = provider.model(type: .fast, inference: .direct)
    #expect(model == .claude45Haiku)
}

@Test func `provider model anthropic fast reasoning`() {
    let provider = LLM.Provider.anthropic(apiKey: "test-key")
    let model = provider.model(type: .fast, inference: .reasoning)
    #expect(model == .claude45Haiku)
}

@Test func `provider model anthropic flagship direct`() {
    let provider = LLM.Provider.anthropic(apiKey: "test-key")
    let model = provider.model(type: .flagship, inference: .direct)
    #expect(model == .claude47Opus)
}

@Test func `provider model anthropic flagship reasoning`() {
    let provider = LLM.Provider.anthropic(apiKey: "test-key")
    let model = provider.model(type: .flagship, inference: .reasoning)
    #expect(model == .claude47Opus)
}

@Test func `provider model lm studio returns placeholder`() {
    let provider = LLM.Provider.lmStudio
    #expect(provider.model(type: .fast, inference: .direct) == .placeholder)
    #expect(provider.model(type: .flagship, inference: .direct) == .placeholder)
    #expect(provider.model(type: .fast, inference: .reasoning) == .placeholder)
    #expect(provider.model(type: .flagship, inference: .reasoning) == .placeholder)
}

@Test func `provider model localhost returns placeholder`() {
    let provider = LLM.Provider.localhost(port: 8080)
    #expect(provider.model(type: .fast, inference: .direct) == .placeholder)
    #expect(provider.model(type: .flagship, inference: .reasoning) == .placeholder)
}

@Test func `provider model other returns placeholder`() throws {
    let provider = try LLM.Provider.other(#require(URL(string: "https://example.com")), apiKey: "key")
    #expect(provider.model(type: .fast, inference: .direct) == .placeholder)
    #expect(provider.model(type: .flagship, inference: .reasoning) == .placeholder)
}
