//
//  ChatConfigurationTests.swift
//  LLMTests
//
//  Tests for ChatConfiguration.request(for:)
//

import Foundation
@testable import LLM
import Testing

// MARK: - ChatConfiguration Construction Tests

@Test func chatConfiguration_construction() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "You are helpful",
        user: "Hello!",
        modelType: .fast,
        inference: .direct
    )

    #expect(config.systemPrompt == "You are helpful")
    #expect(config.user == "Hello!")
    #expect(config.modelType == .fast)
    #expect(config.inference == .direct)
}

@Test func chatConfiguration_optionalParameters() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "System",
        user: "User",
        modelType: .flagship,
        inference: .reasoning,
        temperature: 0.7,
        frequencyPenalty: 0.5,
        repeatPenalty: 0.3,
        topP: 0.9,
        maxTokens: 1000,
        maxReasoningTokens: 500,
        reasoningEffort: .high,
        stopTokens: ["###", "END"]
    )

    #expect(config.temperature == 0.7)
    #expect(config.frequencyPenalty == 0.5)
    #expect(config.repeatPenalty == 0.3)
    #expect(config.topP == 0.9)
    #expect(config.maxTokens == 1000)
    #expect(config.maxReasoningTokens == 500)
    #expect(config.reasoningEffort == .high)
    #expect(config.stopTokens == ["###", "END"])
}

// MARK: - ChatConfiguration.request(for:) - OpenAI Tests

@Test func chatConfigRequest_openAI_systemAsMessage() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "You are helpful",
        user: "Hello!",
        modelType: .fast,
        inference: .direct
    )

    let provider = LLM.Provider.openAI(apiKey: "test")
    let request = config.request(for: provider)

    // OpenAI: system in messages array, not in system field
    #expect(request.system == nil)
    #expect(request.messages.count == 2)
    #expect(request.messages[0].role == .system)
    #expect(request.messages[0].textContent == "You are helpful")
    #expect(request.messages[1].role == .user)
    #expect(request.messages[1].textContent == "Hello!")
}

@Test func chatConfigRequest_openAI_usesMaxCompletionTokens() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "System",
        user: "User",
        modelType: .fast,
        inference: .direct,
        maxTokens: 500
    )

    let provider = LLM.Provider.openAI(apiKey: "test")
    let request = config.request(for: provider)

    #expect(request.max_tokens == nil)
    #expect(request.max_completion_tokens == 500)
}

@Test func chatConfigRequest_openAI_usesStop() {
    // GPT-5 models skip the stop parameter, so stop tokens are nil for OpenAI GPT-5
    let config = LLM.ChatConfiguration(
        systemPrompt: "System",
        user: "User",
        modelType: .fast,
        inference: .direct,
        stopTokens: ["END"]
    )

    let provider = LLM.Provider.openAI(apiKey: "test")
    let request = config.request(for: provider)

    #expect(request.stop == nil)
    #expect(request.stop_sequences == nil)
}

@Test func chatConfigRequest_openAI_reasoning_autoSetsReasoningEffort() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "System",
        user: "Think hard",
        modelType: .flagship,
        inference: .reasoning
    )

    let provider = LLM.Provider.openAI(apiKey: "test")
    let request = config.request(for: provider)

    // GPT-5.2 with reasoning should auto-set reasoning_effort to .high
    #expect(request.reasoning_effort == .high)
}

@Test func chatConfigRequest_openAI_reasoning_respectsExplicitReasoningEffort() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "System",
        user: "Think",
        modelType: .flagship,
        inference: .reasoning,
        reasoningEffort: .medium
    )

    let provider = LLM.Provider.openAI(apiKey: "test")
    let request = config.request(for: provider)

    // Should use explicitly set value
    #expect(request.reasoning_effort == .medium)
}

@Test func chatConfigRequest_openAI_reasoning_skipsTemperatureAndTopP() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "System",
        user: "User",
        modelType: .fast,
        inference: .reasoning,
        temperature: 0.5,
        topP: 0.9
    )

    let provider = LLM.Provider.openAI(apiKey: "test")
    let request = config.request(for: provider)

    #expect(request.temperature == nil)
    #expect(request.top_p == nil)
}

@Test func chatConfigRequest_openAI_direct_includesTemperatureAndTopP() {
    // GPT-5 models skip temperature and topP — they only support default values
    let config = LLM.ChatConfiguration(
        systemPrompt: "System",
        user: "User",
        modelType: .fast,
        inference: .direct,
        temperature: 0.5,
        topP: 0.9
    )

    let provider = LLM.Provider.openAI(apiKey: "test")
    let request = config.request(for: provider)

    #expect(request.temperature == nil)
    #expect(request.top_p == nil)
}

@Test func chatConfigRequest_openAI_includesFrequencyPenalty() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "System",
        user: "User",
        modelType: .fast,
        inference: .direct,
        frequencyPenalty: 0.3
    )

    let provider = LLM.Provider.openAI(apiKey: "test")
    let request = config.request(for: provider)

    #expect(request.frequency_penalty == 0.3)
}

@Test func chatConfigRequest_openAI_noThinking() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "System",
        user: "User",
        modelType: .fast,
        inference: .reasoning
    )

    let provider = LLM.Provider.openAI(apiKey: "test")
    let request = config.request(for: provider)

    // OpenAI doesn't use thinking parameter
    #expect(request.thinking == nil)
}

@Test func chatConfigRequest_openAI_selectsCorrectModel() {
    let fastConfig = LLM.ChatConfiguration(
        systemPrompt: "S", user: "U", modelType: .fast, inference: .direct
    )
    let flagshipConfig = LLM.ChatConfiguration(
        systemPrompt: "S", user: "U", modelType: .flagship, inference: .direct
    )

    let provider = LLM.Provider.openAI(apiKey: "test")

    #expect(fastConfig.request(for: provider).model == .gpt54Nano)
    #expect(flagshipConfig.request(for: provider).model == .gpt54)
}

// MARK: - ChatConfiguration.request(for:) - Anthropic Tests

@Test func chatConfigRequest_anthropic_systemFieldPopulated() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "You are helpful",
        user: "Hello!",
        modelType: .fast,
        inference: .direct
    )

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = config.request(for: provider)

    // Anthropic with caching (default): system = nil, systemBlocks with cache_control on blocks
    #expect(request.system == nil)

    #expect(request.systemBlocks != nil)
    #expect(request.systemBlocks?[0].text == "You are helpful")
    #expect(request.systemBlocks?[0].cache_control != nil)
    #expect(request.messages.count == 1)
    #expect(request.messages[0].role == .user)
}

@Test func chatConfigRequest_anthropic_usesMaxTokens() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "System",
        user: "User",
        modelType: .fast,
        inference: .direct,
        maxTokens: 500
    )

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = config.request(for: provider)

    #expect(request.max_tokens == 500)
    #expect(request.max_completion_tokens == nil)
}

@Test func chatConfigRequest_anthropic_usesStopSequences() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "System",
        user: "User",
        modelType: .fast,
        inference: .direct,
        stopTokens: ["END"]
    )

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = config.request(for: provider)

    #expect(request.stop == nil)
    #expect(request.stop_sequences == ["END"])
}

@Test func chatConfigRequest_anthropic_skipsFrequencyPenalty() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "System",
        user: "User",
        modelType: .fast,
        inference: .direct,
        frequencyPenalty: 0.5
    )

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = config.request(for: provider)

    #expect(request.frequency_penalty == nil)
}

@Test func chatConfigRequest_anthropic_reasoning_includesThinking() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "System",
        user: "Think about this",
        modelType: .fast,
        inference: .reasoning,
        maxReasoningTokens: 2048
    )

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = config.request(for: provider)

    #expect(request.thinking != nil)
    #expect(request.thinking?.type == .enabled)
    #expect(request.thinking?.budget_tokens == 2048)
}

@Test func chatConfigRequest_anthropic_reasoning_defaultBudget() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "System",
        user: "Think",
        modelType: .fast,
        inference: .reasoning
        // No maxReasoningTokens - should default to 1024
    )

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = config.request(for: provider)

    #expect(request.thinking?.budget_tokens == 1024)
}

@Test func chatConfigRequest_anthropic_reasoning_noReasoningEffort() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "System",
        user: "User",
        modelType: .fast,
        inference: .reasoning,
        reasoningEffort: .high // Should be ignored for Anthropic
    )

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = config.request(for: provider)

    // Anthropic doesn't use reasoning_effort
    #expect(request.reasoning_effort == nil)
}

@Test func chatConfigRequest_anthropic_selectsCorrectModel_fastDirect() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "S", user: "U", modelType: .fast, inference: .direct
    )

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = config.request(for: provider)

    #expect(request.model == .claude45Haiku)
}

@Test func chatConfigRequest_anthropic_selectsCorrectModel_fastReasoning() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "S", user: "U", modelType: .fast, inference: .reasoning
    )

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = config.request(for: provider)

    #expect(request.model == .claude45Haiku)
}

@Test func chatConfigRequest_anthropic_selectsCorrectModel_flagship() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "S", user: "U", modelType: .flagship, inference: .direct
    )

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = config.request(for: provider)

    #expect(request.model == .claude46Opus)
}

// MARK: - Token Calculation Tests

@Test func chatConfigRequest_tokenCalculation_withReasoningTokens() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "S",
        user: "U",
        modelType: .fast,
        inference: .reasoning,
        maxTokens: 1000,
        maxReasoningTokens: 500
    )

    let openAIProvider = LLM.Provider.openAI(apiKey: "test")
    let anthropicProvider = LLM.Provider.anthropic(apiKey: "test")

    let openAIRequest = config.request(for: openAIProvider)
    let anthropicRequest = config.request(for: anthropicProvider)

    // OpenAI: reasoning + output share max_completion_tokens = 1000 + 500 = 1500
    #expect(openAIRequest.max_completion_tokens == 1500)
    // Anthropic: only maxTokens for output (thinking has separate budget)
    #expect(anthropicRequest.max_tokens == 1000)
}

@Test func chatConfigRequest_tokenCalculation_noMaxTokens() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "S",
        user: "U",
        modelType: .fast,
        inference: .reasoning,
        maxReasoningTokens: 500
    )

    let openAIProvider = LLM.Provider.openAI(apiKey: "test")
    let anthropicProvider = LLM.Provider.anthropic(apiKey: "test")

    let openAIRequest = config.request(for: openAIProvider)
    let anthropicRequest = config.request(for: anthropicProvider)

    // GPT-5 with reasoning and no maxTokens: don't set a limit (returns nil)
    #expect(openAIRequest.max_completion_tokens == nil)
    // Anthropic: maxTokens is nil → model's maxOutputTokens (claude45Haiku = 64,000)
    #expect(anthropicRequest.max_tokens == 64000)
}

// MARK: - Caching Tests

@Test func chatConfiguration_cachingDefaults() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "System",
        user: "User",
        modelType: .fast,
        inference: .direct
    )

    // Caching should be enabled by default
    #expect(config.enableCaching == true)
    #expect(config.cacheTTL == nil)
}

@Test func chatConfiguration_cachingEnabled() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "System",
        user: "User",
        modelType: .fast,
        inference: .direct,
        enableCaching: true,
        cacheTTL: .oneHour
    )

    #expect(config.enableCaching == true)
    #expect(config.cacheTTL == .oneHour)
}

@Test func chatConfigRequest_anthropic_cachingDisabled_usesStringSystem() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "You are helpful",
        user: "Hello!",
        modelType: .fast,
        inference: .direct,
        enableCaching: false
    )

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = config.request(for: provider)

    // Without caching, should use string system prompt
    #expect(request.system == "You are helpful")
    #expect(request.systemBlocks == nil)
}

@Test func chatConfigRequest_anthropic_cachingEnabled_usesSystemBlocks() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "You are helpful",
        user: "Hello!",
        modelType: .fast,
        inference: .direct,
        enableCaching: true
    )

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = config.request(for: provider)

    // With caching, should use systemBlocks with per-block cache_control
    #expect(request.system == nil)
    #expect(request.systemBlocks != nil)
    #expect(request.systemBlocks?[0].text == "You are helpful")
    #expect(request.systemBlocks?[0].cache_control != nil)
}

@Test func chatConfigRequest_anthropic_cachingEnabled_withTTL() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "System",
        user: "User",
        modelType: .fast,
        inference: .direct,
        enableCaching: true,
        cacheTTL: .oneHour
    )

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = config.request(for: provider)

    #expect(request.systemBlocks?[0].cache_control?.ttl == .oneHour)
}

@Test func chatConfigRequest_openAI_cachingHasNoEffect() {
    let config = LLM.ChatConfiguration(
        systemPrompt: "You are helpful",
        user: "Hello!",
        modelType: .fast,
        inference: .direct,
        enableCaching: true // Should have no effect for OpenAI
    )

    let provider = LLM.Provider.openAI(apiKey: "test")
    let request = config.request(for: provider)

    // OpenAI doesn't use system field or systemBlocks - caching is automatic
    #expect(request.system == nil)
    #expect(request.systemBlocks == nil)
    #expect(request.messages[0].role == .system)
    #expect(request.messages[0].textContent == "You are helpful")
}
