//
//  ConversationTests.swift
//  LLMTests
//
//  Tests for Conversation helpers and request building
//

import Testing
import Foundation
@testable import LLM

// MARK: - Conversation Helper Tests

@Test func conversation_addingUserMessage_appendsMessage() {
    let conversation = LLM.Conversation(systemPrompt: "You are a helpful assistant.")
    let updated = conversation.addingUserMessage("Hello!")

    #expect(updated.messages.count == 1)
    #expect(updated.messages[0].content == "Hello!")
    #expect(updated.messages[0].role == .user)
}

@Test func conversation_addingUserMessage_preservesExisting() {
    var conversation = LLM.Conversation(systemPrompt: "System")
    conversation = conversation.addingUserMessage("First")
    conversation = conversation.addingUserMessage("Second")

    #expect(conversation.messages.count == 2)
    #expect(conversation.messages[0].content == "First")
    #expect(conversation.messages[1].content == "Second")
}

@Test func conversation_addingAssistantMessage_appendsMessage() {
    let conversation = LLM.Conversation(systemPrompt: "System")
    let updated = conversation.addingAssistantMessage("Hello, how can I help?")

    #expect(updated.messages.count == 1)
    #expect(updated.messages[0].content == "Hello, how can I help?")
    #expect(updated.messages[0].role == .assistant)
}

@Test func conversation_addingAssistantMessage_preservesExisting() {
    var conversation = LLM.Conversation(systemPrompt: "System")
    conversation = conversation.addingUserMessage("Hi")
    conversation = conversation.addingAssistantMessage("Hello!")

    #expect(conversation.messages.count == 2)
    #expect(conversation.messages[0].role == .user)
    #expect(conversation.messages[1].role == .assistant)
}

@Test func conversation_chaining_userAssistantUser() {
    let conversation = LLM.Conversation(systemPrompt: "System")
        .addingUserMessage("Question 1")
        .addingAssistantMessage("Answer 1")
        .addingUserMessage("Question 2")

    #expect(conversation.messages.count == 3)
    #expect(conversation.messages[0].role == .user)
    #expect(conversation.messages[0].content == "Question 1")
    #expect(conversation.messages[1].role == .assistant)
    #expect(conversation.messages[1].content == "Answer 1")
    #expect(conversation.messages[2].role == .user)
    #expect(conversation.messages[2].content == "Question 2")
}

@Test func conversation_originalNotMutated() {
    let original = LLM.Conversation(systemPrompt: "System")
    let _ = original.addingUserMessage("New message")

    #expect(original.messages.isEmpty)
}

@Test func conversation_preservesSystemPrompt() {
    let conversation = LLM.Conversation(systemPrompt: "Be helpful")
        .addingUserMessage("Hi")

    #expect(conversation.systemPrompt == "Be helpful")
}

@Test func conversation_preservesConfiguration() {
    let config = LLM.ConversationConfiguration(
        modelType: .flagship,
        inference: .reasoning,
        temperature: 0.7
    )
    let conversation = LLM.Conversation(
        systemPrompt: "System",
        configuration: config
    ).addingUserMessage("Hi")

    #expect(conversation.configuration.modelType == .flagship)
    #expect(conversation.configuration.inference == .reasoning)
    #expect(conversation.configuration.temperature == 0.7)
}

// MARK: - Conversation.request(for:) - OpenAI Tests

@Test func conversationRequest_openAI_systemMessageIncludedInMessages() {
    let conversation = LLM.Conversation(systemPrompt: "You are helpful")
        .addingUserMessage("Hello")

    let provider = LLM.Provider.openAI(apiKey: "test")
    let request = conversation.request(for: provider)

    // OpenAI: system should be in messages array, not in system field
    #expect(request.system == nil)
    #expect(request.messages.count == 2)
    #expect(request.messages[0].role == .system)
    #expect(request.messages[0].content == "You are helpful")
    #expect(request.messages[1].role == .user)
    #expect(request.messages[1].content == "Hello")
}

@Test func conversationRequest_openAI_usesMaxCompletionTokens() {
    let config = LLM.ConversationConfiguration(maxTokens: 500)
    let conversation = LLM.Conversation(
        systemPrompt: "System",
        configuration: config
    ).addingUserMessage("Hi")

    let provider = LLM.Provider.openAI(apiKey: "test")
    let request = conversation.request(for: provider)

    #expect(request.max_tokens == nil)
    #expect(request.max_completion_tokens == 500)
}

@Test func conversationRequest_openAI_usesStop() {
    let config = LLM.ConversationConfiguration(stopTokens: ["END", "STOP"])
    let conversation = LLM.Conversation(
        systemPrompt: "System",
        configuration: config
    ).addingUserMessage("Hi")

    let provider = LLM.Provider.openAI(apiKey: "test")
    let request = conversation.request(for: provider)

    #expect(request.stop == ["END", "STOP"])
    #expect(request.stop_sequences == nil)
}

@Test func conversationRequest_openAI_reasoning_autoSetsReasoningEffort() {
    let config = LLM.ConversationConfiguration(
        modelType: .flagship,
        inference: .reasoning
    )
    let conversation = LLM.Conversation(
        systemPrompt: "System",
        configuration: config
    ).addingUserMessage("Think about this")

    let provider = LLM.Provider.openAI(apiKey: "test")
    let request = conversation.request(for: provider)

    // GPT-5.2 with reasoning should auto-set reasoning_effort to .high
    #expect(request.reasoning_effort == .high)
}

@Test func conversationRequest_openAI_reasoning_skipsTemperatureAndTopP() {
    let config = LLM.ConversationConfiguration(
        inference: .reasoning,
        temperature: 0.5,
        topP: 0.9
    )
    let conversation = LLM.Conversation(
        systemPrompt: "System",
        configuration: config
    ).addingUserMessage("Hi")

    let provider = LLM.Provider.openAI(apiKey: "test")
    let request = conversation.request(for: provider)

    #expect(request.temperature == nil)
    #expect(request.top_p == nil)
}

@Test func conversationRequest_openAI_direct_includesTemperatureAndTopP() {
    let config = LLM.ConversationConfiguration(
        inference: .direct,
        temperature: 0.5,
        topP: 0.9
    )
    let conversation = LLM.Conversation(
        systemPrompt: "System",
        configuration: config
    ).addingUserMessage("Hi")

    let provider = LLM.Provider.openAI(apiKey: "test")
    let request = conversation.request(for: provider)

    #expect(request.temperature == 0.5)
    #expect(request.top_p == 0.9)
}

@Test func conversationRequest_openAI_noThinking() {
    let config = LLM.ConversationConfiguration(inference: .reasoning)
    let conversation = LLM.Conversation(
        systemPrompt: "System",
        configuration: config
    ).addingUserMessage("Hi")

    let provider = LLM.Provider.openAI(apiKey: "test")
    let request = conversation.request(for: provider)

    // OpenAI doesn't use thinking parameter
    #expect(request.thinking == nil)
}

// MARK: - Conversation.request(for:) - Anthropic Tests

@Test func conversationRequest_anthropic_systemFieldPopulated() {
    let conversation = LLM.Conversation(systemPrompt: "You are helpful")
        .addingUserMessage("Hello")

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = conversation.request(for: provider)

    // Anthropic: system should be in system field, not in messages
    #expect(request.system == "You are helpful")
    #expect(request.messages.count == 1)
    #expect(request.messages[0].role == .user)
    #expect(request.messages[0].content == "Hello")
}

@Test func conversationRequest_anthropic_usesMaxTokens() {
    let config = LLM.ConversationConfiguration(maxTokens: 500)
    let conversation = LLM.Conversation(
        systemPrompt: "System",
        configuration: config
    ).addingUserMessage("Hi")

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = conversation.request(for: provider)

    #expect(request.max_tokens == 500)
    #expect(request.max_completion_tokens == nil)
}

@Test func conversationRequest_anthropic_usesStopSequences() {
    let config = LLM.ConversationConfiguration(stopTokens: ["END", "STOP"])
    let conversation = LLM.Conversation(
        systemPrompt: "System",
        configuration: config
    ).addingUserMessage("Hi")

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = conversation.request(for: provider)

    #expect(request.stop == nil)
    #expect(request.stop_sequences == ["END", "STOP"])
}

@Test func conversationRequest_anthropic_skipsFrequencyPenalty() {
    let config = LLM.ConversationConfiguration(frequencyPenalty: 0.5)
    let conversation = LLM.Conversation(
        systemPrompt: "System",
        configuration: config
    ).addingUserMessage("Hi")

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = conversation.request(for: provider)

    #expect(request.frequency_penalty == nil)
}

@Test func conversationRequest_anthropic_reasoning_includesThinking() {
    let config = LLM.ConversationConfiguration(
        inference: .reasoning,
        maxReasoningTokens: 2048
    )
    let conversation = LLM.Conversation(
        systemPrompt: "System",
        configuration: config
    ).addingUserMessage("Think about this")

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = conversation.request(for: provider)

    #expect(request.thinking != nil)
    #expect(request.thinking?.type == .enabled)
    #expect(request.thinking?.budget_tokens == 2048)
}

@Test func conversationRequest_anthropic_reasoning_noReasoningEffort() {
    let config = LLM.ConversationConfiguration(inference: .reasoning)
    let conversation = LLM.Conversation(
        systemPrompt: "System",
        configuration: config
    ).addingUserMessage("Hi")

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = conversation.request(for: provider)

    // Anthropic doesn't use reasoning_effort
    #expect(request.reasoning_effort == nil)
}

@Test func conversationRequest_anthropic_selectsCorrectModel_fastDirect() {
    let config = LLM.ConversationConfiguration(modelType: .fast, inference: .direct)
    let conversation = LLM.Conversation(
        systemPrompt: "System",
        configuration: config
    ).addingUserMessage("Hi")

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = conversation.request(for: provider)

    #expect(request.model == .claude45Haiku)
}

@Test func conversationRequest_anthropic_selectsCorrectModel_fastReasoning() {
    let config = LLM.ConversationConfiguration(modelType: .fast, inference: .reasoning)
    let conversation = LLM.Conversation(
        systemPrompt: "System",
        configuration: config
    ).addingUserMessage("Hi")

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = conversation.request(for: provider)

    #expect(request.model == .claude45Haiku)
}

@Test func conversationRequest_anthropic_selectsCorrectModel_flagship() {
    let config = LLM.ConversationConfiguration(modelType: .flagship, inference: .direct)
    let conversation = LLM.Conversation(
        systemPrompt: "System",
        configuration: config
    ).addingUserMessage("Hi")

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = conversation.request(for: provider)

    #expect(request.model == .claude45Opus)
}

// MARK: - Conversation JSON Serialization Tests

@Test func conversationRequest_serializesToCorrectJSON_openAI() throws {
    let conversation = LLM.Conversation(systemPrompt: "You are a helpful assistant")
        .addingUserMessage("What is 2+2?")
        .addingAssistantMessage("4")
        .addingUserMessage("And 3+3?")

    let request = conversation.request(for: .openAI(apiKey: "test"))

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let data = try encoder.encode(request)
    let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

    // Verify model
    #expect(json["model"] as? String == "gpt-5-mini")

    // Verify system is nil for OpenAI (system prompt is in messages)
    #expect(json["system"] == nil || json["system"] is NSNull)

    // Verify messages array structure
    let messages = json["messages"] as! [[String: Any]]
    #expect(messages.count == 4)

    #expect(messages[0]["role"] as? String == "system")
    #expect(messages[0]["content"] as? String == "You are a helpful assistant")

    #expect(messages[1]["role"] as? String == "user")
    #expect(messages[1]["content"] as? String == "What is 2+2?")

    #expect(messages[2]["role"] as? String == "assistant")
    #expect(messages[2]["content"] as? String == "4")

    #expect(messages[3]["role"] as? String == "user")
    #expect(messages[3]["content"] as? String == "And 3+3?")
}

@Test func conversationRequest_serializesToCorrectJSON_anthropic() throws {
    let conversation = LLM.Conversation(systemPrompt: "You are a helpful assistant")
        .addingUserMessage("What is 2+2?")
        .addingAssistantMessage("4")
        .addingUserMessage("And 3+3?")

    let request = conversation.request(for: .anthropic(apiKey: "test"))

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let data = try encoder.encode(request)
    let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

    // Verify model
    #expect(json["model"] as? String == "claude-haiku-4-5")

    // Verify system is in dedicated field for Anthropic
    #expect(json["system"] as? String == "You are a helpful assistant")

    // Verify messages array (should NOT include system message)
    let messages = json["messages"] as! [[String: Any]]
    #expect(messages.count == 3)

    #expect(messages[0]["role"] as? String == "user")
    #expect(messages[0]["content"] as? String == "What is 2+2?")

    #expect(messages[1]["role"] as? String == "assistant")
    #expect(messages[1]["content"] as? String == "4")

    #expect(messages[2]["role"] as? String == "user")
    #expect(messages[2]["content"] as? String == "And 3+3?")
}

@Test func conversationRequest_serializesToCorrectJSON_withReasoning() throws {
    let config = LLM.ConversationConfiguration(
        modelType: .flagship,
        inference: .reasoning,
        maxTokens: 1000,
        maxReasoningTokens: 2048
    )
    let conversation = LLM.Conversation(
        systemPrompt: "Think step by step",
        configuration: config
    ).addingUserMessage("Solve this puzzle")

    // Test OpenAI serialization
    let openAIRequest = conversation.request(for: .openAI(apiKey: "test"))
    let openAIData = try JSONEncoder().encode(openAIRequest)
    let openAIJson = try JSONSerialization.jsonObject(with: openAIData) as! [String: Any]

    #expect(openAIJson["model"] as? String == "gpt-5.2")
    #expect(openAIJson["reasoning_effort"] as? String == "high")
    #expect(openAIJson["max_completion_tokens"] as? Int == 3048)  // 1000 + 2048
    #expect(openAIJson["thinking"] == nil || openAIJson["thinking"] is NSNull)

    // Test Anthropic serialization
    let anthropicRequest = conversation.request(for: .anthropic(apiKey: "test"))
    let anthropicData = try JSONEncoder().encode(anthropicRequest)
    let anthropicJson = try JSONSerialization.jsonObject(with: anthropicData) as! [String: Any]

    #expect(anthropicJson["model"] as? String == "claude-opus-4-5")
    #expect(anthropicJson["reasoning_effort"] == nil || anthropicJson["reasoning_effort"] is NSNull)
    #expect(anthropicJson["max_tokens"] as? Int == 3048)
    let thinking = anthropicJson["thinking"] as? [String: Any]
    #expect(thinking?["type"] as? String == "enabled")
    #expect(thinking?["budget_tokens"] as? Int == 2048)
}

// MARK: - ConversationConfiguration Tests

@Test func conversationConfiguration_defaults() {
    let config = LLM.ConversationConfiguration()

    #expect(config.modelType == .fast)
    #expect(config.inference == .direct)
    #expect(config.temperature == nil)
    #expect(config.frequencyPenalty == nil)
    #expect(config.topP == nil)
    #expect(config.maxTokens == nil)
    #expect(config.maxReasoningTokens == nil)
    #expect(config.reasoningEffort == nil)
    #expect(config.stopTokens == nil)
}

@Test func conversationConfiguration_customValues() {
    let config = LLM.ConversationConfiguration(
        modelType: .flagship,
        inference: .reasoning,
        temperature: 0.8,
        frequencyPenalty: 0.3,
        repeatPenalty: 0.2,
        topP: 0.95,
        maxTokens: 1000,
        maxReasoningTokens: 500,
        reasoningEffort: .medium,
        stopTokens: ["END"]
    )

    #expect(config.modelType == .flagship)
    #expect(config.inference == .reasoning)
    #expect(config.temperature == 0.8)
    #expect(config.frequencyPenalty == 0.3)
    #expect(config.repeatPenalty == 0.2)
    #expect(config.topP == 0.95)
    #expect(config.maxTokens == 1000)
    #expect(config.maxReasoningTokens == 500)
    #expect(config.reasoningEffort == .medium)
    #expect(config.stopTokens == ["END"])
}

// MARK: - Caching Tests

@Test func conversationConfiguration_cachingDefaults() {
    let config = LLM.ConversationConfiguration()

    #expect(config.enableCaching == false)
    #expect(config.cacheTTL == nil)
}

@Test func conversationConfiguration_cachingEnabled() {
    let config = LLM.ConversationConfiguration(
        enableCaching: true,
        cacheTTL: .oneHour
    )

    #expect(config.enableCaching == true)
    #expect(config.cacheTTL == .oneHour)
}

@Test func conversationRequest_anthropic_cachingDisabled_usesStringSystem() {
    let config = LLM.ConversationConfiguration(enableCaching: false)
    let conversation = LLM.Conversation(
        systemPrompt: "You are helpful",
        configuration: config
    ).addingUserMessage("Hello")

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = conversation.request(for: provider)

    #expect(request.system == "You are helpful")
    #expect(request.systemBlocks == nil)
}

@Test func conversationRequest_anthropic_cachingEnabled_usesSystemBlocks() {
    let config = LLM.ConversationConfiguration(enableCaching: true)
    let conversation = LLM.Conversation(
        systemPrompt: "You are helpful",
        configuration: config
    ).addingUserMessage("Hello")

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = conversation.request(for: provider)

    #expect(request.system == nil)
    #expect(request.systemBlocks != nil)
    #expect(request.systemBlocks?.count == 1)
    #expect(request.systemBlocks?[0].text == "You are helpful")
    #expect(request.systemBlocks?[0].cache_control != nil)
}

@Test func conversationRequest_anthropic_cachingEnabled_withTTL() {
    let config = LLM.ConversationConfiguration(
        enableCaching: true,
        cacheTTL: .fiveMinutes
    )
    let conversation = LLM.Conversation(
        systemPrompt: "System",
        configuration: config
    ).addingUserMessage("Hi")

    let provider = LLM.Provider.anthropic(apiKey: "test")
    let request = conversation.request(for: provider)

    #expect(request.systemBlocks?[0].cache_control?.ttl == .fiveMinutes)
}

@Test func conversationRequest_openAI_cachingHasNoEffect() {
    let config = LLM.ConversationConfiguration(enableCaching: true)
    let conversation = LLM.Conversation(
        systemPrompt: "You are helpful",
        configuration: config
    ).addingUserMessage("Hello")

    let provider = LLM.Provider.openAI(apiKey: "test")
    let request = conversation.request(for: provider)

    // OpenAI doesn't use system field or systemBlocks
    #expect(request.system == nil)
    #expect(request.systemBlocks == nil)
    #expect(request.messages[0].role == .system)
    #expect(request.messages[0].content == "You are helpful")
}
