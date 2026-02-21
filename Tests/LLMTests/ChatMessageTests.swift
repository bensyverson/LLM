//
//  ChatMessageTests.swift
//  LLMTests
//
//  Tests for ChatMessage, ChatCompletion, Role enum
//

import Foundation
@testable import LLM
import Testing

// MARK: - ChatMessage Tests

@Test func chatMessage_backwardCompatibleInit() {
    let message = LLM.OpenAICompatibleAPI.ChatMessage(content: "Hello", role: .user)

    #expect(message.content == "Hello")
    #expect(message.role == .user)
    #expect(message.name == nil)
    #expect(message.tool_calls == nil)
    #expect(message.tool_call_id == nil)
}

@Test func chatMessage_backwardCompatibleInit_withName() {
    let message = LLM.OpenAICompatibleAPI.ChatMessage(
        content: "Hello",
        role: .user,
        name: "test_user"
    )

    #expect(message.content == "Hello")
    #expect(message.role == .user)
    #expect(message.name == "test_user")
}

@Test func chatMessage_fullInit_withOptionalContent() {
    let message = LLM.OpenAICompatibleAPI.ChatMessage(
        content: nil,
        role: .assistant,
        tool_calls: []
    )

    #expect(message.content == nil)
    #expect(message.role == .assistant)
    #expect(message.tool_calls != nil)
}

@Test func chatMessage_fullInit_withToolCalls() {
    let funcCall = LLM.OpenAICompatibleAPI.FunctionCall(
        name: "get_weather",
        arguments: "{\"location\":\"NYC\"}"
    )
    let toolCall = LLM.OpenAICompatibleAPI.ToolCall(
        id: "call_123",
        function: funcCall
    )

    let message = LLM.OpenAICompatibleAPI.ChatMessage(
        content: nil,
        role: .assistant,
        tool_calls: [toolCall]
    )

    #expect(message.content == nil)
    #expect(message.tool_calls?.count == 1)
    #expect(message.tool_calls?[0].id == "call_123")
}

@Test func chatMessage_fullInit_withToolCallId() {
    let message = LLM.OpenAICompatibleAPI.ChatMessage(
        content: "{\"temperature\":72}",
        role: .tool,
        tool_call_id: "call_123"
    )

    #expect(message.content == "{\"temperature\":72}")
    #expect(message.role == .tool)
    #expect(message.tool_call_id == "call_123")
}

@Test func chatMessage_contentLength_returnsZeroForNil() {
    let message = LLM.OpenAICompatibleAPI.ChatMessage(
        content: nil,
        role: .assistant
    )

    #expect(message.contentLength == 0)
}

@Test func chatMessage_contentLength_returnsActualLength() {
    let message = LLM.OpenAICompatibleAPI.ChatMessage(
        content: "Hello, world!",
        role: .user
    )

    #expect(message.contentLength == 13)
}

@Test func chatMessage_contentLength_emptyString() {
    let message = LLM.OpenAICompatibleAPI.ChatMessage(
        content: "",
        role: .user
    )

    #expect(message.contentLength == 0)
}

@Test func chatMessage_codable_roundTrip() throws {
    let message = LLM.OpenAICompatibleAPI.ChatMessage(
        content: "Hello",
        role: .user,
        name: "test"
    )

    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    let data = try encoder.encode(message)
    let decoded = try decoder.decode(LLM.OpenAICompatibleAPI.ChatMessage.self, from: data)

    #expect(decoded.content == message.content)
    #expect(decoded.role == message.role)
    #expect(decoded.name == message.name)
}

@Test func chatMessage_codable_withToolCalls() throws {
    let funcCall = LLM.OpenAICompatibleAPI.FunctionCall(name: "test", arguments: "{}")
    let toolCall = LLM.OpenAICompatibleAPI.ToolCall(id: "call_1", function: funcCall)
    let message = LLM.OpenAICompatibleAPI.ChatMessage(
        content: nil,
        role: .assistant,
        tool_calls: [toolCall]
    )

    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    let data = try encoder.encode(message)
    let decoded = try decoder.decode(LLM.OpenAICompatibleAPI.ChatMessage.self, from: data)

    #expect(decoded.tool_calls?.count == 1)
    #expect(decoded.tool_calls?[0].id == "call_1")
}

// MARK: - Role Tests

@Test func role_rawValues() {
    #expect(LLM.OpenAICompatibleAPI.Role.system.rawValue == "system")
    #expect(LLM.OpenAICompatibleAPI.Role.user.rawValue == "user")
    #expect(LLM.OpenAICompatibleAPI.Role.assistant.rawValue == "assistant")
    #expect(LLM.OpenAICompatibleAPI.Role.tool.rawValue == "tool")
}

@Test func role_codable_roundTrip() throws {
    let roles: [LLM.OpenAICompatibleAPI.Role] = [.system, .user, .assistant, .tool]
    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    for role in roles {
        let data = try encoder.encode(role)
        let decoded = try decoder.decode(LLM.OpenAICompatibleAPI.Role.self, from: data)
        #expect(decoded == role)
    }
}

// MARK: - ChatCompletion Tests

@Test func chatCompletion_defaultValues() {
    let messages = [
        LLM.OpenAICompatibleAPI.ChatMessage(content: "Hi", role: .user),
    ]
    let completion = LLM.OpenAICompatibleAPI.ChatCompletion(messages: messages)

    #expect(completion.model == .gpt35turbo)
    #expect(completion.system == nil)
    #expect(completion.response_format?.type == .json_object)
    #expect(completion.temperature == nil)
    #expect(completion.frequency_penalty == nil)
    #expect(completion.top_p == nil)
    #expect(completion.max_tokens == nil)
    #expect(completion.max_completion_tokens == nil)
    #expect(completion.stop == nil)
    #expect(completion.stop_sequences == nil)
    #expect(completion.thinking == nil)
    #expect(completion.reasoning_effort == nil)
    #expect(completion.tools == nil)
    #expect(completion.tool_choice == nil)
}

@Test func chatCompletion_customValues() {
    let messages = [
        LLM.OpenAICompatibleAPI.ChatMessage(content: "Hi", role: .user),
    ]
    let completion = LLM.OpenAICompatibleAPI.ChatCompletion(
        model: .gpt52,
        system: "Be helpful",
        messages: messages,
        response_format: nil,
        temperature: 0.7,
        frequency_penalty: 0.5,
        top_p: 0.9,
        max_tokens: 500,
        max_completion_tokens: 1000,
        stop: ["###"],
        stop_sequences: nil,
        thinking: .init(budget_tokens: 2048),
        reasoning_effort: .high
    )

    #expect(completion.model == .gpt52)
    #expect(completion.system == "Be helpful")
    #expect(completion.temperature == 0.7)
    #expect(completion.frequency_penalty == 0.5)
    #expect(completion.top_p == 0.9)
    #expect(completion.max_tokens == 500)
    #expect(completion.max_completion_tokens == 1000)
    #expect(completion.stop == ["###"])
    #expect(completion.thinking?.budget_tokens == 2048)
    #expect(completion.reasoning_effort == .high)
}

@Test func chatCompletion_withTools() {
    let params = LLM.OpenAICompatibleAPI.JSONSchema.object(
        properties: ["query": .string()],
        required: ["query"]
    )
    let funcDef = LLM.OpenAICompatibleAPI.FunctionDefinition(
        name: "search",
        description: "Search for info",
        parameters: params
    )
    let toolDef = LLM.OpenAICompatibleAPI.ToolDefinition(function: funcDef)

    let completion = LLM.OpenAICompatibleAPI.ChatCompletion(
        messages: [LLM.OpenAICompatibleAPI.ChatMessage(content: "Search for X", role: .user)],
        tools: [toolDef],
        tool_choice: .auto
    )

    #expect(completion.tools?.count == 1)
    #expect(completion.tools?[0].function.name == "search")
    #expect(completion.tool_choice == .auto)
}

// MARK: - JsonObject Tests

@Test func jsonObject_defaultType() {
    let jsonObject = LLM.OpenAICompatibleAPI.ChatCompletion.JsonObject()
    #expect(jsonObject.type == .json_object)
}

@Test func jsonObject_codable_roundTrip() throws {
    let jsonObject = LLM.OpenAICompatibleAPI.ChatCompletion.JsonObject()

    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    let data = try encoder.encode(jsonObject)
    let decoded = try decoder.decode(
        LLM.OpenAICompatibleAPI.ChatCompletion.JsonObject.self,
        from: data
    )

    #expect(decoded.type == jsonObject.type)
}

// MARK: - Thinking Tests

@Test func thinking_defaults() {
    let thinking = LLM.OpenAICompatibleAPI.ChatCompletion.Thinking()
    #expect(thinking.type == .enabled)
    #expect(thinking.budget_tokens == 1024)
}

@Test func thinking_customBudget() {
    let thinking = LLM.OpenAICompatibleAPI.ChatCompletion.Thinking(budget_tokens: 4096)
    #expect(thinking.type == .enabled)
    #expect(thinking.budget_tokens == 4096)
}

@Test func thinking_disabled() {
    let thinking = LLM.OpenAICompatibleAPI.ChatCompletion.Thinking(type: .disabled)
    #expect(thinking.type == .disabled)
}

@Test func thinking_codable_roundTrip() throws {
    let thinking = LLM.OpenAICompatibleAPI.ChatCompletion.Thinking(
        type: .enabled,
        budget_tokens: 2048
    )

    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    let data = try encoder.encode(thinking)
    let decoded = try decoder.decode(
        LLM.OpenAICompatibleAPI.ChatCompletion.Thinking.self,
        from: data
    )

    #expect(decoded.type == thinking.type)
    #expect(decoded.budget_tokens == thinking.budget_tokens)
}

// MARK: - ReasoningEffort Tests

@Test func reasoningEffort_rawValues() {
    #expect(LLM.OpenAICompatibleAPI.ChatCompletion.ReasoningEffort.none.rawValue == "none")
    #expect(LLM.OpenAICompatibleAPI.ChatCompletion.ReasoningEffort.low.rawValue == "low")
    #expect(LLM.OpenAICompatibleAPI.ChatCompletion.ReasoningEffort.medium.rawValue == "medium")
    #expect(LLM.OpenAICompatibleAPI.ChatCompletion.ReasoningEffort.high.rawValue == "high")
    #expect(LLM.OpenAICompatibleAPI.ChatCompletion.ReasoningEffort.xhigh.rawValue == "xhigh")
    #expect(LLM.OpenAICompatibleAPI.ChatCompletion.ReasoningEffort.minimal.rawValue == "minimal")
}

@Test func reasoningEffort_codable_roundTrip() throws {
    let efforts: [LLM.OpenAICompatibleAPI.ChatCompletion.ReasoningEffort] = [
        .none, .low, .medium, .high, .xhigh, .minimal,
    ]

    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    for effort in efforts {
        let data = try encoder.encode(effort)
        let decoded = try decoder.decode(
            LLM.OpenAICompatibleAPI.ChatCompletion.ReasoningEffort.self,
            from: data
        )
        #expect(decoded == effort)
    }
}

// MARK: - CacheControl Tests

@Test func cacheControl_defaults() {
    let cache = LLM.OpenAICompatibleAPI.CacheControl()
    #expect(cache.type == .ephemeral)
    #expect(cache.ttl == nil)
}

@Test func cacheControl_withTTL() {
    let cache = LLM.OpenAICompatibleAPI.CacheControl(ttl: .oneHour)
    #expect(cache.type == .ephemeral)
    #expect(cache.ttl == .oneHour)
}

@Test func cacheControl_ttlRawValues() {
    #expect(LLM.OpenAICompatibleAPI.CacheControl.TTL.fiveMinutes.rawValue == "5m")
    #expect(LLM.OpenAICompatibleAPI.CacheControl.TTL.oneHour.rawValue == "1h")
}

@Test func cacheControl_codable_roundTrip() throws {
    let cache = LLM.OpenAICompatibleAPI.CacheControl(ttl: .oneHour)

    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    let data = try encoder.encode(cache)
    let decoded = try decoder.decode(LLM.OpenAICompatibleAPI.CacheControl.self, from: data)

    #expect(decoded.type == cache.type)
    #expect(decoded.ttl == cache.ttl)
}

@Test func cacheControl_jsonSerialization() throws {
    let cache = LLM.OpenAICompatibleAPI.CacheControl(ttl: .fiveMinutes)

    let encoder = JSONEncoder()
    let data = try encoder.encode(cache)
    let jsonString = try #require(String(data: data, encoding: .utf8))

    #expect(jsonString.contains("\"type\":\"ephemeral\""))
    #expect(jsonString.contains("\"ttl\":\"5m\""))
}

// MARK: - SystemContentBlock Tests

@Test func systemContentBlock_defaults() {
    let block = LLM.OpenAICompatibleAPI.SystemContentBlock(text: "Hello")
    #expect(block.type == "text")
    #expect(block.text == "Hello")
    #expect(block.cache_control == nil)
}

@Test func systemContentBlock_withCacheControl() {
    let cache = LLM.OpenAICompatibleAPI.CacheControl(ttl: .oneHour)
    let block = LLM.OpenAICompatibleAPI.SystemContentBlock(text: "System", cache_control: cache)

    #expect(block.type == "text")
    #expect(block.text == "System")
    #expect(block.cache_control?.type == .ephemeral)
    #expect(block.cache_control?.ttl == .oneHour)
}

@Test func systemContentBlock_codable_roundTrip() throws {
    let cache = LLM.OpenAICompatibleAPI.CacheControl(ttl: .fiveMinutes)
    let block = LLM.OpenAICompatibleAPI.SystemContentBlock(text: "Test prompt", cache_control: cache)

    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    let data = try encoder.encode(block)
    let decoded = try decoder.decode(LLM.OpenAICompatibleAPI.SystemContentBlock.self, from: data)

    #expect(decoded.type == block.type)
    #expect(decoded.text == block.text)
    #expect(decoded.cache_control?.type == block.cache_control?.type)
    #expect(decoded.cache_control?.ttl == block.cache_control?.ttl)
}

@Test func systemContentBlock_jsonSerialization() throws {
    let cache = LLM.OpenAICompatibleAPI.CacheControl(ttl: .oneHour)
    let block = LLM.OpenAICompatibleAPI.SystemContentBlock(text: "Be helpful", cache_control: cache)

    let encoder = JSONEncoder()
    let data = try encoder.encode(block)
    let jsonString = try #require(String(data: data, encoding: .utf8))

    #expect(jsonString.contains("\"type\":\"text\""))
    #expect(jsonString.contains("\"text\":\"Be helpful\""))
    #expect(jsonString.contains("cache_control"))
    #expect(jsonString.contains("ephemeral"))
}

// MARK: - ChatCompletion with SystemBlocks Tests

@Test func chatCompletion_withSystemBlocks_encodesAsArray() throws {
    let cache = LLM.OpenAICompatibleAPI.CacheControl(ttl: .fiveMinutes)
    let blocks = [LLM.OpenAICompatibleAPI.SystemContentBlock(text: "You are helpful", cache_control: cache)]

    let completion = LLM.OpenAICompatibleAPI.ChatCompletion(
        model: .claude45Opus,
        system: nil,
        systemBlocks: blocks,
        messages: [LLM.OpenAICompatibleAPI.ChatMessage(content: "Hello", role: .user)]
    )

    let encoder = JSONEncoder()
    let data = try encoder.encode(completion)
    let jsonString = try #require(String(data: data, encoding: .utf8))

    // System should be encoded as an array, not a string
    #expect(jsonString.contains("\"system\":[{"))
    #expect(jsonString.contains("\"type\":\"text\""))
    #expect(jsonString.contains("\"text\":\"You are helpful\""))
    #expect(jsonString.contains("cache_control"))
}

@Test func chatCompletion_withStringSystem_encodesAsString() throws {
    let completion = LLM.OpenAICompatibleAPI.ChatCompletion(
        model: .claude45Opus,
        system: "You are helpful",
        systemBlocks: nil,
        messages: [LLM.OpenAICompatibleAPI.ChatMessage(content: "Hello", role: .user)]
    )

    let encoder = JSONEncoder()
    let data = try encoder.encode(completion)
    let jsonString = try #require(String(data: data, encoding: .utf8))

    // System should be encoded as a string
    #expect(jsonString.contains("\"system\":\"You are helpful\""))
    #expect(!jsonString.contains("cache_control"))
}

@Test func chatCompletion_withBothSystemTypes_prefersBlocks() throws {
    let blocks = [LLM.OpenAICompatibleAPI.SystemContentBlock(text: "Block system")]

    let completion = LLM.OpenAICompatibleAPI.ChatCompletion(
        model: .claude45Opus,
        system: "String system", // Should be ignored when blocks are present
        systemBlocks: blocks,
        messages: [LLM.OpenAICompatibleAPI.ChatMessage(content: "Hello", role: .user)]
    )

    let encoder = JSONEncoder()
    let data = try encoder.encode(completion)
    let jsonString = try #require(String(data: data, encoding: .utf8))

    // Should use blocks, not string
    #expect(jsonString.contains("\"system\":[{"))
    #expect(jsonString.contains("Block system"))
    #expect(!jsonString.contains("String system"))
}
