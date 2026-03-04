//
//  ConversationToolTests.swift
//  LLMTests
//
//  Tests for Conversation tool support additions
//

import Foundation
@testable import LLM
import Testing

// MARK: - ConversationConfiguration Tool Fields

@Test func conversationConfiguration_toolsDefaultToNil() {
    let config = LLM.ConversationConfiguration()

    #expect(config.tools == nil)
    #expect(config.toolChoice == nil)
}

@Test func conversationConfiguration_toolsCanBeSet() {
    let toolDef = LLM.OpenAICompatibleAPI.ToolDefinition(
        function: LLM.OpenAICompatibleAPI.FunctionDefinition(
            name: "get_weather",
            description: "Get the weather",
            parameters: LLM.OpenAICompatibleAPI.JSONSchema.object(
                properties: [
                    "location": LLM.OpenAICompatibleAPI.JSONSchema.string(),
                ],
                required: ["location"]
            )
        )
    )
    let config = LLM.ConversationConfiguration(
        tools: [toolDef],
        toolChoice: .auto
    )

    #expect(config.tools?.count == 1)
    #expect(config.tools?[0].function.name == "get_weather")
    #expect(config.toolChoice == .auto)
}

// MARK: - Conversation.request(for:) passes tools

@Test func conversationRequest_passesToolsToRequest() {
    let toolDef = LLM.OpenAICompatibleAPI.ToolDefinition(
        function: LLM.OpenAICompatibleAPI.FunctionDefinition(
            name: "search",
            description: "Search the web",
            parameters: LLM.OpenAICompatibleAPI.JSONSchema.object(properties: [:])
        )
    )
    let config = LLM.ConversationConfiguration(
        tools: [toolDef],
        toolChoice: .required
    )
    let conversation = LLM.Conversation(
        systemPrompt: "System",
        configuration: config
    ).addingUserMessage("Find something")

    let request = conversation.request(for: .openAI(apiKey: "test"))

    #expect(request.tools?.count == 1)
    #expect(request.tools?[0].function.name == "search")
    #expect(request.tool_choice == .required)
}

@Test func conversationRequest_nilToolsNotIncluded() {
    let config = LLM.ConversationConfiguration()
    let conversation = LLM.Conversation(
        systemPrompt: "System",
        configuration: config
    ).addingUserMessage("Hello")

    let request = conversation.request(for: .openAI(apiKey: "test"))

    #expect(request.tools == nil)
    #expect(request.tool_choice == nil)
}

// MARK: - Conversation Tool Call Helpers

@Test func conversation_addingAssistantToolCallMessage_appendsMessage() {
    let conversation = LLM.Conversation(systemPrompt: "System")
        .addingUserMessage("Do something")

    let toolCalls = [
        LLM.OpenAICompatibleAPI.ToolCall(
            id: "call_123",
            function: LLM.OpenAICompatibleAPI.FunctionCall(
                name: "get_weather",
                arguments: "{\"location\":\"NYC\"}"
            )
        ),
    ]

    let updated = conversation.addingAssistantToolCallMessage(toolCalls)

    #expect(updated.messages.count == 2)
    #expect(updated.messages[1].role == .assistant)
    #expect(updated.messages[1].content.isEmpty)
    #expect(updated.messages[1].tool_calls?.count == 1)
    #expect(updated.messages[1].tool_calls?[0].id == "call_123")
    #expect(updated.messages[1].tool_calls?[0].function.name == "get_weather")
}

@Test func conversation_addingToolResultMessage_appendsMessage() {
    let conversation = LLM.Conversation(systemPrompt: "System")
        .addingUserMessage("Do something")

    let updated = conversation.addingToolResultMessage(
        toolCallId: "call_123",
        content: "The weather in NYC is sunny, 72°F"
    )

    #expect(updated.messages.count == 2)
    #expect(updated.messages[1].role == .tool)
    #expect(updated.messages[1].textContent == "The weather in NYC is sunny, 72°F")
    #expect(updated.messages[1].tool_call_id == "call_123")
}

@Test func conversation_toolCallRoundTrip_chainedMessages() {
    let toolCalls = [
        LLM.OpenAICompatibleAPI.ToolCall(
            id: "call_abc",
            function: LLM.OpenAICompatibleAPI.FunctionCall(
                name: "search",
                arguments: "{\"query\":\"Swift concurrency\"}"
            )
        ),
    ]

    let conversation = LLM.Conversation(systemPrompt: "System")
        .addingUserMessage("Search for Swift concurrency")
        .addingAssistantToolCallMessage(toolCalls)
        .addingToolResultMessage(toolCallId: "call_abc", content: "Found 10 results")
        .addingAssistantMessage("I found 10 results about Swift concurrency.")

    #expect(conversation.messages.count == 4)
    #expect(conversation.messages[0].role == .user)
    #expect(conversation.messages[1].role == .assistant)
    #expect(conversation.messages[1].tool_calls != nil)
    #expect(conversation.messages[2].role == .tool)
    #expect(conversation.messages[3].role == .assistant)
    #expect(conversation.messages[3].textContent == "I found 10 results about Swift concurrency.")
}

// MARK: - ConversationResponse Optional Text + ToolCalls

@Test func conversationResponse_textCanBeNil() {
    let response = LLM.ConversationResponse(
        text: nil,
        thinking: nil,
        toolCalls: [
            LLM.OpenAICompatibleAPI.ToolCall(
                id: "call_1",
                function: LLM.OpenAICompatibleAPI.FunctionCall(
                    name: "test",
                    arguments: "{}"
                )
            ),
        ],
        conversation: LLM.Conversation(systemPrompt: "System"),
        rawResponse: LLM.OpenAICompatibleAPI.ChatCompletionResponse(
            id: "test",
            object: nil,
            system_fingerprint: nil,
            usage: LLM.OpenAICompatibleAPI.ChatCompletionResponse.Usage(
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                input_tokens: nil,
                output_tokens: nil,
                cache_creation_input_tokens: nil,
                cache_read_input_tokens: nil,
                prompt_tokens_details: nil
            ),
            model: "test",
            created: nil,
            choices: nil,
            content: nil
        )
    )

    #expect(response.text == nil)
    #expect(response.toolCalls.count == 1)
}

@Test func conversationResponse_textWithNoToolCalls() {
    let response = LLM.ConversationResponse(
        text: "Hello!",
        thinking: nil,
        toolCalls: [],
        conversation: LLM.Conversation(systemPrompt: "System"),
        rawResponse: LLM.OpenAICompatibleAPI.ChatCompletionResponse(
            id: "test",
            object: nil,
            system_fingerprint: nil,
            usage: LLM.OpenAICompatibleAPI.ChatCompletionResponse.Usage(
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                input_tokens: nil,
                output_tokens: nil,
                cache_creation_input_tokens: nil,
                cache_read_input_tokens: nil,
                prompt_tokens_details: nil
            ),
            model: "test",
            created: nil,
            choices: nil,
            content: nil
        )
    )

    #expect(response.text == "Hello!")
    #expect(response.toolCalls.isEmpty)
}

// MARK: - JSONValue Tests

@Test func jsonValue_decodesString() throws {
    let json = "\"hello\"".data(using: .utf8)!
    let value = try JSONDecoder().decode(JSONValue.self, from: json)
    #expect(value == .string("hello"))
    #expect(value.stringValue == "hello")
}

@Test func jsonValue_decodesInteger() throws {
    let json = "42".data(using: .utf8)!
    let value = try JSONDecoder().decode(JSONValue.self, from: json)
    #expect(value == .integer(42))
    #expect(value.intValue == 42)
}

@Test func jsonValue_decodesDouble() throws {
    let json = "3.14".data(using: .utf8)!
    let value = try JSONDecoder().decode(JSONValue.self, from: json)
    #expect(value == .number(3.14))
    #expect(value.doubleValue == 3.14)
}

@Test func jsonValue_decodesBool() throws {
    let json = "true".data(using: .utf8)!
    let value = try JSONDecoder().decode(JSONValue.self, from: json)
    #expect(value == .bool(true))
    #expect(value.boolValue == true)
}

@Test func jsonValue_decodesNull() throws {
    let json = "null".data(using: .utf8)!
    let value = try JSONDecoder().decode(JSONValue.self, from: json)
    #expect(value == .null)
}

@Test func jsonValue_decodesArray() throws {
    let json = "[1, \"two\", true]".data(using: .utf8)!
    let value = try JSONDecoder().decode(JSONValue.self, from: json)
    #expect(value == .array([.integer(1), .string("two"), .bool(true)]))
}

@Test func jsonValue_decodesObject() throws {
    let json = "{\"name\": \"test\", \"count\": 5}".data(using: .utf8)!
    let value = try JSONDecoder().decode(JSONValue.self, from: json)
    #expect(value == .object(["name": .string("test"), "count": .integer(5)]))
}

@Test func jsonValue_roundTrips() throws {
    let original: JSONValue = .object([
        "string": .string("hello"),
        "number": .integer(42),
        "nested": .object(["inner": .bool(false)]),
        "list": .array([.string("a"), .null]),
    ])

    let data = try JSONEncoder().encode(original)
    let decoded = try JSONDecoder().decode(JSONValue.self, from: data)
    #expect(decoded == original)
}

@Test func jsonValue_integerDoubleValue() {
    let value = JSONValue.integer(42)
    #expect(value.doubleValue == 42.0)
}
