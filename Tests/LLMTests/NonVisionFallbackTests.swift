//
//  NonVisionFallbackTests.swift
//  LLMTests
//
//  Tests for non-vision model fallback behavior and conversation API overloads
//

import Foundation
@testable import LLM
import Testing

// MARK: - Conversation API with ContentPart Tests

@Test func conversation_addingUserMessage_contentParts() {
    let conversation = LLM.Conversation(systemPrompt: "System")
        .addingUserMessage([
            .text("Describe this"),
            .image(data: Data(), mediaType: "image/jpeg"),
        ])

    #expect(conversation.messages.count == 1)
    #expect(conversation.messages[0].role == .user)
    #expect(conversation.messages[0].content.count == 2)
    #expect(conversation.messages[0].textContent == "Describe this")
    #expect(conversation.messages[0].hasMedia == true)
}

@Test func conversation_addingToolResultMessage_contentParts() {
    let conversation = LLM.Conversation(systemPrompt: "System")
        .addingToolResultMessage(toolCallId: "call_1", content: [
            .text("Result text"),
            .image(data: Data(), mediaType: "image/png"),
        ])

    #expect(conversation.messages.count == 1)
    #expect(conversation.messages[0].role == .tool)
    #expect(conversation.messages[0].tool_call_id == "call_1")
    #expect(conversation.messages[0].content.count == 2)
    #expect(conversation.messages[0].hasMedia == true)
}

// MARK: - ConversationResponse Warnings

@Test func conversationResponse_warnings_defaultEmpty() {
    let response = LLM.ConversationResponse(
        text: "Hello",
        thinking: nil,
        toolCalls: [],
        conversation: LLM.Conversation(systemPrompt: "System"),
        rawResponse: LLM.OpenAICompatibleAPI.ChatCompletionResponse(model: "test")
    )

    #expect(response.warnings.isEmpty)
}

@Test func conversationResponse_warnings_canBePopulated() {
    let response = LLM.ConversationResponse(
        text: "Hello",
        thinking: nil,
        toolCalls: [],
        conversation: LLM.Conversation(systemPrompt: "System"),
        rawResponse: LLM.OpenAICompatibleAPI.ChatCompletionResponse(model: "test"),
        warnings: ["Images were stripped"]
    )

    #expect(response.warnings.count == 1)
    #expect(response.warnings[0] == "Images were stripped")
}

// MARK: - Model Vision Detection in Conversation Context

@Test func conversation_withImages_nonVisionModel_strippedAtSendTime() async throws {
    // This tests the stripping logic directly (not the full API call)
    let llm = LLM(provider: .openAI(apiKey: "test"))

    let conversation = LLM.Conversation(systemPrompt: "System")
        .addingUserMessage([
            .text("Describe this"),
            .image(data: Data([0xFF, 0xD8]), mediaType: "image/jpeg", filename: "photo.jpg"),
        ])

    // Verify the conversation has media
    #expect(conversation.messages[0].hasMedia == true)

    // Strip using the LLM method
    let stripped = try await llm.strippingMedia(conversation)

    // Verify media is replaced with text stubs
    #expect(stripped.messages[0].hasMedia == false)
    #expect(stripped.messages[0].content.count == 2)
    #expect(stripped.messages[0].content[0].textContent == "Describe this")
    #expect(stripped.messages[0].content[1].textContent?.contains("image") == true)
    #expect(stripped.messages[0].content[1].textContent?.contains("photo.jpg") == true)
}

@Test func conversation_visionModel_passesThrough() {
    // Vision models don't need stripping — the hasMedia + supportsVision check
    // ensures media passes through to the provider
    let model = LLM.OpenAICompatibleAPI.ModelName.gpt4o
    #expect(model.supportsVision == true)

    let conversation = LLM.Conversation(systemPrompt: "System")
        .addingUserMessage([
            .text("Describe this"),
            .image(data: Data(), mediaType: "image/jpeg"),
        ])

    // Verify media is present and would pass through
    #expect(conversation.messages[0].hasMedia == true)
}

@Test func conversation_unknownModel_defaultsToPassthrough() {
    let model = LLM.OpenAICompatibleAPI.ModelName(rawValue: "my-custom-model")
    #expect(model.supportsVision == nil)
    // nil means unknown — we don't strip, we let the provider handle it
}

// MARK: - Multimodal JSON Serialization Tests

@Test func conversationRequest_openAI_multimodalMessage_serializesCorrectly() throws {
    let imageData = Data([0xFF, 0xD8, 0xFF, 0xE0])
    let conversation = LLM.Conversation(systemPrompt: "You are helpful")
        .addingUserMessage([
            .text("What's in this image?"),
            .image(data: imageData, mediaType: "image/jpeg"),
        ])

    let request = conversation.request(for: .openAI(apiKey: "test"))
    let encoder = JSONEncoder()
    let data = try encoder.encode(request)
    let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let messages = try #require(json["messages"] as? [[String: Any]])

    // System message is still text
    #expect(messages[0]["content"] as? String == "You are helpful")

    // User message should be an array (multimodal)
    let userContent = try #require(messages[1]["content"] as? [[String: Any]])
    #expect(userContent.count == 2)
    #expect(userContent[0]["type"] as? String == "text")
    #expect(userContent[1]["type"] as? String == "image_url")
}

@Test func conversationRequest_anthropic_multimodalMessage_serializesCorrectly() throws {
    let imageData = Data([0xFF, 0xD8, 0xFF, 0xE0])
    let conversation = LLM.Conversation(systemPrompt: "You are helpful")
        .addingUserMessage([
            .text("What's in this image?"),
            .image(data: imageData, mediaType: "image/jpeg"),
        ])

    let request = conversation.request(for: .anthropic(apiKey: "test"))
    let encoder = JSONEncoder()
    let data = try encoder.encode(request)
    let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let messages = try #require(json["messages"] as? [[String: Any]])

    #expect(messages.count == 1)
    let content = try #require(messages[0]["content"] as? [[String: Any]])
    #expect(content.count == 2)
    #expect(content[0]["type"] as? String == "text")
    #expect(content[0]["text"] as? String == "What's in this image?")
    #expect(content[1]["type"] as? String == "image")
    let source = try #require(content[1]["source"] as? [String: Any])
    #expect(source["type"] as? String == "base64")
    #expect(source["media_type"] as? String == "image/jpeg")
}
