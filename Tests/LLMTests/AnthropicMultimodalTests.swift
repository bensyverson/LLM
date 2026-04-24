//
//  AnthropicMultimodalTests.swift
//  LLMTests
//
//  Tests for Anthropic-format encoding of multimodal messages
//

import Foundation
@testable import LLM
import Testing

// MARK: - Anthropic Image Encoding

@Test func `anthropic encoding image block encodes base 64 source`() throws {
    let imageData = Data([0xFF, 0xD8, 0xFF, 0xE0])
    let msg = LLM.OpenAICompatibleAPI.ChatMessage(
        content: [
            .text("Describe this"),
            .image(data: imageData, mediaType: "image/jpeg"),
        ],
        role: .user,
    )

    let messages = LLM.OpenAICompatibleAPI.AnthropicMessageConverter.convert([msg])

    #expect(messages.count == 1)
    #expect(messages[0].role == "user")

    // Encode to JSON and verify structure
    let data = try JSONEncoder().encode(messages[0])
    let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let content = try #require(json["content"] as? [[String: Any]])

    #expect(content.count == 2)

    // Text block
    #expect(content[0]["type"] as? String == "text")
    #expect(content[0]["text"] as? String == "Describe this")

    // Image block
    #expect(content[1]["type"] as? String == "image")
    let source = try #require(content[1]["source"] as? [String: Any])
    #expect(source["type"] as? String == "base64")
    #expect(source["media_type"] as? String == "image/jpeg")
    let base64 = try #require(source["data"] as? String)
    #expect(base64 == imageData.base64EncodedString())
}

// MARK: - Anthropic Document Encoding

@Test func `anthropic encoding document block encodes with title`() throws {
    let pdfData = Data([0x25, 0x50, 0x44, 0x46])
    let msg = LLM.OpenAICompatibleAPI.ChatMessage(
        content: [.pdf(data: pdfData, title: "report.pdf")],
        role: .user,
    )

    let messages = LLM.OpenAICompatibleAPI.AnthropicMessageConverter.convert([msg])
    let data = try JSONEncoder().encode(messages[0])
    let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let content = try #require(json["content"] as? [[String: Any]])

    #expect(content.count == 1)
    #expect(content[0]["type"] as? String == "document")
    #expect(content[0]["title"] as? String == "report.pdf")

    let source = try #require(content[0]["source"] as? [String: Any])
    #expect(source["type"] as? String == "base64")
    #expect(source["media_type"] as? String == "application/pdf")
}

// MARK: - Anthropic Tool Result with Images

@Test func `anthropic encoding tool result with image encodes content array`() throws {
    let imageData = Data([0x89, 0x50, 0x4E, 0x47])
    let msg = LLM.OpenAICompatibleAPI.ChatMessage(
        content: [
            .text("Here is the screenshot"),
            .image(data: imageData, mediaType: "image/png"),
        ],
        role: .tool,
        tool_call_id: "call_123",
    )

    let messages = LLM.OpenAICompatibleAPI.AnthropicMessageConverter.convert([msg])

    #expect(messages.count == 1)
    #expect(messages[0].role == "user") // Anthropic: tool results are user messages

    let data = try JSONEncoder().encode(messages[0])
    let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let content = try #require(json["content"] as? [[String: Any]])

    #expect(content.count == 1)
    #expect(content[0]["type"] as? String == "tool_result")
    #expect(content[0]["tool_use_id"] as? String == "call_123")

    // Content should be an array of blocks
    let innerContent = try #require(content[0]["content"] as? [[String: Any]])
    #expect(innerContent.count == 2)
    #expect(innerContent[0]["type"] as? String == "text")
    #expect(innerContent[0]["text"] as? String == "Here is the screenshot")
    #expect(innerContent[1]["type"] as? String == "image")
}

// MARK: - Consecutive User Messages Merged

@Test func `anthropic encoding consecutive user messages merged`() {
    let msg1 = LLM.OpenAICompatibleAPI.ChatMessage(content: "First", role: .user)
    let msg2 = LLM.OpenAICompatibleAPI.ChatMessage(
        content: [.text("Second"), .image(data: Data(), mediaType: "image/jpeg")],
        role: .user,
    )

    let messages = LLM.OpenAICompatibleAPI.AnthropicMessageConverter.convert([msg1, msg2])

    #expect(messages.count == 1)
    #expect(messages[0].role == "user")
    // Should have 3 blocks: text("First"), text("Second"), image
}

// MARK: - Text-only Messages Still Work

@Test func `anthropic encoding text only message still works`() throws {
    let msg = LLM.OpenAICompatibleAPI.ChatMessage(content: "Hello", role: .user)
    let messages = LLM.OpenAICompatibleAPI.AnthropicMessageConverter.convert([msg])

    let data = try JSONEncoder().encode(messages[0])
    let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let content = try #require(json["content"] as? [[String: Any]])

    #expect(content.count == 1)
    #expect(content[0]["type"] as? String == "text")
    #expect(content[0]["text"] as? String == "Hello")
}
