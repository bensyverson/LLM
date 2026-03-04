//
//  MultimodalEncodingTests.swift
//  LLMTests
//
//  Tests for OpenAI-format encoding of multimodal ChatMessages
//

import Foundation
@testable import LLM
import Testing

// MARK: - OpenAI Encoding Tests

@Test func openAIEncoding_textOnlyMessage_encodesAsString() throws {
    let msg = LLM.OpenAICompatibleAPI.ChatMessage(content: "Hello", role: .user)

    let data = try JSONEncoder().encode(msg)
    let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])

    // Text-only should encode content as a plain string
    #expect(json["content"] as? String == "Hello")
    #expect(json["role"] as? String == "user")
}

@Test func openAIEncoding_nilContent_encodesAsNull() throws {
    let msg = LLM.OpenAICompatibleAPI.ChatMessage(content: nil, role: .assistant, tool_calls: [])

    let data = try JSONEncoder().encode(msg)
    let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])

    // nil content → JSON null
    #expect(json["content"] is NSNull)
}

@Test func openAIEncoding_multimodalMessage_encodesAsArray() throws {
    let imageData = Data([0xFF, 0xD8, 0xFF, 0xE0])
    let msg = LLM.OpenAICompatibleAPI.ChatMessage(
        content: [
            .text("Describe this image"),
            .image(data: imageData, mediaType: "image/jpeg"),
        ],
        role: .user
    )

    let data = try JSONEncoder().encode(msg)
    let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])

    // Multimodal should encode content as an array
    let contentArray = try #require(json["content"] as? [[String: Any]])
    #expect(contentArray.count == 2)

    // First part: text
    #expect(contentArray[0]["type"] as? String == "text")
    #expect(contentArray[0]["text"] as? String == "Describe this image")

    // Second part: image_url with data URI
    #expect(contentArray[1]["type"] as? String == "image_url")
    let imageURL = try #require(contentArray[1]["image_url"] as? [String: Any])
    let url = try #require(imageURL["url"] as? String)
    #expect(url.hasPrefix("data:image/jpeg;base64,"))
    #expect(imageURL["detail"] as? String == "auto")
}

@Test func openAIEncoding_imageDataURI_correctBase64() throws {
    let imageData = Data([0x01, 0x02, 0x03])
    let msg = LLM.OpenAICompatibleAPI.ChatMessage(
        content: [.image(data: imageData, mediaType: "image/png")],
        role: .user
    )

    let data = try JSONEncoder().encode(msg)
    let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let contentArray = try #require(json["content"] as? [[String: Any]])
    let imageURL = try #require(contentArray[0]["image_url"] as? [String: Any])
    let url = try #require(imageURL["url"] as? String)

    let expectedBase64 = imageData.base64EncodedString()
    #expect(url == "data:image/png;base64,\(expectedBase64)")
}

@Test func openAIEncoding_pdfContent_encodedAsTextPlaceholder() throws {
    let msg = LLM.OpenAICompatibleAPI.ChatMessage(
        content: [.pdf(data: Data([0x25, 0x50]), title: "doc.pdf")],
        role: .user
    )

    let data = try JSONEncoder().encode(msg)
    let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let contentArray = try #require(json["content"] as? [[String: Any]])

    #expect(contentArray[0]["type"] as? String == "text")
    #expect(contentArray[0]["text"] as? String == "[Unsupported: PDF content]")
}

// MARK: - Decoding Tests

@Test func openAIDecoding_stringContent_decodesCorrectly() throws {
    let jsonStr = #"{"content":"Hello","role":"user"}"#
    let msg = try JSONDecoder().decode(
        LLM.OpenAICompatibleAPI.ChatMessage.self,
        from: Data(jsonStr.utf8)
    )

    #expect(msg.textContent == "Hello")
    #expect(msg.content.count == 1)
    #expect(msg.role == .user)
}

@Test func openAIDecoding_arrayContent_decodesTextParts() throws {
    let jsonStr = #"{"content":[{"type":"text","text":"Hello"},{"type":"text","text":" world"}],"role":"user"}"#
    let msg = try JSONDecoder().decode(
        LLM.OpenAICompatibleAPI.ChatMessage.self,
        from: Data(jsonStr.utf8)
    )

    #expect(msg.content.count == 2)
    #expect(msg.textContent == "Hello world")
}

@Test func openAIDecoding_nullContent_decodesAsEmpty() throws {
    let jsonStr = #"{"content":null,"role":"assistant"}"#
    let msg = try JSONDecoder().decode(
        LLM.OpenAICompatibleAPI.ChatMessage.self,
        from: Data(jsonStr.utf8)
    )

    #expect(msg.content.isEmpty)
    #expect(msg.textContent == nil)
}

@Test func openAIDecoding_imageURLContent_decodesAsImage() throws {
    let base64 = Data([0xFF, 0xD8]).base64EncodedString()
    let jsonStr = """
    {"content":[{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,\(base64)","detail":"auto"}}],"role":"user"}
    """
    let msg = try JSONDecoder().decode(
        LLM.OpenAICompatibleAPI.ChatMessage.self,
        from: Data(jsonStr.utf8)
    )

    #expect(msg.content.count == 1)
    if case let .image(data, mediaType, _, _) = msg.content[0] {
        #expect(mediaType == "image/jpeg")
        #expect(data == Data([0xFF, 0xD8]))
    } else {
        Issue.record("Expected .image case")
    }
}

// MARK: - Round-trip Tests

@Test func openAIEncoding_textOnlyRoundTrip() throws {
    let original = LLM.OpenAICompatibleAPI.ChatMessage(content: "Hello", role: .user, name: "test")
    let data = try JSONEncoder().encode(original)
    let decoded = try JSONDecoder().decode(LLM.OpenAICompatibleAPI.ChatMessage.self, from: data)

    #expect(decoded.textContent == original.textContent)
    #expect(decoded.role == original.role)
    #expect(decoded.name == original.name)
}

@Test func openAIEncoding_multimodalRoundTrip() throws {
    let imageData = Data([0xFF, 0xD8, 0xFF, 0xE0])
    let original = LLM.OpenAICompatibleAPI.ChatMessage(
        content: [
            .text("Look at this"),
            .image(data: imageData, mediaType: "image/jpeg", filename: "test.jpg", description: "A test"),
        ],
        role: .user
    )

    let data = try JSONEncoder().encode(original)
    let decoded = try JSONDecoder().decode(LLM.OpenAICompatibleAPI.ChatMessage.self, from: data)

    #expect(decoded.content.count == 2)
    #expect(decoded.content[0].textContent == "Look at this")
    if case let .image(d, mt, _, _) = decoded.content[1] {
        #expect(d == imageData)
        #expect(mt == "image/jpeg")
    } else {
        Issue.record("Expected .image case after round-trip")
    }
}
