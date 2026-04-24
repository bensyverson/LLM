//
//  ContentPartTests.swift
//  LLMTests
//
//  Tests for ContentPart enum, helpers, and convenience initializers
//

import Foundation
@testable import LLM
import Testing

// MARK: - Creation Tests

@Test func `content part text creation`() {
    let part = LLM.OpenAICompatibleAPI.ContentPart.text("Hello")
    #expect(part == .text("Hello"))
}

@Test func `content part image creation`() {
    let data = Data([0xFF, 0xD8, 0xFF, 0xE0])
    let part = LLM.OpenAICompatibleAPI.ContentPart.image(data: data, mediaType: "image/jpeg")
    if case let .image(d, mt, fn, desc) = part {
        #expect(d == data)
        #expect(mt == "image/jpeg")
        #expect(fn == nil)
        #expect(desc == nil)
    } else {
        Issue.record("Expected .image case")
    }
}

@Test func `content part image with filename and description`() {
    let data = Data([0xFF, 0xD8])
    let part = LLM.OpenAICompatibleAPI.ContentPart.image(
        data: data, mediaType: "image/jpeg",
        filename: "photo.jpg", description: "A sunset",
    )
    if case let .image(_, _, fn, desc) = part {
        #expect(fn == "photo.jpg")
        #expect(desc == "A sunset")
    } else {
        Issue.record("Expected .image case")
    }
}

@Test func `content part pdf creation`() {
    let data = Data([0x25, 0x50, 0x44, 0x46])
    let part = LLM.OpenAICompatibleAPI.ContentPart.pdf(data: data, title: "report.pdf")
    if case let .pdf(d, title) = part {
        #expect(d == data)
        #expect(title == "report.pdf")
    } else {
        Issue.record("Expected .pdf case")
    }
}

// MARK: - Equality Tests

@Test func `content part equality text`() {
    let a = LLM.OpenAICompatibleAPI.ContentPart.text("Hello")
    let b = LLM.OpenAICompatibleAPI.ContentPart.text("Hello")
    let c = LLM.OpenAICompatibleAPI.ContentPart.text("World")
    #expect(a == b)
    #expect(a != c)
}

@Test func `content part equality image`() {
    let data = Data([0xFF, 0xD8])
    let a = LLM.OpenAICompatibleAPI.ContentPart.image(data: data, mediaType: "image/jpeg")
    let b = LLM.OpenAICompatibleAPI.ContentPart.image(data: data, mediaType: "image/jpeg")
    let c = LLM.OpenAICompatibleAPI.ContentPart.image(data: data, mediaType: "image/png")
    #expect(a == b)
    #expect(a != c)
}

// MARK: - textContent Helper

@Test func `content part text content returns text for text part`() {
    let part = LLM.OpenAICompatibleAPI.ContentPart.text("Hello")
    #expect(part.textContent == "Hello")
}

@Test func `content part text content returns nil for image part`() {
    let part = LLM.OpenAICompatibleAPI.ContentPart.image(data: Data(), mediaType: "image/jpeg")
    #expect(part.textContent == nil)
}

@Test func `content part text content returns nil for pdf part`() {
    let part = LLM.OpenAICompatibleAPI.ContentPart.pdf(data: Data())
    #expect(part.textContent == nil)
}

// MARK: - isMedia Helper

@Test func `content part is media false for text`() {
    let part = LLM.OpenAICompatibleAPI.ContentPart.text("Hello")
    #expect(part.isMedia == false)
}

@Test func `content part is media true for image`() {
    let part = LLM.OpenAICompatibleAPI.ContentPart.image(data: Data(), mediaType: "image/jpeg")
    #expect(part.isMedia == true)
}

@Test func `content part is media true for pdf`() {
    let part = LLM.OpenAICompatibleAPI.ContentPart.pdf(data: Data())
    #expect(part.isMedia == true)
}

// MARK: - filename Helper

@Test func `content part filename returns filename for image`() {
    let part = LLM.OpenAICompatibleAPI.ContentPart.image(
        data: Data(), mediaType: "image/jpeg", filename: "photo.jpg",
    )
    #expect(part.filename == "photo.jpg")
}

@Test func `content part filename returns title for pdf`() {
    let part = LLM.OpenAICompatibleAPI.ContentPart.pdf(data: Data(), title: "report.pdf")
    #expect(part.filename == "report.pdf")
}

@Test func `content part filename returns nil for text`() {
    let part = LLM.OpenAICompatibleAPI.ContentPart.text("Hello")
    #expect(part.filename == nil)
}

@Test func `content part filename returns nil for image without filename`() {
    let part = LLM.OpenAICompatibleAPI.ContentPart.image(data: Data(), mediaType: "image/jpeg")
    #expect(part.filename == nil)
}

// MARK: - Media Type Inference

@Test func `content part infer media type jpeg`() {
    let data = Data([0xFF, 0xD8, 0xFF, 0xE0])
    #expect(LLM.OpenAICompatibleAPI.ContentPart.inferMediaType(from: data) == "image/jpeg")
}

@Test func `content part infer media type png`() {
    let data = Data([0x89, 0x50, 0x4E, 0x47])
    #expect(LLM.OpenAICompatibleAPI.ContentPart.inferMediaType(from: data) == "image/png")
}

@Test func `content part infer media type gif`() {
    let data = Data([0x47, 0x49, 0x46, 0x38])
    #expect(LLM.OpenAICompatibleAPI.ContentPart.inferMediaType(from: data) == "image/gif")
}

@Test func `content part infer media type webp`() {
    let data = Data([0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50])
    #expect(LLM.OpenAICompatibleAPI.ContentPart.inferMediaType(from: data) == "image/webp")
}

@Test func `content part infer media type unknown`() {
    let data = Data([0x00, 0x01, 0x02, 0x03])
    #expect(LLM.OpenAICompatibleAPI.ContentPart.inferMediaType(from: data) == nil)
}

@Test func `content part infer media type too short`() {
    let data = Data([0xFF])
    #expect(LLM.OpenAICompatibleAPI.ContentPart.inferMediaType(from: data) == nil)
}

// MARK: - Media Type from Extension

@Test func `content part media type for extension known types`() {
    #expect(LLM.OpenAICompatibleAPI.ContentPart.mediaType(forExtension: "jpg") == "image/jpeg")
    #expect(LLM.OpenAICompatibleAPI.ContentPart.mediaType(forExtension: "jpeg") == "image/jpeg")
    #expect(LLM.OpenAICompatibleAPI.ContentPart.mediaType(forExtension: "png") == "image/png")
    #expect(LLM.OpenAICompatibleAPI.ContentPart.mediaType(forExtension: "gif") == "image/gif")
    #expect(LLM.OpenAICompatibleAPI.ContentPart.mediaType(forExtension: "webp") == "image/webp")
    #expect(LLM.OpenAICompatibleAPI.ContentPart.mediaType(forExtension: "pdf") == "application/pdf")
}

@Test func `content part media type for extension case insensitive`() {
    #expect(LLM.OpenAICompatibleAPI.ContentPart.mediaType(forExtension: "JPG") == "image/jpeg")
    #expect(LLM.OpenAICompatibleAPI.ContentPart.mediaType(forExtension: "PNG") == "image/png")
}

@Test func `content part media type for extension unknown`() {
    #expect(LLM.OpenAICompatibleAPI.ContentPart.mediaType(forExtension: "xyz") == nil)
}

// MARK: - Data Convenience Init

@Test func `content part image from data infers jpeg`() throws {
    let data = Data([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10])
    let part = try LLM.OpenAICompatibleAPI.ContentPart.image(data: data, filename: "test.jpg")
    if case let .image(_, mt, fn, _) = part {
        #expect(mt == "image/jpeg")
        #expect(fn == "test.jpg")
    } else {
        Issue.record("Expected .image case")
    }
}

@Test func `content part image from data throws for unknown`() {
    let data = Data([0x00, 0x01, 0x02, 0x03])
    #expect(throws: LLM.OpenAICompatibleAPI.ContentPartError.self) {
        _ = try LLM.OpenAICompatibleAPI.ContentPart.image(data: data)
    }
}

// MARK: - ChatMessage Multimodal Helpers

@Test func `chat message text content single text`() {
    let msg = LLM.OpenAICompatibleAPI.ChatMessage(content: "Hello", role: .user)
    #expect(msg.textContent == "Hello")
}

@Test func `chat message text content multiple text parts`() {
    let msg = LLM.OpenAICompatibleAPI.ChatMessage(
        content: [.text("Hello "), .text("world")],
        role: .user,
    )
    #expect(msg.textContent == "Hello world")
}

@Test func `chat message text content nil for empty`() {
    let msg = LLM.OpenAICompatibleAPI.ChatMessage(content: nil, role: .assistant)
    #expect(msg.textContent == nil)
}

@Test func `chat message text content nil for media only`() {
    let msg = LLM.OpenAICompatibleAPI.ChatMessage(
        content: [.image(data: Data(), mediaType: "image/jpeg")],
        role: .user,
    )
    #expect(msg.textContent == nil)
}

@Test func `chat message has media false for text only`() {
    let msg = LLM.OpenAICompatibleAPI.ChatMessage(content: "Hello", role: .user)
    #expect(msg.hasMedia == false)
}

@Test func `chat message has media true for image`() {
    let msg = LLM.OpenAICompatibleAPI.ChatMessage(
        content: [.text("Describe this"), .image(data: Data(), mediaType: "image/jpeg")],
        role: .user,
    )
    #expect(msg.hasMedia == true)
}

@Test func `chat message content length sums text parts`() {
    let msg = LLM.OpenAICompatibleAPI.ChatMessage(
        content: [.text("Hello"), .image(data: Data(), mediaType: "image/jpeg"), .text(" world")],
        role: .user,
    )
    #expect(msg.contentLength == 11)
}
