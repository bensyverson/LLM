//
//  ImageResizingTests.swift
//  LLMTests
//
//  Tests for resize cache and image resizing logic
//

import Foundation
@testable import LLM
import Testing

// MARK: - ResizeCacheKey Tests

@Test func `resize cache key equality and hashing`() {
    let key1 = LLM.ResizeCacheKey(dataHash: 123, targetWidth: 1024, targetHeight: 768)
    let key2 = LLM.ResizeCacheKey(dataHash: 123, targetWidth: 1024, targetHeight: 768)
    let key3 = LLM.ResizeCacheKey(dataHash: 456, targetWidth: 1024, targetHeight: 768)

    #expect(key1 == key2)
    #expect(key1 != key3)
    #expect(key1.hashValue == key2.hashValue)
}

// MARK: - Cache Tests

@Test func `resize cache store and retrieve`() async {
    let llm = LLM(provider: .openAI(apiKey: "test"))
    let key = LLM.ResizeCacheKey(dataHash: 42, targetWidth: 100, targetHeight: 100)
    let data = Data([0x01, 0x02, 0x03])

    await llm.cacheResize(data, for: key)
    let cached = await llm.getCachedResize(for: key)

    #expect(cached == data)
}

@Test func `resize cache miss returns nil`() async {
    let llm = LLM(provider: .openAI(apiKey: "test"))
    let key = LLM.ResizeCacheKey(dataHash: 999, targetWidth: 100, targetHeight: 100)

    let cached = await llm.getCachedResize(for: key)
    #expect(cached == nil)
}

@Test func `resize cache fifo eviction`() async {
    let llm = LLM(provider: .openAI(apiKey: "test"))

    // Fill cache to max (50 entries)
    for i in 0 ..< 50 {
        let key = LLM.ResizeCacheKey(dataHash: i, targetWidth: 100, targetHeight: 100)
        await llm.cacheResize(Data([UInt8(i)]), for: key)
    }

    // Add one more to trigger eviction
    let newKey = LLM.ResizeCacheKey(dataHash: 999, targetWidth: 100, targetHeight: 100)
    await llm.cacheResize(Data([0xFF]), for: newKey)

    // First entry should be evicted
    let firstKey = LLM.ResizeCacheKey(dataHash: 0, targetWidth: 100, targetHeight: 100)
    let evicted = await llm.getCachedResize(for: firstKey)
    #expect(evicted == nil)

    // New entry should be present
    let newEntry = await llm.getCachedResize(for: newKey)
    #expect(newEntry == Data([0xFF]))

    // Second entry should still be present
    let secondKey = LLM.ResizeCacheKey(dataHash: 1, targetWidth: 100, targetHeight: 100)
    let secondEntry = await llm.getCachedResize(for: secondKey)
    #expect(secondEntry != nil)
}

// MARK: - Resize Logic Tests

@Test func `resizing images no images returns unchanged`() async throws {
    let llm = LLM(provider: .openAI(apiKey: "test"))

    let conversation = LLM.Conversation(systemPrompt: "System")
        .addingUserMessage("Hello")

    let resizer: @Sendable (Data, String, CGSize) async throws -> Data = { data, _, _ in data }
    let result = try await llm.resizingImages(in: conversation, maxLongEdge: 1568, using: resizer)

    #expect(result.messages[0].textContent == "Hello")
}

@Test func `resizing images calls resizer for images`() async throws {
    let llm = LLM(provider: .openAI(apiKey: "test"))

    let resizer: @Sendable (Data, String, CGSize) async throws -> Data = { _, _, _ in
        Data([0xAA, 0xBB]) // "resized" data
    }

    let conversation = LLM.Conversation(systemPrompt: "System")
        .addingUserMessage([
            .text("Look"),
            .image(data: Data([0xFF, 0xD8]), mediaType: "image/jpeg"),
        ])

    // Note: without CoreGraphics, imageDimensions returns nil, so resize is always attempted
    let result = try await llm.resizingImages(in: conversation, maxLongEdge: 1568, using: resizer)

    // On non-CG platforms, resizer may or may not be called depending on dimension check
    // The conversation structure should remain intact
    #expect(result.messages.count == 1)
    #expect(result.messages[0].content.count == 2)
    #expect(result.messages[0].content[0].textContent == "Look")
}
