//
//  SSEParserTests.swift
//  LLMTests
//
//  Tests for the SSE (Server-Sent Events) parser
//

import Foundation
@testable import LLM
import Testing

// MARK: - SSE Parser Tests

@Test func `sse parser parses data lines`() async throws {
    let lines = [
        "data: hello world",
        "",
        "data: second event",
        "",
    ]
    let events = try await parseSSE(lines)
    #expect(events.count == 2)
    #expect(events[0].data == "hello world")
    #expect(events[0].event == nil)
    #expect(events[1].data == "second event")
}

@Test func `sse parser parses event and data`() async throws {
    let lines = [
        "event: message_start",
        "data: {\"type\":\"message_start\"}",
        "",
        "event: content_block_delta",
        "data: {\"type\":\"content_block_delta\"}",
        "",
    ]
    let events = try await parseSSE(lines)
    #expect(events.count == 2)
    #expect(events[0].event == "message_start")
    #expect(events[0].data == "{\"type\":\"message_start\"}")
    #expect(events[1].event == "content_block_delta")
}

@Test func `sse parser skips comments`() async throws {
    let lines = [
        ": this is a comment",
        "data: actual data",
        "",
    ]
    let events = try await parseSSE(lines)
    #expect(events.count == 1)
    #expect(events[0].data == "actual data")
}

@Test func `sse parser handles done signal`() async throws {
    let lines = [
        "data: {\"content\":\"hello\"}",
        "",
        "data: [DONE]",
        "",
    ]
    let events = try await parseSSE(lines)
    #expect(events.count == 1)
    #expect(events[0].data == "{\"content\":\"hello\"}")
}

@Test func `sse parser handles multi line data`() async throws {
    let lines = [
        "data: line one",
        "data: line two",
        "",
    ]
    let events = try await parseSSE(lines)
    #expect(events.count == 1)
    #expect(events[0].data == "line one\nline two")
}

@Test func `sse parser skips empty data blocks`() async throws {
    let lines = [
        "",
        "",
        "data: after empty lines",
        "",
    ]
    let events = try await parseSSE(lines)
    #expect(events.count == 1)
    #expect(events[0].data == "after empty lines")
}

@Test func `sse parser ignores id and retry fields`() async throws {
    let lines = [
        "id: 123",
        "retry: 5000",
        "data: the data",
        "",
    ]
    let events = try await parseSSE(lines)
    #expect(events.count == 1)
    #expect(events[0].data == "the data")
}

@Test func `sse parser handles trailing data without blank line`() async throws {
    let lines = [
        "data: trailing",
    ]
    let events = try await parseSSE(lines)
    #expect(events.count == 1)
    #expect(events[0].data == "trailing")
}

@Test func `sse parser done in trailing position`() async throws {
    let lines = [
        "data: [DONE]",
    ]
    let events = try await parseSSE(lines)
    #expect(events.count == 0)
}

@Test func `sse parser multiple events with done`() async throws {
    let lines = [
        "data: {\"id\":\"1\"}",
        "",
        "data: {\"id\":\"2\"}",
        "",
        "data: {\"id\":\"3\"}",
        "",
        "data: [DONE]",
        "",
    ]
    let events = try await parseSSE(lines)
    #expect(events.count == 3)
    #expect(events[0].data == "{\"id\":\"1\"}")
    #expect(events[1].data == "{\"id\":\"2\"}")
    #expect(events[2].data == "{\"id\":\"3\"}")
}

@Test func `sse parser event field resets after dispatch`() async throws {
    let lines = [
        "event: first_type",
        "data: first data",
        "",
        "data: second data",
        "",
    ]
    let events = try await parseSSE(lines)
    #expect(events.count == 2)
    #expect(events[0].event == "first_type")
    #expect(events[1].event == nil)
}

// MARK: - Helpers

/// Parse SSE from an array of lines using an AsyncStream
private func parseSSE(_ inputLines: [String]) async throws -> [LLM.OpenAICompatibleAPI.SSEEvent] {
    let stream = AsyncStream<String> { continuation in
        for line in inputLines {
            continuation.yield(line)
        }
        continuation.finish()
    }
    let parser = LLM.OpenAICompatibleAPI.SSEParser(lines: stream)
    var events: [LLM.OpenAICompatibleAPI.SSEEvent] = []
    for try await event in parser {
        events.append(event)
    }
    return events
}
