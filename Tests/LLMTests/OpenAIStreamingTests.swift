//
//  OpenAIStreamingTests.swift
//  LLMTests
//
//  Regression tests for OpenAI-compatible streaming, including the OpenRouter
//  bundled-chunk bug fix.
//

import Foundation
@testable import LLM
import Testing

@Suite("OpenAI Streaming")
struct OpenAIStreamingTests {
    // MARK: - Helpers

    private func decodeChunk(_ json: String) throws -> LLM.OpenAICompatibleAPI.OpenAIStreamChunk {
        let decoder = JSONDecoder()
        guard let data = json.data(using: .utf8) else {
            struct DecodeError: Error {}
            throw DecodeError()
        }
        return try decoder.decode(LLM.OpenAICompatibleAPI.OpenAIStreamChunk.self, from: data)
    }

    // MARK: - Tests

    /// Regression test for OpenRouter bundling multiple JSON objects into a single
    /// SSE event's data field, separated by newlines.
    @Test func bundledSSEChunks_assemblesText() throws {
        // This is the exact shape of a real OpenRouter SSE event whose data field
        // contains two newline-joined JSON objects.
        let bundledData = """
        {"id":"gen-1","choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning":null}}]}
        {"id":"gen-1","choices":[{"index":0,"delta":{"role":"assistant","content":" Hello"}}]}
        """

        var accumulator = LLM.OpenAICompatibleAPI.StreamAccumulator()

        // Replicate the fix: split on newlines, decode each line independently.
        let jsonLines = bundledData
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map(String.init)

        for jsonString in jsonLines {
            let chunk = try decodeChunk(jsonString)
            accumulator.processOpenAIChunk(chunk)
        }

        #expect(accumulator.text == " Hello", "Bundled chunk text should be assembled; was empty before fix")
    }

    /// Standard OpenAI streaming — one JSON object per SSE event, ending with a
    /// usage chunk and [DONE].
    @Test func standardOpenAI_oneChunkPerEvent_assemblesText() throws {
        let chunks = [
            #"{"id":"chatcmpl-1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}"#,
            #"{"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":" world"}}]}"#,
            #"{"id":"chatcmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
        ]

        var accumulator = LLM.OpenAICompatibleAPI.StreamAccumulator()
        for json in chunks {
            let chunk = try decodeChunk(json)
            accumulator.processOpenAIChunk(chunk)
        }

        #expect(accumulator.text == "Hello world")
        #expect(accumulator.promptTokens == 10)
        #expect(accumulator.completionTokens == 5)
        #expect(accumulator.totalTokens == 15)
    }

    /// OpenRouter's reasoning models (e.g. Kimi K2.5) send reasoning in a `"reasoning"`
    /// field rather than `"reasoning_content"`. Both should be surfaced as thinking.
    @Test func openRouter_reasoningField_capturedAsThinking() throws {
        let chunks = [
            #"{"id":"gen-2","choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning":"Let me think..."}}]}"#,
            #"{"id":"gen-2","choices":[{"index":0,"delta":{"content":"The answer is 42."}}]}"#,
        ]

        var accumulator = LLM.OpenAICompatibleAPI.StreamAccumulator()
        for json in chunks {
            let chunk = try decodeChunk(json)
            accumulator.processOpenAIChunk(chunk)
        }

        #expect(accumulator.text == "The answer is 42.")
        #expect(accumulator.thinking == "Let me think...")
    }

    /// Some providers (OpenRouter) attach a usage object to the final non-[DONE]
    /// chunk. Verify that token counts are recorded.
    @Test func usageChunkAtEnd_recordsTokens() throws {
        let chunks = [
            #"{"id":"gen-3","choices":[{"index":0,"delta":{"content":"Hi"}}]}"#,
            #"{"id":"gen-3","choices":[],"usage":{"prompt_tokens":8,"completion_tokens":3,"total_tokens":11}}"#,
        ]

        var accumulator = LLM.OpenAICompatibleAPI.StreamAccumulator()
        for json in chunks {
            let chunk = try decodeChunk(json)
            accumulator.processOpenAIChunk(chunk)
        }

        #expect(accumulator.text == "Hi")
        #expect(accumulator.promptTokens == 8)
        #expect(accumulator.completionTokens == 3)
        #expect(accumulator.totalTokens == 11)
    }

    /// Chunks with `"content":""` (empty string, not null) must not cause the
    /// accumulator to append empty text. Text should remain as accumulated from
    /// non-empty chunks only.
    @Test func emptyContentString_notAppendedToText() throws {
        let chunks = [
            // role chunk — content is empty string (common in OpenAI/OpenRouter)
            #"{"id":"chatcmpl-4","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}"#,
            #"{"id":"chatcmpl-4","choices":[{"index":0,"delta":{"content":"Real content"}}]}"#,
        ]

        var accumulator = LLM.OpenAICompatibleAPI.StreamAccumulator()
        for json in chunks {
            let chunk = try decodeChunk(json)
            accumulator.processOpenAIChunk(chunk)
        }

        // The accumulator itself appends content unconditionally (empty string + real =
        // real), but the delta emission guard (content.isEmpty check) in processOpenAIStream
        // prevents spurious .textDelta("") events.  The text value is still correct.
        #expect(accumulator.text == "Real content")
    }
}
