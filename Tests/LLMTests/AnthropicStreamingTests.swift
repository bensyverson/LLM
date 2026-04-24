//
//  AnthropicStreamingTests.swift
//  LLMTests
//
//  Regression tests for Anthropic streaming event processing.
//

import Foundation
@testable import LLM
import Testing

@Suite("Anthropic Streaming")
struct AnthropicStreamingTests {
    // MARK: - Helpers

    private func decodeEvent(_ json: String) throws -> LLM.OpenAICompatibleAPI.AnthropicStreamEvent {
        let decoder = JSONDecoder()
        guard let data = json.data(using: .utf8) else {
            struct DecodeError: Error {}
            throw DecodeError()
        }
        return try decoder.decode(LLM.OpenAICompatibleAPI.AnthropicStreamEvent.self, from: data)
    }

    // MARK: - Tests

    /// Full happy-path sequence: message_start → content_block_start (text) →
    /// content_block_delta × N → message_delta (stop_reason + output_tokens) →
    /// message_stop.
    @Test func `full conversation assembles text`() throws {
        let events = [
            #"{"type":"message_start","message":{"id":"msg_01","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":25,"output_tokens":0}}}"#,
            #"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            #"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
            #"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":", world"}}"#,
            #"{"type":"content_block_stop","index":0}"#,
            #"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":12}}"#,
            #"{"type":"message_stop"}"#,
        ]

        var accumulator = LLM.OpenAICompatibleAPI.StreamAccumulator()
        for json in events {
            let event = try decodeEvent(json)
            accumulator.processAnthropicEvent(event)
        }

        #expect(accumulator.text == "Hello, world")
        #expect(accumulator.outputTokens == 12)
        #expect(accumulator.finishReason == "end_turn")
    }

    /// Extended thinking: a thinking block followed by a text block. Both
    /// accumulator fields should be populated independently.
    @Test func `thinking block then text`() throws {
        let events = [
            #"{"type":"message_start","message":{"id":"msg_02","model":"claude-3-7-sonnet-20250219","usage":{"input_tokens":30,"output_tokens":0}}}"#,
            #"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","text":""}}"#,
            #"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me reason about this."}}"#,
            #"{"type":"content_block_stop","index":0}"#,
            #"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
            #"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"The answer is 4."}}"#,
            #"{"type":"content_block_stop","index":1}"#,
            #"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}"#,
            #"{"type":"message_stop"}"#,
        ]

        var accumulator = LLM.OpenAICompatibleAPI.StreamAccumulator()
        for json in events {
            let event = try decodeEvent(json)
            accumulator.processAnthropicEvent(event)
        }

        #expect(accumulator.thinking == "Let me reason about this.")
        #expect(accumulator.text == "The answer is 4.")
    }

    /// Tool-use sequence: content_block_start identifies the tool call (id + name),
    /// then input_json_delta events stream the argument JSON fragments.
    @Test func `tool use assembles arguments`() throws {
        let events = [
            #"{"type":"message_start","message":{"id":"msg_03","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":40,"output_tokens":0}}}"#,
            #"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather"}}"#,
            #"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"loc"}}"#,
            #"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"ation\":\"London\"}"}}"#,
            #"{"type":"content_block_stop","index":0}"#,
            #"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":8}}"#,
            #"{"type":"message_stop"}"#,
        ]

        var accumulator = LLM.OpenAICompatibleAPI.StreamAccumulator()
        for json in events {
            let event = try decodeEvent(json)
            accumulator.processAnthropicEvent(event)
        }

        let toolCall = accumulator.toolCalls[0]
        #expect(toolCall != nil)
        #expect(toolCall?.id == "toolu_01")
        #expect(toolCall?.name == "get_weather")
        #expect(toolCall?.arguments == #"{"location":"London"}"#)
    }
}
