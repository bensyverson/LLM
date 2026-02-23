//
//  StreamingTypeTests.swift
//  LLMTests
//
//  Tests for streaming chunk decoding and stream accumulator
//

import Foundation
@testable import LLM
import Testing

// MARK: - OpenAI Stream Chunk Decoding

@Test func openAIStreamChunk_decodesTextDelta() throws {
    let json = """
    {"id":"chatcmpl-abc","object":"chat.completion.chunk","model":"gpt-5.2","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}
    """.data(using: .utf8)!

    let chunk = try JSONDecoder().decode(LLM.OpenAICompatibleAPI.OpenAIStreamChunk.self, from: json)
    #expect(chunk.id == "chatcmpl-abc")
    #expect(chunk.model == "gpt-5.2")
    #expect(chunk.choices?.first?.delta.content == "Hello")
    #expect(chunk.choices?.first?.finish_reason == nil)
}

@Test func openAIStreamChunk_decodesReasoningContent() throws {
    let json = """
    {"id":"chatcmpl-r1","object":"chat.completion.chunk","model":"gpt-5.2","choices":[{"index":0,"delta":{"reasoning_content":"thinking..."},"finish_reason":null}]}
    """.data(using: .utf8)!

    let chunk = try JSONDecoder().decode(LLM.OpenAICompatibleAPI.OpenAIStreamChunk.self, from: json)
    #expect(chunk.choices?.first?.delta.reasoning_content == "thinking...")
}

@Test func openAIStreamChunk_decodesToolCallDelta() throws {
    let json = """
    {"id":"chatcmpl-tc","object":"chat.completion.chunk","model":"gpt-5.2","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}
    """.data(using: .utf8)!

    let chunk = try JSONDecoder().decode(LLM.OpenAICompatibleAPI.OpenAIStreamChunk.self, from: json)
    let tc = try #require(chunk.choices?.first?.delta.tool_calls?.first)
    #expect(tc.index == 0)
    #expect(tc.id == "call_123")
    #expect(tc.function?.name == "get_weather")
}

@Test func openAIStreamChunk_decodesUsage() throws {
    let json = """
    {"id":"chatcmpl-u","object":"chat.completion.chunk","model":"gpt-5.2","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}
    """.data(using: .utf8)!

    let chunk = try JSONDecoder().decode(LLM.OpenAICompatibleAPI.OpenAIStreamChunk.self, from: json)
    #expect(chunk.usage?.prompt_tokens == 10)
    #expect(chunk.usage?.completion_tokens == 5)
    #expect(chunk.usage?.total_tokens == 15)
}

@Test func openAIStreamChunk_decodesFinishReason() throws {
    let json = """
    {"id":"chatcmpl-f","object":"chat.completion.chunk","model":"gpt-5.2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
    """.data(using: .utf8)!

    let chunk = try JSONDecoder().decode(LLM.OpenAICompatibleAPI.OpenAIStreamChunk.self, from: json)
    #expect(chunk.choices?.first?.finish_reason == "stop")
}

// MARK: - Anthropic Stream Event Decoding

@Test func anthropicStreamEvent_decodesMessageStart() throws {
    let json = """
    {"type":"message_start","message":{"id":"msg_abc","model":"claude-opus-4-5","usage":{"input_tokens":12}}}
    """.data(using: .utf8)!

    let event = try JSONDecoder().decode(LLM.OpenAICompatibleAPI.AnthropicStreamEvent.self, from: json)
    #expect(event.type == "message_start")
    #expect(event.message?.id == "msg_abc")
    #expect(event.message?.model == "claude-opus-4-5")
    #expect(event.message?.usage?.input_tokens == 12)
}

@Test func anthropicStreamEvent_decodesContentBlockStart() throws {
    let json = """
    {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}
    """.data(using: .utf8)!

    let event = try JSONDecoder().decode(LLM.OpenAICompatibleAPI.AnthropicStreamEvent.self, from: json)
    #expect(event.type == "content_block_start")
    #expect(event.index == 0)
    #expect(event.content_block?.type == "text")
}

@Test func anthropicStreamEvent_decodesToolUseBlockStart() throws {
    let json = """
    {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_abc","name":"get_weather"}}
    """.data(using: .utf8)!

    let event = try JSONDecoder().decode(LLM.OpenAICompatibleAPI.AnthropicStreamEvent.self, from: json)
    #expect(event.content_block?.type == "tool_use")
    #expect(event.content_block?.id == "toolu_abc")
    #expect(event.content_block?.name == "get_weather")
}

@Test func anthropicStreamEvent_decodesTextDelta() throws {
    let json = """
    {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}
    """.data(using: .utf8)!

    let event = try JSONDecoder().decode(LLM.OpenAICompatibleAPI.AnthropicStreamEvent.self, from: json)
    #expect(event.type == "content_block_delta")
    #expect(event.delta?.type == "text_delta")
    #expect(event.delta?.text == "Hello")
}

@Test func anthropicStreamEvent_decodesThinkingDelta() throws {
    let json = """
    {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think..."}}
    """.data(using: .utf8)!

    let event = try JSONDecoder().decode(LLM.OpenAICompatibleAPI.AnthropicStreamEvent.self, from: json)
    #expect(event.delta?.type == "thinking_delta")
    #expect(event.delta?.thinking == "Let me think...")
}

@Test func anthropicStreamEvent_decodesInputJsonDelta() throws {
    let json = """
    {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\\"loc"}}
    """.data(using: .utf8)!

    let event = try JSONDecoder().decode(LLM.OpenAICompatibleAPI.AnthropicStreamEvent.self, from: json)
    #expect(event.delta?.type == "input_json_delta")
    #expect(event.delta?.partial_json == "{\"loc")
}

@Test func anthropicStreamEvent_decodesMessageDelta() throws {
    let json = """
    {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}
    """.data(using: .utf8)!

    let event = try JSONDecoder().decode(LLM.OpenAICompatibleAPI.AnthropicStreamEvent.self, from: json)
    #expect(event.type == "message_delta")
    #expect(event.delta?.stop_reason == "end_turn")
    #expect(event.usage?.output_tokens == 15)
}

// MARK: - Stream Accumulator Tests

@Test func accumulator_openAI_textOnly() throws {
    var acc = LLM.OpenAICompatibleAPI.StreamAccumulator()
    let decoder = JSONDecoder()

    let chunks = [
        """
        {"id":"chatcmpl-1","model":"gpt-5.2","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}
        """,
        """
        {"id":"chatcmpl-1","model":"gpt-5.2","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}
        """,
        """
        {"id":"chatcmpl-1","model":"gpt-5.2","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}
        """,
        """
        {"id":"chatcmpl-1","model":"gpt-5.2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
        """,
        """
        {"id":"chatcmpl-1","model":"gpt-5.2","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}
        """,
    ]

    for chunkJSON in chunks {
        let chunk = try decoder.decode(LLM.OpenAICompatibleAPI.OpenAIStreamChunk.self, from: #require(chunkJSON.data(using: .utf8)))
        acc.processOpenAIChunk(chunk)
    }

    #expect(acc.text == "Hello world")
    #expect(acc.finishReason == "stop")
    #expect(acc.promptTokens == 10)
    #expect(acc.completionTokens == 2)

    let response = acc.buildResponse(isAnthropic: false)
    #expect(response.choices?.first?.message.content == "Hello world")
    #expect(response.choices?.first?.finish_reason == "stop")
    #expect(response.usage.prompt_tokens == 10)
    #expect(response.usage.completion_tokens == 2)
    #expect(response.id == "chatcmpl-1")
    #expect(response.model == "gpt-5.2")
}

@Test func accumulator_openAI_toolCalls() throws {
    var acc = LLM.OpenAICompatibleAPI.StreamAccumulator()
    let decoder = JSONDecoder()

    let chunks = [
        """
        {"id":"tc1","model":"gpt-5.2","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}
        """,
        """
        {"id":"tc1","model":"gpt-5.2","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"loc"}}]},"finish_reason":null}]}
        """,
        """
        {"id":"tc1","model":"gpt-5.2","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\\":\\"NYC\\"}"}}]},"finish_reason":null}]}
        """,
        """
        {"id":"tc1","model":"gpt-5.2","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}
        """,
    ]

    for chunkJSON in chunks {
        let chunk = try decoder.decode(LLM.OpenAICompatibleAPI.OpenAIStreamChunk.self, from: #require(chunkJSON.data(using: .utf8)))
        acc.processOpenAIChunk(chunk)
    }

    #expect(acc.toolCalls.count == 1)
    #expect(acc.toolCalls[0]?.id == "call_1")
    #expect(acc.toolCalls[0]?.name == "get_weather")
    #expect(acc.toolCalls[0]?.arguments == "{\"location\":\"NYC\"}")

    let response = acc.buildResponse(isAnthropic: false)
    let toolCall = try #require(response.choices?.first?.message.tool_calls?.first)
    #expect(toolCall.id == "call_1")
    #expect(toolCall.function.name == "get_weather")
    #expect(toolCall.function.arguments == "{\"location\":\"NYC\"}")
}

@Test func accumulator_openAI_reasoning() throws {
    var acc = LLM.OpenAICompatibleAPI.StreamAccumulator()
    let decoder = JSONDecoder()

    let chunks = [
        """
        {"id":"r1","model":"gpt-5.2","choices":[{"index":0,"delta":{"reasoning_content":"Think"},"finish_reason":null}]}
        """,
        """
        {"id":"r1","model":"gpt-5.2","choices":[{"index":0,"delta":{"reasoning_content":"ing..."},"finish_reason":null}]}
        """,
        """
        {"id":"r1","model":"gpt-5.2","choices":[{"index":0,"delta":{"content":"Answer"},"finish_reason":null}]}
        """,
        """
        {"id":"r1","model":"gpt-5.2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
        """,
    ]

    for chunkJSON in chunks {
        let chunk = try decoder.decode(LLM.OpenAICompatibleAPI.OpenAIStreamChunk.self, from: #require(chunkJSON.data(using: .utf8)))
        acc.processOpenAIChunk(chunk)
    }

    #expect(acc.thinking == "Thinking...")
    #expect(acc.text == "Answer")

    let response = acc.buildResponse(isAnthropic: false)
    #expect(response.choices?.first?.message.reasoning_content == "Thinking...")
    #expect(response.choices?.first?.message.content == "Answer")
}

@Test func accumulator_anthropic_textOnly() throws {
    var acc = LLM.OpenAICompatibleAPI.StreamAccumulator()
    let decoder = JSONDecoder()

    let events = [
        """
        {"type":"message_start","message":{"id":"msg_1","model":"claude-opus-4-5","usage":{"input_tokens":12}}}
        """,
        """
        {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}
        """,
        """
        {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}
        """,
        """
        {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}
        """,
        """
        {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}
        """,
    ]

    for eventJSON in events {
        let event = try decoder.decode(LLM.OpenAICompatibleAPI.AnthropicStreamEvent.self, from: #require(eventJSON.data(using: .utf8)))
        acc.processAnthropicEvent(event)
    }

    #expect(acc.text == "Hello world")
    #expect(acc.id == "msg_1")
    #expect(acc.inputTokens == 12)
    #expect(acc.outputTokens == 5)

    let response = acc.buildResponse(isAnthropic: true)
    let textContent = try #require(response.content?.first(where: { $0.type == .text }))
    #expect(textContent.text == "Hello world")
    #expect(response.usage.input_tokens == 12)
    #expect(response.usage.output_tokens == 5)
}

@Test func accumulator_anthropic_thinking() throws {
    var acc = LLM.OpenAICompatibleAPI.StreamAccumulator()
    let decoder = JSONDecoder()

    let events = [
        """
        {"type":"message_start","message":{"id":"msg_t","model":"claude-sonnet-4-5","usage":{"input_tokens":10}}}
        """,
        """
        {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me think"}}
        """,
        """
        {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":" carefully."}}
        """,
        """
        {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer"}}
        """,
        """
        {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20}}
        """,
    ]

    for eventJSON in events {
        let event = try decoder.decode(LLM.OpenAICompatibleAPI.AnthropicStreamEvent.self, from: #require(eventJSON.data(using: .utf8)))
        acc.processAnthropicEvent(event)
    }

    #expect(acc.thinking == "Let me think carefully.")
    #expect(acc.text == "Answer")

    let response = acc.buildResponse(isAnthropic: true)
    let thinkingContent = response.content?.first(where: { $0.type == .thinking })
    #expect(thinkingContent?.thinking == "Let me think carefully.")
    let textContent = response.content?.first(where: { $0.type == .text })
    #expect(textContent?.text == "Answer")
}

@Test func accumulator_anthropic_toolUse() throws {
    var acc = LLM.OpenAICompatibleAPI.StreamAccumulator()
    let decoder = JSONDecoder()

    let events = [
        """
        {"type":"message_start","message":{"id":"msg_tu","model":"claude-opus-4-5","usage":{"input_tokens":25}}}
        """,
        """
        {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather"}}
        """,
        """
        {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"loc"}}
        """,
        """
        {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"ation\\":\\"NYC\\"}"}}
        """,
        """
        {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":10}}
        """,
    ]

    for eventJSON in events {
        let event = try decoder.decode(LLM.OpenAICompatibleAPI.AnthropicStreamEvent.self, from: #require(eventJSON.data(using: .utf8)))
        acc.processAnthropicEvent(event)
    }

    #expect(acc.toolCalls.count == 1)
    #expect(acc.toolCalls[0]?.id == "toolu_1")
    #expect(acc.toolCalls[0]?.name == "get_weather")
    #expect(acc.toolCalls[0]?.arguments == "{\"location\":\"NYC\"}")

    let response = acc.buildResponse(isAnthropic: true)
    let toolUse = try #require(response.content?.first(where: { $0.type == .tool_use }))
    #expect(toolUse.id == "toolu_1")
    #expect(toolUse.name == "get_weather")
}

@Test func accumulator_openAI_noUsage() throws {
    // Local servers may not send usage data
    var acc = LLM.OpenAICompatibleAPI.StreamAccumulator()
    let decoder = JSONDecoder()

    let chunks = [
        """
        {"id":"local1","model":"local-model","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}
        """,
        """
        {"id":"local1","model":"local-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
        """,
    ]

    for chunkJSON in chunks {
        let chunk = try decoder.decode(LLM.OpenAICompatibleAPI.OpenAIStreamChunk.self, from: #require(chunkJSON.data(using: .utf8)))
        acc.processOpenAIChunk(chunk)
    }

    let response = acc.buildResponse(isAnthropic: false)
    #expect(response.choices?.first?.message.content == "Hi")
    #expect(response.usage.prompt_tokens == nil)
    #expect(response.usage.completion_tokens == nil)
}

@Test func accumulator_anthropic_cacheUsage() throws {
    var acc = LLM.OpenAICompatibleAPI.StreamAccumulator()
    let decoder = JSONDecoder()

    let events = [
        """
        {"type":"message_start","message":{"id":"msg_c","model":"claude-opus-4-5","usage":{"input_tokens":100,"cache_creation_input_tokens":80,"cache_read_input_tokens":20}}}
        """,
        """
        {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}
        """,
        """
        {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}
        """,
    ]

    for eventJSON in events {
        let event = try decoder.decode(LLM.OpenAICompatibleAPI.AnthropicStreamEvent.self, from: #require(eventJSON.data(using: .utf8)))
        acc.processAnthropicEvent(event)
    }

    let response = acc.buildResponse(isAnthropic: true)
    #expect(response.usage.input_tokens == 100)
    #expect(response.usage.cache_creation_input_tokens == 80)
    #expect(response.usage.cache_read_input_tokens == 20)
    #expect(response.usage.output_tokens == 5)
}

// MARK: - StreamEvent Type Tests

@Test func streamEvent_textDelta() {
    let event = LLM.StreamEvent.textDelta("hello")
    if case let .textDelta(text) = event {
        #expect(text == "hello")
    } else {
        Issue.record("Expected textDelta")
    }
}

@Test func streamEvent_thinkingDelta() {
    let event = LLM.StreamEvent.thinkingDelta("thinking...")
    if case let .thinkingDelta(text) = event {
        #expect(text == "thinking...")
    } else {
        Issue.record("Expected thinkingDelta")
    }
}

@Test func toolCallDelta_properties() {
    let delta = LLM.ToolCallDelta(index: 0, id: "call_1", name: "func", argumentsFragment: "{}")
    #expect(delta.index == 0)
    #expect(delta.id == "call_1")
    #expect(delta.name == "func")
    #expect(delta.argumentsFragment == "{}")
}

@Test func toolCallDelta_subsequentChunk() {
    let delta = LLM.ToolCallDelta(index: 0, id: nil, name: nil, argumentsFragment: "partial")
    #expect(delta.id == nil)
    #expect(delta.name == nil)
    #expect(delta.argumentsFragment == "partial")
}
