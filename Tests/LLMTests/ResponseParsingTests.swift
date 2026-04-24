//
//  ResponseParsingTests.swift
//  LLMTests
//
//  Tests for parsing real API response fixtures
//

import Foundation
@testable import LLM
import Testing

// MARK: - Test Fixture Helper

/// Loads a JSON fixture from the Fixtures directory
func loadFixture(_ name: String) throws -> Data {
    // Get the path to the test bundle
    let bundle = Bundle.module
    guard let url = bundle.url(forResource: name.replacingOccurrences(of: ".json", with: ""), withExtension: "json", subdirectory: "Fixtures") else {
        throw FixtureError.notFound(name)
    }
    return try Data(contentsOf: url)
}

enum FixtureError: Error {
    case notFound(String)
}

// MARK: - OpenAI Response Parsing Tests

@Test func `parse open AI chat response`() throws {
    let json = """
    {
      "id": "chatcmpl-abc123def456",
      "object": "chat.completion",
      "created": 1706745600,
      "model": "gpt-5.2",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "Hello! I'm here to help. How can I assist you today?"
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 14,
        "total_tokens": 29
      },
      "system_fingerprint": "fp_abc123"
    }
    """.data(using: .utf8)!

    let decoder = JSONDecoder()
    let response = try decoder.decode(LLM.OpenAICompatibleAPI.ChatCompletionResponse.self, from: json)

    #expect(response.id == "chatcmpl-abc123def456")
    #expect(response.object == .chatCompletion)
    #expect(response.model == "gpt-5.2")
    #expect(response.system_fingerprint == "fp_abc123")

    // Usage
    #expect(response.usage.prompt_tokens == 15)
    #expect(response.usage.completion_tokens == 14)
    #expect(response.usage.total_tokens == 29)

    // Choices
    #expect(response.choices?.count == 1)
    let choice = try #require(response.choices?[0])
    #expect(choice.index == 0)
    #expect(choice.message.role == .assistant)
    #expect(choice.message.textContent == "Hello! I'm here to help. How can I assist you today?")
    #expect(choice.finish_reason == "stop")
}

@Test func `parse open AI tool call response`() throws {
    let json = """
    {
      "id": "chatcmpl-xyz789",
      "object": "chat.completion",
      "created": 1706745700,
      "model": "gpt-5.2",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": null,
            "tool_calls": [
              {
                "id": "call_abc123",
                "type": "function",
                "function": {
                  "name": "get_weather",
                  "arguments": "{\\"location\\":\\"New York City\\",\\"unit\\":\\"fahrenheit\\"}"
                }
              }
            ]
          },
          "finish_reason": "tool_calls"
        }
      ],
      "usage": {
        "prompt_tokens": 50,
        "completion_tokens": 25,
        "total_tokens": 75
      }
    }
    """.data(using: .utf8)!

    let decoder = JSONDecoder()
    let response = try decoder.decode(LLM.OpenAICompatibleAPI.ChatCompletionResponse.self, from: json)

    #expect(response.id == "chatcmpl-xyz789")
    #expect(response.model == "gpt-5.2")

    // Usage
    #expect(response.usage.prompt_tokens == 50)
    #expect(response.usage.completion_tokens == 25)
    #expect(response.usage.total_tokens == 75)

    // Choices
    #expect(response.choices?.count == 1)
    let choice = try #require(response.choices?[0])
    #expect(choice.message.content.isEmpty)
    #expect(choice.finish_reason == "tool_calls")

    // Tool calls
    let toolCalls = choice.message.tool_calls
    #expect(toolCalls?.count == 1)
    let toolCall = try #require(toolCalls?[0])
    #expect(toolCall.id == "call_abc123")
    #expect(toolCall.type == "function")
    #expect(toolCall.function.name == "get_weather")
    #expect(toolCall.function.arguments.contains("New York City"))
    #expect(toolCall.function.arguments.contains("fahrenheit"))
}

// MARK: - Anthropic Response Parsing Tests

@Test func `parse anthropic chat response`() throws {
    let json = """
    {
      "id": "msg_01abc123def456",
      "model": "claude-opus-4-5",
      "content": [
        {
          "type": "text",
          "text": "Hello! I'm Claude, an AI assistant. How can I help you today?"
        }
      ],
      "usage": {
        "input_tokens": 12,
        "output_tokens": 18
      }
    }
    """.data(using: .utf8)!

    let decoder = JSONDecoder()
    let response = try decoder.decode(LLM.OpenAICompatibleAPI.ChatCompletionResponse.self, from: json)

    #expect(response.id == "msg_01abc123def456")
    #expect(response.model == "claude-opus-4-5")

    // Usage (Anthropic format)
    #expect(response.usage.input_tokens == 12)
    #expect(response.usage.output_tokens == 18)

    // Content array (Anthropic format)
    #expect(response.content?.count == 1)
    let content = try #require(response.content?[0])
    #expect(content.type == .text)
    #expect(content.text == "Hello! I'm Claude, an AI assistant. How can I help you today?")
}

@Test func `parse anthropic thinking response`() throws {
    let json = """
    {
      "id": "msg_thinking123",
      "model": "claude-sonnet-4-5",
      "content": [
        {
          "type": "thinking",
          "thinking": "Let me analyze this step by step."
        },
        {
          "type": "text",
          "text": "The capital of France is Paris."
        }
      ],
      "usage": {
        "input_tokens": 10,
        "output_tokens": 45
      }
    }
    """.data(using: .utf8)!

    let decoder = JSONDecoder()
    let response = try decoder.decode(LLM.OpenAICompatibleAPI.ChatCompletionResponse.self, from: json)

    #expect(response.id == "msg_thinking123")
    #expect(response.model == "claude-sonnet-4-5")

    // Content with thinking
    #expect(response.content?.count == 2)

    let thinkingContent = try #require(response.content?[0])
    #expect(thinkingContent.type == .thinking)

    let textContent = try #require(response.content?[1])
    #expect(textContent.type == .text)
    #expect(textContent.text == "The capital of France is Paris.")
}

@Test func `parse anthropic tool use response`() throws {
    let json = """
    {
      "id": "msg_tool456",
      "model": "claude-opus-4-5",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_01abc123",
          "name": "get_weather",
          "input": {
            "location": "San Francisco",
            "unit": "celsius"
          }
        }
      ],
      "usage": {
        "input_tokens": 25,
        "output_tokens": 30
      }
    }
    """.data(using: .utf8)!

    let decoder = JSONDecoder()
    let response = try decoder.decode(LLM.OpenAICompatibleAPI.ChatCompletionResponse.self, from: json)

    #expect(response.id == "msg_tool456")
    #expect(response.model == "claude-opus-4-5")

    // Usage
    #expect(response.usage.input_tokens == 25)
    #expect(response.usage.output_tokens == 30)

    // Tool use content
    #expect(response.content?.count == 1)
    let toolUse = try #require(response.content?[0])
    #expect(toolUse.type == .tool_use)
    #expect(toolUse.id == "toolu_01abc123")
    #expect(toolUse.name == "get_weather")
    #expect(toolUse.input?["location"] == .string("San Francisco"))
    #expect(toolUse.input?["unit"] == .string("celsius"))
}

// MARK: - ChatCompletionResponse Usage Tests

@Test func `chat completion response usage fields open AI format`() throws {
    let json = """
    {
      "id": "test",
      "model": "gpt-5.2",
      "choices": [],
      "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150
      }
    }
    """.data(using: .utf8)!

    let decoder = JSONDecoder()
    let response = try decoder.decode(LLM.OpenAICompatibleAPI.ChatCompletionResponse.self, from: json)

    #expect(response.usage.prompt_tokens == 100)
    #expect(response.usage.completion_tokens == 50)
    #expect(response.usage.total_tokens == 150)
    #expect(response.usage.input_tokens == nil)
    #expect(response.usage.output_tokens == nil)
}

@Test func `chat completion response usage fields anthropic format`() throws {
    let json = """
    {
      "id": "test",
      "model": "claude-opus-4-5",
      "content": [],
      "usage": {
        "input_tokens": 100,
        "output_tokens": 50
      }
    }
    """.data(using: .utf8)!

    let decoder = JSONDecoder()
    let response = try decoder.decode(LLM.OpenAICompatibleAPI.ChatCompletionResponse.self, from: json)

    #expect(response.usage.input_tokens == 100)
    #expect(response.usage.output_tokens == 50)
    #expect(response.usage.prompt_tokens == nil)
    #expect(response.usage.completion_tokens == nil)
    #expect(response.usage.total_tokens == nil)
}

// MARK: - Content Type Tests

@Test func `content type all cases`() {
    #expect(LLM.OpenAICompatibleAPI.ChatCompletionResponse.Content.ContentType.text.rawValue == "text")
    #expect(LLM.OpenAICompatibleAPI.ChatCompletionResponse.Content.ContentType.thinking.rawValue == "thinking")
    #expect(LLM.OpenAICompatibleAPI.ChatCompletionResponse.Content.ContentType.redacted_thinking.rawValue == "redacted_thinking")
    #expect(LLM.OpenAICompatibleAPI.ChatCompletionResponse.Content.ContentType.tool_use.rawValue == "tool_use")
    #expect(LLM.OpenAICompatibleAPI.ChatCompletionResponse.Content.ContentType.tool_result.rawValue == "tool_result")
}

// MARK: - Real Fixture File Tests

@Test func `parse real open AI chat fixture`() throws {
    let url = try #require(Bundle.module.url(forResource: "openai_chat_response", withExtension: "json", subdirectory: "Fixtures"))
    let data = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    let response = try decoder.decode(LLM.OpenAICompatibleAPI.ChatCompletionResponse.self, from: data)

    #expect(response.id == "chatcmpl-D3oebJe6uqC797reNAhiYVinD5T3J")
    #expect(response.model == "gpt-4o-mini-2024-07-18")
    #expect(response.choices?.first?.message.textContent == "Hello there, how are you?")
    #expect(response.usage.prompt_tokens == 15)
    #expect(response.usage.completion_tokens == 7)
    #expect(response.usage.total_tokens == 22)
}

@Test func `parse real open AI tool call fixture`() throws {
    let url = try #require(Bundle.module.url(forResource: "openai_tool_call_response", withExtension: "json", subdirectory: "Fixtures"))
    let data = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    let response = try decoder.decode(LLM.OpenAICompatibleAPI.ChatCompletionResponse.self, from: data)

    #expect(response.choices?.first?.message.content.isEmpty == true)
    #expect(response.choices?.first?.finish_reason == "tool_calls")
    let toolCall = response.choices?.first?.message.tool_calls?.first
    #expect(toolCall?.function.name == "get_weather")
    #expect(toolCall?.function.arguments.contains("San Francisco") == true)
}

@Test func `parse real anthropic chat fixture`() throws {
    let url = try #require(Bundle.module.url(forResource: "anthropic_chat_response", withExtension: "json", subdirectory: "Fixtures"))
    let data = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    let response = try decoder.decode(LLM.OpenAICompatibleAPI.ChatCompletionResponse.self, from: data)

    #expect(response.id == "msg_01TvAiUy8H9G7mdKcQ7Wjg1V")
    #expect(response.model == "claude-3-5-haiku-20241022")
    #expect(response.content?.first?.type == .text)
    #expect(response.content?.first?.text == "Hi there, how are you today?")
    #expect(response.usage.input_tokens == 16)
    #expect(response.usage.output_tokens == 11)
}

@Test func `parse real anthropic thinking fixture`() throws {
    let url = try #require(Bundle.module.url(forResource: "anthropic_thinking_response", withExtension: "json", subdirectory: "Fixtures"))
    let data = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    let response = try decoder.decode(LLM.OpenAICompatibleAPI.ChatCompletionResponse.self, from: data)

    #expect(response.model == "claude-3-7-sonnet-20250219")
    #expect(response.content?.count == 2)

    let thinkingContent = response.content?[0]
    #expect(thinkingContent?.type == .thinking)
    #expect(thinkingContent?.thinking?.contains("15 + 27 = 42") == true)
    #expect(thinkingContent?.signature != nil)

    let textContent = response.content?[1]
    #expect(textContent?.type == .text)
    #expect(textContent?.text == "15 + 27 = 42")
}

@Test func `parse real anthropic tool use fixture`() throws {
    let url = try #require(Bundle.module.url(forResource: "anthropic_tool_use_response", withExtension: "json", subdirectory: "Fixtures"))
    let data = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    let response = try decoder.decode(LLM.OpenAICompatibleAPI.ChatCompletionResponse.self, from: data)

    #expect(response.model == "claude-3-5-haiku-20241022")
    let toolUse = response.content?.first
    #expect(toolUse?.type == .tool_use)
    #expect(toolUse?.id == "toolu_01GKH6LVsrEkQonsFXCgpTGh")
    #expect(toolUse?.name == "get_weather")
}

// MARK: - Edge Cases

@Test func `parse response with minimal fields`() throws {
    let json = """
    {
      "model": "test-model",
      "usage": {}
    }
    """.data(using: .utf8)!

    let decoder = JSONDecoder()
    let response = try decoder.decode(LLM.OpenAICompatibleAPI.ChatCompletionResponse.self, from: json)

    #expect(response.model == "test-model")
    #expect(response.id == nil)
    #expect(response.choices == nil)
    #expect(response.content == nil)
}

@Test func `parse response multiple choices`() throws {
    let json = """
    {
      "id": "test",
      "model": "gpt-5.2",
      "choices": [
        {
          "index": 0,
          "message": {"role": "assistant", "content": "Response 1"},
          "finish_reason": "stop"
        },
        {
          "index": 1,
          "message": {"role": "assistant", "content": "Response 2"},
          "finish_reason": "stop"
        }
      ],
      "usage": {"total_tokens": 50}
    }
    """.data(using: .utf8)!

    let decoder = JSONDecoder()
    let response = try decoder.decode(LLM.OpenAICompatibleAPI.ChatCompletionResponse.self, from: json)

    #expect(response.choices?.count == 2)
    #expect(response.choices?[0].message.textContent == "Response 1")
    #expect(response.choices?[1].message.textContent == "Response 2")
}

@Test func `parse response multiple tool calls`() throws {
    let json = """
    {
      "id": "test",
      "model": "gpt-5.2",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": null,
            "tool_calls": [
              {
                "id": "call_1",
                "type": "function",
                "function": {"name": "func1", "arguments": "{}"}
              },
              {
                "id": "call_2",
                "type": "function",
                "function": {"name": "func2", "arguments": "{}"}
              }
            ]
          },
          "finish_reason": "tool_calls"
        }
      ],
      "usage": {}
    }
    """.data(using: .utf8)!

    let decoder = JSONDecoder()
    let response = try decoder.decode(LLM.OpenAICompatibleAPI.ChatCompletionResponse.self, from: json)

    let toolCalls = response.choices?[0].message.tool_calls
    #expect(toolCalls?.count == 2)
    #expect(toolCalls?[0].function.name == "func1")
    #expect(toolCalls?[1].function.name == "func2")
}

// MARK: - Cache Usage Tests

@Test func `chat completion response anthropic cache usage`() throws {
    let json = """
    {
      "id": "msg_cache_test",
      "model": "claude-opus-4-5",
      "content": [
        {
          "type": "text",
          "text": "Hello!"
        }
      ],
      "usage": {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_creation_input_tokens": 80,
        "cache_read_input_tokens": 20
      }
    }
    """.data(using: .utf8)!

    let decoder = JSONDecoder()
    let response = try decoder.decode(LLM.OpenAICompatibleAPI.ChatCompletionResponse.self, from: json)

    #expect(response.usage.input_tokens == 100)
    #expect(response.usage.output_tokens == 50)
    #expect(response.usage.cache_creation_input_tokens == 80)
    #expect(response.usage.cache_read_input_tokens == 20)
}

@Test func `chat completion response open AI cache usage`() throws {
    let json = """
    {
      "id": "chatcmpl-cache",
      "model": "gpt-5.2",
      "choices": [
        {
          "index": 0,
          "message": {"role": "assistant", "content": "Hello!"},
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "prompt_tokens_details": {
          "cached_tokens": 80
        }
      }
    }
    """.data(using: .utf8)!

    let decoder = JSONDecoder()
    let response = try decoder.decode(LLM.OpenAICompatibleAPI.ChatCompletionResponse.self, from: json)

    #expect(response.usage.prompt_tokens == 100)
    #expect(response.usage.completion_tokens == 50)
    #expect(response.usage.total_tokens == 150)
    #expect(response.usage.prompt_tokens_details?.cached_tokens == 80)
}

@Test func `chat completion response no cache usage`() throws {
    let json = """
    {
      "id": "test",
      "model": "gpt-5.2",
      "choices": [],
      "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150
      }
    }
    """.data(using: .utf8)!

    let decoder = JSONDecoder()
    let response = try decoder.decode(LLM.OpenAICompatibleAPI.ChatCompletionResponse.self, from: json)

    // Cache fields should be nil when not present
    #expect(response.usage.cache_creation_input_tokens == nil)
    #expect(response.usage.cache_read_input_tokens == nil)
    #expect(response.usage.prompt_tokens_details == nil)
}
