//
//  OpenAI+ChatCompletion.swift
//  LLM
//
//  Created by Ben Syverson on 2024-11-13.
//

import Foundation

public extension LLM.OpenAICompatibleAPI {
    // MARK: - Cache Control (Anthropic)

    struct CacheControl: Codable, Sendable {
        public enum CacheType: String, Codable, Sendable {
            case ephemeral
        }

        public enum TTL: String, Codable, Sendable {
            case fiveMinutes = "5m"
            case oneHour = "1h"
        }

        public var type: CacheType = .ephemeral
        public var ttl: TTL?

        public init(type: CacheType = .ephemeral, ttl: TTL? = nil) {
            self.type = type
            self.ttl = ttl
        }
    }

    /// System content block for Anthropic's array-based system prompt with caching
    struct SystemContentBlock: Codable {
        public var type: String = "text"
        public var text: String
        public var cache_control: CacheControl?

        public init(text: String, cache_control: CacheControl? = nil) {
            type = "text"
            self.text = text
            self.cache_control = cache_control
        }
    }

    struct StreamOptions: Codable, Sendable {
        public var include_usage: Bool

        public init(include_usage: Bool = true) {
            self.include_usage = include_usage
        }
    }

    // MARK: - Chat Completion

    struct ChatCompletion: Codable {
        public struct JsonObject: Codable {
            public enum ObjectType: String, Codable {
                case json_object
            }

            public var type: ObjectType = .json_object

            public init(type: LLM.OpenAICompatibleAPI.ChatCompletion.JsonObject.ObjectType = .json_object) {
                self.type = type
            }
        }

        public struct Thinking: Codable {
            public enum ThinkingType: String, Codable {
                case enabled, disabled
            }

            public var type: ThinkingType = .enabled
            public var budget_tokens: Int = 1024
            public init(
                type: ThinkingType = .enabled,
                budget_tokens: Int = 1024
            ) {
                self.type = type
                self.budget_tokens = budget_tokens
            }
        }

        public enum ReasoningEffort: String, Codable, Sendable {
            case none, low, medium, high, xhigh, minimal
        }

        public var model: ModelName = .gpt35turbo
        public var system: String? = nil
        public var systemBlocks: [SystemContentBlock]? = nil // For Anthropic caching
        public var messages: [ChatMessage]
        public var response_format: JsonObject? = JsonObject()
        public var temperature: Double? = 1.0
        public var frequency_penalty: Double? = nil
        public var top_p: Double? = 1.0
        public var max_tokens: Int? = nil
        public var max_completion_tokens: Int? = nil
        public var stop: [String]? = ["###"]
        public var stop_sequences: [String]? = nil
        public var thinking: Thinking? = nil
        public var reasoning_effort: ReasoningEffort? = nil
        public var tools: [ToolDefinition]? = nil
        public var tool_choice: ToolChoice? = nil
        public var stream: Bool? = nil
        public var stream_options: StreamOptions? = nil

        /// When true, tools are encoded in Anthropic's format (name/description/input_schema)
        /// rather than OpenAI's (type/function wrapper). Set by `request(for:)`.
        public var useAnthropicToolFormat: Bool = false

        public init(
            model: LLM.OpenAICompatibleAPI.ModelName = .gpt35turbo,
            system: String? = nil,
            systemBlocks: [SystemContentBlock]? = nil,
            messages: [LLM.OpenAICompatibleAPI.ChatMessage],
            response_format: LLM.OpenAICompatibleAPI.ChatCompletion.JsonObject? = JsonObject(),
            temperature: Double? = nil,
            frequency_penalty: Double? = nil,
            top_p: Double? = nil,
            max_tokens: Int? = nil,
            max_completion_tokens: Int? = nil,
            stop: [String]? = nil,
            stop_sequences: [String]? = nil,
            thinking: Thinking? = nil,
            reasoning_effort: ReasoningEffort? = nil,
            tools: [ToolDefinition]? = nil,
            tool_choice: ToolChoice? = nil
        ) {
            self.model = model
            self.system = system
            self.systemBlocks = systemBlocks
            self.messages = messages
            self.response_format = response_format
            self.temperature = temperature
            self.frequency_penalty = frequency_penalty
            self.top_p = top_p
            self.max_tokens = max_tokens
            self.max_completion_tokens = max_completion_tokens
            self.stop = stop
            self.stop_sequences = stop_sequences
            self.thinking = thinking
            self.reasoning_effort = reasoning_effort
            self.tools = tools
            self.tool_choice = tool_choice
        }

        /// Custom encoding to handle Anthropic's array-based system prompt
        enum CodingKeys: String, CodingKey {
            case model, system, messages, response_format, temperature
            case frequency_penalty, top_p, max_tokens, max_completion_tokens
            case stop, stop_sequences, thinking, reasoning_effort, tools, tool_choice
            case stream, stream_options
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(model, forKey: .model)

            // For Anthropic with caching, use systemBlocks (array format)
            // Otherwise use standard string system prompt
            if let blocks = systemBlocks {
                try container.encode(blocks, forKey: .system)
            } else if let system = system {
                try container.encode(system, forKey: .system)
            }

            if useAnthropicToolFormat {
                try container.encode(AnthropicMessageConverter.convert(messages), forKey: .messages)
            } else {
                try container.encode(messages, forKey: .messages)
            }
            try container.encodeIfPresent(response_format, forKey: .response_format)
            try container.encodeIfPresent(temperature, forKey: .temperature)
            try container.encodeIfPresent(frequency_penalty, forKey: .frequency_penalty)
            try container.encodeIfPresent(top_p, forKey: .top_p)
            try container.encodeIfPresent(max_tokens, forKey: .max_tokens)
            try container.encodeIfPresent(max_completion_tokens, forKey: .max_completion_tokens)
            try container.encodeIfPresent(stop, forKey: .stop)
            try container.encodeIfPresent(stop_sequences, forKey: .stop_sequences)
            try container.encodeIfPresent(thinking, forKey: .thinking)
            try container.encodeIfPresent(reasoning_effort, forKey: .reasoning_effort)
            if useAnthropicToolFormat, let tools {
                let anthropicTools = tools.map { AnthropicToolDefinition(from: $0) }
                try container.encode(anthropicTools, forKey: .tools)
            } else {
                try container.encodeIfPresent(tools, forKey: .tools)
            }
            if useAnthropicToolFormat, let tool_choice {
                try container.encode(AnthropicToolChoice(from: tool_choice), forKey: .tool_choice)
            } else {
                try container.encodeIfPresent(tool_choice, forKey: .tool_choice)
            }
            try container.encodeIfPresent(stream, forKey: .stream)
            try container.encodeIfPresent(stream_options, forKey: .stream_options)
        }
    }

    enum Role: String, Friendly {
        case system, user, assistant, tool
    }

    struct ChatMessage: Friendly {
        public var content: String?
        public var role: Role
        public var name: String?
        public var tool_calls: [ToolCall]?
        public var tool_call_id: String?

        /// Reasoning content from OpenAI o-series and GPT-5 models.
        public var reasoning_content: String?

        /// Full initializer
        public init(
            content: String?,
            role: Role,
            name: String? = nil,
            tool_calls: [ToolCall]? = nil,
            tool_call_id: String? = nil,
            reasoning_content: String? = nil
        ) {
            self.content = content
            self.role = role
            self.name = name
            self.tool_calls = tool_calls
            self.tool_call_id = tool_call_id
            self.reasoning_content = reasoning_content
        }

        /// Backward-compatible convenience init
        public init(
            content: String,
            role: Role,
            name: String? = nil
        ) {
            self.content = content
            self.role = role
            self.name = name
            tool_calls = nil
            tool_call_id = nil
        }

        /// Helper to get content length (for token estimation)
        public var contentLength: Int {
            content?.count ?? 0
        }
    }

    struct ChatCompletionResponse: Friendly {
        public enum ObjectType: String, Friendly {
            case chatCompletion = "chat.completion"
        }

        public struct Usage: Friendly {
            public let prompt_tokens: Int?
            public let completion_tokens: Int?
            public let total_tokens: Int?
            public let input_tokens: Int?
            public let output_tokens: Int?

            // Anthropic cache fields
            public let cache_creation_input_tokens: Int?
            public let cache_read_input_tokens: Int?

            /// OpenAI cache fields (nested in prompt_tokens_details)
            public let prompt_tokens_details: PromptTokensDetails?

            public struct PromptTokensDetails: Friendly {
                public let cached_tokens: Int?
            }
        }

        public struct Choice: Friendly {
            public let index: Int
            public let message: ChatMessage
            public let finish_reason: String?
        }

        /// Anthropic Content
        public struct Content: Friendly {
            public enum ContentType: String, Friendly {
                case text, thinking, redacted_thinking, tool_use, tool_result
            }

            public var type: ContentType
            public var text: String?
            public var thinking: String? // Anthropic thinking content
            public var data: Data?
            public var signature: String?
            // Tool use fields (Anthropic)
            public var id: String?
            public var name: String?
            public var input: [String: JSONValue]?
            // Tool result fields (Anthropic)
            public var tool_use_id: String?
            public var content: String?
        }

        public let id: String?
        public let object: ObjectType?
        public let system_fingerprint: String?
        public let usage: Usage
        public let model: String
        public let created: Date? // unix timestamp
        public let choices: [Choice]?
        public let content: [Content]?
    }
}
