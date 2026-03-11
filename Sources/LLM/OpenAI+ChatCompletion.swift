//
//  OpenAI+ChatCompletion.swift
//  LLM
//
//  Created by Ben Syverson on 2024-11-13.
//

import Foundation

public extension LLM.OpenAICompatibleAPI {
    // MARK: - Cache Control (Anthropic)

    /// Anthropic prompt caching configuration.
    struct CacheControl: Codable, Sendable {
        public enum CacheType: String, Codable, Sendable {
            case ephemeral
        }

        public enum TTL: String, Codable, Sendable {
            case fiveMinutes = "5m"
            case oneHour = "1h"
        }

        public var type: CacheType = .ephemeral
        public var ttl: TTL

        public init(type: CacheType = .ephemeral, ttl: TTL = .fiveMinutes) {
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

    /// The request body for a chat completion API call.
    ///
    /// Supports both OpenAI and Anthropic wire formats. Provider-specific fields
    /// (like `thinking` for Anthropic or `reasoning_effort` for OpenAI) are included
    /// conditionally based on the provider.
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

    /// The role of a message in a chat conversation.
    enum Role: String, Friendly {
        case system, user, assistant, tool
    }

    /// A single message in a chat conversation.
    ///
    /// Content is represented as an array of ``ContentPart`` values, supporting
    /// text, images, PDFs, and other media types. For text-only messages, the
    /// convenience initializers accepting `String` wrap the text automatically.
    struct ChatMessage: Friendly {
        /// The content parts of this message.
        public var content: [ContentPart]
        public var role: Role
        public var name: String?
        public var tool_calls: [ToolCall]?
        public var tool_call_id: String?

        /// Reasoning content from OpenAI o-series and GPT-5 models.
        public var reasoning_content: String?

        /// Primary initializer accepting an array of content parts.
        public init(
            content: [ContentPart],
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

        /// Convenience initializer for optional text content (e.g. tool-call-only messages).
        public init(
            content: String?,
            role: Role,
            name: String? = nil,
            tool_calls: [ToolCall]? = nil,
            tool_call_id: String? = nil,
            reasoning_content: String? = nil
        ) {
            self.content = content.map { [.text($0)] } ?? []
            self.role = role
            self.name = name
            self.tool_calls = tool_calls
            self.tool_call_id = tool_call_id
            self.reasoning_content = reasoning_content
        }

        /// Backward-compatible convenience init for text messages.
        public init(
            content: String,
            role: Role,
            name: String? = nil
        ) {
            self.content = [.text(content)]
            self.role = role
            self.name = name
            tool_calls = nil
            tool_call_id = nil
        }

        /// The concatenated text content of this message, or `nil` if there are no text parts.
        public var textContent: String? {
            let texts = content.compactMap(\.textContent)
            return texts.isEmpty ? nil : texts.joined()
        }

        /// Whether this message contains any media (non-text) parts.
        public var hasMedia: Bool {
            content.contains { $0.isMedia }
        }

        /// Estimated content length in characters (sum of text part lengths).
        public var contentLength: Int {
            content.compactMap(\.textContent).reduce(0) { $0 + $1.count }
        }

        // MARK: - Custom Codable

        enum CodingKeys: String, CodingKey {
            case content, role, name, tool_calls, tool_call_id, reasoning_content
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(role, forKey: .role)
            try container.encodeIfPresent(name, forKey: .name)
            try container.encodeIfPresent(tool_calls, forKey: .tool_calls)
            try container.encodeIfPresent(tool_call_id, forKey: .tool_call_id)
            try container.encodeIfPresent(reasoning_content, forKey: .reasoning_content)

            // Encode content: text-only as plain string, multimodal as array
            if content.isEmpty {
                try container.encodeNil(forKey: .content)
            } else if !hasMedia, let text = textContent {
                try container.encode(text, forKey: .content)
            } else {
                try container.encode(content.map { OpenAIContentPartWrapper($0) }, forKey: .content)
            }
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            role = try container.decode(Role.self, forKey: .role)
            name = try container.decodeIfPresent(String.self, forKey: .name)
            tool_calls = try container.decodeIfPresent([ToolCall].self, forKey: .tool_calls)
            tool_call_id = try container.decodeIfPresent(String.self, forKey: .tool_call_id)
            reasoning_content = try container.decodeIfPresent(String.self, forKey: .reasoning_content)

            // Decode content: handle string, array, or null
            if let text = try? container.decodeIfPresent(String.self, forKey: .content) {
                content = [.text(text)]
            } else if let parts = try? container.decodeIfPresent([OpenAIContentPartWrapper].self, forKey: .content) {
                content = parts.map(\.part)
            } else {
                content = []
            }
        }
    }

    /// Wrapper for encoding/decoding content parts in OpenAI's format.
    struct OpenAIContentPartWrapper: Codable, Equatable, Hashable, Sendable {
        let part: ContentPart

        init(_ part: ContentPart) {
            self.part = part
        }

        enum CodingKeys: String, CodingKey {
            case type, text, image_url
        }

        struct ImageURL: Codable, Equatable, Hashable, Sendable {
            let url: String
            let detail: String?
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            switch part {
            case let .text(text):
                try container.encode("text", forKey: .type)
                try container.encode(text, forKey: .text)
            case let .image(data, mediaType, _, _):
                try container.encode("image_url", forKey: .type)
                let base64 = data.base64EncodedString()
                let dataURI = "data:\(mediaType);base64,\(base64)"
                try container.encode(ImageURL(url: dataURI, detail: "auto"), forKey: .image_url)
            case .pdf:
                // OpenAI doesn't support PDFs — encode as text placeholder
                try container.encode("text", forKey: .type)
                try container.encode("[Unsupported: PDF content]", forKey: .text)
            case .audio, .video:
                try container.encode("text", forKey: .type)
                try container.encode("[Unsupported media]", forKey: .text)
            }
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let type = try container.decode(String.self, forKey: .type)
            switch type {
            case "text":
                let text = try container.decode(String.self, forKey: .text)
                part = .text(text)
            case "image_url":
                let imageURL = try container.decode(ImageURL.self, forKey: .image_url)
                // Parse data URI back to Data if possible
                if imageURL.url.hasPrefix("data:"),
                   let semicolonIdx = imageURL.url.firstIndex(of: ";"),
                   let commaIdx = imageURL.url.firstIndex(of: ",")
                {
                    let mediaType = String(imageURL.url[imageURL.url.index(imageURL.url.startIndex, offsetBy: 5) ..< semicolonIdx])
                    let base64String = String(imageURL.url[imageURL.url.index(after: commaIdx)...])
                    if let data = Data(base64Encoded: base64String) {
                        part = .image(data: data, mediaType: mediaType)
                    } else {
                        part = .text(imageURL.url)
                    }
                } else {
                    part = .text(imageURL.url)
                }
            default:
                let text = try container.decodeIfPresent(String.self, forKey: .text) ?? ""
                part = .text(text)
            }
        }
    }

    /// The response from a chat completion API call.
    ///
    /// Supports both OpenAI format (with `choices`) and Anthropic format (with `content` blocks).
    struct ChatCompletionResponse: Friendly {
        public enum ObjectType: String, Friendly {
            case chatCompletion = "chat.completion"
        }

        /// Token usage statistics from the API response.
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

            public init(
                prompt_tokens: Int? = nil,
                completion_tokens: Int? = nil,
                total_tokens: Int? = nil,
                input_tokens: Int? = nil,
                output_tokens: Int? = nil,
                cache_creation_input_tokens: Int? = nil,
                cache_read_input_tokens: Int? = nil,
                prompt_tokens_details: PromptTokensDetails? = nil
            ) {
                self.prompt_tokens = prompt_tokens
                self.completion_tokens = completion_tokens
                self.total_tokens = total_tokens
                self.input_tokens = input_tokens
                self.output_tokens = output_tokens
                self.cache_creation_input_tokens = cache_creation_input_tokens
                self.cache_read_input_tokens = cache_read_input_tokens
                self.prompt_tokens_details = prompt_tokens_details
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

        public init(
            id: String? = nil,
            object: ObjectType? = nil,
            system_fingerprint: String? = nil,
            usage: Usage = Usage(),
            model: String = "",
            created: Date? = nil,
            choices: [Choice]? = nil,
            content: [Content]? = nil
        ) {
            self.id = id
            self.object = object
            self.system_fingerprint = system_fingerprint
            self.usage = usage
            self.model = model
            self.created = created
            self.choices = choices
            self.content = content
        }
    }
}
