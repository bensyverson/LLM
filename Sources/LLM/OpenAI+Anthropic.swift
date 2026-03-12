//
//  OpenAI+Anthropic.swift
//  LLM
//
//  Anthropic-specific encoding helpers for the OpenAI-compatible API layer.
//  Anthropic's Messages API uses different wire formats for tools, tool choices,
//  and message content blocks. These types handle the conversion.
//

import Foundation

public extension LLM.OpenAICompatibleAPI {
    // MARK: - Anthropic Tool Format

    /// Anthropic's tool definition format: name/description/input_schema (no type/function wrapper).
    struct AnthropicToolDefinition: Encodable {
        let name: String
        let description: String
        let input_schema: JSONSchema

        init(from tool: ToolDefinition) {
            name = tool.function.name
            description = tool.function.description
            input_schema = tool.function.parameters
        }
    }

    /// Anthropic's tool_choice format: `{"type": "auto"}`, `{"type": "any"}`,
    /// `{"type": "tool", "name": "..."}`, or `{"type": "none"}`.
    struct AnthropicToolChoice: Encodable {
        let type: String
        let name: String?

        init(from choice: ToolChoice) {
            switch choice {
            case .auto:
                type = "auto"
                name = nil
            case .required:
                type = "any"
                name = nil
            case .none:
                type = "none"
                name = nil
            case let .tool(toolName):
                type = "tool"
                name = toolName
            }
        }
    }

    // MARK: - Anthropic Message Converter

    /// Converts OpenAI-format messages to Anthropic-format messages.
    ///
    /// Anthropic differences:
    /// - No `role: "tool"` — tool results are `role: "user"` with `tool_result` content blocks
    /// - No `tool_calls` array — assistant tool calls are `tool_use` content blocks
    /// - Consecutive same-role messages must be merged
    enum AnthropicMessageConverter {
        /// An Anthropic message with structured content blocks.
        struct Message: Encodable {
            let role: String
            let content: [ContentBlock]
            /// When set, the last content block is encoded with this cache control.
            var lastBlockCacheControl: CacheControl?

            enum CodingKeys: String, CodingKey {
                case role, content
            }

            func encode(to encoder: Encoder) throws {
                var container = encoder.container(keyedBy: CodingKeys.self)
                try container.encode(role, forKey: .role)
                if let cacheControl = lastBlockCacheControl, !content.isEmpty {
                    // Encode all blocks normally except the last, which gets cache_control
                    var contentContainer = container.nestedUnkeyedContainer(forKey: .content)
                    for (index, block) in content.enumerated() {
                        let blockEncoder = contentContainer.superEncoder()
                        if index == content.count - 1 {
                            try block.encode(to: blockEncoder, cacheControl: cacheControl)
                        } else {
                            try block.encode(to: blockEncoder)
                        }
                    }
                } else {
                    try container.encode(content, forKey: .content)
                }
            }
        }

        /// A content block in an Anthropic message.
        enum ContentBlock: Encodable {
            case text(String)
            case image(data: Data, mediaType: String)
            case document(data: Data, mediaType: String, title: String?)
            case toolUse(id: String, name: String, input: String)
            case toolResult(toolUseId: String, content: [ContentBlock])

            func encode(to encoder: Encoder) throws {
                try encode(to: encoder, cacheControl: nil)
            }

            func encode(to encoder: Encoder, cacheControl: CacheControl?) throws {
                var container = encoder.container(keyedBy: CodingKeys.self)
                switch self {
                case let .text(text):
                    try container.encode("text", forKey: .type)
                    try container.encode(text, forKey: .text)
                case let .image(data, mediaType):
                    try container.encode("image", forKey: .type)
                    try container.encode(
                        ImageSource(type: "base64", media_type: mediaType, data: data.base64EncodedString()),
                        forKey: .source
                    )
                case let .document(data, mediaType, title):
                    try container.encode("document", forKey: .type)
                    try container.encode(
                        ImageSource(type: "base64", media_type: mediaType, data: data.base64EncodedString()),
                        forKey: .source
                    )
                    try container.encodeIfPresent(title, forKey: .title)
                case let .toolUse(id, name, input):
                    try container.encode("tool_use", forKey: .type)
                    try container.encode(id, forKey: .id)
                    try container.encode(name, forKey: .name)
                    // input must be a JSON object, not a string — decode then re-encode
                    let inputData = Data(input.utf8)
                    let decoded = try JSONDecoder().decode([String: JSONValue].self, from: inputData)
                    try container.encode(decoded, forKey: .input)
                case let .toolResult(toolUseId, content):
                    try container.encode("tool_result", forKey: .type)
                    try container.encode(toolUseId, forKey: .tool_use_id)
                    try container.encode(content, forKey: .content)
                }
                try container.encodeIfPresent(cacheControl, forKey: .cache_control)
            }

            enum CodingKeys: String, CodingKey {
                case type, text, id, name, input, tool_use_id, content, source, title, cache_control
            }

            struct ImageSource: Encodable {
                let type: String
                let media_type: String
                let data: String
            }
        }

        /// Converts content parts to Anthropic content blocks.
        private static func contentBlocks(from parts: [ContentPart]) -> [ContentBlock] {
            parts.compactMap { part in
                switch part {
                case let .text(text):
                    return .text(text)
                case let .image(data, mediaType, _, _):
                    return .image(data: data, mediaType: mediaType)
                case let .pdf(data, title):
                    return .document(data: data, mediaType: "application/pdf", title: title)
                case .audio, .video:
                    return nil
                }
            }
        }

        static func convert(_ messages: [ChatMessage]) -> [Message] {
            var result = [Message]()

            for msg in messages {
                switch msg.role {
                case .system:
                    // Skip system messages — handled separately by Anthropic
                    continue

                case .user:
                    let blocks = contentBlocks(from: msg.content)
                    let effectiveBlocks = blocks.isEmpty ? [ContentBlock.text("")] : blocks
                    appendOrMerge(&result, role: "user", blocks: effectiveBlocks)

                case .assistant:
                    var blocks = [ContentBlock]()
                    // Assistant messages only use text (no images in assistant content)
                    if let text = msg.textContent, !text.isEmpty {
                        blocks.append(.text(text))
                    }
                    if let toolCalls = msg.tool_calls {
                        for call in toolCalls {
                            blocks.append(.toolUse(
                                id: call.id,
                                name: call.function.name,
                                input: call.function.arguments
                            ))
                        }
                    }
                    if !blocks.isEmpty {
                        appendOrMerge(&result, role: "assistant", blocks: blocks)
                    }

                case .tool:
                    // Convert to user message with tool_result content block
                    // Tool results can include images (Anthropic supports this)
                    let innerBlocks = contentBlocks(from: msg.content)
                    let effectiveContent = innerBlocks.isEmpty ? [ContentBlock.text("")] : innerBlocks
                    let block = ContentBlock.toolResult(
                        toolUseId: msg.tool_call_id ?? "",
                        content: effectiveContent
                    )
                    appendOrMerge(&result, role: "user", blocks: [block])
                }
            }

            return result
        }

        /// Appends blocks to the last message if it has the same role, otherwise creates a new message.
        private static func appendOrMerge(_ messages: inout [Message], role: String, blocks: [ContentBlock]) {
            if let last = messages.last, last.role == role {
                messages[messages.count - 1] = Message(role: role, content: last.content + blocks)
            } else {
                messages.append(Message(role: role, content: blocks))
            }
        }
    }
}
