//
//  OpenAI+StreamingTypes.swift
//  LLM
//
//  Created by Claude on 2026-02-22.
//

import Foundation

// MARK: - Public Stream Types

public extension LLM {
    enum StreamEvent: Sendable {
        case textDelta(String)
        case thinkingDelta(String)
        case toolCallDelta(ToolCallDelta)
        case completed(ConversationResponse)
    }

    struct ToolCallDelta: Sendable {
        public let index: Int
        public let id: String?
        public let name: String?
        public let argumentsFragment: String
    }
}

// MARK: - OpenAI Stream Chunk

extension LLM.OpenAICompatibleAPI {
    struct OpenAIStreamChunk: Decodable {
        let id: String?
        let object: String?
        let model: String?
        let choices: [Choice]?
        let usage: UsageChunk?

        struct Choice: Decodable {
            let index: Int
            let delta: Delta
            let finish_reason: String?
        }

        struct Delta: Decodable {
            let role: String?
            let content: String?
            let reasoning_content: String?
            let tool_calls: [ToolCallChunk]?
        }

        struct ToolCallChunk: Decodable {
            let index: Int
            let id: String?
            let type: String?
            let function: FunctionChunk?
        }

        struct FunctionChunk: Decodable {
            let name: String?
            let arguments: String?
        }

        struct UsageChunk: Decodable {
            let prompt_tokens: Int?
            let completion_tokens: Int?
            let total_tokens: Int?
        }
    }
}

// MARK: - Anthropic Stream Event

extension LLM.OpenAICompatibleAPI {
    struct AnthropicStreamEvent: Decodable {
        let type: String

        /// message_start
        let message: AnthropicMessage?

        // content_block_start
        let index: Int?
        let content_block: AnthropicContentBlock?

        /// content_block_delta
        let delta: AnthropicDelta?

        /// message_delta (usage)
        let usage: AnthropicUsage?

        struct AnthropicMessage: Decodable {
            let id: String?
            let model: String?
            let usage: AnthropicUsage?
        }

        struct AnthropicContentBlock: Decodable {
            let type: String
            let id: String?
            let name: String?
            let text: String?
        }

        struct AnthropicDelta: Decodable {
            let type: String?
            let text: String?
            let thinking: String?
            let partial_json: String?
            let stop_reason: String?
        }

        struct AnthropicUsage: Decodable {
            let input_tokens: Int?
            let output_tokens: Int?
            let cache_creation_input_tokens: Int?
            let cache_read_input_tokens: Int?
        }
    }
}

// MARK: - Stream Accumulator

extension LLM.OpenAICompatibleAPI {
    struct ToolCallAccumulator {
        var id: String = ""
        var name: String = ""
        var arguments: String = ""
    }

    struct StreamAccumulator {
        var id: String?
        var model: String = ""
        var text: String = ""
        var thinking: String = ""
        var finishReason: String?

        /// OpenAI tool calls keyed by index
        var toolCalls: [Int: ToolCallAccumulator] = [:]

        // Usage
        var promptTokens: Int?
        var completionTokens: Int?
        var totalTokens: Int?

        // Anthropic-specific usage
        var inputTokens: Int?
        var outputTokens: Int?
        var cacheCreationInputTokens: Int?
        var cacheReadInputTokens: Int?

        // MARK: - OpenAI chunk processing

        mutating func processOpenAIChunk(_ chunk: OpenAIStreamChunk) {
            if let chunkId = chunk.id { id = chunkId }
            if let chunkModel = chunk.model { model = chunkModel }

            if let usage = chunk.usage {
                promptTokens = usage.prompt_tokens
                completionTokens = usage.completion_tokens
                totalTokens = usage.total_tokens
            }

            guard let choice = chunk.choices?.first else { return }
            if let reason = choice.finish_reason { finishReason = reason }

            if let content = choice.delta.content {
                text += content
            }
            if let reasoning = choice.delta.reasoning_content {
                thinking += reasoning
            }
            if let tcChunks = choice.delta.tool_calls {
                for tc in tcChunks {
                    var acc = toolCalls[tc.index] ?? ToolCallAccumulator()
                    if let tcId = tc.id { acc.id = tcId }
                    if let name = tc.function?.name { acc.name = name }
                    if let args = tc.function?.arguments { acc.arguments += args }
                    toolCalls[tc.index] = acc
                }
            }
        }

        // MARK: - Anthropic event processing

        mutating func processAnthropicEvent(_ event: AnthropicStreamEvent) {
            switch event.type {
            case "message_start":
                if let msg = event.message {
                    id = msg.id
                    if let m = msg.model { model = m }
                    if let usage = msg.usage {
                        inputTokens = usage.input_tokens
                        cacheCreationInputTokens = usage.cache_creation_input_tokens
                        cacheReadInputTokens = usage.cache_read_input_tokens
                    }
                }

            case "content_block_start":
                if let idx = event.index, let block = event.content_block {
                    if block.type == "tool_use" {
                        var acc = ToolCallAccumulator()
                        acc.id = block.id ?? ""
                        acc.name = block.name ?? ""
                        toolCalls[idx] = acc
                    }
                }

            case "content_block_delta":
                if let delta = event.delta {
                    switch delta.type {
                    case "text_delta":
                        if let t = delta.text { text += t }
                    case "thinking_delta":
                        if let t = delta.thinking { thinking += t }
                    case "input_json_delta":
                        if let json = delta.partial_json, let idx = event.index {
                            toolCalls[idx]?.arguments += json
                        }
                    default:
                        break
                    }
                }

            case "message_delta":
                if let delta = event.delta {
                    finishReason = delta.stop_reason
                }
                if let usage = event.usage {
                    outputTokens = usage.output_tokens
                }

            default:
                break
            }
        }

        // MARK: - Build final response

        func buildResponse(isAnthropic: Bool) -> ChatCompletionResponse {
            let sortedToolCalls = toolCalls.sorted(by: { $0.key < $1.key })
            let completedToolCalls: [ToolCall]? = sortedToolCalls.isEmpty ? nil : sortedToolCalls.map { _, acc in
                ToolCall(
                    id: acc.id,
                    function: FunctionCall(name: acc.name, arguments: acc.arguments)
                )
            }

            if isAnthropic {
                var contentBlocks: [ChatCompletionResponse.Content] = []
                if !thinking.isEmpty {
                    contentBlocks.append(ChatCompletionResponse.Content(
                        type: .thinking,
                        text: nil,
                        thinking: thinking
                    ))
                }
                if !text.isEmpty {
                    contentBlocks.append(ChatCompletionResponse.Content(
                        type: .text,
                        text: text
                    ))
                }
                if let tcs = completedToolCalls {
                    for tc in tcs {
                        let inputDict: [String: JSONValue]?
                        if let data = tc.function.arguments.data(using: .utf8),
                           let decoded = try? JSONDecoder().decode([String: JSONValue].self, from: data)
                        {
                            inputDict = decoded
                        } else {
                            inputDict = nil
                        }
                        contentBlocks.append(ChatCompletionResponse.Content(
                            type: .tool_use,
                            id: tc.id,
                            name: tc.function.name,
                            input: inputDict
                        ))
                    }
                }

                let usage = ChatCompletionResponse.Usage(
                    prompt_tokens: nil,
                    completion_tokens: nil,
                    total_tokens: nil,
                    input_tokens: inputTokens,
                    output_tokens: outputTokens,
                    cache_creation_input_tokens: cacheCreationInputTokens,
                    cache_read_input_tokens: cacheReadInputTokens,
                    prompt_tokens_details: nil
                )
                return ChatCompletionResponse(
                    id: id,
                    object: nil,
                    system_fingerprint: nil,
                    usage: usage,
                    model: model,
                    created: nil,
                    choices: nil,
                    content: contentBlocks.isEmpty ? nil : contentBlocks
                )
            } else {
                // OpenAI format
                let message = ChatMessage(
                    content: text.isEmpty ? nil : text,
                    role: .assistant,
                    tool_calls: completedToolCalls,
                    reasoning_content: thinking.isEmpty ? nil : thinking
                )
                let choice = ChatCompletionResponse.Choice(
                    index: 0,
                    message: message,
                    finish_reason: finishReason
                )
                let usage = ChatCompletionResponse.Usage(
                    prompt_tokens: promptTokens,
                    completion_tokens: completionTokens,
                    total_tokens: totalTokens,
                    input_tokens: nil,
                    output_tokens: nil,
                    cache_creation_input_tokens: nil,
                    cache_read_input_tokens: nil,
                    prompt_tokens_details: nil
                )
                return ChatCompletionResponse(
                    id: id,
                    object: .chatCompletion,
                    system_fingerprint: nil,
                    usage: usage,
                    model: model,
                    created: nil,
                    choices: [choice],
                    content: nil
                )
            }
        }
    }
}
