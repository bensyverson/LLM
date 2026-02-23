//
//  LLM+Streaming.swift
//  LLM
//
//  Created by Claude on 2026-02-22.
//

import Foundation

public extension LLM {
    func streamChat(conversation: Conversation) -> AsyncThrowingStream<StreamEvent, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    try await _streamChat(conversation: conversation, continuation: continuation)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    func streamConversation(
        systemPrompt: String,
        userMessage: String,
        configuration: ConversationConfiguration = .init()
    ) -> AsyncThrowingStream<StreamEvent, Error> {
        let conversation = Conversation(
            systemPrompt: systemPrompt,
            messages: [OpenAICompatibleAPI.ChatMessage(content: userMessage, role: .user)],
            configuration: configuration
        )
        return streamChat(conversation: conversation)
    }

    func streamContinueConversation(
        _ conversation: Conversation,
        userMessage: String
    ) -> AsyncThrowingStream<StreamEvent, Error> {
        let updated = conversation.addingUserMessage(userMessage)
        return streamChat(conversation: updated)
    }
}

// MARK: - Internal Implementation

extension LLM {
    private func _streamChat(
        conversation: Conversation,
        continuation: AsyncThrowingStream<StreamEvent, Error>.Continuation
    ) async throws {
        let api = providerApi
        let isAnthropic = provider.isAnthropic

        // Rate limit (same as non-streaming path)
        let messageTextLength = conversation.messages.reduce(0) { total, msg in
            total + msg.contentLength
        }
        let tokenCount = Int(Double(conversation.systemPrompt.count + messageTextLength) / 2.0)
        try await chatRateLimiter.acquire(tokens: tokenCount)

        // Build request with streaming enabled
        var request = conversation.request(for: provider)
        request.stream = true
        if provider.isOpenAI {
            request.stream_options = OpenAICompatibleAPI.StreamOptions(include_usage: true)
        }

        let jsonData = try JSONEncoder().encode(request)

        let (parser, _) = try await api.streamingChatCompletion(with: jsonData)

        var accumulator = OpenAICompatibleAPI.StreamAccumulator()

        if isAnthropic {
            try await processAnthropicStream(parser: parser, accumulator: &accumulator, continuation: continuation)
        } else {
            try await processOpenAIStream(parser: parser, accumulator: &accumulator, continuation: continuation)
        }

        // Build final response from accumulator
        let rawResponse = accumulator.buildResponse(isAnthropic: isAnthropic)

        let text = rawResponse.content?.first(where: { $0.type == .text })?.text
            ?? rawResponse.choices?.first?.message.content

        let thinking = rawResponse.content?.first(where: { $0.type == .thinking })?.thinking
            ?? rawResponse.choices?.first?.message.reasoning_content

        let toolCalls = Self.extractToolCalls(from: rawResponse)

        // Build updated conversation (same logic as non-streaming path)
        let updatedConversation: Conversation
        if !toolCalls.isEmpty {
            var conv = conversation
            if let text {
                conv = conv.addingAssistantMessage(text)
            }
            updatedConversation = conv.addingAssistantToolCallMessage(toolCalls)
        } else if let text {
            updatedConversation = conversation.addingAssistantMessage(text)
        } else {
            updatedConversation = conversation
        }

        let conversationResponse = ConversationResponse(
            text: text,
            thinking: thinking,
            toolCalls: toolCalls,
            conversation: updatedConversation,
            rawResponse: rawResponse
        )
        continuation.yield(.completed(conversationResponse))
        continuation.finish()
    }

    private func processOpenAIStream<Lines: AsyncSequence>(
        parser: OpenAICompatibleAPI.SSEParser<Lines>,
        accumulator: inout OpenAICompatibleAPI.StreamAccumulator,
        continuation: AsyncThrowingStream<StreamEvent, Error>.Continuation
    ) async throws where Lines.Element == String {
        let decoder = JSONDecoder()

        for try await event in parser {
            guard let data = event.data.data(using: .utf8) else { continue }

            let chunk: OpenAICompatibleAPI.OpenAIStreamChunk
            do {
                chunk = try decoder.decode(OpenAICompatibleAPI.OpenAIStreamChunk.self, from: data)
            } catch {
                continue
            }

            // Emit deltas before accumulating
            if let choice = chunk.choices?.first {
                if let content = choice.delta.content, !content.isEmpty {
                    continuation.yield(.textDelta(content))
                }
                if let reasoning = choice.delta.reasoning_content, !reasoning.isEmpty {
                    continuation.yield(.thinkingDelta(reasoning))
                }
                if let tcChunks = choice.delta.tool_calls {
                    for tc in tcChunks {
                        let delta = ToolCallDelta(
                            index: tc.index,
                            id: tc.id,
                            name: tc.function?.name,
                            argumentsFragment: tc.function?.arguments ?? ""
                        )
                        continuation.yield(.toolCallDelta(delta))
                    }
                }
            }

            accumulator.processOpenAIChunk(chunk)
        }
    }

    private func processAnthropicStream<Lines: AsyncSequence>(
        parser: OpenAICompatibleAPI.SSEParser<Lines>,
        accumulator: inout OpenAICompatibleAPI.StreamAccumulator,
        continuation: AsyncThrowingStream<StreamEvent, Error>.Continuation
    ) async throws where Lines.Element == String {
        let decoder = JSONDecoder()

        for try await sseEvent in parser {
            guard let data = sseEvent.data.data(using: .utf8) else { continue }

            let event: OpenAICompatibleAPI.AnthropicStreamEvent
            do {
                event = try decoder.decode(OpenAICompatibleAPI.AnthropicStreamEvent.self, from: data)
            } catch {
                continue
            }

            // Emit deltas before accumulating
            switch event.type {
            case "content_block_delta":
                if let delta = event.delta {
                    switch delta.type {
                    case "text_delta":
                        if let text = delta.text, !text.isEmpty {
                            continuation.yield(.textDelta(text))
                        }
                    case "thinking_delta":
                        if let thinking = delta.thinking, !thinking.isEmpty {
                            continuation.yield(.thinkingDelta(thinking))
                        }
                    case "input_json_delta":
                        if let json = delta.partial_json, let idx = event.index {
                            let tcDelta = ToolCallDelta(
                                index: idx,
                                id: nil,
                                name: nil,
                                argumentsFragment: json
                            )
                            continuation.yield(.toolCallDelta(tcDelta))
                        }
                    default:
                        break
                    }
                }

            case "content_block_start":
                if let idx = event.index, let block = event.content_block, block.type == "tool_use" {
                    let tcDelta = ToolCallDelta(
                        index: idx,
                        id: block.id,
                        name: block.name,
                        argumentsFragment: ""
                    )
                    continuation.yield(.toolCallDelta(tcDelta))
                }

            default:
                break
            }

            accumulator.processAnthropicEvent(event)
        }
    }
}
