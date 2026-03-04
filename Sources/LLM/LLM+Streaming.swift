//
//  LLM+Streaming.swift
//  LLM
//
//  Created by Claude on 2026-02-22.
//

import Foundation
#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

public extension LLM {
    /// Streams a chat conversation, yielding incremental deltas as they arrive.
    ///
    /// Returns an `AsyncThrowingStream` of ``StreamEvent`` values. Text, thinking, and
    /// tool call fragments are yielded as they arrive, followed by a final
    /// `.completed` event containing the full ``ConversationResponse``.
    ///
    /// - Parameter conversation: The conversation to stream.
    /// - Returns: A stream of events representing the model's incremental response.
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

    /// Starts a new streaming conversation with the given system prompt and user message.
    ///
    /// - Parameters:
    ///   - systemPrompt: The system prompt that guides the model's behavior.
    ///   - userMessage: The initial user message.
    ///   - configuration: Optional conversation configuration.
    /// - Returns: A stream of events representing the model's incremental response.
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

    /// Continues an existing conversation with a new user message, streaming the response.
    ///
    /// - Parameters:
    ///   - conversation: The conversation to continue.
    ///   - userMessage: The new user message to append.
    /// - Returns: A stream of events representing the model's incremental response.
    func streamContinueConversation(
        _ conversation: Conversation,
        userMessage: String
    ) -> AsyncThrowingStream<StreamEvent, Error> {
        let updated = conversation.addingUserMessage(userMessage)
        return streamChat(conversation: updated)
    }

    /// Starts a new streaming conversation with multimodal content.
    func streamConversation(
        systemPrompt: String,
        userMessage: [OpenAICompatibleAPI.ContentPart],
        configuration: ConversationConfiguration = .init()
    ) -> AsyncThrowingStream<StreamEvent, Error> {
        let conversation = Conversation(
            systemPrompt: systemPrompt,
            messages: [OpenAICompatibleAPI.ChatMessage(content: userMessage, role: .user)],
            configuration: configuration
        )
        return streamChat(conversation: conversation)
    }

    /// Continues an existing conversation with multimodal content, streaming the response.
    func streamContinueConversation(
        _ conversation: Conversation,
        userMessage: [OpenAICompatibleAPI.ContentPart]
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

        // Pre-process: non-vision fallback and image resizing
        let model = conversation.configuration.model ?? provider.model(type: conversation.configuration.modelType, inference: conversation.configuration.inference)
        let hasMedia = conversation.messages.contains { $0.hasMedia }
        var effectiveConversation = conversation

        if hasMedia && model.supportsVision == false {
            effectiveConversation = try await strippingMedia(conversation, using: imageDescriber)
        } else if hasMedia, let maxEdge = model.maxImageLongEdge, let resizer = imageResizer {
            effectiveConversation = try await resizingImages(in: effectiveConversation, maxLongEdge: maxEdge, using: resizer)
        }

        // Build request with streaming enabled
        var request = effectiveConversation.request(for: provider)
        request.stream = true
        if provider.isOpenAI {
            request.stream_options = OpenAICompatibleAPI.StreamOptions(include_usage: true)
        }

        let jsonData = try JSONEncoder().encode(request)

        // Platform-conditional streaming setup
        #if canImport(AsyncHTTPClient)
            let (parser, rateLimitInfo) = try await api.streamingChatCompletionLinux(
                with: jsonData, provider: provider
            )
            if let info = rateLimitInfo {
                await chatRateLimiter.updateLimits(
                    maxRequests: info.requestLimit,
                    maxTokens: info.tokenLimit
                )
            }
            try await processStream(
                parser: parser, isAnthropic: isAnthropic,
                continuation: continuation, conversation: conversation
            )
        #else
            let (parser, httpResponse, _) = try await api.streamingChatCompletion(with: jsonData)
            if let info = RateLimitInfo.parse(from: httpResponse, provider: provider) {
                await chatRateLimiter.updateLimits(
                    maxRequests: info.requestLimit,
                    maxTokens: info.tokenLimit
                )
            }
            try await processStream(
                parser: parser, isAnthropic: isAnthropic,
                continuation: continuation, conversation: conversation
            )
        #endif
    }

    /// Processes an SSE parser stream, emitting deltas and building the final response.
    ///
    /// This is the shared implementation used by both the macOS (`URLSession`) and
    /// Linux (`AsyncHTTPClient`) streaming paths.
    private func processStream<Lines: AsyncSequence>(
        parser: OpenAICompatibleAPI.SSEParser<Lines>,
        isAnthropic: Bool,
        continuation: AsyncThrowingStream<StreamEvent, Error>.Continuation,
        conversation: Conversation
    ) async throws where Lines.Element == String {
        var accumulator = OpenAICompatibleAPI.StreamAccumulator()

        if isAnthropic {
            try await processAnthropicStream(parser: parser, accumulator: &accumulator, continuation: continuation)
        } else {
            try await processOpenAIStream(parser: parser, accumulator: &accumulator, continuation: continuation)
        }

        // Build final response from accumulator
        let rawResponse = accumulator.buildResponse(isAnthropic: isAnthropic)

        let text = rawResponse.content?.first(where: { $0.type == .text })?.text
            ?? rawResponse.choices?.first?.message.textContent

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
            // Split on newlines to handle providers (e.g. OpenRouter) that bundle
            // multiple JSON chunks into a single SSE event's data field.
            let jsonLines = event.data
                .split(separator: "\n", omittingEmptySubsequences: true)
                .map(String.init)

            for jsonString in jsonLines.isEmpty ? [event.data] : jsonLines {
                guard let data = jsonString.data(using: .utf8) else { continue }

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
                    if let reasoning = choice.delta.thinking, !reasoning.isEmpty {
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
