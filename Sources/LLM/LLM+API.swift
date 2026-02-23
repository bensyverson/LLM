//
//  LLM+API.swift
//  LLM
//
//  Created by Ben Syverson on 2024-11-13.
//

import Foundation

public extension LLM {
    func embedding(
        input: String,
        model: ModelType = .fast,
        dimensions: Int? = nil
    ) async throws -> [Float] {
        let api = providerApi

        try await embeddingRateLimiter.acquire(tokens: input.count / 2)

        let request = OpenAICompatibleAPI.EmbeddingRequest(
            input: input,
            model: model == .flagship ? .textEmbedding3Large : .textEmbedding3Small,
            encoding_format: .float,
            dimensions: dimensions
        )
        let encoder = JSONEncoder()
        let json = try encoder.encode(request)

        do {
            return try await api.embedding(for: json)
        } catch {
            print(error)
            throw error
        }
    }

    func chat(configuration: ChatConfiguration) async throws -> OpenAICompatibleAPI.ChatCompletionResponse {
        let api = providerApi
        let tokenCount = Int(Double((configuration.systemPrompt + configuration.user).count) / 2.0)
        try await chatRateLimiter.acquire(tokens: tokenCount)

        let request = configuration.request(for: provider)
        let jsonData = try JSONEncoder().encode(request)
        return try await api.chatCompletion(with: jsonData)
    }

    func chat(configuration: ChatConfiguration) async throws -> String {
        do {
            let response: OpenAICompatibleAPI.ChatCompletionResponse = try await chat(configuration: configuration)

            guard let content = response.content?.first(where: { $0.type == .text })?.text ?? response.choices?.first?.message.content else {
                print("Couldn't parse response")
                print(response)
                throw LLMError.parseResponse(response)
            }
            return content
        } catch {
            print("Chat completion error:")
            print(error)
            throw error
        }
    }

    // MARK: - Conversation API

    func chat(conversation: Conversation) async throws -> ConversationResponse {
        let api = providerApi
        let messageTextLength = conversation.messages.reduce(0) { total, msg in
            total + msg.contentLength
        }
        let tokenCount = Int(Double(conversation.systemPrompt.count + messageTextLength) / 2.0)
        try await chatRateLimiter.acquire(tokens: tokenCount)

        let request = conversation.request(for: provider)
        let jsonData = try JSONEncoder().encode(request)
        let response = try await api.chatCompletion(with: jsonData)

        // Extract text content (may be nil if response is tool-calls-only)
        let text = response.content?.first(where: { $0.type == .text })?.text
            ?? response.choices?.first?.message.content

        // Extract thinking content from Anthropic extended thinking blocks or OpenAI reasoning_content
        let thinking = response.content?.first(where: { $0.type == .thinking })?.thinking
            ?? response.choices?.first?.message.reasoning_content

        // Extract tool calls from OpenAI format (choices) or Anthropic format (content)
        let toolCalls: [OpenAICompatibleAPI.ToolCall] = Self.extractToolCalls(from: response)

        // If we got neither text nor tool calls, the response is unparseable
        if text == nil && toolCalls.isEmpty {
            throw LLMError.parseResponse(response)
        }

        // Build updated conversation based on what the response contains
        let updatedConversation: Conversation
        if !toolCalls.isEmpty {
            // Assistant responded with tool calls — may also include text
            var conv = conversation
            if let text {
                conv = conv.addingAssistantMessage(text)
            }
            updatedConversation = conv.addingAssistantToolCallMessage(toolCalls)
        } else {
            updatedConversation = conversation.addingAssistantMessage(text!)
        }

        return ConversationResponse(
            text: text,
            thinking: thinking,
            toolCalls: toolCalls,
            conversation: updatedConversation,
            rawResponse: response
        )
    }

    /// Extract tool calls from either OpenAI or Anthropic response format.
    static func extractToolCalls(
        from response: OpenAICompatibleAPI.ChatCompletionResponse
    ) -> [OpenAICompatibleAPI.ToolCall] {
        // OpenAI format: tool_calls on the message in choices
        if let openAIToolCalls = response.choices?.first?.message.tool_calls, !openAIToolCalls.isEmpty {
            return openAIToolCalls
        }

        // Anthropic format: content blocks with type == .tool_use
        if let contentBlocks = response.content {
            let toolUseBlocks = contentBlocks.filter { $0.type == .tool_use }
            if !toolUseBlocks.isEmpty {
                return toolUseBlocks.compactMap { block -> OpenAICompatibleAPI.ToolCall? in
                    guard let id = block.id, let name = block.name else { return nil }
                    // Convert the structured input dict back to a JSON string for FunctionCall.arguments
                    let argumentsJSON: String
                    if let input = block.input {
                        let data = (try? JSONEncoder().encode(input)) ?? Data()
                        argumentsJSON = String(data: data, encoding: .utf8) ?? "{}"
                    } else {
                        argumentsJSON = "{}"
                    }
                    return OpenAICompatibleAPI.ToolCall(
                        id: id,
                        function: OpenAICompatibleAPI.FunctionCall(
                            name: name,
                            arguments: argumentsJSON
                        )
                    )
                }
            }
        }

        return []
    }

    func startConversation(
        systemPrompt: String,
        userMessage: String,
        configuration: ConversationConfiguration = .init()
    ) async throws -> ConversationResponse {
        let conversation = Conversation(
            systemPrompt: systemPrompt,
            messages: [OpenAICompatibleAPI.ChatMessage(content: userMessage, role: .user)],
            configuration: configuration
        )
        return try await chat(conversation: conversation)
    }

    func continueConversation(
        _ conversation: Conversation,
        userMessage: String
    ) async throws -> ConversationResponse {
        let updated = conversation.addingUserMessage(userMessage)
        return try await chat(conversation: updated)
    }
}
