//
//  LLM+API.swift
//  LLM
//
//  Created by Ben Syverson on 2024-11-13.
//

import Foundation

public extension LLM {
    /// Generates an embedding vector for the given text.
    ///
    /// - Parameters:
    ///   - input: The text to embed.
    ///   - model: The model tier to use (`.fast` for small, `.flagship` for large).
    ///   - dimensions: Optional dimensionality override for the embedding.
    /// - Returns: The embedding as an array of floats.
    func embedding(
        input: String,
        model: ModelType = .fast,
        dimensions: Int? = nil,
    ) async throws -> [Float] {
        let api = providerApi

        try await embeddingRateLimiter.acquire(tokens: input.count / 2)

        let request = OpenAICompatibleAPI.EmbeddingRequest(
            input: input,
            model: model == .flagship ? .textEmbedding3Large : .textEmbedding3Small,
            encoding_format: .float,
            dimensions: dimensions,
        )
        let json = try requestEncoder.encode(request)
        return try await api.embedding(for: json)
    }

    /// Sends a one-shot chat completion and returns the full response.
    ///
    /// - Parameter configuration: The chat configuration including model, prompts, and parameters.
    /// - Returns: The raw ``OpenAICompatibleAPI/ChatCompletionResponse``.
    func chat(configuration: ChatConfiguration) async throws -> OpenAICompatibleAPI.ChatCompletionResponse {
        let api = providerApi
        let tokenCount = Int(Double((configuration.systemPrompt + configuration.user).count) / 2.0)
        try await chatRateLimiter.acquire(tokens: tokenCount)

        let request = configuration.request(for: provider)
        let jsonData = try requestEncoder.encode(request)
        let (response, httpResponse) = try await api.chatCompletion(with: jsonData)

        // Adapt rate limits from response headers
        if let info = RateLimitInfo.parse(from: httpResponse, provider: provider) {
            await chatRateLimiter.updateLimits(
                maxRequests: info.requestLimit,
                maxTokens: info.tokenLimit,
            )
        }

        return response
    }

    /// Sends a one-shot chat completion and returns just the text content.
    ///
    /// - Parameter configuration: The chat configuration including model, prompts, and parameters.
    /// - Returns: The text content from the response.
    /// - Throws: ``LLMError/parseResponse(_:)`` if no text content is found.
    func chat(configuration: ChatConfiguration) async throws -> String {
        let response: OpenAICompatibleAPI.ChatCompletionResponse = try await chat(configuration: configuration)

        guard let content = response.content?.first(where: { $0.type == .text })?.text ?? response.choices?.first?.message.textContent else {
            throw LLMError.parseResponse(response)
        }
        return content
    }

    // MARK: - Conversation API

    /// Sends a conversation to the model and returns the response with an updated conversation history.
    ///
    /// This is the primary conversation driver. It handles rate limiting, request encoding,
    /// response parsing (text, thinking, tool calls) for both OpenAI and Anthropic formats,
    /// and builds an updated ``Conversation`` with the assistant's reply appended.
    ///
    /// - Parameter conversation: The conversation to continue, including system prompt and message history.
    /// - Returns: A ``ConversationResponse`` containing the model's reply and updated conversation.
    /// - Throws: ``LLMError/parseResponse(_:)`` if the response contains neither text nor tool calls.
    func chat(conversation: Conversation) async throws -> ConversationResponse {
        let api = providerApi
        let messageTextLength = conversation.messages.reduce(0) { total, msg in
            total + msg.contentLength
        }
        let tokenCount = Int(Double(conversation.systemPrompt.count + messageTextLength) / 2.0)
        try await chatRateLimiter.acquire(tokens: tokenCount)

        // Pre-process: non-vision fallback and image resizing
        let model = conversation.configuration.model ?? provider.model(type: conversation.configuration.modelType, inference: conversation.configuration.inference)
        let hasMedia = conversation.messages.contains { $0.hasMedia }
        var warnings: [String] = []
        var effectiveConversation = conversation

        if hasMedia, model.supportsVision == false {
            effectiveConversation = try await strippingMedia(conversation, using: imageDescriber)
            warnings.append("Images were converted to text descriptions because \(model.rawValue) does not support vision.")
        } else if hasMedia, let maxEdge = model.maxImageLongEdge, let resizer = imageResizer {
            effectiveConversation = try await resizingImages(in: effectiveConversation, maxLongEdge: maxEdge, using: resizer)
        }

        let request = effectiveConversation.request(for: provider)
        let jsonData = try requestEncoder.encode(request)
        let (response, httpResponse) = try await api.chatCompletion(with: jsonData)

        // Adapt rate limits from response headers
        if let info = RateLimitInfo.parse(from: httpResponse, provider: provider) {
            await chatRateLimiter.updateLimits(
                maxRequests: info.requestLimit,
                maxTokens: info.tokenLimit,
            )
        }

        // Extract text content (may be nil if response is tool-calls-only)
        let text = response.content?.first(where: { $0.type == .text })?.text
            ?? response.choices?.first?.message.textContent

        // Extract thinking content from Anthropic extended thinking blocks or OpenAI reasoning_content
        let thinking = response.content?.first(where: { $0.type == .thinking })?.thinking
            ?? response.choices?.first?.message.reasoning_content

        // Extract tool calls from OpenAI format (choices) or Anthropic format (content)
        let toolCalls: [OpenAICompatibleAPI.ToolCall] = Self.extractToolCalls(from: response)

        // If we got neither text nor tool calls, the response is unparseable
        if text == nil, toolCalls.isEmpty {
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
            rawResponse: response,
            warnings: warnings,
        )
    }

    /// Extracts tool calls from either OpenAI or Anthropic response format.
    ///
    /// OpenAI returns tool calls in `choices[0].message.tool_calls`. Anthropic returns
    /// them as content blocks with `type == "tool_use"`. This method normalizes both
    /// formats into an array of ``OpenAICompatibleAPI/ToolCall``.
    ///
    /// - Parameter response: The raw chat completion response.
    /// - Returns: An array of tool calls, or an empty array if none were found.
    static func extractToolCalls(
        from response: OpenAICompatibleAPI.ChatCompletionResponse,
    ) -> [OpenAICompatibleAPI.ToolCall] {
        // OpenAI format: tool_calls on the message in choices
        if let openAIToolCalls = response.choices?.first?.message.tool_calls, !openAIToolCalls.isEmpty {
            return openAIToolCalls.filter { !$0.function.name.isEmpty }
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
                            arguments: argumentsJSON,
                        ),
                    )
                }
            }
        }

        return []
    }

    /// Starts a new conversation with the given system prompt and user message.
    ///
    /// - Parameters:
    ///   - systemPrompt: The system prompt that guides the model's behavior.
    ///   - userMessage: The initial user message.
    ///   - configuration: Optional conversation configuration (model, temperature, tools, etc.).
    /// - Returns: A ``ConversationResponse`` with the model's reply and the conversation history.
    func startConversation(
        systemPrompt: String,
        userMessage: String,
        configuration: ConversationConfiguration = .init(),
    ) async throws -> ConversationResponse {
        let conversation = Conversation(
            systemPrompt: systemPrompt,
            messages: [OpenAICompatibleAPI.ChatMessage(content: userMessage, role: .user)],
            configuration: configuration,
        )
        return try await chat(conversation: conversation)
    }

    /// Continues an existing conversation with a new user message.
    ///
    /// - Parameters:
    ///   - conversation: The conversation to continue.
    ///   - userMessage: The new user message to append.
    /// - Returns: A ``ConversationResponse`` with the model's reply and updated conversation.
    func continueConversation(
        _ conversation: Conversation,
        userMessage: String,
    ) async throws -> ConversationResponse {
        let updated = conversation.addingUserMessage(userMessage)
        return try await chat(conversation: updated)
    }

    /// Starts a new conversation with multimodal content.
    ///
    /// - Parameters:
    ///   - systemPrompt: The system prompt that guides the model's behavior.
    ///   - userMessage: The initial user message as content parts.
    ///   - configuration: Optional conversation configuration.
    /// - Returns: A ``ConversationResponse`` with the model's reply and the conversation history.
    func startConversation(
        systemPrompt: String,
        userMessage: [OpenAICompatibleAPI.ContentPart],
        configuration: ConversationConfiguration = .init(),
    ) async throws -> ConversationResponse {
        let conversation = Conversation(
            systemPrompt: systemPrompt,
            messages: [OpenAICompatibleAPI.ChatMessage(content: userMessage, role: .user)],
            configuration: configuration,
        )
        return try await chat(conversation: conversation)
    }

    /// Continues an existing conversation with multimodal content.
    ///
    /// - Parameters:
    ///   - conversation: The conversation to continue.
    ///   - userMessage: The new user message as content parts.
    /// - Returns: A ``ConversationResponse`` with the model's reply and updated conversation.
    func continueConversation(
        _ conversation: Conversation,
        userMessage: [OpenAICompatibleAPI.ContentPart],
    ) async throws -> ConversationResponse {
        let updated = conversation.addingUserMessage(userMessage)
        return try await chat(conversation: updated)
    }
}
