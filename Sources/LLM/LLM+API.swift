//
//  File.swift
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

		guard let text = response.content?.first(where: { $0.type == .text })?.text ?? response.choices?.first?.message.content else {
			throw LLMError.parseResponse(response)
		}

		let updatedConversation = conversation.addingAssistantMessage(text)
		return ConversationResponse(text: text, conversation: updatedConversation, rawResponse: response)
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
