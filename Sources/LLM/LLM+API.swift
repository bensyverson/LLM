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
}
