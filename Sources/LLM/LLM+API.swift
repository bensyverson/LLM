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
		model: ModelType = .fastest,
		dimensions: Int? = nil
	) async throws -> [Float] {
		let api = providerApi

		try await embeddingRateLimiter.acquire(tokens: input.count / 2)

		let request = OpenAICompatibleAPI.EmbeddingRequest(
			input: input,
			model: model == .highestInteractive ? .textEmbedding3Large : .textEmbedding3Small,
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

	func chat(
		systemPrompt: String,
		user: String,
		model: ModelType = .fastest,
		temperature: Double = 0.7,
		frequency_penalty: Double? = nil,
		repeat_penalty: Double = 1.0,
		top_p: Double = 0.95,
		maxTokens: Int? = 512,
		stopTokens: [String] = []
	) async throws -> String {
		let api = providerApi

		let tokenCount = Int(Double((systemPrompt + user).count) / 2.0) // over-estimate the token count
		try await chatRateLimiter.acquire(tokens: tokenCount)

		let request = OpenAICompatibleAPI.ChatCompletion(
			model: provider.model(type: model),
			messages: [
				.init(content: systemPrompt, role: .system),
				.init(content: user, role: .user)
			],
			response_format: nil,
			temperature: temperature,
			frequency_penalty: frequency_penalty,
			top_p: top_p,
			max_tokens: maxTokens,
			stop: stopTokens
		)

		let encoder = JSONEncoder()
		let json = try encoder.encode(request)

		do {
			let response = try await api.chatCompletion(with: json)
			guard let content = response.choices.first?.message.content else {
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
