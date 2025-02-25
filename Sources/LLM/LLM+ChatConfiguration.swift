//
//  File.swift
//  LLM
//
//  Created by Ben Syverson on 2025-02-24.
//

import Foundation

public extension LLM {
	struct ChatConfiguration: Friendly {
		public var systemPrompt: String
		public var user: String
		public var modelType: ModelType = .fast
		public var inference: InferenceType = .direct
		public var temperature: Double?
		public var frequencyPenalty: Double?
		public var repeatPenalty: Double?
		public var topP: Double?
		public var maxTokens: Int?
		public var maxReasoningTokens: Int?
		public var stopTokens: [String]?

		public init(
			systemPrompt: String,
			user: String,
			modelType: ModelType,
			inference: InferenceType,
			temperature: Double? = nil,
			frequencyPenalty: Double? = nil,
			repeatPenalty: Double? = nil,
			topP: Double? = nil,
			maxTokens: Int? = nil,
			maxReasoningTokens: Int? = nil,
			stopTokens: [String]? = nil
		) {
			self.systemPrompt = systemPrompt
			self.user = user
			self.modelType = modelType
			self.inference = inference
			self.temperature = temperature
			self.frequencyPenalty = frequencyPenalty
			self.repeatPenalty = repeatPenalty
			self.topP = topP
			self.maxTokens = maxTokens
			self.maxReasoningTokens = maxReasoningTokens
			self.stopTokens = stopTokens
		}
	}
}

public extension LLM.ChatConfiguration {
	func request(for provider: LLM.Provider) -> LLM.OpenAICompatibleAPI.ChatCompletion {
		let isAnthropic = provider.isAnthropic
		let skipTemp = isAnthropic && inference == .reasoning
		let skipTopP = skipTemp
		let skipFreq = isAnthropic

		let thinking: LLM.OpenAICompatibleAPI.ChatCompletion.Thinking? = inference == .reasoning ? .init(budget_tokens: maxReasoningTokens ?? 1024) : nil
		return LLM.OpenAICompatibleAPI.ChatCompletion(
			model: provider.model(
				type: modelType,
				inference: inference
			),
			messages: [
				.init(content: systemPrompt, role: .system),
				.init(content: user, role: .user)
			],
			response_format: nil,
			temperature: skipTemp ? nil : temperature,
			frequency_penalty: skipFreq ? nil : frequencyPenalty,
			top_p: skipTopP ? nil : topP,
			max_tokens: maxTokens,
			stop: isAnthropic ? nil : stopTokens,
			stop_sequences: isAnthropic ? stopTokens : nil,
			thinking: thinking
		)
	}
}
