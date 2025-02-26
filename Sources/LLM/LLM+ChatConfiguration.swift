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
		public var reasoningEffort: LLM.OpenAICompatibleAPI.ChatCompletion.ReasoningEffort?
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
			reasoningEffort: LLM.OpenAICompatibleAPI.ChatCompletion.ReasoningEffort? = nil,
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
			self.reasoningEffort = reasoningEffort
			self.stopTokens = stopTokens
		}
	}
}

public extension LLM.ChatConfiguration {
	func request(for provider: LLM.Provider) -> LLM.OpenAICompatibleAPI.ChatCompletion {
		let isAnthropic = provider.isAnthropic
		let isOpenAI = provider.isOpenAI
		let skipTemp = inference == .reasoning
		let skipTopP = skipTemp
		let skipFreq = isAnthropic
		let maxReasoningTokenCount = inference == .reasoning ? maxReasoningTokens ?? 1024 : 0
		let maxCompletionTokens = (maxTokens ?? 0) + maxReasoningTokenCount
		let thinking: LLM.OpenAICompatibleAPI.ChatCompletion.Thinking? = (isAnthropic && inference == .reasoning) ? .init(budget_tokens: maxReasoningTokenCount) : nil

		let messages: [LLM.OpenAICompatibleAPI.ChatMessage] = isAnthropic ? [
			LLM.OpenAICompatibleAPI.ChatMessage(content: user, role: .user)
		] : [
			LLM.OpenAICompatibleAPI.ChatMessage(content: systemPrompt, role: .system),
			LLM.OpenAICompatibleAPI.ChatMessage(content: user, role: .user)
		]
		return LLM.OpenAICompatibleAPI.ChatCompletion(
			model: provider.model(
				type: modelType,
				inference: inference
			),
			system: isAnthropic ? systemPrompt : nil,
			messages: messages,
			response_format: nil,
			temperature: skipTemp ? nil : temperature,
			frequency_penalty: skipFreq ? nil : frequencyPenalty,
			top_p: skipTopP ? nil : topP,
			max_tokens: isOpenAI ? nil : maxCompletionTokens,
			max_completion_tokens: isOpenAI && maxCompletionTokens > 0 ? maxCompletionTokens : nil,
			stop: isAnthropic ? nil : stopTokens,
			stop_sequences: isAnthropic ? stopTokens : nil,
			thinking: thinking,
			reasoning_effort: isAnthropic ? nil : reasoningEffort
		)
	}
}
