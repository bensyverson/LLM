//
//  LLM+Conversation.swift
//  LLM
//
//  Created by Claude on 2026-01-30.
//

import Foundation

public extension LLM {
	struct Conversation: Friendly {
		public var systemPrompt: String
		public var messages: [OpenAICompatibleAPI.ChatMessage]
		public var configuration: ConversationConfiguration

		public init(
			systemPrompt: String,
			messages: [OpenAICompatibleAPI.ChatMessage] = [],
			configuration: ConversationConfiguration = .init()
		) {
			self.systemPrompt = systemPrompt
			self.messages = messages
			self.configuration = configuration
		}

		public func addingUserMessage(_ content: String) -> Conversation {
			var copy = self
			copy.messages.append(OpenAICompatibleAPI.ChatMessage(content: content, role: .user))
			return copy
		}

		public func addingAssistantMessage(_ content: String) -> Conversation {
			var copy = self
			copy.messages.append(OpenAICompatibleAPI.ChatMessage(content: content, role: .assistant))
			return copy
		}
	}

	struct ConversationConfiguration: Friendly {
		public var modelType: ModelType
		public var inference: InferenceType
		public var temperature: Double?
		public var frequencyPenalty: Double?
		public var repeatPenalty: Double?
		public var topP: Double?
		public var maxTokens: Int?
		public var maxReasoningTokens: Int?
		public var reasoningEffort: LLM.OpenAICompatibleAPI.ChatCompletion.ReasoningEffort?
		public var stopTokens: [String]?

		public init(
			modelType: ModelType = .fast,
			inference: InferenceType = .direct,
			temperature: Double? = nil,
			frequencyPenalty: Double? = nil,
			repeatPenalty: Double? = nil,
			topP: Double? = nil,
			maxTokens: Int? = nil,
			maxReasoningTokens: Int? = nil,
			reasoningEffort: LLM.OpenAICompatibleAPI.ChatCompletion.ReasoningEffort? = nil,
			stopTokens: [String]? = nil
		) {
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

	struct ConversationResponse: Friendly {
		public let text: String
		public let conversation: Conversation
		public let rawResponse: OpenAICompatibleAPI.ChatCompletionResponse
	}
}

public extension LLM.Conversation {
	func request(for provider: LLM.Provider) -> LLM.OpenAICompatibleAPI.ChatCompletion {
		let isAnthropic = provider.isAnthropic
		let isOpenAI = provider.isOpenAI
		let model = provider.model(type: configuration.modelType, inference: configuration.inference)
		let isGPT5 = model.isGPT5
		let skipTemp = configuration.inference == .reasoning
		let skipTopP = skipTemp
		let skipFreq = isAnthropic
		let maxReasoningTokenCount = configuration.inference == .reasoning ? configuration.maxReasoningTokens ?? 1024 : 0
		let maxCompletionTokens = (configuration.maxTokens ?? 0) + maxReasoningTokenCount
		let thinking: LLM.OpenAICompatibleAPI.ChatCompletion.Thinking? = (isAnthropic && configuration.inference == .reasoning) ? .init(budget_tokens: maxReasoningTokenCount) : nil

		// For GPT-5 models with .reasoning inference, auto-set reasoning_effort if not specified
		let effectiveReasoningEffort: LLM.OpenAICompatibleAPI.ChatCompletion.ReasoningEffort? = {
			if isOpenAI && configuration.inference == .reasoning && isGPT5 {
				return configuration.reasoningEffort ?? .high
			}
			return configuration.reasoningEffort
		}()

		// Build messages for the conversation
		var conversationMessages: [LLM.OpenAICompatibleAPI.ChatMessage] = []
		if !isAnthropic {
			conversationMessages.append(LLM.OpenAICompatibleAPI.ChatMessage(content: systemPrompt, role: .system))
		}
		conversationMessages.append(contentsOf: messages)

		return LLM.OpenAICompatibleAPI.ChatCompletion(
			model: model,
			system: isAnthropic ? systemPrompt : nil,
			messages: conversationMessages,
			response_format: nil,
			temperature: skipTemp ? nil : configuration.temperature,
			frequency_penalty: skipFreq ? nil : configuration.frequencyPenalty,
			top_p: skipTopP ? nil : configuration.topP,
			max_tokens: isOpenAI ? nil : maxCompletionTokens,
			max_completion_tokens: isOpenAI && maxCompletionTokens > 0 ? maxCompletionTokens : nil,
			stop: isAnthropic ? nil : configuration.stopTokens,
			stop_sequences: isAnthropic ? configuration.stopTokens : nil,
			thinking: thinking,
			reasoning_effort: isAnthropic ? nil : effectiveReasoningEffort
		)
	}
}
