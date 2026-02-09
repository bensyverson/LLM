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
		public var enableCaching: Bool = false
		public var cacheTTL: LLM.OpenAICompatibleAPI.CacheControl.TTL? = nil

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
			stopTokens: [String]? = nil,
			enableCaching: Bool = false,
			cacheTTL: LLM.OpenAICompatibleAPI.CacheControl.TTL? = nil
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
			self.enableCaching = enableCaching
			self.cacheTTL = cacheTTL
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
		// For GPT-5 models, always skip temperature/topP (they only support default values)
		// For older o-series reasoning models, also skip
		let skipTemp = isGPT5 || configuration.inference == .reasoning
		let skipTopP = skipTemp
		let skipFreq = isAnthropic
		let skipStop = isGPT5  // GPT-5 doesn't support stop parameter
		let maxReasoningTokenCount = configuration.inference == .reasoning ? configuration.maxReasoningTokens ?? 1024 : 0
		// Token budget calculation differs between providers:
		// - OpenAI: reasoning + output share max_completion_tokens budget
		// - Anthropic: reasoning (budget_tokens) is separate from output (max_tokens)
		let maxCompletionTokens: Int = {
			if isAnthropic {
				// Anthropic: only use maxTokens for output (thinking has separate budget)
				return configuration.maxTokens ?? 0
			} else if isGPT5 && configuration.inference == .reasoning && configuration.maxTokens == nil {
				// GPT-5 with reasoning: if user doesn't specify maxTokens, don't set a limit
				return 0
			} else {
				// OpenAI non-GPT5 or with explicit maxTokens: combine reasoning + output
				return (configuration.maxTokens ?? 0) + maxReasoningTokenCount
			}
		}()
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

		// Build system prompt - use array format with cache_control for Anthropic when caching enabled
		let systemString: String? = (isAnthropic && !configuration.enableCaching) ? systemPrompt : (isAnthropic ? nil : nil)
		let systemBlocks: [LLM.OpenAICompatibleAPI.SystemContentBlock]? = (isAnthropic && configuration.enableCaching) ? [
			LLM.OpenAICompatibleAPI.SystemContentBlock(
				text: systemPrompt,
				cache_control: LLM.OpenAICompatibleAPI.CacheControl(ttl: configuration.cacheTTL)
			)
		] : nil

		return LLM.OpenAICompatibleAPI.ChatCompletion(
			model: model,
			system: systemString,
			systemBlocks: systemBlocks,
			messages: conversationMessages,
			response_format: nil,
			temperature: skipTemp ? nil : configuration.temperature,
			frequency_penalty: skipFreq ? nil : configuration.frequencyPenalty,
			top_p: skipTopP ? nil : configuration.topP,
			max_tokens: isOpenAI ? nil : maxCompletionTokens,
			max_completion_tokens: isOpenAI && maxCompletionTokens > 0 ? maxCompletionTokens : nil,
			stop: (isAnthropic || skipStop) ? nil : configuration.stopTokens,
			stop_sequences: isAnthropic ? configuration.stopTokens : nil,
			thinking: thinking,
			reasoning_effort: isAnthropic ? nil : effectiveReasoningEffort
		)
	}
}
