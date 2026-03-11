//
//  LLM+ChatConfiguration.swift
//  LLM
//
//  Created by Ben Syverson on 2025-02-24.
//

import Foundation

public extension LLM {
    /// Configuration for a single-shot chat request (no conversation history).
    ///
    /// Use this for simple prompt/response interactions where you don't need
    /// to maintain a multi-turn conversation.
    struct ChatConfiguration: Friendly {
        /// The system prompt that guides the model's behavior.
        public var systemPrompt: String
        /// The user's message.
        public var user: String
        /// The model tier to use (`.fast`, `.standard` or `.flagship`).
        public var modelType: ModelType = .fast
        /// The inference mode (`.direct` or `.reasoning`).
        public var inference: InferenceType = .direct
        /// An explicit model name override. When set, ``request(for:)``
        /// uses this instead of asking the provider for its default model.
        public var model: LLM.OpenAICompatibleAPI.ModelName?
        public var temperature: Double?
        public var frequencyPenalty: Double?
        public var repeatPenalty: Double?
        public var topP: Double?
        public var maxTokens: Int?
        public var maxReasoningTokens: Int?
        public var reasoningEffort: LLM.OpenAICompatibleAPI.ChatCompletion.ReasoningEffort?
        public var stopTokens: [String]?
        public var enableCaching: Bool = true
        public var cacheTTL: LLM.OpenAICompatibleAPI.CacheControl.TTL?

        public init(
            systemPrompt: String,
            user: String,
            modelType: ModelType,
            inference: InferenceType,
            model: LLM.OpenAICompatibleAPI.ModelName? = nil,
            temperature: Double? = nil,
            frequencyPenalty: Double? = nil,
            repeatPenalty: Double? = nil,
            topP: Double? = nil,
            maxTokens: Int? = nil,
            maxReasoningTokens: Int? = nil,
            reasoningEffort: LLM.OpenAICompatibleAPI.ChatCompletion.ReasoningEffort? = nil,
            stopTokens: [String]? = nil,
            enableCaching: Bool = true,
            cacheTTL: LLM.OpenAICompatibleAPI.CacheControl.TTL? = nil
        ) {
            self.systemPrompt = systemPrompt
            self.user = user
            self.modelType = modelType
            self.inference = inference
            self.model = model
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
}

public extension LLM.ChatConfiguration {
    /// Builds a ``LLM/LLM/OpenAICompatibleAPI/ChatCompletion`` request configured for the given provider.
    func request(for provider: LLM.Provider) -> LLM.OpenAICompatibleAPI.ChatCompletion {
        let isAnthropic = provider.isAnthropic
        let isOpenAI = provider.isOpenAI
        let model = model ?? provider.model(type: modelType, inference: inference)
        let isGPT5 = model.isGPT5
        // For GPT-5 models, always skip temperature/topP (they only support default values)
        // For older o-series reasoning models, also skip
        let skipTemp = isGPT5 || inference == .reasoning
        // Anthropic forbids specifying both temperature and top_p
        let skipTopP = skipTemp || (isAnthropic && temperature != nil)
        let skipFreq = isAnthropic
        let skipStop = isGPT5 // GPT-5 doesn't support stop parameter
        let maxReasoningTokenCount = inference == .reasoning ? maxReasoningTokens ?? 1024 : 0
        // Token budget calculation differs between providers:
        // - OpenAI: reasoning + output share max_completion_tokens budget
        // - Anthropic: reasoning (budget_tokens) is separate from output (max_tokens)
        let maxCompletionTokens: Int = {
            if isAnthropic {
                // Anthropic: only use maxTokens for output (thinking has separate budget)
                // Fall back to the model's documented max, then a safe default
                // Treat 0 as unset since Anthropic requires max_tokens >= 1
                if let maxTokens, maxTokens > 0 { return maxTokens }
                return model.maxOutputTokens ?? 16384
            } else if isGPT5 && inference == .reasoning && maxTokens == nil {
                // GPT-5 with reasoning: if user doesn't specify maxTokens, don't set a limit
                return 0
            } else {
                // OpenAI non-GPT5 or with explicit maxTokens: combine reasoning + output
                return (maxTokens ?? 0) + maxReasoningTokenCount
            }
        }()
        let thinking: LLM.OpenAICompatibleAPI.ChatCompletion.Thinking? = (isAnthropic && inference == .reasoning) ? .init(budget_tokens: maxReasoningTokenCount) : nil

        // For GPT-5 models with .reasoning inference, auto-set reasoning_effort if not specified
        let effectiveReasoningEffort: LLM.OpenAICompatibleAPI.ChatCompletion.ReasoningEffort? = {
            if isOpenAI && inference == .reasoning && isGPT5 {
                return reasoningEffort ?? .high
            }
            return reasoningEffort
        }()

        let messages: [LLM.OpenAICompatibleAPI.ChatMessage] = isAnthropic ? [
            LLM.OpenAICompatibleAPI.ChatMessage(content: user, role: .user),
        ] : [
            LLM.OpenAICompatibleAPI.ChatMessage(content: systemPrompt, role: .system),
            LLM.OpenAICompatibleAPI.ChatMessage(content: user, role: .user),
        ]

        // Build system prompt - use array format with per-block cache_control for Anthropic
        let systemString: String? = (isAnthropic && !enableCaching) ? systemPrompt : (isAnthropic ? nil : nil)
        let systemBlocks: [LLM.OpenAICompatibleAPI.SystemContentBlock]? = (isAnthropic && enableCaching) ? [
            LLM.OpenAICompatibleAPI.SystemContentBlock(
                text: systemPrompt,
                cache_control: LLM.OpenAICompatibleAPI.CacheControl(ttl: cacheTTL ?? .fiveMinutes)
            ),
        ] : nil

        return LLM.OpenAICompatibleAPI.ChatCompletion(
            model: model,
            system: systemString,
            systemBlocks: systemBlocks,
            messages: messages,
            response_format: nil,
            temperature: skipTemp ? nil : temperature,
            frequency_penalty: skipFreq ? nil : frequencyPenalty,
            top_p: skipTopP ? nil : topP,
            max_tokens: isOpenAI || maxCompletionTokens <= 0 ? nil : maxCompletionTokens,
            max_completion_tokens: isOpenAI && maxCompletionTokens > 0 ? maxCompletionTokens : nil,
            stop: (isAnthropic || skipStop) ? nil : stopTokens,
            stop_sequences: isAnthropic ? stopTokens : nil,
            thinking: thinking,
            reasoning_effort: isAnthropic ? nil : effectiveReasoningEffort
        )
    }
}
