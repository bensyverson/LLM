//
//  LLM+Conversation.swift
//  LLM
//
//  Created by Claude on 2026-01-30.
//

import Foundation

public extension LLM {
    /// A multi-turn conversation with a system prompt, message history, and configuration.
    ///
    /// `Conversation` is a value type. Each mutation (adding a message) returns a new copy,
    /// making it safe to branch conversations or retry from any point.
    struct Conversation: Friendly {
        /// The system prompt that guides the model's behavior throughout the conversation.
        public var systemPrompt: String
        /// The ordered list of messages in this conversation.
        public var messages: [OpenAICompatibleAPI.ChatMessage]
        /// Configuration controlling model selection, temperature, tools, etc.
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

        /// Returns a new conversation with a user message appended.
        public func addingUserMessage(_ content: String) -> Conversation {
            var copy = self
            copy.messages.append(OpenAICompatibleAPI.ChatMessage(content: content, role: .user))
            return copy
        }

        /// Returns a new conversation with a multimodal user message appended.
        public func addingUserMessage(_ parts: [OpenAICompatibleAPI.ContentPart]) -> Conversation {
            var copy = self
            copy.messages.append(OpenAICompatibleAPI.ChatMessage(content: parts, role: .user))
            return copy
        }

        /// Returns a new conversation with an assistant message appended.
        public func addingAssistantMessage(_ content: String) -> Conversation {
            var copy = self
            copy.messages.append(OpenAICompatibleAPI.ChatMessage(content: content, role: .assistant))
            return copy
        }

        /// Add an assistant message that contains tool calls (no text content).
        public func addingAssistantToolCallMessage(_ toolCalls: [OpenAICompatibleAPI.ToolCall]) -> Conversation {
            var copy = self
            copy.messages.append(OpenAICompatibleAPI.ChatMessage(
                content: nil,
                role: .assistant,
                tool_calls: toolCalls
            ))
            return copy
        }

        /// Add a tool result message.
        public func addingToolResultMessage(toolCallId: String, content: String) -> Conversation {
            var copy = self
            copy.messages.append(OpenAICompatibleAPI.ChatMessage(
                content: content,
                role: .tool,
                tool_call_id: toolCallId
            ))
            return copy
        }

        /// Add a tool result message with multimodal content.
        public func addingToolResultMessage(toolCallId: String, content: [OpenAICompatibleAPI.ContentPart]) -> Conversation {
            var copy = self
            copy.messages.append(OpenAICompatibleAPI.ChatMessage(
                content: content,
                role: .tool,
                tool_call_id: toolCallId
            ))
            return copy
        }
    }

    /// Configuration for a multi-turn conversation, including model, sampling parameters, and tools.
    struct ConversationConfiguration: Friendly {
        public var modelType: ModelType
        public var inference: InferenceType
        /// An explicit model name override. When set, ``LLM/LLM/Conversation/request(for:)``
        /// uses this instead of asking the provider for its default model.
        public var model: OpenAICompatibleAPI.ModelName?
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
        public var tools: [LLM.OpenAICompatibleAPI.ToolDefinition]?
        public var toolChoice: LLM.OpenAICompatibleAPI.ToolChoice?
        public var parallelToolCalls: Bool?

        public init(
            modelType: ModelType = .fast,
            inference: InferenceType = .direct,
            model: OpenAICompatibleAPI.ModelName? = nil,
            temperature: Double? = nil,
            frequencyPenalty: Double? = nil,
            repeatPenalty: Double? = nil,
            topP: Double? = nil,
            maxTokens: Int? = nil,
            maxReasoningTokens: Int? = nil,
            reasoningEffort: LLM.OpenAICompatibleAPI.ChatCompletion.ReasoningEffort? = nil,
            stopTokens: [String]? = nil,
            enableCaching: Bool = true,
            cacheTTL: LLM.OpenAICompatibleAPI.CacheControl.TTL? = nil,
            tools: [LLM.OpenAICompatibleAPI.ToolDefinition]? = nil,
            toolChoice: LLM.OpenAICompatibleAPI.ToolChoice? = nil,
            parallelToolCalls: Bool? = nil
        ) {
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
            self.tools = tools
            self.toolChoice = toolChoice
            self.parallelToolCalls = parallelToolCalls
        }
    }

    /// The result of a chat or streaming conversation request.
    ///
    /// Contains the model's reply (text and/or tool calls), any reasoning output,
    /// the updated conversation history, and the raw provider response.
    struct ConversationResponse: Friendly {
        /// The text content of the model's reply, or `nil` if the response only contains tool calls.
        public let text: String?
        /// The model's internal reasoning (Anthropic extended thinking or OpenAI reasoning), or `nil`.
        public let thinking: String?
        /// Tool calls requested by the model. Empty if no tools were called.
        public let toolCalls: [OpenAICompatibleAPI.ToolCall]
        /// The conversation with the model's reply appended.
        public let conversation: Conversation
        /// The raw, unprocessed response from the provider.
        public let rawResponse: OpenAICompatibleAPI.ChatCompletionResponse
        /// Warnings generated during request processing (e.g. image stripping for non-vision models).
        public let warnings: [String]

        public init(
            text: String?,
            thinking: String?,
            toolCalls: [OpenAICompatibleAPI.ToolCall],
            conversation: Conversation,
            rawResponse: OpenAICompatibleAPI.ChatCompletionResponse,
            warnings: [String] = []
        ) {
            self.text = text
            self.thinking = thinking
            self.toolCalls = toolCalls
            self.conversation = conversation
            self.rawResponse = rawResponse
            self.warnings = warnings
        }
    }
}

public extension LLM.Conversation {
    func request(for provider: LLM.Provider) -> LLM.OpenAICompatibleAPI.ChatCompletion {
        let isAnthropic = provider.isAnthropic
        let isMistral = provider.isMistral
        let isOpenAI = provider.isOpenAI
        let model = configuration.model ?? provider.model(type: configuration.modelType, inference: configuration.inference)
        let isGPT5 = model.isGPT5
        // For GPT-5 models, always skip temperature/topP (they only support default values)
        // For older o-series reasoning models, also skip
        let skipTemp = isGPT5 || configuration.inference == .reasoning
        // Anthropic forbids specifying both temperature and top_p
        let skipTopP = skipTemp || (isAnthropic && configuration.temperature != nil)
        let skipFreq = isAnthropic
        let skipStop = isGPT5 // GPT-5 doesn't support stop parameter
        let maxReasoningTokenCount = configuration.inference == .reasoning ? configuration.maxReasoningTokens ?? 1024 : 0
        // Token budget calculation differs between providers:
        // - OpenAI: reasoning + output share max_completion_tokens budget
        // - Anthropic: reasoning (budget_tokens) is separate from output (max_tokens)
        let maxCompletionTokens: Int = {
            if isAnthropic {
                // Anthropic: only use maxTokens for output (thinking has separate budget)
                // Fall back to the model's documented max, then a safe default
                // Treat 0 as unset since Anthropic requires max_tokens >= 1
                if let maxTokens = configuration.maxTokens, maxTokens > 0 { return maxTokens }
                return model.maxOutputTokens ?? 16384
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

        // Build system prompt - use array format with per-block cache_control for Anthropic
        let systemString: String? = (isAnthropic && !configuration.enableCaching) ? systemPrompt : (isAnthropic ? nil : nil)
        let systemBlocks: [LLM.OpenAICompatibleAPI.SystemContentBlock]? = (isAnthropic && configuration.enableCaching) ? [
            LLM.OpenAICompatibleAPI.SystemContentBlock(
                text: systemPrompt,
                cache_control: LLM.OpenAICompatibleAPI.CacheControl(ttl: configuration.cacheTTL)
            ),
        ] : nil

        var completion = LLM.OpenAICompatibleAPI.ChatCompletion(
            model: model,
            system: systemString,
            systemBlocks: systemBlocks,
            messages: conversationMessages,
            response_format: nil,
            temperature: skipTemp ? nil : configuration.temperature,
            frequency_penalty: skipFreq ? nil : configuration.frequencyPenalty,
            top_p: skipTopP ? nil : configuration.topP,
            max_tokens: isOpenAI || maxCompletionTokens <= 0 ? nil : maxCompletionTokens,
            max_completion_tokens: isOpenAI && maxCompletionTokens > 0 ? maxCompletionTokens : nil,
            stop: (isAnthropic || skipStop) ? nil : configuration.stopTokens,
            stop_sequences: isAnthropic ? configuration.stopTokens : nil,
            thinking: thinking,
            reasoning_effort: isAnthropic ? nil : effectiveReasoningEffort,
            tools: configuration.tools,
            tool_choice: configuration.toolChoice,
            parallel_tool_calls: isMistral ? configuration.parallelToolCalls : nil
        )
        completion.useAnthropicToolFormat = isAnthropic
        completion.useMistralFormat = isMistral
        if isAnthropic, configuration.enableCaching {
            completion.conversationCacheControl = LLM.OpenAICompatibleAPI.CacheControl(ttl: configuration.cacheTTL)
        }
        return completion
    }
}
