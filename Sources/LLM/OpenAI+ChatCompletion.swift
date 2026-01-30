//
//  File.swift
//  LLM
//
//  Created by Ben Syverson on 2024-11-13.
//

import Foundation

public extension LLM.OpenAICompatibleAPI {
	struct ChatCompletion: Codable {
		public struct JsonObject: Codable {
			public enum ObjectType: String, Codable {
				case json_object
			}
			public var type: ObjectType = .json_object

			public init(type: LLM.OpenAICompatibleAPI.ChatCompletion.JsonObject.ObjectType = .json_object) {
				self.type = type
			}
		}
		public struct Thinking: Codable {
			public enum ThinkingType: String, Codable {
				case enabled, disabled
			}
			public var type: ThinkingType = .enabled
			public var budget_tokens: Int = 1024
			public init(
				type: ThinkingType = .enabled,
				budget_tokens: Int = 1024
			) {
				self.type = type
				self.budget_tokens = budget_tokens
			}
		}
		public enum ReasoningEffort: String, Codable, Sendable {
			case none, low, medium, high, xhigh, minimal
		}
		public var model: ModelName = .gpt35turbo
		public var system: String? = nil
		public var messages: [ChatMessage]
		public var response_format: JsonObject? = JsonObject()
		public var temperature: Double? = 1.0
		public var frequency_penalty: Double? = nil
		public var top_p: Double? = 1.0
		public var max_tokens: Int? = nil
		public var max_completion_tokens: Int? = nil
		public var stop: [String]? = ["###"]
		public var stop_sequences: [String]? = nil
		public var thinking: Thinking? = nil
		public var reasoning_effort: ReasoningEffort? = nil
		public var tools: [ToolDefinition]? = nil
		public var tool_choice: ToolChoice? = nil

		public init(
			model: LLM.OpenAICompatibleAPI.ModelName = .gpt35turbo,
			system: String? = nil,
			messages: [LLM.OpenAICompatibleAPI.ChatMessage],
			response_format: LLM.OpenAICompatibleAPI.ChatCompletion.JsonObject? = JsonObject(),
			temperature: Double? = nil,
			frequency_penalty: Double? = nil,
			top_p: Double? = nil,
			max_tokens: Int? = nil,
			max_completion_tokens: Int? = nil,
			stop: [String]? = nil,
			stop_sequences: [String]? = nil,
			thinking: Thinking? = nil,
			reasoning_effort: ReasoningEffort? = nil,
			tools: [ToolDefinition]? = nil,
			tool_choice: ToolChoice? = nil
		) {
			self.model = model
			self.system = system
			self.messages = messages
			self.response_format = response_format
			self.temperature = temperature
			self.frequency_penalty = frequency_penalty
			self.top_p = top_p
			self.max_tokens = max_tokens
			self.max_completion_tokens = max_completion_tokens
			self.stop = stop
			self.stop_sequences = stop_sequences
			self.thinking = thinking
			self.reasoning_effort = reasoning_effort
			self.tools = tools
			self.tool_choice = tool_choice
		}
	}

	enum Role: String, Friendly {
		case system, user, assistant, tool
	}

	struct ChatMessage: Friendly {
		public var content: String?
		public var role: Role
		public var name: String?
		public var tool_calls: [ToolCall]?
		public var tool_call_id: String?

		// Full initializer
		public init(
			content: String?,
			role: Role,
			name: String? = nil,
			tool_calls: [ToolCall]? = nil,
			tool_call_id: String? = nil
		) {
			self.content = content
			self.role = role
			self.name = name
			self.tool_calls = tool_calls
			self.tool_call_id = tool_call_id
		}

		// Backward-compatible convenience init
		public init(
			content: String,
			role: Role,
			name: String? = nil
		) {
			self.content = content
			self.role = role
			self.name = name
			self.tool_calls = nil
			self.tool_call_id = nil
		}

		// Helper to get content length (for token estimation)
		public var contentLength: Int {
			content?.count ?? 0
		}
	}

	struct ChatCompletionResponse: Friendly {
		public enum ObjectType: String, Friendly {
			case chatCompletion = "chat.completion"
		}

		public struct Usage: Friendly {
			public let prompt_tokens: Int?
			public let completion_tokens: Int?
			public let total_tokens: Int?
			public let input_tokens: Int?
			public let output_tokens: Int?
		}

		public struct Choice: Friendly {
			public let index: Int
			public let message: ChatMessage
			public let finish_reason: String?
		}

		// Anthropic Content
		public struct Content: Friendly {
			public enum ContentType: String, Friendly {
				case text, thinking, redacted_thinking, tool_use, tool_result
			}
			public var type: ContentType
			public var text: String?
			public var thinking: String?  // Anthropic thinking content
			public var data: Data?
			public var signature: String?
			// Tool use fields (Anthropic)
			public var id: String?
			public var name: String?
			public var input: [String: String]?
			// Tool result fields (Anthropic)
			public var tool_use_id: String?
			public var content: String?
		}

		public let id: String?
		public let object: ObjectType?
		public let system_fingerprint: String?
		public let usage: Usage
		public let model: String
		public let created: Date? // unix timestamp
		public let choices: [Choice]?
		public let content: [Content]?
	}
}
