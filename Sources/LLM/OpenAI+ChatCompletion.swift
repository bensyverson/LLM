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
		public var model: ModelName = .gpt35turbo
		public var messages: [ChatMessage]
		public var response_format: JsonObject? = JsonObject()
		public var temperature: Double? = 1.0
		public var frequency_penalty: Double? = nil
		public var top_p: Double? = 1.0
		public var max_tokens: Int? = nil
		public var stop: [String]? = ["###"]
		public var stop_sequences: [String]? = nil
		public var thinking: Thinking? = nil

		public init(
			model: LLM.OpenAICompatibleAPI.ModelName = .gpt35turbo,
			messages: [LLM.OpenAICompatibleAPI.ChatMessage],
			response_format: LLM.OpenAICompatibleAPI.ChatCompletion.JsonObject? = JsonObject(),
			temperature: Double? = nil,
			frequency_penalty: Double? = nil,
			top_p: Double? = nil,
			max_tokens: Int? = nil,
			stop: [String]? = nil,
			stop_sequences: [String]? = nil,
			thinking: Thinking? = nil
		) {
			self.model = model
			self.messages = messages
			self.response_format = response_format
			self.temperature = temperature
			self.frequency_penalty = frequency_penalty
			self.top_p = top_p
			self.max_tokens = max_tokens
			self.stop = stop
			self.stop_sequences = stop_sequences
			self.thinking = thinking
		}
	}

	enum Role: String, Friendly {
		case system, user, assistant, tool
	}

	struct ChatMessage: Friendly {
		public var content: String
		public var role: Role
		public var name: String?

		public init(
			content: String,
			role: Role,
			name: String? = nil
		) {
			self.content = content
			self.role = role
			self.name = name
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
		}

		// Anthropic Content
		public struct Content: Friendly {
			public enum ContentType: String, Friendly {
				case text, thinking, redacted_thinking
			}
			public var type: ContentType
			public var text: String?
			public var data: Data?
			public var signature: String?
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
