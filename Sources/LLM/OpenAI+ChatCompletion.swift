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
		public var model: ModelName = .gpt35turbo
		public var messages: [ChatMessage]
		public var response_format: JsonObject? = JsonObject()
		public var temperature: Double = 1.0
		public var frequency_penalty: Double? = nil
		public var top_p: Double = 1.0
		public var max_tokens: Int? = nil
		public var stop = ["###"]

		public init(
			model: LLM.OpenAICompatibleAPI.ModelName = .gpt35turbo,
			messages: [LLM.OpenAICompatibleAPI.ChatMessage],
			response_format: LLM.OpenAICompatibleAPI.ChatCompletion.JsonObject? = JsonObject(),
			temperature: Double = 1.0,
			frequency_penalty: Double? = nil,
			top_p: Double = 1.0,
			max_tokens: Int? = nil,
			stop: [String] = ["###"]
		) {
			self.model = model
			self.messages = messages
			self.response_format = response_format
			self.temperature = temperature
			self.frequency_penalty = frequency_penalty
			self.top_p = top_p
			self.max_tokens = max_tokens
			self.stop = stop
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
			public let prompt_tokens: Int
			public let completion_tokens: Int
			public let total_tokens: Int
		}

		public struct Choice: Friendly {
			public let index: Int
			public let message: ChatMessage
		}

		public let id: String?
		public let object: ObjectType
		public let system_fingerprint: String?
		public let usage: Usage
		public let model: String
		public let created: Date // unix timestamp
		public let choices: [Choice]
	}
}
