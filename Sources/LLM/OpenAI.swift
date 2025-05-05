//
//  File.swift
//  
//
//  Created by Ben Syverson on 2023-12-27.
//

import Foundation

public extension LLM {
	struct OpenAICompatibleAPI: Friendly {
		public enum ModelName: String, Codable {
			case placeholder = "placeholder"
			case gpt35turbo = "gpt-3.5-turbo"
			case gpt35turbo16k = "gpt-3.5-turbo-16k"
			case gpt4 = "gpt-4"
			case gpt4turbo = "gpt-4-turbo"
			case gpt4o = "gpt-4o"
			case gpt4oMini = "gpt-4o-mini"
			case gpt41 = "gpt-4.1"
			case gpt41Mini = "gpt-4.1-mini"
			case gpt41Nano = "gpt-4.1-nano"
			case o1preview = "o1-preview"
			case o1mini = "o1-mini"
			case o1 = "o1"
			case o3 = "o3"
			case o3mini = "o3-mini"
			case o4mini = "o4-mini"
			case claude37SonnetLatest = "claude-3-7-sonnet-latest"
			case claude35HaikuLatest = "claude-3-5-haiku-latest"
		}

		public enum OpenAIError: Error {
			case noEmbedding
			case badResponse(URLResponse)
			case badResponseCode(Int)
		}

		public enum AuthenticationMethod: Friendly {
			case none
			case bearer(apiKey: String)
			case xApiKey(apiKey: String)
		}

		public var baseURL: URL
		public var authenticationMethod: AuthenticationMethod
		public var chatEndpoint: String
		public var headers: [String: String]?

		public static func openAI(apiKey: String) -> Self {
			.init(
				baseURL: URL(string: "https://api.openai.com/")!,
				authMethod: .bearer(apiKey: apiKey)
			)
		}

		public static func anthropic(apiKey: String) -> Self {
			.init(
				baseURL: URL(string: "https://api.anthropic.com/")!,
				authMethod: .xApiKey(apiKey: apiKey),
				chatEndpoint: "v1/messages",
				headers: ["anthropic-version": "2023-06-01"]
			)
		}

		public static func localhost(port: Int) -> Self {
			.init(baseURL: URL(string: "http://localhost:\(port)/")!)
		}

		public init(
			baseURL: URL,
			authMethod: AuthenticationMethod = .none,
			chatEndpoint: String = "v1/chat/completions",
			headers: [String: String]? = nil
		) {
			self.baseURL = baseURL
			self.authenticationMethod = authMethod
			self.chatEndpoint = chatEndpoint
			self.headers = headers
		}
	}
}
