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
			case o1preview = "o1-preview"
			case o1mini = "o1-mini"
		}

		public enum OpenAIError: Error {
			case noEmbedding
			case badResponse(URLResponse)
			case badResponseCode(Int)
		}
		public var baseURL: URL
		public var apiKey: String? = nil

		public static func openAI(apiKey: String) -> Self {
			.init(
				baseURL: URL(string: "https://api.openai.com/")!,
				apiKey: apiKey
			)
		}

		public static func localhost(port: Int) -> Self {
			.init(baseURL: URL(string: "http://localhost:\(port)/")!)
		}

		public init(baseURL: URL, apiKey: String? = nil) {
			self.baseURL = baseURL
			self.apiKey = apiKey
		}
	}
}
