//
//  LLM+Providers.swift
//
//  Created by Ben Syverson on 2024-01-20.
//

import Foundation

public extension LLM {
	enum SimpleProvider: Friendly {
		case openAI
		case lmStudio
		case localhost(port: Int)
		case other(URL)

		public func fullProvider(using apiKey: String) -> LLM.Provider {
			switch self {
			case .openAI:
				return .openAI(apiKey: apiKey)
			case .lmStudio:
				return .lmStudio
			case .localhost(port: let port):
				return .localhost(port: port)
			case .other(let url):
				return .other(url, apiKey: apiKey)
			}
		}
	}
}

public extension LLM.Provider {
	var simpleProvider: LLM.SimpleProvider {
		switch self {
		case .openAI(apiKey: _):
			return .openAI
		case .lmStudio:
			return .lmStudio
		case .localhost(port: let port):
			return .localhost(port: port)
		case .other(let url, apiKey: _):
			return .other(url)
		}
	}
}
