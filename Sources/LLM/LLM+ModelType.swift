//
//  LLM+ModelName.swift
//  
//
//  Created by Ben Syverson on 2024-01-26.
//

import Foundation

public extension LLM {
	enum ModelType: String, Friendly {
		case fast, flagship
	}

	enum InferenceType: String, Friendly {
		case direct, reasoning
	}
}

public extension LLM.Provider {
	func model(
		type: LLM.ModelType,
		inference: LLM.InferenceType
	) -> LLM.OpenAICompatibleAPI.ModelName {
		switch self {
		case .openAI(apiKey: _):
			switch type {
			case .fast:
				return inference == .direct ? .gpt4oMini : .o3mini
			case .flagship:
				return inference == .direct ? .gpt4o : .o1
			}
		case .anthropic(apiKey: _):
			switch type {
			case .fast:
				return inference == .direct ? .claude35HaikuLatest : .claude37SonnetLatest
			case .flagship:
				return .claude37SonnetLatest
			}
		case .lmStudio:
			return .placeholder
		case .localhost(port: _):
			return .placeholder
		case .other(_, apiKey: _):
			return .placeholder
		}
	}
}
