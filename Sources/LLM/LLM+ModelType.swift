//
//  LLM+ModelName.swift
//  
//
//  Created by Ben Syverson on 2024-01-26.
//

import Foundation

public extension LLM {
	enum ModelType: String, Friendly {
		case fastest, highestInteractive, highestNonInteractive
	}
}

public extension LLM.Provider {
	func model(type: LLM.ModelType) -> LLM.OpenAICompatibleAPI.ModelName {
		switch self {
		case .openAI(apiKey: _):
			switch type {
			case .fastest:
				return .gpt4oMini
			case .highestInteractive:
				return .gpt4o
			case .highestNonInteractive:
				return .o1preview
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
