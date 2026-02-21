//
//  LLM+ModelType.swift
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
        inference _: LLM.InferenceType
    ) -> LLM.OpenAICompatibleAPI.ModelName {
        switch self {
        case .openAI(apiKey: _):
            // GPT-5 models have native reasoning via reasoning_effort parameter
            switch type {
            case .fast:
                return .gpt5Mini
            case .flagship:
                return .gpt52
            }
        case .anthropic(apiKey: _):
            switch type {
            case .fast:
                return .claude45Haiku
            case .flagship:
                return .claude46Opus
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
