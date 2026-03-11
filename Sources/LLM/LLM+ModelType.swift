//
//  LLM+ModelType.swift
//
//
//  Created by Ben Syverson on 2024-01-26.
//

import Foundation

public extension LLM {
    /// The model performance tier. Each provider maps these to specific model names.
    enum ModelType: String, Friendly {
        /// A fast, cost-efficient model (e.g. GPT-5 Mini, Claude Haiku).
        case fast
		/// A balanced mid-tier model (e.g. Claude Sonnet).
		case standard
        /// The most capable model available (e.g. GPT-5.2, Claude Opus).
        case flagship
    }

    /// The inference mode controlling whether the model uses chain-of-thought reasoning.
    enum InferenceType: String, Friendly {
        /// Standard inference without explicit reasoning.
        case direct
        /// Enables extended thinking / chain-of-thought reasoning.
        case reasoning
    }
}

public extension LLM.Provider {
    /// Returns the concrete model name for this provider given a model tier and inference type.
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
			case .standard:
				return .gpt5Mini
            case .flagship:
                return .gpt52
            }
        case .anthropic(apiKey: _):
            switch type {
            case .fast:
                return .claude45Haiku
			case .standard:
				return .claude4Sonnet
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
