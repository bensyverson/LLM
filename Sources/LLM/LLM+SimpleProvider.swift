//
//  LLM+SimpleProvider.swift
//
//  Created by Ben Syverson on 2024-01-20.
//

import Foundation

public extension LLM {
    /// A provider identifier without credentials, useful for configuration and serialization.
    ///
    /// Use ``fullProvider(using:)`` to convert back to a ``Provider`` with an API key.
    enum SimpleProvider: Friendly {
        case openAI
        case anthropic
        case lmStudio
        case localhost(port: Int)
        case other(URL)

        /// Creates a full ``Provider`` by attaching the given API key.
        public func fullProvider(using apiKey: String) -> LLM.Provider {
            switch self {
            case .openAI:
                return .openAI(apiKey: apiKey)
            case .anthropic:
                return .anthropic(apiKey: apiKey)
            case .lmStudio:
                return .lmStudio
            case let .localhost(port: port):
                return .localhost(port: port)
            case let .other(url):
                return .other(url, apiKey: apiKey)
            }
        }
    }
}

public extension LLM.Provider {
    /// The credential-free ``SimpleProvider`` equivalent of this provider.
    var simpleProvider: LLM.SimpleProvider {
        switch self {
        case .openAI(apiKey: _):
            return .openAI
        case .anthropic(apiKey: _):
            return .anthropic
        case .lmStudio:
            return .lmStudio
        case let .localhost(port: port):
            return .localhost(port: port)
        case .other(let url, apiKey: _):
            return .other(url)
        }
    }

    /// Whether this provider sends requests to Anthropic's API.
    var isAnthropic: Bool {
        simpleProvider == .anthropic
    }

    /// Whether this provider sends requests to OpenAI's API.
    var isOpenAI: Bool {
        simpleProvider == .openAI
    }
}
