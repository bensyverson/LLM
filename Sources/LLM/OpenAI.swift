//
//  OpenAI.swift
//
//
//  Created by Ben Syverson on 2023-12-27.
//

import Foundation
#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

public extension LLM {
    /// A configurable client for any OpenAI-compatible chat completions API.
    ///
    /// This type handles URL construction, authentication headers, and endpoint routing.
    /// Use the static factory methods (``openAI(apiKey:)``, ``anthropic(apiKey:)``,
    /// ``localhost(port:)``) for common providers, or the full initializer for custom endpoints.
    struct OpenAICompatibleAPI: Friendly {
        /// A model identifier for OpenAI-compatible APIs.
        ///
        /// Use the provided static constants for well-known models, or create
        /// custom identifiers with `ModelName(rawValue:)` for arbitrary model
        /// strings (e.g. when using OpenRouter or other proxy services).
        public struct ModelName: RawRepresentable, Friendly {
            public var rawValue: String

            public init(rawValue: String) {
                self.rawValue = rawValue
            }

            // MARK: - Placeholder

            public static let placeholder = ModelName(rawValue: "placeholder")

            // MARK: - OpenAI GPT-3.5

            public static let gpt35turbo = ModelName(rawValue: "gpt-3.5-turbo")
            public static let gpt35turbo16k = ModelName(rawValue: "gpt-3.5-turbo-16k")

            // MARK: - OpenAI GPT-4

            public static let gpt4 = ModelName(rawValue: "gpt-4")
            public static let gpt4turbo = ModelName(rawValue: "gpt-4-turbo")
            public static let gpt4o = ModelName(rawValue: "gpt-4o")
            public static let gpt4oMini = ModelName(rawValue: "gpt-4o-mini")

            // MARK: - OpenAI GPT-4.1

            public static let gpt41 = ModelName(rawValue: "gpt-4.1")
            public static let gpt41Mini = ModelName(rawValue: "gpt-4.1-mini")
            public static let gpt41Nano = ModelName(rawValue: "gpt-4.1-nano")

            // MARK: - OpenAI o-series

            public static let o1preview = ModelName(rawValue: "o1-preview")
            public static let o1mini = ModelName(rawValue: "o1-mini")
            public static let o1 = ModelName(rawValue: "o1")
            public static let o3 = ModelName(rawValue: "o3")
            public static let o3mini = ModelName(rawValue: "o3-mini")
            public static let o4mini = ModelName(rawValue: "o4-mini")

            // MARK: - OpenAI GPT-5 (2026)

            public static let gpt52 = ModelName(rawValue: "gpt-5.2")
            public static let gpt5Mini = ModelName(rawValue: "gpt-5-mini")
            public static let gpt5Nano = ModelName(rawValue: "gpt-5-nano")

            public static let gpt54 = ModelName(rawValue: "gpt-5.4")
            public static let gpt54Mini = ModelName(rawValue: "gpt-5.4-mini")
            public static let gpt54Nano = ModelName(rawValue: "gpt-5.4-nano")

            // MARK: - Anthropic Claude 4.6

            public static let claude46Opus = ModelName(rawValue: "claude-opus-4-6")
            public static let claude46Sonnet = ModelName(rawValue: "claude-sonnet-4-6")

            // MARK: - Anthropic Claude 4.5

            public static let claude45Opus = ModelName(rawValue: "claude-opus-4-5")
            public static let claude45Sonnet = ModelName(rawValue: "claude-sonnet-4-5")
            public static let claude45Haiku = ModelName(rawValue: "claude-haiku-4-5")

            // MARK: - Mistral

            public static let mistralSmall = ModelName(rawValue: "mistral-small-latest")
            public static let mistralMedium = ModelName(rawValue: "mistral-medium-latest")
            public static let mistralLarge = ModelName(rawValue: "mistral-large-latest")
            public static let devstral = ModelName(rawValue: "devstral-latest")

            // MARK: - Computed Properties

            /// Whether this model is in the GPT-5 family.
            public var isGPT5: Bool {
                rawValue.hasPrefix("gpt-5")
            }
        }

        /// Errors returned by the OpenAI-compatible API layer.
        public enum OpenAIError: Error {
            /// The embedding response contained no embedding data.
            case noEmbedding
            /// The server returned a non-HTTP response.
            case badResponse(URLResponse)
            /// The server returned a non-200 HTTP status code.
            case badResponseCode(Int)
        }

        /// The authentication method used for API requests.
        public enum AuthenticationMethod: Friendly {
            /// No authentication.
            case none
            /// Bearer token authentication (`Authorization: Bearer <key>`).
            case bearer(apiKey: String)
            /// Anthropic-style API key header (`x-api-key: <key>`).
            case xApiKey(apiKey: String)
        }

        /// The base URL for the API (e.g. `https://api.openai.com/`).
        public var baseURL: URL
        /// The authentication method to use for requests.
        public var authenticationMethod: AuthenticationMethod
        /// The chat completions endpoint path (e.g. `v1/chat/completions`).
        public var chatEndpoint: String
        /// Additional HTTP headers to include on every request.
        public var headers: [String: String]?

        /// Creates a client configured for OpenAI's API.
        public static func openAI(apiKey: String) -> Self {
            .init(
                baseURL: URL(string: "https://api.openai.com/")!,
                authMethod: .bearer(apiKey: apiKey)
            )
        }

        /// Creates a client configured for Anthropic's Messages API.
        public static func anthropic(apiKey: String, baseURL: URL? = nil) -> Self {
            .init(
                baseURL: baseURL ?? URL(string: "https://api.anthropic.com/")!,
                authMethod: .xApiKey(apiKey: apiKey),
                chatEndpoint: "v1/messages",
                headers: [
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "prompt-caching-2024-07-31",
                ]
            )
        }

        /// Creates a client configured for Mistral's API.
        public static func mistral(apiKey: String) -> Self {
            .init(
                baseURL: URL(string: "https://api.mistral.ai/")!,
                authMethod: .bearer(apiKey: apiKey)
            )
        }

        /// Creates a client configured for a local server on the given port.
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
            authenticationMethod = authMethod
            self.chatEndpoint = chatEndpoint
            self.headers = headers
        }
    }
}
