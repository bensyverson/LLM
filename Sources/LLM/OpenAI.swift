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
        /// Known model identifiers for OpenAI and Anthropic.
        public enum ModelName: String, Codable {
            case placeholder
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
            case o1
            case o3
            case o3mini = "o3-mini"
            case o4mini = "o4-mini"
            // OpenAI GPT-5 family (2026 - all have native reasoning)
            case gpt52 = "gpt-5.2" // Flagship
            case gpt5Mini = "gpt-5-mini" // Fast, cost-efficient
            case gpt5Nano = "gpt-5-nano" // Fastest, most cost-efficient
            // Anthropic Claude 4.6
            case claude46Opus = "claude-opus-4-6"
            case claude46Sonnet = "claude-sonnet-4-6"
            // Anthropic Claude 4.5
            case claude45Opus = "claude-opus-4-5"
            case claude45Sonnet = "claude-sonnet-4-5"
            case claude45Haiku = "claude-haiku-4-5"
            // Anthropic Claude 4
            case claude4Opus = "claude-opus-4"
            case claude4Sonnet = "claude-sonnet-4"
            // Legacy Anthropic models
            case claude37SonnetLatest = "claude-3-7-sonnet-latest"
            case claude35HaikuLatest = "claude-3-5-haiku-latest"

            public var isGPT5: Bool {
                switch self {
                case .gpt52, .gpt5Mini, .gpt5Nano:
                    return true
                default:
                    return false
                }
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
        public static func anthropic(apiKey: String) -> Self {
            .init(
                baseURL: URL(string: "https://api.anthropic.com/")!,
                authMethod: .xApiKey(apiKey: apiKey),
                chatEndpoint: "v1/messages",
                headers: ["anthropic-version": "2023-06-01"]
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
