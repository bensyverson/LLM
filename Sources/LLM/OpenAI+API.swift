//
//  OpenAI+API.swift
//  LLM
//
//  Created by Ben Syverson on 2024-11-13.
//

import Foundation
#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

public extension LLM.OpenAICompatibleAPI {
    /// Sends a chat completion request and returns the parsed response with HTTP metadata.
    ///
    /// - Parameter body: JSON-encoded request body.
    /// - Returns: A tuple of the decoded response and the raw HTTP response (for header inspection).
    /// - Throws: ``OpenAIError/badResponse(_:)`` if the response isn't HTTP,
    ///   or ``OpenAIError/badResponseCode(_:)`` on non-200 status codes.
    func chatCompletion(with body: Data) async throws -> (ChatCompletionResponse, HTTPURLResponse) {
        let url = baseURL.appending(path: chatEndpoint)

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        for (header, headerValue) in headers ?? [:] {
            request.addValue(headerValue, forHTTPHeaderField: header)
        }
        switch authenticationMethod {
        case let .bearer(apiKey):
            request.addValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        case let .xApiKey(apiKey):
            request.addValue(apiKey, forHTTPHeaderField: "x-api-key")
        default:
            break
        }
        request.httpBody = body

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw OpenAIError.badResponse(response)
        }
        let statusCode = httpResponse.statusCode

        guard statusCode == 200 else {
            if let errorBody = String(data: data, encoding: .utf8) {
                FileHandle.standardError.write(Data("[LLM] HTTP \(statusCode): \(errorBody)\n".utf8))
            }
            throw OpenAIError.badResponseCode(statusCode)
        }

        let decoder = JSONDecoder()
        decoder.dataDecodingStrategy = .base64
        let decoded = try decoder.decode(ChatCompletionResponse.self, from: data)
        return (decoded, httpResponse)
    }

    /// Generates an embedding vector for the given input.
    ///
    /// - Parameter body: JSON-encoded embedding request body.
    /// - Returns: The embedding as an array of floats.
    /// - Throws: ``OpenAIError/noEmbedding`` if the response contains no embedding data.
    func embedding(for body: Data) async throws -> [Float] {
        let url = baseURL.appending(components: "v1", "embeddings")

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        switch authenticationMethod {
        case let .bearer(apiKey):
            request.addValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        case let .xApiKey(apiKey):
            request.addValue(apiKey, forHTTPHeaderField: "x-api-key")
        default:
            break
        }
        request.httpBody = body

        let (data, _) = try await URLSession.shared.data(for: request)
        let response = try JSONDecoder().decode(EmbeddingList.self, from: data)
        guard let embedding = response.data.first?.embedding else {
            throw OpenAIError.noEmbedding
        }
        return embedding
    }

    /// Builds a one-shot chat completion request with a system message and user prompt.
    ///
    /// - Parameters:
    ///   - model: The model to use. Defaults to GPT-4o Mini.
    ///   - systemMessage: The system message content.
    ///   - prompt: The user prompt.
    /// - Returns: A configured ``ChatCompletion`` ready to encode and send.
    static func oneShot(
        model: ModelName = .gpt4oMini,
        systemMessage: String,
        prompt: String
    ) -> ChatCompletion {
        ChatCompletion(
            model: model,
            messages: [
                ChatMessage(content: systemMessage, role: .system),
                ChatMessage(content: prompt, role: .user),
            ]
        )
    }
}
