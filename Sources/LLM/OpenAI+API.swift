//
//  OpenAI+API.swift
//  LLM
//
//  Created by Ben Syverson on 2024-11-13.
//

import Foundation

public extension LLM.OpenAICompatibleAPI {
    func chatCompletion(with body: Data) async throws -> ChatCompletionResponse {
        let url = baseURL.appending(path: chatEndpoint)

        var request = URLRequest(url: url, timeoutInterval: 120)
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
            throw OpenAIError.badResponseCode(statusCode)
        }
        do {
            let decoder = JSONDecoder()
            decoder.dataDecodingStrategy = .base64
            return try decoder.decode(ChatCompletionResponse.self, from: data)
        } catch {
            print("Couldn't decode \(String(bytes: data, encoding: .utf8)!)")
            print(error)
            throw error
        }
    }

    func embedding(for body: Data) async throws -> [Float] {
        let url = baseURL.appending(components: "v1", "embeddings")

        var request = URLRequest(url: url, timeoutInterval: 120)
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
        do {
            let response = try JSONDecoder().decode(EmbeddingList.self, from: data)
            guard let embedding = response.data.first?.embedding else {
                throw OpenAIError.noEmbedding
            }
            return embedding
        } catch {
            print(error)
            throw error
        }
    }

    static func oneShot(
        model: ModelName = .gpt4oMini,
        systemMessage: String,
        prompt: String
    ) -> ChatCompletion {
        .init(
            model: model,
            messages: [
                .init(content: systemMessage, role: .system),
                .init(content: prompt, role: .user),
            ]
        )
    }
}
