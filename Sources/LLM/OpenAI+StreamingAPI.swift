//
//  OpenAI+StreamingAPI.swift
//  LLM
//
//  Created by Claude on 2026-02-22.
//

import Foundation

public extension LLM.OpenAICompatibleAPI {
    func streamingChatCompletion(with body: Data) async throws -> (SSEParser<AsyncLineSequence<URLSession.AsyncBytes>>, URLSession) {
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

        let session = URLSession.shared
        let (bytes, response) = try await session.bytes(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw OpenAIError.badResponse(response)
        }

        guard httpResponse.statusCode == 200 else {
            // Collect error body from the byte stream
            var errorData = Data()
            for try await byte in bytes {
                errorData.append(byte)
            }
            if let errorBody = String(data: errorData, encoding: .utf8) {
                FileHandle.standardError.write(Data("[LLM] HTTP \(httpResponse.statusCode): \(errorBody)\n".utf8))
            }
            throw OpenAIError.badResponseCode(httpResponse.statusCode)
        }

        let parser = SSEParser(bytes: bytes)
        return (parser, session)
    }
}
