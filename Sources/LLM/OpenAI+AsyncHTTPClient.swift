//
//  OpenAI+AsyncHTTPClient.swift
//  LLM
//
//  Created by Claude on 2026-02-25.
//

#if canImport(AsyncHTTPClient)
    import AsyncHTTPClient
    import Foundation
    import NIOCore
    import NIOFoundationCompat
    import NIOHTTP1

    // MARK: - ByteBuffer → lines adapter

    /// An `AsyncSequence` that reads `ByteBuffer` chunks and yields individual
    /// lines (`String`), handling partial lines that span chunk boundaries.
    ///
    /// Empty lines (blank-line delimiters critical for SSE) are preserved.
    public struct ByteBufferLineSequence<Body: AsyncSequence>: AsyncSequence
        where Body.Element == ByteBuffer
    {
        public typealias Element = String

        let body: Body

        public func makeAsyncIterator() -> AsyncIterator {
            AsyncIterator(base: body.makeAsyncIterator())
        }

        public struct AsyncIterator: AsyncIteratorProtocol {
            var base: Body.AsyncIterator
            var buffer = ""
            var pendingLines: [String] = []
            var finished = false

            init(base: Body.AsyncIterator) {
                self.base = base
            }

            public mutating func next() async throws -> String? {
                // Drain any buffered lines first
                if !pendingLines.isEmpty {
                    return pendingLines.removeFirst()
                }

                guard !finished else { return nil }

                while true {
                    guard let chunk = try await base.next() else {
                        // Stream ended — emit any remaining partial line
                        finished = true
                        if !buffer.isEmpty {
                            let remaining = buffer
                            buffer = ""
                            return remaining
                        }
                        return nil
                    }

                    let text = String(buffer: chunk)
                    buffer.append(text)

                    // Split on \n (handles \r\n because SSE treats \r as whitespace)
                    var lines: [String] = []
                    while let newlineIndex = buffer.firstIndex(of: "\n") {
                        let line = String(buffer[buffer.startIndex ..< newlineIndex])
                            .trimmingCharacters(in: CharacterSet(charactersIn: "\r"))
                        lines.append(line)
                        buffer = String(buffer[buffer.index(after: newlineIndex)...])
                    }

                    if !lines.isEmpty {
                        // Return the first, buffer the rest
                        let first = lines.removeFirst()
                        pendingLines = lines
                        return first
                    }
                    // No complete line yet — fetch another chunk
                }
            }
        }
    }

    // MARK: - Linux streaming implementation

    public extension LLM.OpenAICompatibleAPI {
        /// Linux-specific streaming chat completion using AsyncHTTPClient.
        ///
        /// Builds and sends the streaming request via `HTTPClient.shared`,
        /// returning an SSE parser over the response body and any rate limit info.
        ///
        /// - Parameters:
        ///   - body: JSON-encoded request body (must have `stream: true`).
        ///   - provider: The provider, used for rate limit header parsing.
        /// - Returns: A tuple of the SSE parser and optional rate limit info.
        /// - Throws: ``OpenAIError/badResponseCode(_:)`` on non-200 status codes.
        func streamingChatCompletionLinux(
            with body: Data,
            provider: LLM.Provider
        ) async throws -> (
            SSEParser<ByteBufferLineSequence<HTTPClientResponse.Body>>,
            LLM.RateLimitInfo?
        ) {
            let url = baseURL.appending(path: chatEndpoint).absoluteString

            var request = HTTPClientRequest(url: url)
            request.method = .POST
            request.headers.add(name: "Content-Type", value: "application/json")
            for (header, headerValue) in headers ?? [:] {
                request.headers.add(name: header, value: headerValue)
            }
            switch authenticationMethod {
            case let .bearer(apiKey):
                request.headers.add(name: "Authorization", value: "Bearer \(apiKey)")
            case let .xApiKey(apiKey):
                request.headers.add(name: "x-api-key", value: apiKey)
            default:
                break
            }
            request.body = .bytes(ByteBuffer(data: body))

            let response = try await HTTPClient.shared.execute(request, timeout: .seconds(300))

            guard response.status == .ok else {
                // Collect error body
                var errorData = Data()
                for try await chunk in response.body {
                    errorData.append(contentsOf: chunk.readableBytesView)
                }
                if let errorBody = String(data: errorData, encoding: .utf8) {
                    FileHandle.standardError.write(Data("[LLM] HTTP \(response.status.code): \(errorBody)\n".utf8))
                }
                throw OpenAIError.badResponseCode(Int(response.status.code))
            }

            // Parse rate limit headers
            let rateLimitInfo = LLM.RateLimitInfo.parse(
                headerLookup: { response.headers.first(name: $0) },
                provider: provider
            )

            let lineSequence = ByteBufferLineSequence(body: response.body)
            let parser = SSEParser(lines: lineSequence)
            return (parser, rateLimitInfo)
        }
    }
#endif
