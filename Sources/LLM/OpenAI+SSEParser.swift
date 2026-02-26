//
//  OpenAI+SSEParser.swift
//  LLM
//
//  Created by Claude on 2026-02-22.
//

import Foundation

public extension LLM.OpenAICompatibleAPI {
    /// A single Server-Sent Event with an optional event type and data payload.
    struct SSEEvent: Sendable {
        /// The event type (from `event:` field), or `nil` if not specified.
        public let event: String?
        /// The event data (from `data:` field(s)).
        public let data: String
    }

    /// An async sequence that parses Server-Sent Events from a stream of lines.
    struct SSEParser<Lines: AsyncSequence>: AsyncSequence where Lines.Element == String {
        public typealias Element = SSEEvent

        let lines: Lines

        public func makeAsyncIterator() -> AsyncIterator {
            AsyncIterator(lines: lines.makeAsyncIterator())
        }

        public struct AsyncIterator: AsyncIteratorProtocol {
            var linesIterator: Lines.AsyncIterator
            var finished = false

            init(lines: Lines.AsyncIterator) {
                linesIterator = lines
            }

            public mutating func next() async throws -> SSEEvent? {
                guard !finished else { return nil }

                var currentEvent: String?
                var dataLines: [String] = []

                while let line = try await linesIterator.next() {
                    // Skip comments
                    if line.hasPrefix(":") {
                        continue
                    }

                    // Empty line = dispatch event
                    if line.isEmpty {
                        guard !dataLines.isEmpty else { continue }
                        let data = dataLines.joined(separator: "\n")
                        dataLines = []

                        // [DONE] signals end of stream (OpenAI convention)
                        if data == "[DONE]" {
                            finished = true
                            return nil
                        }

                        let event = SSEEvent(event: currentEvent, data: data)
                        currentEvent = nil
                        return event
                    }

                    // Parse field
                    if line.hasPrefix("event:") {
                        // If we already have data buffered, dispatch the previous event
                        // before starting a new one (handles streams without blank separators)
                        if !dataLines.isEmpty {
                            let data = dataLines.joined(separator: "\n")
                            dataLines = []
                            if data == "[DONE]" {
                                finished = true
                                return nil
                            }
                            let event = SSEEvent(event: currentEvent, data: data)
                            currentEvent = String(line.dropFirst(6)).trimmingCharacters(in: CharacterSet.whitespaces)
                            return event
                        }
                        currentEvent = String(line.dropFirst(6)).trimmingCharacters(in: CharacterSet.whitespaces)
                    } else if line.hasPrefix("data:") {
                        dataLines.append(String(line.dropFirst(5)).trimmingCharacters(in: CharacterSet.whitespaces))
                    } else if line.hasPrefix("id:") || line.hasPrefix("retry:") {
                        // Ignored per SSE spec
                    }
                }

                // Stream ended — dispatch any remaining buffered data
                if !dataLines.isEmpty {
                    let data = dataLines.joined(separator: "\n")
                    if data == "[DONE]" {
                        finished = true
                        return nil
                    }
                    finished = true
                    return SSEEvent(event: currentEvent, data: data)
                }

                finished = true
                return nil
            }
        }
    }
}

// MARK: - Convenience initializer for URLSession.AsyncBytes

public extension LLM.OpenAICompatibleAPI.SSEParser where Lines == AsyncLineSequence<URLSession.AsyncBytes> {
    init(bytes: URLSession.AsyncBytes) {
        self.init(lines: bytes.lines)
    }
}
