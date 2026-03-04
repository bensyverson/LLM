//
//  LLM+ImageStripping.swift
//  LLM
//
//  Strips images and PDFs from conversations for non-vision models.
//

import Foundation

public extension LLM {
    /// Strips images and PDFs from a conversation, replacing them with XML-tag descriptions.
    ///
    /// Images are numbered sequentially across the entire conversation.
    /// The XML stub format:
    /// - `<image number="1"/>` — no filename or description
    /// - `<image number="2" name="DSC1234.jpg"/>` — with filename
    /// - `<image number="3" name="photo.jpg" alt="A dog"/>` — with description
    /// - `<document number="1" title="report.pdf"/>` — for PDFs
    ///
    /// - Parameters:
    ///   - conversation: The conversation to strip media from.
    ///   - describer: An optional closure that generates a text description of an image.
    ///     Called for images that don't already have a description.
    /// - Returns: A new conversation with all media parts replaced by text stubs.
    func strippingMedia(
        _ conversation: Conversation,
        using describer: (@Sendable (Data, String) async throws -> String)? = nil
    ) async throws -> Conversation {
        var imageCounter = 0
        var documentCounter = 0
        var newMessages: [OpenAICompatibleAPI.ChatMessage] = []

        for msg in conversation.messages {
            var newParts: [OpenAICompatibleAPI.ContentPart] = []
            for part in msg.content {
                switch part {
                case .text:
                    newParts.append(part)

                case let .image(data, mediaType, filename, description):
                    imageCounter += 1
                    var resolvedDescription = description
                    if resolvedDescription == nil, let describer {
                        resolvedDescription = try await describer(data, mediaType)
                    }
                    let stub = Self.imageStub(number: imageCounter, filename: filename, description: resolvedDescription)
                    newParts.append(.text(stub))

                case let .pdf(_, title):
                    documentCounter += 1
                    let stub = Self.documentStub(number: documentCounter, title: title)
                    newParts.append(.text(stub))

                case .audio, .video:
                    // Drop unsupported media silently
                    continue
                }
            }
            var newMsg = msg
            newMsg.content = newParts
            newMessages.append(newMsg)
        }

        var result = conversation
        result.messages = newMessages
        return result
    }

    /// Generates an XML stub for an image.
    static func imageStub(number: Int, filename: String?, description: String?) -> String {
        var attrs = "number=\"\(number)\""
        if let filename { attrs += " name=\"\(filename)\"" }
        if let description { attrs += " alt=\"\(description)\"" }
        return "<image \(attrs)/>"
    }

    /// Generates an XML stub for a document.
    static func documentStub(number: Int, title: String?) -> String {
        var attrs = "number=\"\(number)\""
        if let title { attrs += " title=\"\(title)\"" }
        return "<document \(attrs)/>"
    }
}
