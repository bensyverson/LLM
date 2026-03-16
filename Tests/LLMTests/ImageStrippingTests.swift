//
//  ImageStrippingTests.swift
//  LLMTests
//
//  Tests for LLM.strippingMedia — XML stub generation for non-vision models
//

import Foundation
@testable import LLM
import Testing

// MARK: - XML Stub Generation

@Test func imageStub_noFilenameOrDescription() {
    let stub = LLM.imageStub(number: 1, filename: nil, description: nil)
    #expect(stub == #"<image number="1"/>"#)
}

@Test func imageStub_withFilename() {
    let stub = LLM.imageStub(number: 2, filename: "DSC1234.jpg", description: nil)
    #expect(stub == #"<image number="2" name="DSC1234.jpg"/>"#)
}

@Test func imageStub_withFilenameAndDescription() {
    let stub = LLM.imageStub(number: 3, filename: "photo.jpg", description: "A dog with a stick")
    #expect(stub == #"<image number="3" name="photo.jpg" alt="A dog with a stick"/>"#)
}

@Test func documentStub_noTitle() {
    let stub = LLM.documentStub(number: 1, title: nil)
    #expect(stub == #"<document number="1"/>"#)
}

@Test func documentStub_withTitle() {
    let stub = LLM.documentStub(number: 1, title: "report.pdf")
    #expect(stub == #"<document number="1" title="report.pdf"/>"#)
}

// MARK: - Conversation Stripping

@Test func strippingMedia_singleImage_replacesWithStub() async throws {
    let llm = LLM(provider: .openAI(apiKey: "test"))

    let conversation = LLM.Conversation(systemPrompt: "System")
        .addingUserMessage([
            .text("Describe this"),
            .image(data: Data([0xFF, 0xD8]), mediaType: "image/jpeg"),
        ])

    let stripped = try await llm.strippingMedia(conversation)

    #expect(stripped.messages.count == 1)
    #expect(stripped.messages[0].content.count == 2)
    #expect(stripped.messages[0].content[0].textContent == "Describe this")
    #expect(stripped.messages[0].content[1].textContent == #"<image number="1"/>"#)
    #expect(stripped.messages[0].hasMedia == false)
}

@Test func strippingMedia_numberedAcrossMessages() async throws {
    let llm = LLM(provider: .openAI(apiKey: "test"))

    let conversation = LLM.Conversation(systemPrompt: "System")
        .addingUserMessage([.image(data: Data([0xFF, 0xD8]), mediaType: "image/jpeg")])
        .addingAssistantMessage("I see an image.")
        .addingUserMessage([.image(data: Data([0x89, 0x50]), mediaType: "image/png", filename: "chart.png")])

    let stripped = try await llm.strippingMedia(conversation)

    // First image: number 1
    #expect(stripped.messages[0].content[0].textContent == #"<image number="1"/>"#)
    // Second image: number 2 (across messages)
    #expect(stripped.messages[2].content[0].textContent == #"<image number="2" name="chart.png"/>"#)
}

@Test func strippingMedia_withDescription_includesAlt() async throws {
    let llm = LLM(provider: .openAI(apiKey: "test"))

    let conversation = LLM.Conversation(systemPrompt: "System")
        .addingUserMessage([
            .image(data: Data(), mediaType: "image/jpeg", filename: "photo.jpg", description: "A sunset"),
        ])

    let stripped = try await llm.strippingMedia(conversation)

    #expect(stripped.messages[0].content[0].textContent == #"<image number="1" name="photo.jpg" alt="A sunset"/>"#)
}

@Test func strippingMedia_pdf_replacesWithDocumentStub() async throws {
    let llm = LLM(provider: .openAI(apiKey: "test"))

    let conversation = LLM.Conversation(systemPrompt: "System")
        .addingUserMessage([
            .text("Summarize this"),
            .pdf(data: Data(), title: "report.pdf"),
        ])

    let stripped = try await llm.strippingMedia(conversation)

    #expect(stripped.messages[0].content[1].textContent == #"<document number="1" title="report.pdf"/>"#)
}

@Test func strippingMedia_mixedImageAndPdf_separateCounters() async throws {
    let llm = LLM(provider: .openAI(apiKey: "test"))

    let conversation = LLM.Conversation(systemPrompt: "System")
        .addingUserMessage([
            .image(data: Data(), mediaType: "image/jpeg"),
            .pdf(data: Data(), title: "doc.pdf"),
            .image(data: Data(), mediaType: "image/png"),
        ])

    let stripped = try await llm.strippingMedia(conversation)

    #expect(stripped.messages[0].content[0].textContent == #"<image number="1"/>"#)
    #expect(stripped.messages[0].content[1].textContent == #"<document number="1" title="doc.pdf"/>"#)
    #expect(stripped.messages[0].content[2].textContent == #"<image number="2"/>"#)
}

@Test func strippingMedia_describerCalled_forUndescribedImages() async throws {
    let llm = LLM(provider: .openAI(apiKey: "test"))

    let describer: @Sendable (Data, String) async throws -> String = { _, _ in
        "A generated description"
    }

    let conversation = LLM.Conversation(systemPrompt: "System")
        .addingUserMessage([
            .image(data: Data([0xFF, 0xD8]), mediaType: "image/jpeg", filename: "test.jpg"),
        ])

    let stripped = try await llm.strippingMedia(conversation, using: describer)

    #expect(stripped.messages[0].content[0].textContent == #"<image number="1" name="test.jpg" alt="A generated description"/>"#)
}

@Test func strippingMedia_textOnlyConversation_unchanged() async throws {
    let llm = LLM(provider: .openAI(apiKey: "test"))

    let conversation = LLM.Conversation(systemPrompt: "System")
        .addingUserMessage("Hello")
        .addingAssistantMessage("Hi there")

    let stripped = try await llm.strippingMedia(conversation)

    #expect(stripped.messages.count == 2)
    #expect(stripped.messages[0].textContent == "Hello")
    #expect(stripped.messages[1].textContent == "Hi there")
}
