//
//  ModelVisionTests.swift
//  LLMTests
//
//  Tests for ModelName.supportsVision and maxImageLongEdge
//

import Foundation
@testable import LLM
import Testing

typealias ModelName = LLM.OpenAICompatibleAPI.ModelName

// MARK: - supportsVision Tests

@Test func `supports vision non vision models return false`() {
    #expect(ModelName.gpt35turbo.supportsVision == false)
    #expect(ModelName.gpt35turbo16k.supportsVision == false)
    #expect(ModelName.gpt4.supportsVision == false)
}

@Test func `supports vision open AI vision models return true`() {
    #expect(ModelName.gpt4turbo.supportsVision == true)
    #expect(ModelName.gpt4o.supportsVision == true)
    #expect(ModelName.gpt4oMini.supportsVision == true)
    #expect(ModelName.gpt41.supportsVision == true)
    #expect(ModelName.gpt41Mini.supportsVision == true)
    #expect(ModelName.gpt41Nano.supportsVision == true)
    #expect(ModelName.gpt52.supportsVision == true)
    #expect(ModelName.gpt5Mini.supportsVision == true)
    #expect(ModelName.gpt5Nano.supportsVision == true)
    #expect(ModelName.gpt54.supportsVision == true)
    #expect(ModelName.gpt54Mini.supportsVision == true)
    #expect(ModelName.gpt54Nano.supportsVision == true)
    #expect(ModelName.gpt55.supportsVision == true)
}

@Test func `supports vision reasoning models return true`() {
    #expect(ModelName.o1.supportsVision == true)
    #expect(ModelName.o3.supportsVision == true)
    #expect(ModelName.o3mini.supportsVision == true)
    #expect(ModelName.o4mini.supportsVision == true)
}

@Test func `supports vision claude models return true`() {
    #expect(ModelName.claude46Opus.supportsVision == true)
    #expect(ModelName.claude46Sonnet.supportsVision == true)
    #expect(ModelName.claude45Opus.supportsVision == true)
    #expect(ModelName.claude45Sonnet.supportsVision == true)
    #expect(ModelName.claude45Haiku.supportsVision == true)
}

@Test func `supports vision custom claude return true`() {
    let custom = ModelName(rawValue: "claude-custom-model")
    #expect(custom.supportsVision == true)
}

@Test func `supports vision unknown model return nil`() {
    let custom = ModelName(rawValue: "my-custom-model")
    #expect(custom.supportsVision == nil)
}

// MARK: - maxImageLongEdge Tests

@Test func `max image long edge claude returns 1568`() {
    #expect(ModelName.claude46Opus.maxImageLongEdge == 1568)
    #expect(ModelName.claude45Sonnet.maxImageLongEdge == 1568)
}

@Test func `max image long edge open AI vision returns 2048`() {
    #expect(ModelName.gpt4o.maxImageLongEdge == 2048)
    #expect(ModelName.gpt41.maxImageLongEdge == 2048)
    #expect(ModelName.gpt52.maxImageLongEdge == 2048)
}

@Test func `max image long edge non vision return nil`() {
    #expect(ModelName.gpt35turbo.maxImageLongEdge == nil)
    #expect(ModelName.gpt4.maxImageLongEdge == nil)
}

@Test func `max image long edge unknown return nil`() {
    let custom = ModelName(rawValue: "my-custom-model")
    #expect(custom.maxImageLongEdge == nil)
}
