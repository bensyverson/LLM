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

@Test func supportsVision_nonVisionModels_returnFalse() {
    #expect(ModelName.gpt35turbo.supportsVision == false)
    #expect(ModelName.gpt35turbo16k.supportsVision == false)
    #expect(ModelName.gpt4.supportsVision == false)
}

@Test func supportsVision_openAIVisionModels_returnTrue() {
    #expect(ModelName.gpt4turbo.supportsVision == true)
    #expect(ModelName.gpt4o.supportsVision == true)
    #expect(ModelName.gpt4oMini.supportsVision == true)
    #expect(ModelName.gpt41.supportsVision == true)
    #expect(ModelName.gpt41Mini.supportsVision == true)
    #expect(ModelName.gpt41Nano.supportsVision == true)
    #expect(ModelName.gpt52.supportsVision == true)
    #expect(ModelName.gpt5Mini.supportsVision == true)
    #expect(ModelName.gpt5Nano.supportsVision == true)
}

@Test func supportsVision_reasoningModels_returnTrue() {
    #expect(ModelName.o1.supportsVision == true)
    #expect(ModelName.o3.supportsVision == true)
    #expect(ModelName.o3mini.supportsVision == true)
    #expect(ModelName.o4mini.supportsVision == true)
}

@Test func supportsVision_claudeModels_returnTrue() {
    #expect(ModelName.claude46Opus.supportsVision == true)
    #expect(ModelName.claude46Sonnet.supportsVision == true)
    #expect(ModelName.claude45Opus.supportsVision == true)
    #expect(ModelName.claude45Sonnet.supportsVision == true)
    #expect(ModelName.claude45Haiku.supportsVision == true)
    #expect(ModelName.claude4Opus.supportsVision == true)
    #expect(ModelName.claude4Sonnet.supportsVision == true)
    #expect(ModelName.claude37SonnetLatest.supportsVision == true)
    #expect(ModelName.claude35HaikuLatest.supportsVision == true)
}

@Test func supportsVision_customClaude_returnTrue() {
    let custom = ModelName(rawValue: "claude-custom-model")
    #expect(custom.supportsVision == true)
}

@Test func supportsVision_unknownModel_returnNil() {
    let custom = ModelName(rawValue: "my-custom-model")
    #expect(custom.supportsVision == nil)
}

// MARK: - maxImageLongEdge Tests

@Test func maxImageLongEdge_claude_returns1568() {
    #expect(ModelName.claude46Opus.maxImageLongEdge == 1568)
    #expect(ModelName.claude45Sonnet.maxImageLongEdge == 1568)
}

@Test func maxImageLongEdge_openAIVision_returns2048() {
    #expect(ModelName.gpt4o.maxImageLongEdge == 2048)
    #expect(ModelName.gpt41.maxImageLongEdge == 2048)
    #expect(ModelName.gpt52.maxImageLongEdge == 2048)
}

@Test func maxImageLongEdge_nonVision_returnNil() {
    #expect(ModelName.gpt35turbo.maxImageLongEdge == nil)
    #expect(ModelName.gpt4.maxImageLongEdge == nil)
}

@Test func maxImageLongEdge_unknown_returnNil() {
    let custom = ModelName(rawValue: "my-custom-model")
    #expect(custom.maxImageLongEdge == nil)
}
