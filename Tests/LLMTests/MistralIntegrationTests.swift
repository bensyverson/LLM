//
//  MistralIntegrationTests.swift
//  LLMTests
//
//  Integration tests for Mistral API
//

import Foundation
@testable import LLM
import Testing

// MARK: - Integration Tests (require real API key)

@Test func mistral_integration_basicChat() async throws {
    #if canImport(FoundationNetworking)
        try? await Task.sleep(nanoseconds: 100_000_000) // Avoid rate limiting

        guard ProcessInfo.processInfo.environment["MISTRAL_API_KEY"] != nil else {
            print("Skipping Mistral integration test: MISTRAL_API_KEY not set")
            return
        }

        let provider = try LLM.Provider.mistral(apiKey: #require(ProcessInfo.processInfo.environment["MISTRAL_API_KEY"]))
        let llm = LLM(provider: provider)

        let response = try await llm.startConversation(
            systemPrompt: "You are a helpful assistant.",
            userMessage: "What is the capital of France?"
        )

        #expect(response.text?.lowercased().contains("paris") == true, "Should mention Paris")
        print("Mistral basic chat test passed: " + (response.text ?? "No response"))
    #endif
}

@Test func mistral_integration_withTools() async throws {
    #if canImport(FoundationNetworking)
        try? await Task.sleep(nanoseconds: 100_000_000) // Avoid rate limiting

        guard ProcessInfo.processInfo.environment["MISTRAL_API_KEY"] != nil else {
            print("Skipping Mistral integration test: MISTRAL_API_KEY not set")
            return
        }

        let provider = try LLM.Provider.mistral(apiKey: #require(ProcessInfo.processInfo.environment["MISTRAL_API_KEY"]))
        let llm = LLM(provider: provider)

        let tools = [
            LLM.OpenAICompatibleAPI.ToolDefinition(
                function: LLM.OpenAICompatibleAPI.FunctionDefinition(
                    name: "get_weather",
                    description: "Get weather information for a location",
                    parameters: LLM.OpenAICompatibleAPI.JSONSchema.object(
                        properties: [
                            "location": LLM.OpenAICompatibleAPI.JSONSchema.string(description: "The city and state"),
                        ],
                        required: ["location"],
                        description: "Weather query parameters"
                    )
                )
            ),
        ]

        var config = LLM.ConversationConfiguration()
        config.tools = tools
        config.toolChoice = .auto
        config.parallelToolCalls = true

        let response = try await llm.startConversation(
            systemPrompt: "You are a helpful assistant with access to weather tools.",
            userMessage: "What's the weather in San Francisco?",
            configuration: config
        )

        // The model should either call the tool or explain it can't
        if !response.toolCalls.isEmpty {
            #expect(response.toolCalls.count == 1, "Should make one tool call")
            #expect(response.toolCalls[0].function.name == "get_weather", "Should call get_weather tool")
            print("Mistral tool test passed: Made tool call")
        } else if let text = response.text {
            #expect(text.lowercased().contains("tool") || text.lowercased().contains("weather"), "Should mention tools or weather")
            print("Mistral tool test passed: &#123;text&#125;")
        }
    #endif
}

@Test func mistral_integration_requestEncoding() throws {
    #if canImport(FoundationNetworking)
        guard ProcessInfo.processInfo.environment["MISTRAL_API_KEY"] != nil else {
            print("Skipping Mistral integration test: MISTRAL_API_KEY not set")
            return
        }

        let provider = try LLM.Provider.mistral(apiKey: #require(ProcessInfo.processInfo.environment["MISTRAL_API_KEY"]))
        let conversation = LLM.Conversation(
            systemPrompt: "You are a helpful assistant.",
            messages: [.init(content: "Hello from Mistral!", role: .user)],
            configuration: LLM.ConversationConfiguration()
        )

        let request = conversation.request(for: provider)

        // Verify the request is properly configured for Mistral
        #expect(request.useMistralFormat == true, "Should use Mistral format")
        #expect(request.useAnthropicToolFormat == false, "Should not use Anthropic format")
        #expect(request.model == .mistralSmall, "Should use Mistral small model by default")

        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let data = try encoder.encode(request)
        let jsonString = String(data: data, encoding: .utf8)

        #expect(jsonString?.contains("\"model\":\"mistral-small-latest\"") == true, "Should include Mistral model")
        print("Mistral request encoding test passed")
    #endif
}
