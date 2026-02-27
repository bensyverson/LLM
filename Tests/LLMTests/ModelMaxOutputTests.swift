@testable import LLM
import Testing

@Suite("ModelName.maxOutputTokens")
struct ModelMaxOutputTests {
    // MARK: - Anthropic models return correct values

    @Test func claude46Opus_returns128k() {
        #expect(LLM.OpenAICompatibleAPI.ModelName.claude46Opus.maxOutputTokens == 128_000)
    }

    @Test func claude46Sonnet_returns64k() {
        #expect(LLM.OpenAICompatibleAPI.ModelName.claude46Sonnet.maxOutputTokens == 64000)
    }

    @Test func claude45Haiku_returns64k() {
        #expect(LLM.OpenAICompatibleAPI.ModelName.claude45Haiku.maxOutputTokens == 64000)
    }

    @Test func claude4Opus_returns32k() {
        #expect(LLM.OpenAICompatibleAPI.ModelName.claude4Opus.maxOutputTokens == 32000)
    }

    @Test func claude37SonnetLatest_returns8192() {
        #expect(LLM.OpenAICompatibleAPI.ModelName.claude37SonnetLatest.maxOutputTokens == 8192)
    }

    // MARK: - OpenAI models return nil

    @Test func openAI_models_returnNil() {
        #expect(LLM.OpenAICompatibleAPI.ModelName.gpt52.maxOutputTokens == nil)
        #expect(LLM.OpenAICompatibleAPI.ModelName.gpt5Mini.maxOutputTokens == nil)
        #expect(LLM.OpenAICompatibleAPI.ModelName.gpt4o.maxOutputTokens == nil)
        #expect(LLM.OpenAICompatibleAPI.ModelName.o3.maxOutputTokens == nil)
    }

    @Test func placeholder_returnsNil() {
        #expect(LLM.OpenAICompatibleAPI.ModelName.placeholder.maxOutputTokens == nil)
    }

    // MARK: - Conversation request uses model max when maxTokens is nil

    @Test func conversationRequest_usesModelMax_whenMaxTokensNil() {
        var conversation = LLM.Conversation(
            systemPrompt: "Test",
            configuration: LLM.ConversationConfiguration(
                modelType: .flagship,
                inference: .direct
            )
        )
        conversation.configuration.maxTokens = nil

        let provider = LLM.Provider.anthropic(apiKey: "test")
        let request = conversation.request(for: provider)
        // flagship Anthropic = claude46Opus → 128,000
        #expect(request.max_tokens == 128_000)
    }

    // MARK: - Conversation request uses explicit maxTokens when set

    @Test func conversationRequest_usesExplicit_whenMaxTokensSet() {
        var conversation = LLM.Conversation(
            systemPrompt: "Test",
            configuration: LLM.ConversationConfiguration(
                modelType: .flagship,
                inference: .direct
            )
        )
        conversation.configuration.maxTokens = 2048

        let provider = LLM.Provider.anthropic(apiKey: "test")
        let request = conversation.request(for: provider)
        #expect(request.max_tokens == 2048)
    }

    // MARK: - OpenAI max_tokens remains nil

    @Test func conversationRequest_openAI_maxTokensNil() {
        var conversation = LLM.Conversation(
            systemPrompt: "Test",
            configuration: LLM.ConversationConfiguration(
                modelType: .fast,
                inference: .reasoning
            )
        )
        conversation.configuration.maxTokens = nil

        let provider = LLM.Provider.openAI(apiKey: "test")
        let request = conversation.request(for: provider)
        // GPT-5 with reasoning and no maxTokens: don't set a limit
        #expect(request.max_completion_tokens == nil)
    }
}
