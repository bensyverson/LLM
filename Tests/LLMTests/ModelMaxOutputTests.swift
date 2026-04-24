@testable import LLM
import Testing

@Suite("ModelName.maxOutputTokens")
struct ModelMaxOutputTests {
    // MARK: - Anthropic models return correct values

    @Test func `claude 46 opus returns 128 k`() {
        #expect(LLM.OpenAICompatibleAPI.ModelName.claude46Opus.maxOutputTokens == 128_000)
    }

    @Test func `claude 46 sonnet returns 64 k`() {
        #expect(LLM.OpenAICompatibleAPI.ModelName.claude46Sonnet.maxOutputTokens == 64000)
    }

    @Test func `claude 45 haiku returns 64 k`() {
        #expect(LLM.OpenAICompatibleAPI.ModelName.claude45Haiku.maxOutputTokens == 64000)
    }

    // MARK: - OpenAI models return nil

    @Test func `open AI models return nil`() {
        #expect(LLM.OpenAICompatibleAPI.ModelName.gpt52.maxOutputTokens == nil)
        #expect(LLM.OpenAICompatibleAPI.ModelName.gpt5Mini.maxOutputTokens == nil)
        #expect(LLM.OpenAICompatibleAPI.ModelName.gpt4o.maxOutputTokens == nil)
        #expect(LLM.OpenAICompatibleAPI.ModelName.o3.maxOutputTokens == nil)
    }

    @Test func `placeholder returns nil`() {
        #expect(LLM.OpenAICompatibleAPI.ModelName.placeholder.maxOutputTokens == nil)
    }

    // MARK: - Conversation request uses model max when maxTokens is nil

    @Test func `conversation request uses model max when max tokens nil`() {
        var conversation = LLM.Conversation(
            systemPrompt: "Test",
            configuration: LLM.ConversationConfiguration(
                modelType: .flagship,
                inference: .direct,
            ),
        )
        conversation.configuration.maxTokens = nil

        let provider = LLM.Provider.anthropic(apiKey: "test")
        let request = conversation.request(for: provider)
        // flagship Anthropic = claude46Opus → 128,000
        #expect(request.max_tokens == 128_000)
    }

    // MARK: - Conversation request uses explicit maxTokens when set

    @Test func `conversation request uses explicit when max tokens set`() {
        var conversation = LLM.Conversation(
            systemPrompt: "Test",
            configuration: LLM.ConversationConfiguration(
                modelType: .flagship,
                inference: .direct,
            ),
        )
        conversation.configuration.maxTokens = 2048

        let provider = LLM.Provider.anthropic(apiKey: "test")
        let request = conversation.request(for: provider)
        #expect(request.max_tokens == 2048)
    }

    // MARK: - OpenAI max_tokens remains nil

    @Test func `conversation request open AI max tokens nil`() {
        var conversation = LLM.Conversation(
            systemPrompt: "Test",
            configuration: LLM.ConversationConfiguration(
                modelType: .fast,
                inference: .reasoning,
            ),
        )
        conversation.configuration.maxTokens = nil

        let provider = LLM.Provider.openAI(apiKey: "test")
        let request = conversation.request(for: provider)
        // GPT-5 with reasoning and no maxTokens: don't set a limit
        #expect(request.max_completion_tokens == nil)
    }
}
