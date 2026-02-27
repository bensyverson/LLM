public extension LLM.OpenAICompatibleAPI.ModelName {
    /// The context window size in tokens for this model, if known.
    ///
    /// Returns `nil` for `.placeholder`, custom model strings, or models
    /// whose context window size cannot be determined at compile time.
    var contextWindowTokens: Int? {
        // GPT-3.5 family
        if self == .gpt35turbo { return 16385 }
        if self == .gpt35turbo16k { return 16385 }
        // GPT-4 family
        if self == .gpt4 { return 8192 }
        if self == .gpt4turbo { return 128_000 }
        if self == .gpt4o { return 128_000 }
        if self == .gpt4oMini { return 128_000 }
        // GPT-4.1 family
        if self == .gpt41 { return 1_047_576 }
        if self == .gpt41Mini { return 1_047_576 }
        if self == .gpt41Nano { return 1_047_576 }
        // O-series reasoning models
        if self == .o1preview { return 128_000 }
        if self == .o1mini { return 128_000 }
        if self == .o1 { return 200_000 }
        if self == .o3 { return 200_000 }
        if self == .o3mini { return 200_000 }
        if self == .o4mini { return 200_000 }
        // GPT-5 family
        if self == .gpt52 { return 1_047_576 }
        if self == .gpt5Mini { return 1_047_576 }
        if self == .gpt5Nano { return 1_047_576 }
        // Anthropic Claude 4.6
        if self == .claude46Opus { return 200_000 }
        if self == .claude46Sonnet { return 200_000 }
        // Anthropic Claude 4.5
        if self == .claude45Opus { return 200_000 }
        if self == .claude45Sonnet { return 200_000 }
        if self == .claude45Haiku { return 200_000 }
        // Anthropic Claude 4
        if self == .claude4Opus { return 200_000 }
        if self == .claude4Sonnet { return 200_000 }
        // Legacy Anthropic
        if self == .claude37SonnetLatest { return 200_000 }
        if self == .claude35HaikuLatest { return 200_000 }
        // Unknown / custom models
        return nil
    }
}
