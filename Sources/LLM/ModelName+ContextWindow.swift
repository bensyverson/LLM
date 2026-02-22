public extension LLM.OpenAICompatibleAPI.ModelName {
    /// The context window size in tokens for this model, if known.
    ///
    /// Returns `nil` for `.placeholder` or models whose context window
    /// size cannot be determined at compile time.
    var contextWindowTokens: Int? {
        switch self {
        // GPT-3.5 family
        case .gpt35turbo:
            return 16_385
        case .gpt35turbo16k:
            return 16_385

        // GPT-4 family
        case .gpt4:
            return 8_192
        case .gpt4turbo:
            return 128_000
        case .gpt4o:
            return 128_000
        case .gpt4oMini:
            return 128_000

        // GPT-4.1 family
        case .gpt41:
            return 1_047_576
        case .gpt41Mini:
            return 1_047_576
        case .gpt41Nano:
            return 1_047_576

        // O-series reasoning models
        case .o1preview:
            return 128_000
        case .o1mini:
            return 128_000
        case .o1:
            return 200_000
        case .o3:
            return 200_000
        case .o3mini:
            return 200_000
        case .o4mini:
            return 200_000

        // GPT-5 family
        case .gpt52:
            return 1_047_576
        case .gpt5Mini:
            return 1_047_576
        case .gpt5Nano:
            return 1_047_576

        // Anthropic Claude 4.6
        case .claude46Opus:
            return 200_000
        case .claude46Sonnet:
            return 200_000

        // Anthropic Claude 4.5
        case .claude45Opus:
            return 200_000
        case .claude45Sonnet:
            return 200_000
        case .claude45Haiku:
            return 200_000

        // Anthropic Claude 4
        case .claude4Opus:
            return 200_000
        case .claude4Sonnet:
            return 200_000

        // Legacy Anthropic
        case .claude37SonnetLatest:
            return 200_000
        case .claude35HaikuLatest:
            return 200_000

        // Unknown
        case .placeholder:
            return nil
        }
    }
}
