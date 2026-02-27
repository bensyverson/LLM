public extension LLM.OpenAICompatibleAPI.ModelName {
    /// The maximum output token count for this model, if known.
    ///
    /// Returns `nil` for OpenAI models (which don't require an explicit
    /// `max_tokens`), `.placeholder`, or models whose max output cannot
    /// be determined at compile time.
    var maxOutputTokens: Int? {
        switch self {
        // Anthropic Claude 4.6
        case .claude46Opus:
            return 128_000
        case .claude46Sonnet:
            return 64000
        // Anthropic Claude 4.5
        case .claude45Opus:
            return 64000
        case .claude45Sonnet:
            return 64000
        case .claude45Haiku:
            return 64000
        // Anthropic Claude 4
        case .claude4Opus:
            return 32000
        case .claude4Sonnet:
            return 64000
        // Legacy Anthropic
        case .claude37SonnetLatest:
            return 8192
        case .claude35HaikuLatest:
            return 8192
        // OpenAI and unknown models
        default:
            return nil
        }
    }
}
