public extension LLM.OpenAICompatibleAPI.ModelName {
    /// The maximum output token count for this model, if known.
    ///
    /// Returns `nil` for OpenAI models (which don't require an explicit
    /// `max_tokens`), `.placeholder`, or models whose max output cannot
    /// be determined at compile time.
    var maxOutputTokens: Int? {
        // Anthropic Claude 4.6
        if self == .claude46Opus { return 128_000 }
        if self == .claude46Sonnet { return 64000 }
        // Anthropic Claude 4.5
        if self == .claude45Opus { return 64000 }
        if self == .claude45Sonnet { return 64000 }
        if self == .claude45Haiku { return 64000 }
        // OpenAI and unknown models
        return nil
    }
}
