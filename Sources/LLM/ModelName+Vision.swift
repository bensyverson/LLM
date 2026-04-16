public extension LLM.OpenAICompatibleAPI.ModelName {
    /// Whether this model supports vision (image input), if known.
    ///
    /// Returns `nil` for custom or unknown models where vision support
    /// cannot be determined at compile time.
    var supportsVision: Bool? {
        // Non-vision models
        if self == .gpt35turbo || self == .gpt35turbo16k || self == .gpt4 { return false }
        // Vision models — OpenAI
        if self == .gpt4turbo || self == .gpt4o || self == .gpt4oMini { return true }
        if self == .gpt41 || self == .gpt41Mini || self == .gpt41Nano { return true }
        if self == .gpt52 || self == .gpt5Mini || self == .gpt5Nano { return true }
        if self == .gpt54 || self == .gpt54Mini || self == .gpt54Nano { return true }
        if self == .o1 || self == .o3 || self == .o3mini || self == .o4mini { return true }
        // All Claude 3+ models support vision
        if rawValue.hasPrefix("claude-") { return true }
        return nil
    }

    /// Maximum image long-edge in pixels before the provider resizes or rejects.
    ///
    /// Anthropic limits images to 1568px on the long edge.
    /// OpenAI auto-scales to 2048px for high-detail mode.
    var maxImageLongEdge: Int? {
        if self == .claude47Opus { return 2576 }
        if rawValue.hasPrefix("claude-") { return 1568 }
        if supportsVision == true { return 2048 }
        return nil
    }
}
