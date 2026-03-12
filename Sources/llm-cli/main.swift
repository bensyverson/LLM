import Foundation
import LLM

// MARK: - Argument Parsing

let args = CommandLine.arguments.dropFirst() // drop executable name

func flag(_ name: String) -> Bool {
    args.contains("--\(name)")
}

func value(_ name: String) -> String? {
    guard let index = args.firstIndex(of: "--\(name)"),
          args.index(after: index) < args.endIndex else { return nil }
    return args[args.index(after: index)]
}

// MARK: - Configuration

let providerName = value("provider") ?? "anthropic"
let tierName = value("tier") ?? "fast"
let explicitModel = value("model")
let systemPromptPath = value("system")
let message = value("message")
let thinkingEnabled = flag("thinking")
let maxTokens = value("max-tokens").flatMap(Int.init)
let maxReasoningTokens = value("max-reasoning-tokens").flatMap(Int.init)
let temperature = value("temperature").flatMap(Double.init)
let cachingEnabled = !flag("no-cache") // caching on by default
let cacheTTLString = value("cache-ttl")
let rawRequest = flag("raw-request")
let rawResponse = flag("raw-response")

guard let message else {
    fputs("Error: --message is required\n", stderr)
    fputs("Usage: llm-cli --message \"Hello\" [options]\n", stderr)
    fputs("""

    Options:
      --provider <name>            Provider: anthropic, openai, local (default: anthropic)
      --model <name>               Explicit model name
      --tier <tier>                Model tier: fast, standard, flagship (default: fast)
      --system <path>              System prompt file path
      --message <text>             User message (required)
      --thinking                   Enable extended thinking / reasoning
      --max-tokens <n>             Max output tokens
      --max-reasoning-tokens <n>   Max reasoning tokens
      --temperature <n>            Temperature (0.0-1.0)
      --no-cache                   Disable prompt caching
      --cache-ttl <ttl>            Cache TTL: 5m or 1h
      --raw-request                Print outgoing JSON request
      --raw-response               Print full JSON response

    """, stderr)
    exit(1)
}

// MARK: - Provider Setup

let provider: LLM.Provider
switch providerName {
case "anthropic":
    guard let apiKey = ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"] else {
        fputs("Error: ANTHROPIC_API_KEY environment variable not set\n", stderr)
        exit(1)
    }
    provider = .anthropic(apiKey: apiKey)
case "openai":
    guard let apiKey = ProcessInfo.processInfo.environment["OPENAI_API_KEY"] else {
        fputs("Error: OPENAI_API_KEY environment variable not set\n", stderr)
        exit(1)
    }
    provider = .openAI(apiKey: apiKey)
case "local":
    provider = .lmStudio
default:
    fputs("Error: Unknown provider '\(providerName)'. Use: anthropic, openai, local\n", stderr)
    exit(1)
}

// MARK: - Model Tier

let modelType: LLM.ModelType
switch tierName {
case "fast": modelType = .fast
case "standard": modelType = .standard
case "flagship": modelType = .flagship
default:
    fputs("Error: Unknown tier '\(tierName)'. Use: fast, standard, flagship\n", stderr)
    exit(1)
}

// MARK: - Cache TTL

let cacheTTL: LLM.OpenAICompatibleAPI.CacheControl.TTL?
if let cacheTTLString {
    switch cacheTTLString {
    case "5m": cacheTTL = .fiveMinutes
    case "1h": cacheTTL = .oneHour
    default:
        fputs("Error: Unknown cache TTL '\(cacheTTLString)'. Use: 5m, 1h\n", stderr)
        exit(1)
    }
} else {
    cacheTTL = nil
}

// MARK: - System Prompt

let systemPrompt: String
if let systemPromptPath {
    do {
        systemPrompt = try String(contentsOfFile: systemPromptPath, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    } catch {
        fputs("Error: Could not read system prompt file '\(systemPromptPath)': \(error.localizedDescription)\n", stderr)
        exit(1)
    }
} else {
    systemPrompt = "You are a helpful assistant."
}

// MARK: - Build Configuration

let config = LLM.ConversationConfiguration(
    modelType: modelType,
    inference: thinkingEnabled ? .reasoning : .direct,
    model: explicitModel.map { LLM.OpenAICompatibleAPI.ModelName(rawValue: $0) },
    temperature: temperature,
    maxTokens: maxTokens,
    maxReasoningTokens: maxReasoningTokens,
    enableCaching: cachingEnabled,
    cacheTTL: cacheTTL
)

// MARK: - Pretty-Print JSON Helper

let prettyEncoder: JSONEncoder = {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    return encoder
}()

func prettyJSON<T: Encodable>(_ value: T) -> String {
    guard let data = try? prettyEncoder.encode(value),
          let string = String(data: data, encoding: .utf8) else {
        return "<failed to encode JSON>"
    }
    return string
}

// MARK: - Raw Request

if rawRequest {
    let conversation = LLM.Conversation(
        systemPrompt: systemPrompt,
        messages: [.init(content: message, role: .user)],
        configuration: config
    )
    let request = conversation.request(for: provider)
    print("── Raw Request ──")
    print(prettyJSON(request))
    print()
}

// MARK: - Send Request

let llm = LLM(provider: provider)

do {
    let response = try await llm.startConversation(
        systemPrompt: systemPrompt,
        userMessage: message,
        configuration: config
    )

    // MARK: - Raw Response
    if rawResponse {
        print("── Raw Response ──")
        print(prettyJSON(response.rawResponse))
        print()
    }

    // MARK: - Usage Summary
    let usage = response.rawResponse.usage
    print("── Usage ──")
    if let input = usage.input_tokens ?? usage.prompt_tokens {
        print("  Input tokens:        \(input)")
    }
    if let output = usage.output_tokens ?? usage.completion_tokens {
        print("  Output tokens:       \(output)")
    }
    if let cacheWrite = usage.cache_creation_input_tokens, cacheWrite > 0 {
        print("  Cache write tokens:  \(cacheWrite)")
    }
    if let cacheRead = usage.cache_read_input_tokens, cacheRead > 0 {
        print("  Cache read tokens:   \(cacheRead)")
    }
    if let cached = usage.prompt_tokens_details?.cached_tokens, cached > 0 {
        print("  Cached tokens (OAI): \(cached)")
    }
    print()

    // MARK: - Thinking
    if let thinking = response.thinking {
        print("── Thinking ──")
        print(thinking)
        print()
    }

    // MARK: - Response Text
    if let text = response.text {
        print("── Response ──")
        print(text)
    } else {
        print("── Response ──")
        print("<no text content>")
    }
} catch {
    fputs("Error: \(error)\n", stderr)
    exit(1)
}
