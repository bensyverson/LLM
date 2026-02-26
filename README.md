# LLM

A unified Swift async interface for OpenAI, Anthropic, and OpenAI-compatible LLM providers.

## Supported Providers

- **OpenAI** — GPT-4o, GPT-4.1, GPT-5, o-series reasoning models
- **Anthropic** — Claude 4.5, Claude 4.6 (via Messages API)
- **LM Studio** — Local models on `localhost:1234`
- **Any OpenAI-compatible API** — Ollama, vLLM, Azure OpenAI, etc.

## Quick Start

```swift
import LLM

let llm = LLM(provider: .openAI(apiKey: "sk-..."))

// One-shot chat
let config = LLM.ChatConfiguration(
    systemPrompt: "You are a helpful assistant.",
    user: "What is the capital of France?",
    modelType: .fast,
    inference: .direct
)
let answer: String = try await llm.chat(configuration: config)

// Multi-turn conversation
var result = try await llm.startConversation(
    systemPrompt: "You are a helpful assistant.",
    userMessage: "Hello!"
)
result = try await llm.continueConversation(result.conversation, userMessage: "Tell me more.")

// Streaming
for try await event in llm.streamChat(conversation: result.conversation) {
    switch event {
    case .textDelta(let text): print(text, terminator: "")
    case .completed(let response): print("\nDone: \(response.text ?? "")")
    default: break
    }
}
```

## Features

- **Adaptive rate limiting** — reads limits from response headers and adjusts automatically
- **Streaming** — incremental text, thinking, and tool call deltas via `AsyncThrowingStream`
- **Tool calling** — unified interface across OpenAI and Anthropic formats
- **Conversation management** — immutable conversation history with easy branching
- **Extended thinking** — supports Anthropic's extended thinking and OpenAI's reasoning

## Installation

Add LLM as a Swift Package Manager dependency:

```swift
.package(url: "https://github.com/your-org/LLM.git", branch: "main")
```

## Documentation

[Browse the documentation online](https://bensyverson.com/documentation/LLM/), or generate the full API documentation with DocC:

```bash
swift package generate-documentation --target LLM

# Serve the docs:
swift package --disable-sandbox preview-documentation --target LLM

```

The DocC catalog includes guides for:
- [Getting Started](Sources/LLM/Documentation.docc/GettingStarted.md)
- [Streaming](Sources/LLM/Documentation.docc/StreamingGuide.md)
- [Tools](Sources/LLM/Documentation.docc/ToolsGuide.md)
- [Rate Limiting](Sources/LLM/Documentation.docc/RateLimiting.md)

## About LLM

I created this library in 2023 to experiment with OpenAI models in my macOS and iOS projects, and later extended the library to support local models and then Anthropic models. As a result, one of the core types is `OpenAICompatibleAPI`, though under the hood, LLM is really altering the shape of its requests to match different model capabilities and vendor flavors.

LLM is designed to be lightweight (zero dependencies outside of DocC), and tries not to make any assumptions about how you'll use it; it can be used to one-shot answers to questions, and it's also the engine for my agentic library, [Operator](https://github.com/bensyverson/Operator). Hope it's helpful!

[Ben Syverson](https://bensyverson.com/)
Feb 2025

---

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Ben Syverson
