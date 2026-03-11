# Getting Started

Set up a provider and make your first chat request.

## Overview

LLM requires a provider to know which API to talk to. Each provider case carries
the credentials needed to authenticate.

### Create an LLM instance

```swift
import LLM

// OpenAI
let openai = LLM(provider: .openAI(apiKey: "sk-..."))

// Anthropic
let anthropic = LLM(provider: .anthropic(apiKey: "sk-ant-..."))

// Local model via LM Studio
let local = LLM(provider: .lmStudio)
```

### One-shot chat

For a simple prompt/response without maintaining conversation history, use
``LLM/LLM/ChatConfiguration``:

```swift
let config = LLM.ChatConfiguration(
    systemPrompt: "You are a helpful assistant.",
    user: "What is the capital of France?",
    modelType: .fast,
    inference: .direct
)

let response: String = try await openai.chat(configuration: config)
print(response) // "The capital of France is Paris."
```

### Conversations

For multi-turn conversations, use the ``LLM/LLM/Conversation`` API:

```swift
// Start a conversation
var result = try await openai.startConversation(
    systemPrompt: "You are a helpful assistant.",
    userMessage: "Hello!"
)
print(result.text ?? "")

// Continue the conversation
result = try await openai.continueConversation(
    result.conversation,
    userMessage: "What's the weather like?"
)
print(result.text ?? "")
```

The ``LLM/LLM/ConversationResponse`` carries the updated conversation history, so you
can keep calling ``LLM/LLM/continueConversation(_:userMessage:)`` to add turns.

### Sending images

Add images and other media to conversations using ``LLM/LLM/OpenAICompatibleAPI/ContentPart``:

```swift
let imageData = try Data(contentsOf: imageURL)

let result = try await openai.startConversation(
    systemPrompt: "You are a helpful assistant.",
    userMessage: [
        .text("Describe this image."),
        .image(data: imageData, mediaType: "image/jpeg"),
    ]
)
print(result.text ?? "")
```

See <doc:MultimodalContent> for the full guide on multimodal support.

### Choosing a model

LLM uses ``LLM/LLM/ModelType`` (`.fast`, `.standard` or `.flagship`) and ``LLM/LLM/InferenceType``
(`.direct` or `.reasoning`) to select the right model for each provider. This lets
you write provider-agnostic code that automatically picks the best model available.
