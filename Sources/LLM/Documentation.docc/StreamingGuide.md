# Streaming

Receive incremental response deltas as the model generates text.

## Overview

Streaming lets you display partial results to the user as the model generates them,
rather than waiting for the full response. LLM provides streaming through
`AsyncThrowingStream<StreamEvent, Error>`.

### Stream a conversation

```swift
let stream = openai.streamChat(conversation: conversation)

for try await event in stream {
    switch event {
    case .textDelta(let text):
        print(text, terminator: "")
    case .thinkingDelta(let thinking):
        // Model's internal reasoning (Anthropic extended thinking, OpenAI reasoning)
        break
    case .toolCallDelta(let delta):
        // Incremental tool call data
        break
    case .completed(let response):
        // Final ConversationResponse with full text and updated conversation
        print("\n---")
        print("Total tokens: \(response.rawResponse.usage.total_tokens ?? 0)")
    }
}
```

### Convenience methods

Start a new streaming conversation or continue an existing one:

```swift
// Start new
let stream = openai.streamConversation(
    systemPrompt: "You are a helpful assistant.",
    userMessage: "Tell me a story."
)

// Continue existing
let stream = openai.streamContinueConversation(
    conversation,
    userMessage: "Make it longer."
)
```

### Stream events

- ``LLM/LLM/StreamEvent/textDelta(_:)`` — A fragment of the model's text response.
- ``LLM/LLM/StreamEvent/thinkingDelta(_:)`` — A fragment of the model's reasoning.
- ``LLM/LLM/StreamEvent/toolCallDelta(_:)`` — Incremental tool call data (id, name, arguments).
- ``LLM/LLM/StreamEvent/completed(_:)`` — The final ``LLM/LLM/ConversationResponse`` with accumulated results.

The `.completed` event is always the last event in the stream and contains the same
``LLM/LLM/ConversationResponse`` you would get from the non-streaming API.

### Multimodal messages

Streaming works identically with multimodal messages — no special handling is needed.
Use the `[ContentPart]` overloads to include images in streamed conversations:

```swift
let stream = openai.streamConversation(
    systemPrompt: "You are a helpful assistant.",
    userMessage: [
        .text("What's in this image?"),
        .image(data: imageData, mediaType: "image/jpeg"),
    ]
)
```
