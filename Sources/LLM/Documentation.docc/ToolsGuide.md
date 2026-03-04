# Tools

Define tools that the model can call, and handle tool results.

## Overview

LLM supports function calling (tool use) across both OpenAI and Anthropic. You define
tools with JSON Schema parameters, and the model returns structured tool call requests
that your code executes.

### Define a tool

```swift
let weatherTool = LLM.OpenAICompatibleAPI.ToolDefinition(
    function: LLM.OpenAICompatibleAPI.FunctionDefinition(
        name: "get_weather",
        description: "Get the current weather for a city.",
        parameters: .object(
            properties: [
                "city": .string(description: "The city name"),
                "unit": .string(description: "Temperature unit", enum: ["celsius", "fahrenheit"]),
            ],
            required: ["city"]
        )
    )
)
```

### Use tools in a conversation

Pass tools via ``LLM/LLM/ConversationConfiguration``:

```swift
let config = LLM.ConversationConfiguration(
    tools: [weatherTool],
    toolChoice: .auto
)

var result = try await openai.startConversation(
    systemPrompt: "You are a weather assistant.",
    userMessage: "What's the weather in Tokyo?",
    configuration: config
)
```

### Handle tool calls

When the model wants to call a tool, the response will have non-empty
``LLM/LLM/ConversationResponse/toolCalls``:

```swift
if !result.toolCalls.isEmpty {
    for call in result.toolCalls {
        let args = call.function.arguments // JSON string
        let toolResult = executeMyTool(name: call.function.name, args: args)

        // Feed the result back
        let updated = result.conversation.addingToolResultMessage(
            toolCallId: call.id,
            content: toolResult
        )
        result = try await openai.chat(conversation: updated)
    }
}
```

### Tool choice

Control how the model uses tools with ``LLM/LLM/OpenAICompatibleAPI/ToolChoice``:

- `.auto` — The model decides whether to call a tool (default).
- `.required` — The model must call at least one tool.
- `.none` — The model must not call any tools.
- `.tool(name:)` — The model must call the specific named tool.

### Images in tool results

Anthropic supports returning images in tool results, which is useful for tools that
generate screenshots or charts. Use the `[ContentPart]` overload:

```swift
let updated = result.conversation.addingToolResultMessage(
    toolCallId: call.id,
    content: [
        .text("Here is the screenshot"),
        .image(data: screenshotData, mediaType: "image/png"),
    ]
)
```

> Note: OpenAI only supports text in tool results. When sending to OpenAI, image parts
> in tool results are automatically converted to text placeholders.
