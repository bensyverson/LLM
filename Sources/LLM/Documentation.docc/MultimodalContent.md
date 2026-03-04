# Multimodal Content

Send images, PDFs, and other media alongside text in conversations.

## Overview

LLM supports multimodal content through the ``LLM/LLM/OpenAICompatibleAPI/ContentPart`` type.
Messages can contain a mix of text, images, and documents, and the library automatically
encodes them in the correct format for each provider.

### Sending an image

```swift
let imageData = try Data(contentsOf: imageURL)

let result = try await llm.startConversation(
    systemPrompt: "You are a helpful assistant.",
    userMessage: [
        .text("What's in this image?"),
        .image(data: imageData, mediaType: "image/jpeg"),
    ]
)
```

### Convenience initializers

Create image parts from URLs (local or remote):

```swift
let part = try LLM.OpenAICompatibleAPI.ContentPart.image(
    url: URL(fileURLWithPath: "/path/to/photo.jpg")
)
```

Or from raw data with automatic format detection:

```swift
let part = try LLM.OpenAICompatibleAPI.ContentPart.image(data: someData)
// Infers JPEG, PNG, GIF, or WebP from magic bytes
```

### PDF documents

```swift
let result = try await llm.startConversation(
    systemPrompt: "Summarize documents.",
    userMessage: [
        .text("Summarize this report."),
        .pdf(data: pdfData, title: "Q4 Report"),
    ]
)
```

> Note: PDFs are only supported by Anthropic. When sent to OpenAI, they are replaced
> with a text placeholder.

### Provider differences

The library handles encoding differences automatically:

| Feature | OpenAI | Anthropic |
|---|---|---|
| Images | `image_url` with base64 data URI | `image` with base64 source object |
| PDFs | Not supported (placeholder text) | `document` with base64 source |
| Tool results with images | Text-only (images converted) | Full image support |
| Max image long edge | 2048px | 1568px |

### Which models support vision?

Use ``LLM/LLM/OpenAICompatibleAPI/ModelName/supportsVision`` to check:

```swift
let model: LLM.OpenAICompatibleAPI.ModelName = .gpt4o
if model.supportsVision == true {
    // Safe to send images
}
```

All GPT-4+ models and all Claude 3+ models support vision. Older models like GPT-3.5
and GPT-4 (non-turbo) do not.

## Topics

### Content Parts

- ``LLM/LLM/OpenAICompatibleAPI/ContentPart``

### Vision Support

- ``LLM/LLM/OpenAICompatibleAPI/ModelName/supportsVision``
- ``LLM/LLM/OpenAICompatibleAPI/ModelName/maxImageLongEdge``
