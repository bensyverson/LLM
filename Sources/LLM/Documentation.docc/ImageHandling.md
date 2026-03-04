# Image Handling

Automatic resizing, non-vision fallback, and the image description workflow.

## Overview

LLM automatically handles image preprocessing before sending to providers, including
resizing oversized images and gracefully degrading when a model doesn't support vision.

### Automatic resizing

When sending images to a model, LLM checks the model's
``LLM/LLM/OpenAICompatibleAPI/ModelName/maxImageLongEdge`` and resizes any image that
exceeds the limit. On Apple platforms, a CoreGraphics-based resizer is used by default.

Resized images are cached (up to 50 entries) to avoid redundant work when the same
image appears in multiple conversation turns.

You can provide a custom resizer:

```swift
llm.imageResizer = { data, mediaType, targetSize in
    // Your custom resize logic
    return resizedData
}
```

### Non-vision model fallback

When a conversation contains images but the target model doesn't support vision
(e.g. GPT-3.5), LLM automatically strips the images and replaces them with XML stubs:

```xml
<image number="1" name="photo.jpg" alt="A sunset over the ocean"/>
<document number="1" title="report.pdf"/>
```

Images are numbered sequentially across the entire conversation. The response includes
a warning in ``LLM/LLM/ConversationResponse/warnings``:

```swift
let result = try await llm.chat(conversation: conversation)
for warning in result.warnings {
    print("Warning: \(warning)")
}
```

### Image descriptions

Set ``LLM/LLM/imageDescriber`` to automatically generate text descriptions for images
that don't already have one. This is called during media stripping for non-vision models:

```swift
llm.imageDescriber = { imageData, mediaType in
    // Call a vision model to describe the image
    return "A photo of a sunset"
}
```

Images with an existing `description` parameter skip the describer.

## Topics

### Properties

- ``LLM/LLM/imageResizer``
- ``LLM/LLM/imageDescriber``

### Related

- ``LLM/LLM/ConversationResponse/warnings``
