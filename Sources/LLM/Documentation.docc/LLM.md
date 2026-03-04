# ``LLM``

A unified Swift async interface for various LLM providers.

## Overview

LLM wraps multiple language model APIs behind a single async/await interface. It handles
provider-specific wire formats, rate limiting, streaming, tool calling, and conversation
management so you can focus on building your application.

### Supported Providers

- **OpenAI** - GPT
- **Anthropic** - Claude
- **LM Studio** and **localhost** — Local models
- **Any OpenAI-compatible API** — Ollama, vLLM, Azure OpenAI, etc

### Key Features

- **Adaptive rate limiting** that reads limits from response headers and adjusts automatically
- **Streaming** with incremental text, thinking, and tool call deltas
- **Tool calling** with a unified interface across OpenAI and Anthropic formats
- **Conversation management** that tracks message history and handles assistant replies
- **Multimodal** support for images and PDFs with automatic provider-specific encoding, resizing, and non-vision fallback

## Topics

### Essentials

- <doc:GettingStarted>
- ``LLM/LLM``
- ``LLM/LLM/Provider``

### Chat and Conversations

- ``LLM/LLM/Conversation``
- ``LLM/LLM/ConversationConfiguration``
- ``LLM/LLM/ConversationResponse``
- ``LLM/LLM/ChatConfiguration``

### Streaming

- <doc:StreamingGuide>
- ``LLM/LLM/StreamEvent``
- ``LLM/LLM/ToolCallDelta``

### Tools

- <doc:ToolsGuide>
- ``LLM/LLM/OpenAICompatibleAPI/ToolDefinition``
- ``LLM/LLM/OpenAICompatibleAPI/ToolCall``
- ``LLM/LLM/OpenAICompatibleAPI/JSONSchema``
- ``LLM/LLM/OpenAICompatibleAPI/ToolChoice``

### Multimodal

- <doc:MultimodalContent>
- <doc:ImageHandling>
- ``LLM/LLM/OpenAICompatibleAPI/ContentPart``

### Rate Limiting

- <doc:RateLimiting>
- ``LLM/LLM/RateLimiter``
- ``LLM/LLM/RateLimitInfo``

### Models

- ``LLM/LLM/ModelType``
- ``LLM/LLM/InferenceType``
- ``LLM/LLM/OpenAICompatibleAPI/ModelName``
