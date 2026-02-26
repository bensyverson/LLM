# ``LLM``

A unified Swift async interface for OpenAI, Anthropic, and OpenAI-compatible LLM providers.

## Overview

LLM wraps multiple language model APIs behind a single async/await interface. It handles
provider-specific wire formats, rate limiting, streaming, tool calling, and conversation
management so you can focus on building your application.

### Supported Providers

- **OpenAI** — GPT-4o, GPT-4.1, GPT-5, o-series reasoning models
- **Anthropic** — Claude 4.5, Claude 4.6 (via Anthropic's Messages API)
- **LM Studio** — Local models on `localhost:1234`
- **Any OpenAI-compatible API** — Ollama, vLLM, Azure OpenAI, etc.

### Key Features

- **Adaptive rate limiting** that reads limits from response headers and adjusts automatically
- **Streaming** with incremental text, thinking, and tool call deltas
- **Tool calling** with a unified interface across OpenAI and Anthropic formats
- **Conversation management** that tracks message history and handles assistant replies

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

### Rate Limiting

- <doc:RateLimiting>
- ``LLM/LLM/RateLimiter``
- ``LLM/LLM/RateLimitInfo``

### Models

- ``LLM/LLM/ModelType``
- ``LLM/LLM/InferenceType``
- ``LLM/LLM/OpenAICompatibleAPI/ModelName``
