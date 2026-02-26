# Rate Limiting

How LLM manages API rate limits automatically.

## Overview

LLM includes a built-in ``LLM/LLM/RateLimiter`` that prevents your application from
exceeding provider rate limits. It starts with conservative defaults and **adapts
automatically** by reading rate limit headers from API responses.

### How it works

1. **Conservative start** — New LLM instances begin with safe defaults (50 requests
   and 40,000 tokens per minute for cloud providers). These limits are low enough to
   avoid 429 errors on any tier.

2. **Adaptive adjustment** — After each API response, LLM parses the provider's rate
   limit headers and updates the limiter's ceilings. For example, an OpenAI Tier 3
   user will see limits increase to 5,000+ requests per minute after the first response.

3. **Automatic throttling** — When the limiter's request or token budget is exhausted,
   subsequent calls sleep until the rate window resets.

### Supported headers

| Provider | Request limit header | Token limit header |
|----------|---------------------|--------------------|
| OpenAI | `x-ratelimit-limit-requests` | `x-ratelimit-limit-tokens` |
| Anthropic | `anthropic-ratelimit-requests-limit` | `anthropic-ratelimit-tokens-limit` |

### Custom rate limiters

You can override the defaults by passing your own limiters:

```swift
let limiter = LLM.RateLimiter(
    maxRequests: 10000,
    maxTokens: 2_000_000,
    interval: 60
)

let llm = LLM(
    provider: .openAI(apiKey: "sk-..."),
    chatLimiter: limiter
)
```

Custom limiters still benefit from adaptive adjustment — their ceilings will be updated
if the provider's response headers indicate different limits.

### Local providers

Local providers (`.lmStudio`, `.localhost`) use permissive defaults that effectively
disable rate limiting, since local inference servers typically have no rate limits.
