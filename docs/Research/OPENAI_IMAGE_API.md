# OpenAI Image API — Research Notes


## Summary

The OpenAI Image and Audio APIs provide REST endpoints for image generation/editing and audio speech/transcription. This document covers the exact request/response JSON schemas for all endpoints: `/v1/images/generations`, `/v1/images/edits`, `/v1/images/variations`, `/v1/audio/speech`, `/v1/audio/transcriptions`, and `/v1/audio/translations`. All schemas are sourced from the official OpenAI OpenAPI specification (openapi.yaml) and corroborated with OpenAI guide documentation. SharpInference.Server must implement these schemas to be a drop-in OpenAI-compatible backend.

Sources:
- [OpenAI OpenAPI Spec (GitHub)](https://github.com/openai/openai-openapi) — canonical machine-readable spec (openapi.yaml, version 2.3.0+)
- [OpenAI Image Generation Guide](https://platform.openai.com/docs/guides/image-generation)
- [OpenAI Text-to-Speech Guide](https://platform.openai.com/docs/guides/text-to-speech)
- [OpenAI Speech-to-Text Guide](https://platform.openai.com/docs/guides/speech-to-text)
- [OpenAI API Reference — Create Image](https://developers.openai.com/api/reference/resources/images/methods/generate)
- [OpenAI API Reference — Create Speech](https://platform.openai.com/docs/api-reference/audio/createSpeech)
- [OpenAI API Reference — Create Transcription](https://platform.openai.com/docs/api-reference/audio/createTranscription)
- [LocalAI Image Generation](https://localai.io/features/image-generation/)
- [vLLM-Omni Image Generation API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/image_generation_api/)

## Detailed Findings

### Endpoint 1: POST /v1/images/generations

Creates one or more images from a text prompt. Accepts `application/json`.

#### Request — CreateImageRequest

| Parameter | Type | Required | Default | Allowed Values | Description |
|-----------|------|----------|---------|----------------|-------------|
| `prompt` | string | **Yes** | — | max 32000 chars (GPT image), 4000 (dall-e-3), 1000 (dall-e-2) | Text description of desired image(s) |
| `model` | string | No | `"dall-e-2"` | `"gpt-image-1.5"`, `"gpt-image-1"`, `"gpt-image-1-mini"`, `"dall-e-3"`, `"dall-e-2"` | Model to use |
| `n` | integer | No | `1` | 1–10 (dall-e-3 only supports n=1) | Number of images to generate |
| `size` | string | No | `"auto"` | See Size Values table below | Output image dimensions |
| `quality` | string | No | `"auto"` | `"auto"`, `"high"`, `"medium"`, `"low"`, `"hd"`, `"standard"` | Image quality level |
| `response_format` | string | No | `"url"` | `"url"`, `"b64_json"` | Return format (dall-e-2/3 only; GPT image models always return b64_json) |
| `output_format` | string | No | `"png"` | `"png"`, `"jpeg"`, `"webp"` | Image file format (GPT image models only) |
| `output_compression` | integer | No | `100` | 0–100 | JPEG/WebP compression % (GPT image models only) |
| `style` | string | No | `"vivid"` | `"vivid"`, `"natural"` | Image style (dall-e-3 only) |
| `background` | string | No | `"auto"` | `"transparent"`, `"opaque"`, `"auto"` | Background transparency (GPT image models only) |
| `moderation` | string | No | `"auto"` | `"low"`, `"auto"` | Content moderation level (GPT image models only) |
| `stream` | boolean | No | `false` | `true`, `false` | Enable streaming (GPT image models only) |
| `partial_images` | integer\|null | No | `0` | 0–3 | Number of partial images in streaming mode |
| `user` | string | No | — | any string | End-user identifier for abuse monitoring |

#### Response — ImagesResponse

```json
{
  "created": 1713833628,
  "data": [
    {
      "b64_json": "<base64-encoded image data>",
      "url": "https://...",
      "revised_prompt": "A cute baby sea otter swimming..."
    }
  ],
  "background": "transparent",
  "output_format": "png",
  "size": "1024x1024",
  "quality": "high",
  "usage": {
    "total_tokens": 100,
    "input_tokens": 50,
    "output_tokens": 50,
    "input_tokens_details": {
      "text_tokens": 10,
      "image_tokens": 40
    }
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `created` | integer | **Yes** | Unix timestamp (seconds) |
| `data` | array of Image | No | List of generated images |
| `data[].b64_json` | string | No | Base64-encoded image (GPT image models default; dall-e-2/3 when response_format=b64_json) |
| `data[].url` | string | No | Temporary URL valid for 60 min (dall-e-2/3 when response_format=url) |
| `data[].revised_prompt` | string | No | Revised prompt (dall-e-3 only) |
| `background` | string | No | `"transparent"` or `"opaque"` (GPT image models) |
| `output_format` | string | No | `"png"`, `"webp"`, or `"jpeg"` (GPT image models) |
| `size` | string | No | Size used: `"1024x1024"`, `"1024x1536"`, `"1536x1024"` |
| `quality` | string | No | Quality used: `"low"`, `"medium"`, `"high"` |
| `usage` | object | No | Token usage (GPT image models only) |

---

### Endpoint 2: POST /v1/images/edits

Edits an image given a prompt. Accepts `multipart/form-data`.

#### Request — CreateImageEditRequest

| Parameter | Type | Required | Default | Allowed Values | Description |
|-----------|------|----------|---------|----------------|-------------|
| `image` | file \| file[] | **Yes** | — | PNG/WebP/JPG < 50MB (GPT image), PNG < 4MB square (dall-e-2). Up to 16 images for GPT image models | Source image(s) to edit |
| `prompt` | string | **Yes** | — | max 32000 chars (GPT image), 1000 (dall-e-2) | Text description of desired edit |
| `mask` | file | No | — | PNG < 4MB, same dims as image. Transparent areas = edit region | Inpainting mask |
| `model` | string | No | `"gpt-image-1.5"` | `"gpt-image-1.5"`, `"gpt-image-1"`, `"gpt-image-1-mini"`, `"chatgpt-image-latest"`, `"dall-e-2"` | Model to use |
| `n` | integer | No | `1` | 1–10 | Number of images |
| `size` | string | No | `"1024x1024"` | `"auto"`, `"1024x1024"`, `"1536x1024"`, `"1024x1536"`, `"256x256"`, `"512x512"` | Output dimensions |
| `quality` | string | No | `"auto"` | `"auto"`, `"high"`, `"medium"`, `"low"`, `"standard"` | Quality level |
| `response_format` | string | No | `"url"` (dall-e-2) | `"url"`, `"b64_json"` | Return format (dall-e-2 only; GPT image models always return b64_json) |
| `output_format` | string | No | `"png"` | `"png"`, `"jpeg"`, `"webp"` | File format (GPT image models only) |
| `output_compression` | integer | No | `100` | 0–100 | Compression % (GPT image models, webp/jpeg only) |
| `background` | string | No | `"auto"` | `"transparent"`, `"opaque"`, `"auto"` | Background transparency (GPT image models only) |
| `input_fidelity` | string\|null | No | `"low"` | `"high"`, `"low"` | How closely to preserve input details (faces, logos). GPT image models only, not gpt-image-1-mini |
| `stream` | boolean | No | `false` | `true`, `false` | Enable streaming |
| `partial_images` | integer\|null | No | `0` | 0–3 | Partial images in streaming |
| `user` | string | No | — | any string | End-user ID |

Response: Same `ImagesResponse` format as `/v1/images/generations`.

---

### Endpoint 3: POST /v1/images/variations

Creates variations of an existing image. Accepts `multipart/form-data`. **dall-e-2 only**.

#### Request — CreateImageVariationRequest

| Parameter | Type | Required | Default | Allowed Values | Description |
|-----------|------|----------|---------|----------------|-------------|
| `image` | file | **Yes** | — | PNG < 4MB, square | Source image |
| `model` | string | No | `"dall-e-2"` | `"dall-e-2"` | Only dall-e-2 supported |
| `n` | integer | No | `1` | 1–10 | Number of variations |
| `response_format` | string | No | `"url"` | `"url"`, `"b64_json"` | Return format |
| `size` | string | No | `"1024x1024"` | `"256x256"`, `"512x512"`, `"1024x1024"` | Output dimensions |
| `user` | string | No | — | any string | End-user ID |

Response: Same `ImagesResponse` format.

---

### Endpoint 4: POST /v1/audio/speech

Text-to-speech. Accepts `application/json`. Returns raw audio bytes (or SSE stream).

#### Request — CreateSpeechRequest

| Parameter | Type | Required | Default | Allowed Values | Description |
|-----------|------|----------|---------|----------------|-------------|
| `model` | string | **Yes** | — | `"tts-1"`, `"tts-1-hd"`, `"gpt-4o-mini-tts"`, `"gpt-4o-mini-tts-2025-12-15"` | TTS model |
| `input` | string | **Yes** | — | max 4096 chars | Text to synthesize |
| `voice` | string \| object | **Yes** | — | Built-in: `"alloy"`, `"ash"`, `"ballad"`, `"coral"`, `"echo"`, `"fable"`, `"onyx"`, `"nova"`, `"sage"`, `"shimmer"`, `"verse"`, `"marin"`, `"cedar"`. Custom: `{"id": "voice_1234"}` | Voice selection |
| `instructions` | string | No | — | max 4096 chars | Voice control instructions (not supported for tts-1/tts-1-hd) |
| `response_format` | string | No | `"mp3"` | `"mp3"`, `"opus"`, `"aac"`, `"flac"`, `"wav"`, `"pcm"` | Output audio format |
| `speed` | number | No | `1.0` | 0.25–4.0 | Playback speed |
| `stream_format` | string | No | `"audio"` | `"sse"`, `"audio"` | Streaming transport (sse not supported for tts-1/tts-1-hd) |

#### Response

HTTP 200 with `Content-Type` matching the requested format (e.g., `audio/mpeg` for mp3, `audio/opus` for opus, `audio/wav` for wav, `audio/flac` for flac, `audio/aac` for aac, `audio/pcm` for pcm). Body is the raw audio byte stream.

When `stream_format=sse`, response is `text/event-stream` with `SpeechAudioDeltaEvent` and `SpeechAudioDoneEvent` events.

---

### Endpoint 5: POST /v1/audio/transcriptions

Speech-to-text. Accepts `multipart/form-data`.

#### Request — CreateTranscriptionRequest

| Parameter | Type | Required | Default | Allowed Values | Description |
|-----------|------|----------|---------|----------------|-------------|
| `file` | file | **Yes** | — | flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm | Audio file to transcribe |
| `model` | string | **Yes** | — | `"whisper-1"`, `"gpt-4o-transcribe"`, `"gpt-4o-mini-transcribe"`, `"gpt-4o-mini-transcribe-2025-12-15"`, `"gpt-4o-transcribe-diarize"` | Transcription model |
| `language` | string | No | — | ISO-639-1 code (e.g., `"en"`, `"fr"`, `"ja"`) | Input audio language hint |
| `prompt` | string | No | — | any string | Style/context hint (not supported for diarize model) |
| `response_format` | string | No | `"json"` | `"json"`, `"text"`, `"srt"`, `"verbose_json"`, `"vtt"`, `"diarized_json"` | Output format. gpt-4o-transcribe/mini only support json/text. diarize model supports json/text/diarized_json |
| `temperature` | number | No | `0` | 0.0–1.0 | Sampling temperature |
| `timestamp_granularities` | string[] | No | `["segment"]` | `"word"`, `"segment"` | Requires verbose_json. Not available for diarize model |
| `stream` | boolean\|null | No | `false` | `true`, `false` | SSE streaming (not supported for whisper-1) |
| `include` | string[] | No | — | `"logprobs"` | Extra response data. Requires json format. Only gpt-4o-transcribe/mini |
| `chunking_strategy` | string\|object\|null | No | — | `"auto"` or VadConfig object | Audio chunking control. Required for diarize inputs > 30s |
| `known_speaker_names` | string[] | No | — | max 4 items | Speaker labels for diarization |
| `known_speaker_references` | file[] | No | — | max 4 items, 2–10s each | Audio samples of known speakers |

#### Response — CreateTranscriptionResponseJson (response_format=json)

```json
{
  "text": "The transcribed text content...",
  "logprobs": [
    {
      "token": "The",
      "logprob": -0.123,
      "bytes": [84, 104, 101]
    }
  ],
  "usage": {
    "type": "tokens",
    "input_tokens": 14,
    "input_token_details": {
      "text_tokens": 0,
      "audio_tokens": 14
    }
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | **Yes** | Transcribed text |
| `logprobs` | array | No | Token log probabilities (only if `include` contains `"logprobs"`) |
| `usage` | object | No | Token or duration usage |

#### Response — CreateTranscriptionResponseVerboseJson (response_format=verbose_json)

```json
{
  "language": "english",
  "duration": 8.47,
  "text": "The beach was a popular spot on a hot summer day.",
  "words": [
    { "word": "The", "start": 0.0, "end": 0.24 },
    { "word": "beach", "start": 0.24, "end": 0.56 }
  ],
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 3.32,
      "text": " The beach was a popular spot on a hot summer day.",
      "tokens": [50364, 440, 7534, 390, 257, 3743, 4008, 322, 257, 2368, 4266, 786, 13, 50530],
      "temperature": 0.0,
      "avg_logprob": -0.286,
      "compression_ratio": 1.236,
      "no_speech_prob": 0.0099
    }
  ],
  "usage": {
    "type": "duration",
    "seconds": 9
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `language` | string | **Yes** | Detected language |
| `duration` | number | **Yes** | Audio duration in seconds |
| `text` | string | **Yes** | Full transcribed text |
| `words` | TranscriptionWord[] | No | Word-level timestamps (if requested) |
| `segments` | TranscriptionSegment[] | No | Segment-level detail |
| `usage` | object | No | Duration-based usage |

**TranscriptionWord**: `{ word: string, start: float, end: float }`

**TranscriptionSegment**: `{ id: int, seek: int, start: float, end: float, text: string, tokens: int[], temperature: float, avg_logprob: float, compression_ratio: float, no_speech_prob: float }`

#### Response — text, srt, vtt formats

When `response_format` is `"text"`, `"srt"`, or `"vtt"`, the response is returned as `text/plain` with the appropriate format content (raw text, SubRip, or WebVTT respectively).

---

### Endpoint 6: POST /v1/audio/translations

Translates audio into English. Accepts `multipart/form-data`. **whisper-1 only**.

#### Request — CreateTranslationRequest

| Parameter | Type | Required | Default | Allowed Values | Description |
|-----------|------|----------|---------|----------------|-------------|
| `file` | file | **Yes** | — | flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm | Audio file |
| `model` | string | **Yes** | — | `"whisper-1"` | Only whisper-1 supported |
| `prompt` | string | No | — | any string (should be in English) | Context hint |
| `response_format` | string | No | `"json"` | `"json"`, `"text"`, `"srt"`, `"verbose_json"`, `"vtt"` | Output format |
| `temperature` | number | No | `0` | 0.0–1.0 | Sampling temperature |

#### Response

Same as transcription: `{ "text": "..." }` for json, or raw text/srt/vtt.

---

### Error Response Format

All endpoints return errors in this format:

```json
{
  "error": {
    "message": "Human-readable error description",
    "type": "invalid_request_error",
    "param": "prompt",
    "code": "invalid_api_key"
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `error.type` | string | **Yes** | Error category: `"invalid_request_error"`, `"authentication_error"`, `"permission_error"`, `"not_found_error"`, `"rate_limit_error"`, `"server_error"` |
| `error.message` | string | **Yes** | Human-readable message |
| `error.param` | string\|null | **Yes** | Which parameter caused the error (null if N/A) |
| `error.code` | string\|null | **Yes** | Machine-readable error code (null if N/A) |

Common HTTP status codes: 400 (bad request), 401 (auth), 403 (permission), 404 (not found), 429 (rate limit), 500 (server error).

## Key Numbers/Constants

### Image Size Values by Model

| Model | Supported Sizes | Default |
|-------|----------------|---------|
| dall-e-2 | `256x256`, `512x512`, `1024x1024` | `1024x1024` |
| dall-e-3 | `1024x1024`, `1792x1024`, `1024x1792` | `1024x1024` |
| gpt-image-1 / 1-mini / 1.5 | `1024x1024`, `1536x1024` (landscape), `1024x1536` (portrait), `auto` | `auto` |

### Image Quality Values by Model

| Model | Supported Qualities | Default |
|-------|-------------------|---------|
| dall-e-2 | `standard` | `standard` |
| dall-e-3 | `standard`, `hd` | `standard` |
| gpt-image-1 / 1-mini / 1.5 | `low`, `medium`, `high`, `auto` | `auto` |

### Audio Formats

**TTS output formats:**
| Format | MIME Type | Notes |
|--------|-----------|-------|
| mp3 | audio/mpeg | Default. General purpose |
| opus | audio/opus | Low latency, streaming/communication |
| aac | audio/aac | Broad device compatibility |
| flac | audio/flac | Lossless compression |
| wav | audio/wav | Uncompressed, low-latency decode |
| pcm | audio/pcm | Raw samples, no header |

**Transcription input formats:** flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm

**Transcription output formats:**
| Format | Content-Type | Notes |
|--------|-------------|-------|
| json | application/json | `{"text": "..."}` (default) |
| verbose_json | application/json | Includes language, duration, segments, words |
| text | text/plain | Raw transcript text |
| srt | text/plain | SubRip subtitle format |
| vtt | text/plain | WebVTT subtitle format |
| diarized_json | application/json | With speaker annotations (diarize model only) |

### TTS Voice Options

Built-in voices: `alloy`, `ash`, `ballad`, `coral`, `echo`, `fable`, `onyx`, `nova`, `sage`, `shimmer`, `verse`, `marin`, `cedar`

Note: `fable` and `onyx` are from the original set. `ash`, `ballad`, `coral`, `sage`, `verse`, `marin`, `cedar` were added later. Custom voices via `{"id": "voice_1234"}` are also supported.

### TTS Speed Range

Minimum: 0.25, Maximum: 4.0, Default: 1.0

### Prompt Length Limits

| Model | Max Prompt Length |
|-------|-----------------|
| dall-e-2 | 1,000 characters |
| dall-e-3 | 4,000 characters |
| GPT image models | 32,000 characters |
| TTS input | 4,096 characters |

### Image File Size Limits

| Model | Max File Size | Format Requirements |
|-------|-------------|-------------------|
| dall-e-2 | 4MB | PNG, square |
| GPT image models | 50MB | PNG, WebP, JPG |
| GPT image edit (multi-image) | 50MB each | Up to 16 images |
| dall-e-2 mask | 4MB | PNG, same dimensions as image, alpha channel for transparency |

## Data Layouts/Formats

### Image Endpoint Content Types

| Endpoint | Request Content-Type | Response Content-Type |
|----------|---------------------|----------------------|
| /v1/images/generations | application/json | application/json (or text/event-stream if streaming) |
| /v1/images/edits | multipart/form-data | application/json (or text/event-stream if streaming) |
| /v1/images/variations | multipart/form-data | application/json |
| /v1/audio/speech | application/json | audio/* (mp3/opus/aac/flac/wav/pcm) or text/event-stream |
| /v1/audio/transcriptions | multipart/form-data | application/json or text/plain |
| /v1/audio/translations | multipart/form-data | application/json or text/plain |

### Base64 Image Encoding

GPT image models return images as raw base64 strings in `data[].b64_json`. The string does NOT include a data URI prefix (no `data:image/png;base64,`). The caller must know the output format from the `output_format` response field or request parameter.

### Multipart Form Data Fields

For `/v1/images/edits`, the `image` field can be a single file or multiple files (up to 16 for GPT image models). When sending multiple images, use repeated `image` fields or array-style naming depending on client library. The `mask` field is always a single PNG file.

## Algorithm Steps

### SharpInference.Server Image Generation Flow

1. Accept POST /v1/images/generations with JSON body
2. Parse and validate `CreateImageRequest` — check model, size, quality, n constraints
3. Map `size` string to width/height integers (e.g., "1536x1024" -> 1536, 1024)
4. For each of `n` images:
   a. Run the appropriate diffusion pipeline (SD 1.5, SDXL, Flux, etc.)
   b. Encode output to requested `output_format` (png/jpeg/webp) with `output_compression`
   c. If `background=transparent`, ensure alpha channel is preserved (png/webp only)
5. Base64-encode each image and place in `data[].b64_json`
6. Return `ImagesResponse` with `created` timestamp and metadata

### SharpInference.Server Audio TTS Flow

1. Accept POST /v1/audio/speech with JSON body
2. Parse and validate `CreateSpeechRequest`
3. Run TTS model (Kokoro or similar) with specified voice and speed
4. Encode audio to requested format (mp3/opus/flac/wav/aac/pcm)
5. Return raw audio bytes with appropriate Content-Type header

### SharpInference.Server Transcription Flow

1. Accept POST /v1/audio/transcriptions with multipart form data
2. Extract audio file and parameters
3. Run Whisper model on audio
4. Format response according to `response_format`:
   - json: `{"text": "..."}`
   - verbose_json: include language, duration, segments, words
   - text: raw string
   - srt: SubRip formatted
   - vtt: WebVTT formatted
5. Return with appropriate Content-Type

## Reference Implementations

### LocalAI

- Supports `/v1/images/generations` only (not edits or variations)
- Uses `stablediffusion-ggml` and `diffusers` backends
- Supports negative prompts via pipe syntax in prompt field: `"positive prompt|negative prompt"`
- Extra parameters: `mode`, `step` (not in OpenAI spec)
- Size handled as direct passthrough to backend
- Source: [LocalAI Image Generation](https://localai.io/features/image-generation/)

### vLLM-Omni

- Supports `/v1/images/generations` endpoint
- Parameters passed directly to diffusion pipeline without model-specific validation
- Unsupported parameters silently ignored; incompatible values cause pipeline errors
- Returns base64-encoded PNG images
- Source: [vLLM-Omni Image Generation API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/image_generation_api/)

### openedai-speech

- OpenAI TTS-compatible server using Coqui XTTS v2 and/or Piper TTS
- Implements `/v1/audio/speech` endpoint
- Source: [openedai-speech (GitHub)](https://github.com/matatonic/openedai-speech)

## Differences Between Implementations

| Feature | OpenAI Official | LocalAI | vLLM-Omni | SharpInference (planned) |
|---------|----------------|---------|-----------|-------------------------|
| /v1/images/generations | Full spec | Supported (subset params) | Supported (passthrough) | Full spec |
| /v1/images/edits | Full spec | Not supported | Not supported | Full spec |
| /v1/images/variations | dall-e-2 only | Not supported | Not supported | Optional (low priority) |
| /v1/audio/speech | Full spec | Not documented | Not supported | Full spec |
| /v1/audio/transcriptions | Full spec | Supported (whisper backend) | Not supported | Full spec |
| Negative prompts | Not in API spec | Pipe syntax in prompt | N/A | Consider separate param |
| Streaming images | GPT image models | Not supported | Not supported | Planned |
| output_format (png/jpeg/webp) | GPT image models | N/A | PNG only | Full spec |
| background transparency | GPT image models | N/A | N/A | Full spec |

## Open Questions

- [ ] How to map OpenAI model names to local SharpInference models (e.g., `"dall-e-3"` -> which local diffusion pipeline?)
- [ ] Whether to support the `stream` parameter for image generation (requires SSE with partial image events)
- [ ] Whether to implement the `instructions` field for TTS (requires model support beyond basic Kokoro)
- [ ] Whether to support custom voices via `{"id": "voice_1234"}` object syntax
- [ ] How to handle `moderation` parameter — implement content safety filtering or ignore?
- [ ] Whether to support `input_fidelity` for image edits (requires model-specific face/detail preservation)
- [ ] Audio translation endpoint (`/v1/audio/translations`) — whisper-1 only, translates to English

## Implementation Notes

### Priority Order for SharpInference.Server

1. **Phase 1**: `/v1/images/generations` with `prompt`, `model`, `n`, `size`, `response_format` (b64_json), `output_format`. This covers the core use case.
2. **Phase 2**: `/v1/audio/speech` with `model`, `input`, `voice`, `response_format`, `speed`.
3. **Phase 3**: `/v1/audio/transcriptions` with `file`, `model`, `response_format`, `language`, `temperature`.
4. **Phase 4**: `/v1/images/edits` with multipart form data, mask support, `input_fidelity`.
5. **Phase 5**: Streaming support for images and audio, `/v1/audio/translations`, `/v1/images/variations`.

### Model Name Mapping Strategy

SharpInference should accept OpenAI model names and map them to local pipelines:
- `"dall-e-2"` / `"dall-e-3"` -> Stable Diffusion 1.5 or SDXL (closest local equivalent)
- `"gpt-image-1"` / `"gpt-image-1.5"` -> FLUX or SD3 (highest quality local model)
- `"gpt-image-1-mini"` -> SDXL or SD 1.5 (faster model)
- `"tts-1"` / `"tts-1-hd"` / `"gpt-4o-mini-tts"` -> Kokoro TTS
- `"whisper-1"` -> Local Whisper model

### Size Parsing

All size strings follow the pattern `"{width}x{height}"`. Parse with simple string split on `"x"`. The `"auto"` value should map to a sensible default (e.g., 1024x1024 or model-native resolution).

### Content-Type Handling

- Image endpoints: Always return `application/json` (unless streaming -> `text/event-stream`)
- Speech endpoint: Return `audio/{format}` matching the requested response_format
- Transcription: Return `application/json` for json/verbose_json/diarized_json, `text/plain` for text/srt/vtt

### URL Expiry for response_format=url

OpenAI URLs expire after 60 minutes. If SharpInference.Server implements URL-based responses, it should either:
- Serve images from a temporary file store with cleanup, or
- Simply always return b64_json (which is what GPT image models do anyway)

### Backward Compatibility

The `quality` and `style` parameters have different valid values depending on the model. SharpInference should accept all values and silently map unsupported ones to the closest equivalent. For example, if a client sends `quality=hd` (dall-e-3 style) but the local model is SDXL, map to `quality=high`.
