# Phase 2 — Deviations from Design Plan

## 1. T5 Tokenizer — Protobuf bos_id Patching

**Design assumption**: `SentencePieceTokenizer.Create()` from Microsoft.ML.Tokenizers v2.0.0 would load T5 models directly.

**Deviation**: T5 SentencePiece models set `bos_id = -1` in their TrainerSpec protobuf, meaning "no BOS token". Microsoft.ML.Tokenizers interprets -1 as a vocabulary array index, causing `IndexOutOfRangeException`. This affects all T5 models (T5-small through T5-XXL) across v1.0.0 through v3.0.0-preview of the library.

**Workaround**: `T5Tokenizer.PatchT5ProtobufStream()` reads the model into a MemoryStream and patches the protobuf binary to rename the `bos_id` field tag (field 41 → unused field 99) before passing to `SentencePieceTokenizer.Create()`. The patch finds the exact byte pattern (field tag `0xC8 0x02` followed by varint -1 `0xFF*9 0x01`) and changes the tag to `0x98 0x06`.

**Risk**: If Microsoft.ML.Tokenizers fixes this upstream, the patch becomes unnecessary but harmless — it only modifies a field that isn't used by T5 tokenization.

## 2. CLIP Token IDs — Microsoft.ML.Tokenizers vs Python CLIP

**Design assumption**: Token IDs from `BpeTokenizer.Create()` would match OpenAI's Python CLIP tokenizer exactly.

**Deviation**: Microsoft.ML.Tokenizers BPE implementation produces different token IDs than Python CLIP for the same input. Example: `"a photo of a cat"` → `[64, 1153, 684, 64, 1481]` (C#) vs `[320, 1125, 539, 320, 2368]` (Python). This is because the BPE implementation handles vocabulary mapping differently (likely the `bytes_to_unicode` and `</w>` end-of-word marker conventions).

**Impact**: Token IDs are internally consistent — the same input always produces the same output, and the tokenizer correctly handles SOT/EOT/padding. However, the IDs are not byte-identical to Python CLIP output. This means weight compatibility with pretrained CLIP text encoders must be verified end-to-end (encoder weights may expect the Python token ID space).

**Mitigation**: Integration tests verify round-trip consistency (encode → decode preserves text). Full end-to-end validation against Python pipeline output will happen when the CLIP text encoder is implemented in a later phase.

## 3. T5 Default Max Length

**Design assumption**: T5 tokenizer should default to 77 tokens (same as CLIP).

**Note**: The T5 research doc specifies SD3 uses `max_sequence_length=256` and Flux uses `max_sequence_length=512`. The default of 77 in `T5Tokenizer.DefaultMaxLength` is a generic default; pipelines will override this with the correct value (256 for SD3, 512 for Flux) when constructing the tokenizer.
