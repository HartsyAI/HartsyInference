# Tokenizers — Research Notes

## CLIP BPE Tokenizer

The CLIP tokenizer is a modified GPT-2 BPE tokenizer defined in `openai/CLIP` (`clip/simple_tokenizer.py`). Key differences from GPT-2: CLIP lowercases all text, uses `</w>` end-of-word marker instead of space-prefix, adds `<|startoftext|>` and `<|endoftext|>` special tokens, and uses Unicode-aware regex with `\p{L}` and `\p{N}`.

### Key Constants

| Constant | Value |
|---|---|
| Vocabulary size | 49,408 |
| SOT token (`<\|startoftext\|>`) | 49,406 |
| EOT token (`<\|endoftext\|>`) | 49,407 |
| Context length (max sequence) | 77 |
| Padding value | 0 |
| Number of BPE merges | 48,894 |
| Base byte tokens | 256 |
| Base byte + `</w>` variants | 512 |
| BPE merge file | `bpe_simple_vocab_16e6.txt.gz` |

### Vocabulary Structure (49,408 tokens)

Built in this exact order:

| Index Range | Count | Contents |
|---|---|---|
| 0–255 | 256 | Base byte-to-unicode characters |
| 256–511 | 256 | Same characters with `</w>` suffix |
| 512–49,405 | 48,894 | BPE merge results |
| 49,406 | 1 | `<\|startoftext\|>` |
| 49,407 | 1 | `<\|endoftext\|>` |

### The `bytes_to_unicode()` Mapping

Maps all 256 byte values to printable Unicode characters:

- Bytes 33–126 (ASCII `!` to `~`): 94 values → map to themselves
- Bytes 161–172 (Latin `¡` to `¬`): 12 values → map to themselves
- Bytes 174–255 (Latin `®` to `ÿ`): 82 values → map to themselves
- Remaining 68 bytes (control chars, whitespace): mapped to U+0100 onwards (e.g., space → `Ġ` at U+0120)

```csharp
Dictionary<byte, char> BytesToUnicode()
{
    List<int> bs = new List<int>();
    for (int i = 33; i <= 126; i++) bs.Add(i);
    for (int i = 161; i <= 172; i++) bs.Add(i);
    for (int i = 174; i <= 255; i++) bs.Add(i);

    List<int> cs = new List<int>(bs);
    int n = 0;
    for (int b = 0; b < 256; b++)
    {
        if (!bs.Contains(b))
        {
            bs.Add(b);
            cs.Add(256 + n);
            n++;
        }
    }
    Dictionary<byte, char> result = new Dictionary<byte, char>();
    for (int i = 0; i < bs.Count; i++)
        result[(byte)bs[i]] = (char)cs[i];
    return result;
}
```

### Text Preprocessing Pipeline

Applied in this exact order:

1. **basic_clean**: `ftfy.fix_text(text)` → `html.unescape(html.unescape(text))` → `strip()`
2. **whitespace_clean**: collapse all whitespace to single space → strip
3. **lowercase**: `text.lower()` (CLIP lowercases AFTER cleaning — key difference from GPT-2)

### Regex Tokenization Pattern

```
<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+
```

Matches in priority order:
1. Special tokens (`<|startoftext|>`, `<|endoftext|>`)
2. English contractions (`'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`)
3. Unicode letter runs (`[\p{L}]+`)
4. Single Unicode digits (`[\p{N}]`)
5. Non-whitespace, non-letter, non-digit runs (punctuation)

.NET `System.Text.RegularExpressions` supports `\p{L}` and `\p{N}` natively.

### BPE Encoding Algorithm

For each regex-matched token:

#### Step 1: Byte-encode and map to unicode
```
"hello" → UTF-8 bytes [104, 101, 108, 108, 111] → mapped chars "hello"
```

#### Step 2: Add end-of-word marker
Split into characters, append `</w>` to the LAST character:
```
word = ('h', 'e', 'l', 'l', 'o</w>')
```
**CRITICAL**: `</w>` is appended as a suffix to the last char, not as a separate symbol.

#### Step 3: Iteratively apply BPE merges
```
1. Get all adjacent pairs in word
2. Find pair with LOWEST rank in bpe_ranks
3. If no pair has a rank → STOP
4. Merge all occurrences of that pair
5. Go to step 1
```

#### Step 4: Look up token IDs
Each final BPE piece maps to a token ID via the encoder dictionary.

### Full Tokenization Pipeline

```
Input: text string
Output: int[] of length 77, zero-padded

1. Clean text (ftfy → HTML unescape ×2 → strip)
2. Whitespace clean (collapse → strip)
3. Lowercase
4. Regex findall → list of token strings
5. For each token: UTF-8 encode → bytes_to_unicode → BPE → token IDs
6. Prepend SOT (49406)
7. Append EOT (49407)
8. Truncate to 77 (last token forced to EOT)
9. Pad with zeros to length 77
```

**Truncation**: At most 75 "content" tokens fit (positions 1–75), SOT at 0, EOT at 76.

**Padding**: Zeros (0), NOT the EOT token.

### Implementation Notes

1. **Current approach**: We wrap `Microsoft.ML.Tokenizers.BpeTokenizer` which handles the core BPE algorithm. Our `ClipTokenizer` adds CLIP-specific preprocessing (lowercase, SOT/EOT wrapping, 77-token padding).
2. **ftfy substitute**: For SD prompts, `string.Normalize(NormalizationForm.FormC)` is a minimal substitute. Most user prompts don't have encoding issues.
3. **HTML unescaping**: `System.Net.WebUtility.HtmlDecode()` called twice.
4. **BPE cache**: The Python implementation caches BPE results per unique token string — important for performance.
5. **Merge file**: `bpe_simple_vocab_16e6.txt.gz` — skip line 0 (header), read lines 1–48894.
6. **Gzip**: `System.IO.Compression.GZipStream` for decompression.

### Reference Implementations

- [openai/CLIP `simple_tokenizer.py`](https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py) — primary reference
- HuggingFace `transformers/models/clip/tokenization_clip.py`
- Merge file: `bpe_simple_vocab_16e6.txt.gz` bundled with CLIP repo

---

## T5 SentencePiece Tokenizer

T5 uses a **SentencePiece Unigram** model (not BPE). The T5-XXL model used in SD3 and Flux has a **vocabulary size of 32,128 tokens**. The tokenizer file is a `.model` file in protobuf binary format. Tokenization uses the Viterbi algorithm to find the optimal segmentation by maximum log probability. T5 does NOT use a BOS token — only EOS is appended.

### Key Constants

| Token | String | ID | Type |
|-------|--------|-----|------|
| PAD | `<pad>` | 0 | CONTROL |
| EOS | `</s>` | 1 | CONTROL |
| UNK | `<unk>` | 2 | UNKNOWN |

- **Vocabulary size**: 32,128 tokens
- **No BOS token** — T5 does NOT prepend a start token
- **EOS appended** at end of sequence (ID 1)
- **Padding** with PAD (ID 0) to max length
- **SD3 max_sequence_length**: 256 (hard limit 512)
- **Flux max_sequence_length**: 512

### SD3 and Flux Pipeline Usage

Both use `padding="max_length"`, `truncation=True`, `add_special_tokens=True`. The EOS token (ID 1) is appended, then padded with PAD tokens (ID 0) to max length.

### Text Preprocessing / Normalization Pipeline

#### Step 1: NFKC Normalization
The T5 model includes a `precompiled_charsmap` in the normalizer_spec implementing NFKC Unicode normalization. Practical C# alternative: `string.Normalize(NormalizationForm.FormKC)`.

#### Step 2: Remove Extra Whitespaces
Leading whitespace stripped, consecutive whitespace collapsed to single space, trailing whitespace stripped.

#### Step 3: Add Dummy Prefix
A space character is **prepended** to input text. `"hello world"` → `" hello world"`. Ensures the first word gets the same `\u2581` prefix as other words.

#### Step 4: Escape Whitespaces
All spaces (0x20) replaced with `\u2581` (LOWER ONE EIGHTH BLOCK, UTF-8: `\xe2\x96\x81`).

**Net effect**: `"hello world"` → `"\u2581hello\u2581world"` (with leading meta-space from dummy prefix).

### SentencePiece Model File Format

#### Protobuf Schema (key fields)

```protobuf
message ModelProto {
  message SentencePiece {
    enum Type {
      NORMAL = 1;       // Regular vocabulary piece
      UNKNOWN = 2;      // Unknown token (<unk>)
      CONTROL = 3;      // Control tokens (<pad>, </s>)
      USER_DEFINED = 4; // User-defined tokens
      UNUSED = 5;       // Unused placeholder
      BYTE = 6;         // Byte fallback tokens (<0x00> through <0xFF>)
    }
    optional string piece = 1;   // token string
    optional float score = 2;    // log probability score
    optional Type type = 3;      // piece type [default = NORMAL]
  }
  repeated SentencePiece pieces = 1;          // vocabulary entries
  optional TrainerSpec trainer_spec = 2;
  optional NormalizerSpec normalizer_spec = 3;
}
```

**The piece index in the repeated array IS the token ID.** Index 0 = token ID 0, etc.

#### Protobuf Wire Format

- Tag encoding: `(field_number << 3) | wire_type` as varint
- Wire type 0 (VARINT): int, bool, enum
- Wire type 2 (LEN): string, bytes, embedded messages
- Wire type 5 (I32): float (4 bytes little-endian IEEE 754)

### Unigram Tokenization Algorithm (Viterbi)

#### Core Algorithm

```
function Encode(normalized_text):
    n = length(text) in bytes
    best[0].score = 0.0
    best[1..n].score = -infinity

    for pos = 0 to n-1 (UTF-8 char boundaries only):
        if best[pos].score == -infinity: continue

        // Try all vocabulary pieces starting at this position
        for each matching piece (id, score) starting at pos:
            end_pos = pos + piece_length_bytes
            candidate = best[pos].score + score
            if candidate > best[end_pos].score:
                best[end_pos] = {id, candidate, pos}

        // Byte fallback for unmatched characters
        if no single-char match at pos:
            char_len = utf8_char_length(text[pos])
            unk_score = min_vocab_score - 10.0
            candidate = best[pos].score + unk_score
            if candidate > best[pos + char_len].score:
                best[pos + char_len] = {unk_id, candidate, pos}

    // Backtrack from position n to reconstruct token sequence
    result = backtrack(best, n)
    return result
```

#### Key Details

- **Scores are log probabilities** (negative floats). Viterbi finds maximum total score.
- **UNK penalty**: `min_score - 10.0` where min_score is the minimum across all NORMAL pieces.
- **Positions are byte offsets** in UTF-8 encoded text, respecting character boundaries.
- **max_sentencepiece_length** default is 16, limiting substring lookups per position.

### Byte Fallback Mechanism

When `byte_fallback` is enabled (T5 uses this), unknown characters are decomposed into byte tokens instead of UNK:

- **Format**: `<0xHH>` where HH is uppercase hex, zero-padded to 2 digits
- **256 byte tokens** in vocabulary (type = BYTE)
- After Viterbi, any UNK piece is replaced with byte tokens for each UTF-8 byte of the original surface text

### Complete End-to-End Pipeline

```
Input: "A cat sat."

1. NFKC normalize     → "A cat sat."
2. Remove extra space  → "A cat sat."
3. Add dummy prefix    → " A cat sat."
4. Escape whitespace   → "▁A▁cat▁sat."
5. Viterbi tokenize    → ["▁A", "▁cat", "▁sat", "."] → token IDs
6. Byte fallback       → replace any UNK with <0xHH> sequences
7. Append EOS          → [...ids..., 1]
8. Pad to max_length   → [...ids..., 1, 0, 0, 0, ...]
```

### Verification Checklist

1. Apply NFKC, remove extra whitespace, add prefix space, escape to `\u2581`
2. Viterbi with float32 scores
3. Byte fallback format: `<0x%02X>` (uppercase hex, zero-padded)
4. EOS (ID 1) appended after content tokens
5. No BOS — T5 does NOT prepend any start token
6. Padding with PAD (ID 0) to max_sequence_length
7. Token IDs = array indices in the .model file's pieces array

### Implementation Notes

1. **Current approach**: We wrap `Microsoft.ML.Tokenizers.SentencePieceTokenizer` which handles the protobuf parsing, Viterbi algorithm, and byte fallback internally. Our `T5Tokenizer` adds T5-specific conventions (EOS appending, padding, no BOS).
2. **NFKC normalization**: `SentencePieceTokenizer` handles this via the precompiled charsmap in the model file.
3. **The `\u2581` character**: All word-initial vocabulary pieces start with this character. It's the SentencePiece convention for marking word boundaries.
4. **Scores use float32** (not double). The original implementation uses float.
5. **Sentinel tokens**: T5 has 100 extra sentinel tokens (`<extra_id_0>` through `<extra_id_99>`) appended in reverse order. These are used for T5's span-corruption pre-training objective but are irrelevant for SD3/Flux text encoding.

### Reference Implementations

- [google/sentencepiece](https://github.com/google/sentencepiece) — `sentencepiece_model.proto`, `unigram_model.cc`, `normalizer.cc`
- HuggingFace `transformers/models/t5/tokenization_t5.py`
- SD3 pipeline: `diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py`
- Flux pipeline: `diffusers/pipelines/flux/pipeline_flux.py`
