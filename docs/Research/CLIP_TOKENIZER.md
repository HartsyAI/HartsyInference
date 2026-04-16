# CLIP BPE Tokenizer — Research Notes

> Status: Complete
> Last Updated: 2026-04-16
> Needed Before: SharpInference.Tokenizers (ClipTokenizer)

## Summary

The CLIP tokenizer is a modified GPT-2 BPE tokenizer defined in `openai/CLIP` (`clip/simple_tokenizer.py`). Key differences from GPT-2: CLIP lowercases all text, uses `</w>` end-of-word marker instead of space-prefix, adds `<|startoftext|>` and `<|endoftext|>` special tokens, and uses Unicode-aware regex with `\p{L}` and `\p{N}`.

## Key Constants

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

## Vocabulary Structure (49,408 tokens)

Built in this exact order:

| Index Range | Count | Contents |
|---|---|---|
| 0–255 | 256 | Base byte-to-unicode characters |
| 256–511 | 256 | Same characters with `</w>` suffix |
| 512–49,405 | 48,894 | BPE merge results |
| 49,406 | 1 | `<\|startoftext\|>` |
| 49,407 | 1 | `<\|endoftext\|>` |

## The `bytes_to_unicode()` Mapping

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

## Text Preprocessing Pipeline

Applied in this exact order:

1. **basic_clean**: `ftfy.fix_text(text)` → `html.unescape(html.unescape(text))` → `strip()`
2. **whitespace_clean**: collapse all whitespace to single space → strip
3. **lowercase**: `text.lower()` (CLIP lowercases AFTER cleaning — key difference from GPT-2)

## Regex Tokenization Pattern

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

## BPE Encoding Algorithm

For each regex-matched token:

### Step 1: Byte-encode and map to unicode
```
"hello" → UTF-8 bytes [104, 101, 108, 108, 111] → mapped chars "hello"
```

### Step 2: Add end-of-word marker
Split into characters, append `</w>` to the LAST character:
```
word = ('h', 'e', 'l', 'l', 'o</w>')
```
**CRITICAL**: `</w>` is appended as a suffix to the last char, not as a separate symbol.

### Step 3: Iteratively apply BPE merges
```
1. Get all adjacent pairs in word
2. Find pair with LOWEST rank in bpe_ranks
3. If no pair has a rank → STOP
4. Merge all occurrences of that pair
5. Go to step 1
```

### Step 4: Look up token IDs
Each final BPE piece maps to a token ID via the encoder dictionary.

## Full Tokenization Pipeline

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

## Implementation Notes for SharpInference

1. **Current approach**: We wrap `Microsoft.ML.Tokenizers.BpeTokenizer` which handles the core BPE algorithm. Our `ClipTokenizer` adds CLIP-specific preprocessing (lowercase, SOT/EOT wrapping, 77-token padding).

2. **ftfy substitute**: For SD prompts, `string.Normalize(NormalizationForm.FormC)` is a minimal substitute. Most user prompts don't have encoding issues.

3. **HTML unescaping**: `System.Net.WebUtility.HtmlDecode()` called twice.

4. **BPE cache**: The Python implementation caches BPE results per unique token string — important for performance.

5. **Merge file**: `bpe_simple_vocab_16e6.txt.gz` — skip line 0 (header), read lines 1–48894.

6. **Gzip**: `System.IO.Compression.GZipStream` for decompression.

## Reference Implementations

- **Primary**: [openai/CLIP `simple_tokenizer.py`](https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py)
- **HuggingFace transformers**: `CLIPTokenizer` in `transformers/models/clip/tokenization_clip.py`
- **Merge file**: `bpe_simple_vocab_16e6.txt.gz` bundled with CLIP repo

## Open Questions

- [x] Vocabulary size: 49,408 (confirmed)
- [x] Padding value: 0 (zeros, not EOT)
- [x] Truncation behavior: last token forced to EOT (49407)
- [x] `</w>` marker: suffix on last char, not separate token
