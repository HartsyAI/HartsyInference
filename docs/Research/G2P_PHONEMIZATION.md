# G2P / Phonemization — Research Notes

> Status: Complete | Last Updated: 2026-05-17 | Needed Before: HartsyInference.Audio (any phoneme-input TTS — Kokoro, StyleTTS2, MeloTTS)

## Summary

Most modern TTS models that take phoneme input depend on **espeak-ng** (a GPL/LGPL C library) or **phonemizer** (its Python wrapper). HartsyInference is pure C# / .NET 10 with no native binaries, so we cannot link or ship espeak. This document surveys G2P (grapheme-to-phoneme) options that can be implemented in pure C# and recommends a per-language strategy.

The short version: for **English** we ship CMUDict (134k entries, public domain, ARPABET) with a deterministic ARPABET→IPA conversion table and POS-tag-based heteronym disambiguation; OOV words go to a small (~5-6 M param) ByT5-style neural G2P model run through HartsyInference itself. For **Mandarin** we ship jieba.NET + a pinyin dictionary + a pinyin→IPA table with tone digits. For **Korean** we do pure rule-based Hangul-jamo decomposition (trivial). For **Japanese** we either skip it on day one or port pyopenjtalk's MeCab+UniDic pipeline (large project — flagged). For **German / Spanish / French / Italian / Portuguese / Russian / Hindi** we port `epitran`-style rule tables. Notably, **F5-TTS, XTTS-v2, Bark, and IndexTTS-2** require no G2P at all — they take raw text — so for those models we ship nothing and just normalize text. Kokoro is the most demanding consumer here: its phoneme vocabulary is exactly 178 tokens (indices 0-177, pad=0) and its expected IPA inventory matches misaki's 49-phoneme English set plus diacritics, prosody arrows, and punctuation.

## Detailed Findings

### 1. Why TTS Needs G2P, and What an IPA String Looks Like

Most neural TTS models do not learn from raw text. They learn from a sequence of **phonemes** — abstract sound units. This is because:

- English orthography is famously irregular ("though", "through", "thought", "thorough" all spell `ough` differently). A phoneme-input model never has to learn this.
- Phoneme vocabularies are small (40-100 symbols), so the model's input embedding table is tiny.
- Phonemes generalize across speakers and accents; characters do not.
- Many TTS architectures (StyleTTS 2, MeloTTS, GPT-SoVITS in part) have an explicit "PLBERT" (phoneme-level BERT) that only makes sense over phoneme tokens.

A G2P engine maps written text to an IPA (or ARPABET) string. Example:

```
input:  "Hello world."
IPA:    "həlˈoʊ wˈɜːld."          (US English)
IPA:    "həlˈəʊ wˈɜːld."          (UK English)
ARPABET: "HH AH0 L OW1  W ER1 L D"
```

`ˈ` is U+02C8 PRIMARY STRESS, `ˌ` is U+02CC SECONDARY STRESS, `ː` is U+02D0 LENGTH MARK, `oʊ` is the diphthong "oh".

**IPA vs ARPABET**:

| Feature | IPA | ARPABET (CMU variant) |
|---|---|---|
| Character set | Unicode (combining diacritics common) | ASCII letters + digits |
| Stress | Prefix mark `ˈ`/`ˌ` before stressed syllable | Suffix digit `0`/`1`/`2` on the vowel |
| Phoneme count for English | ~49 (misaki), ~40 (compact) | exactly 39 (CMU) |
| Example | `həlˈoʊ` | `HH AH0 L OW1` |
| Granularity | Tonality, length, voicing diacritics available | Coarse — one symbol per phoneme |

CMU ARPABET → IPA is a fixed table (no context dependence). It's a one-line lookup per token. We always have to do this conversion because **every modern TTS model that takes phonemes expects IPA**, not ARPABET.

### 2. espeak-ng — The De-Facto Reference (and Why We Can't Use It)

espeak-ng (https://github.com/espeak-ng/espeak-ng) is a C library that does TTS for ~100 languages. The bulk of TTS research piggy-backs on it for the G2P stage (calling it via `phonemizer` Python wrapper, then training a model on the IPA output). It is the implicit gold-standard reference for what "correct" phonemization looks like.

**Architecture** (per the [dictionary docs](https://github.com/espeak-ng/espeak-ng/blob/master/docs/dictionary.md)):

- One pair of text files per language: `<lang>_rules` (rewrite rules) and `<lang>_list` (lexicon overrides).
- These compile into a binary `<lang>_dict` blob, loaded at runtime.
- Rule syntax: `[<pre>)] <match> [(<post>] <phoneme string>`, with letter classes (`A` = vowel, `C` = consonant, `_` = word boundary, `%` = doubled letter, `@` = syllable count).
- Rules are organized into "groups" by leading letter; the matcher iterates left-to-right, longest match wins per group.
- Special operators handle suffix removal (`S<n>`) and prefix removal (`P<n>`) with reanalysis, so e.g. `running` becomes `run` + `+ing` recursively.
- The lexicon overrides individual words (heteronyms with POS, foreign words, abbreviations).

**Why we can't link or ship it**:

1. **Native code, no managed binding**. espeak-ng is C; the only C# binding (`espeak-sharp` etc.) P/Invokes the shared library. Per CLAUDE.md: "Pure C# only — no native shared libraries". Disqualifying.
2. **License**. espeak-ng is GPLv3, espeak-data is also GPLv3. We're aiming at MIT-ish for HartsyInference — GPL would force the entire engine to GPL.
3. **Binary `.so` / `.dll` deployment**. Even if licensing were fine, we'd need per-platform binaries and would lose AOT-friendliness.

**What we *can* take from espeak-ng**:

- The compiled `_dict` binary files are GPL data, so still off-limits.
- The **source rule files** (`<lang>_rules`, `<lang>_list`) are also GPL — also off-limits for redistribution.
- The **algorithmic ideas** (rule format, suffix/prefix reanalysis, letter classes) are not copyrightable. We can reimplement an equivalent rule engine.
- The **gruut** project (MIT) ships espeak-ng-derived IPA lexicons that have been re-licensed — they are reusable. See section 5.

### 3. misaki — Kokoro's G2P Engine

**Repository**: https://github.com/hexgrad/misaki — Apache-2.0.

misaki is the recommended G2P for Kokoro 82M. It is the most polished pure-Python G2P that targets IPA and explicitly handles heteronyms.

**Architecture**:

- **Per-language modules** (`misaki.en`, `misaki.ja`, `misaki.ko`, `misaki.zh`, `misaki.vi`). Each has its own dependency footprint.
- **English (`misaki.en`)**:
  - **spaCy** (`en_core_web_sm`) for tokenization, POS tagging, lemmatization. Used purely for heteronym disambiguation.
  - **Two JSON dictionaries** at `misaki/data/`: `us_gold.json` (~hand-curated entries) and `us_silver.json` (machine-derived bulk). Same structure for British: `gb_gold.json`, `gb_silver.json`.
  - **Lookup order**: gold → silver → rules → fallback model.
  - **Heteronym map**: gold dictionary entries can have per-POS variants — `{"axes": {"NOUN": "ˈæksɪz", "VERB": "ˈæksɪz"}}` style.
  - **Fallback**: an optional transformer model (`PhonoGlyphe`, ~6 M params) that runs when a word isn't in either dict. See section 8.
  - **num2words** for "1234" → "one thousand two hundred thirty four" expansion.
- **Japanese**: depends on `pyopenjtalk` + `unidic-py` (~750 MB of dictionary data).
- **Chinese**: depends on `paddlespeech` for the front-end, then `jieba` + `pypinyin`.
- **Korean**: adapted from `g2pkc`, rule-based — no heavy dependency.

**English phoneme inventory** ([EN_PHONES.md](https://github.com/hexgrad/misaki/blob/main/EN_PHONES.md)):

- **49 total**: 41 shared US/UK + 4 US-only + 4 UK-only.
- Consonants: `b d f h j k l m n p s t v w z ɡ ŋ ɹ ʃ ʒ ð θ ʤ ʧ`
- Shared vowels: `i ɪ ɛ æ ə ɝ ʌ u ʊ ɔ ɑ`
- Diphthongs (custom capital-letter encoding so each diphthong is one token, not two): `A` = `eɪ`, `I` = `aɪ`, `W` = `aʊ`, `Y` = `ɔɪ`
- Schwa variant: `ᵊ` (superscript schwa, U+1D4A) for r-coloured/syllabic schwa
- US-only: `æ`, `O` (= `oʊ`), `ᵻ` (near-close central unrounded vowel, U+1DFB), `ɾ` (flap T)
- UK-only: `a`, `Q` (= `əʊ`), `ɒ`, plus length marker `ː`
- Stress: `ˈ` (primary), `ˌ` (secondary), prefixed before the stressed syllable

This is the inventory Kokoro's 178-token vocab is built around (see section 9).

### 4. CMUDict + ARPABET→IPA Conversion (our English baseline)

**CMU Pronouncing Dictionary** ([cmusphinx/cmudict](https://github.com/cmusphinx/cmudict)):

- **134,000+ entries**, plain-text, **public domain** ("free for research or commercial use, provided acknowledgement").
- **Format**: `WORD  PH1 PH2 PH3 ...` — uppercase headword, two spaces, space-separated ARPABET phonemes.
- **Variants**: alternate pronunciations marked `WORD(2)`, `WORD(3)`, etc.
- **39 phonemes** total: 15 vowels + 24 consonants.
- **Stress**: appended digit on each vowel — `0` = unstressed, `1` = primary, `2` = secondary. E.g. `AH0` vs `AH1`.
- **Example entry**:
  ```
  HELLO  HH AH0 L OW1
  HELLO(2)  HH EH0 L OW1
  WORLD  W ER1 L D
  ```
- File size: ~3.5 MB raw, ~1.5 MB gzipped — perfectly fine to ship in a NuGet package.

**ARPABET → IPA mapping table** (used in `g2p-en`, NLTK, balacoon, and the gruut project — all agree):

Vowels (stress 0/1/2 maps to IPA: none / `ˈ`-prefix / `ˌ`-prefix):

| ARPABET | IPA | Example |
|---|---|---|
| AA | ɑ | "father" |
| AE | æ | "cat" |
| AH | ʌ (stressed), ə (unstressed) | "but", "sofa" |
| AO | ɔ | "thought" |
| AW | aʊ | "cow" |
| AX | ə | "sofa" (extension) |
| AY | aɪ | "hide" |
| EH | ɛ | "red" |
| ER | ɝ (stressed), ɚ (unstressed) | "bird", "letter" |
| EY | eɪ | "say" |
| IH | ɪ | "big" |
| IY | i | "see" |
| OW | oʊ | "go" |
| OY | ɔɪ | "boy" |
| UH | ʊ | "book" |
| UW | u | "blue" |

Consonants:

| ARPABET | IPA | Example |
|---|---|---|
| B | b | "be" |
| CH | tʃ | "cheese" |
| D | d | "do" |
| DH | ð | "this" |
| F | f | "fee" |
| G | ɡ | "go" (note U+0261, not ASCII g) |
| HH | h | "he" |
| JH | dʒ | "joy" |
| K | k | "key" |
| L | l | "lee" |
| M | m | "me" |
| N | n | "no" |
| NG | ŋ | "sing" |
| P | p | "pea" |
| R | ɹ | "red" |
| S | s | "sea" |
| SH | ʃ | "she" |
| T | t | "tea" |
| TH | θ | "thin" |
| V | v | "vee" |
| W | w | "we" |
| Y | j | "yes" |
| Z | z | "zee" |
| ZH | ʒ | "vision" |

Stress conversion: rewrite `HH AH0 L OW1` (ARPABET) as `həlˈoʊ` (IPA) — drop the digit and prefix `ˈ` on the syllable containing that vowel. Syllable boundaries are implicit (insertion of `ˈ` before the consonants leading up to the stressed vowel; the standard rule is "maximal onset" — assign as many preceding consonants to the same syllable as form a legal onset).

**Caveats**:
- CMUDict's `AH` is the only vowel whose IPA depends on stress: `AH0` → `ə`, `AH1` → `ʌ`. Most converters hard-code this rule.
- `R` is `ɹ` in IPA (the American alveolar approximant), not `r` (the rolled-r trill).
- Use the IPA Latin small letter script g `ɡ` (U+0261), not ASCII `g`. Many models tokenize these as different IDs.

**Heteronym handling on top of CMUDict**:

CMUDict gives you all pronunciations of "lead" but doesn't tell you which one to use in context. Standard approach (used by `g2p-en`):

1. POS-tag the input sentence with a small averaged-perceptron tagger (NLTK's tagger is ~3 MB).
2. Look up the headword in a hand-curated **heteronym table** keyed by `(word, POS)` — covers ~1,000 common heteronyms (lead/lead, read/read, bow/bow, tear/tear, wind/wind, axes, bass, present, conduct, object, ...).
3. Fall through to CMUDict's first entry otherwise.

The NLTK averaged-perceptron tagger has been ported to pure C# already (search "PerceptronTagger C#"); alternatively, a small CRF over POS would be a tiny side model.

### 5. gruut, OpenPhonemizer, Phonikud — Standalone Phonemizers

**gruut** ([rhasspy/gruut](https://github.com/rhasspy/gruut), MIT):

- Pure-Python tokenizer + IPA phonemizer for ~13 languages: ar, cs-cz, de-de, en-us, en-gb, es-es, fa, it-it, lb-lb, nl, pt, ru-ru, sw.
- Ships a lexicon per language plus a pre-trained seq2seq G2P model for OOV (small CRF, not transformer).
- IPA inventory generated from espeak-ng output, but the **lexicons are re-derived/re-licensed**, so the data is usable.
- **This is the single most relevant prior art**. If we port one project to C#, this is the one.
- StyleTTS 2's official inference code switched from espeak to gruut for exactly this licensing reason.

**OpenPhonemizer** ([NeuralVox/OpenPhonemizer](https://github.com/NeuralVox/OpenPhonemizer), MIT):

- Drop-in espeak replacement built on **DeepPhonemizer** (transformer G2P).
- English-only. Ships a ~100 MB built-in word→phoneme dict for speed; the transformer is the fallback.
- **No longer maintained** per the README — useful as inspiration, not as an ongoing source.

**Phonikud** ([thewh1teagle/phonikud](https://github.com/thewh1teagle/phonikud), MIT + arXiv 2506.12311):

- Hebrew-specific. Tiny diacritization-then-rule pipeline.
- Currently being ported to a pure FST (finite-state transducer) form to drop Python entirely.
- Relevant signal: **FST-based phonemizers are practical at ~50k words/sec** in interpreted languages and would be even faster in C#. The FST runtime is a few hundred lines.

### 6. Per-Language Strategy

#### English (US + UK) — `[high confidence]`

**Approach**: CMUDict + ARPABET→IPA table + heteronym disambiguation + neural fallback.

- **Primary lookup**: CMUDict (134k entries, public domain). Ship as a binary trie or perfect-hash structure (~1.5 MB packed).
- **Heteronym table**: ~1,000 entries keyed by `(word, POS)`. Hand-curated; both `g2p-en` and `misaki` have starter sets we can reference (Apache-licensed in misaki's case — directly reusable).
- **POS tagger**: pure-C# averaged-perceptron, ~3 MB model.
- **Punctuation / numerals**: normalize numbers via a small expansion module (port of `num2words` — 600 LOC for English).
- **OOV fallback**: small ByT5-style G2P (see section 8), running through HartsyInference itself as a `.safetensors`.
- **UK variant**: same pipeline, different lexicon file (BEEP or gruut's en-gb).

Expected accuracy vs espeak reference: ≥98 % on in-vocab, ≥92 % on OOV with the neural fallback. CMUDict alone (no neural) is ~90 % on the typical inference text because most OOV is proper nouns.

#### Mandarin Chinese — `[high confidence, lots of work]`

**Approach**: jieba.NET + character→pinyin dictionary + pinyin→IPA table.

- **Segmenter**: `jieba.NET` ([anderscui/jieba.NET](https://github.com/anderscui/jieba.NET)) — pure C#, on NuGet, MIT, equivalent to Python jieba up to v0.42. Segments Chinese strings into words.
- **Char→pinyin dict**: equivalent of `pypinyin` — ship the CC-CEDICT-derived char→pinyin map (~14 MB raw, ~3 MB packed). Multi-character words first (jieba output), then fallback char-by-char.
- **Pinyin→IPA**: a deterministic table (~400 entries — every legal pinyin syllable in Mandarin). Encode tones with **digits 1-4** (and 5 / 0 for neutral) appended to the syllable, or with IPA tone-letter diacritics `˥ ˧˥ ˨˩˦ ˥˩`.
- **Tone sandhi**: third-tone before third-tone becomes second-tone — single rule, ~5 lines.
- **Polyphone handling**: words like 行 (xíng vs háng) need word-context. CC-CEDICT entries already include this when the word is a known multi-char compound.
- Chao tone numbers (5-3-5-1 scale) are used for fine-grained IPA. Mandarin's four lexical tones in Chao: 55 (T1), 35 (T2), 214 (T3), 51 (T4), neutral (no contour).

Reference: the GPT-SoVITS Chinese frontend is the simplest reusable spec.

#### Korean — `[high confidence, easy]`

**Approach**: Pure rule-based jamo decomposition + assimilation rules.

- Hangul characters in Unicode block U+AC00-U+D7A3 are composed of (initial, medial, final) jamo. The decomposition is arithmetic:
  ```
  s = code - 0xAC00       (0 ≤ s < 11172)
  initial = s / 588       (0..18, 19 choices)
  medial = (s / 28) % 21  (0..20, 21 choices)
  final = s % 28          (0..27, 28 choices including no-final)
  ```
- Each jamo maps to an IPA segment via a fixed ~70-row table.
- Apply ~12 phonological rules (nasal assimilation, palatalization, intervocalic /h/ deletion, lenis-tense alternations). All deterministic.
- Reference: `g2pkc` / `g2pkk` Python projects, ~500 LOC of rules.
- Hangul→IPA is the easiest non-English language; estimate **1-2 days of work in C#**.

#### Japanese — `[medium-high effort, gate this]`

**Approach**: Port pyopenjtalk's MeCab+UniDic pipeline, OR punt to a future phase.

- Japanese G2P requires **morphological analysis** because there are no spaces, kanji have multiple readings, and pitch accent depends on the word.
- `pyopenjtalk` wraps OpenJTalk's MeCab + a custom Naist-JDIC dictionary (~50 MB).
- A pure-C# MeCab port exists: **NMeCab** (https://github.com/komutan/NMeCab) — MIT-licensed, mature. But the dictionary it normally uses is IPAdic (older). For TTS-quality readings we'd need to repack UniDic-CSJ or Naist-JDIC into NMeCab's binary format.
- **Estimated effort**: 2-3 weeks (dictionary repackaging is the painful part).
- **Recommendation**: ship Kokoro's Japanese support in a later phase. Until then, Japanese TTS support routes through F5-TTS / XTTS-v2 (both character-level) or fails fast with a clear error.

#### Spanish / Italian / Portuguese — `[high confidence, easy]`

**Approach**: Rule-based, no dictionary needed.

- All three have **highly regular orthography → phoneme** mappings (Italian and Spanish especially). 50-100 rewrite rules cover ~99 % of words.
- Stress: usually predictable from the orthography (Spanish: penultimate unless marked with acute; Italian: penultimate by default; Portuguese: penultimate-or-final rules + diacritics).
- Reference: `epitran` rule files for `spa-Latn`, `ita-Latn`, `por-Latn` (Apache 2.0) — these are plain TSV-style rules we can port directly.
- Estimated effort: **2-3 days each**.

#### French — `[medium effort]`

**Approach**: Rule-based + ~10k-entry lexicon for irregular cases.

- French has many silent letters and liaison phenomena. Rule-based gets you to ~85 %, but loanwords and exceptions need a lexicon.
- Reference: `epitran` `fra-Latn`, gruut `fr-fr`. Lexicon ~5 MB.
- Estimated effort: **1 week**.

#### German — `[medium effort]`

**Approach**: Rule-based + exception lexicon.

- German orthography is fairly regular but has compound-word splitting issues and irregular vowel quality on loanwords.
- Reference: gruut `de-de`, epitran `deu-Latn`. Lexicon ~10k entries.
- Estimated effort: **1 week**.

#### Russian — `[medium-high effort]`

**Approach**: Stress dictionary + reduction rules.

- The hard part is **stress placement**, which is lexical and unpredictable from the orthography. Without stress, vowel reduction can't be computed. So we need a stress dictionary (~120k entries, ~5 MB) — Wiktionary-derived ones exist under permissive licenses.
- Once stress is known, vowel reduction (akanye/ikanye) and consonant palatalization/voicing-assimilation rules are deterministic (~30 rules).
- Reference: gruut `ru-ru`, the "russtress" project.
- Estimated effort: **2 weeks**.

#### Hindi — `[high confidence, mostly easy]`

**Approach**: Rule-based Devanāgarī decomposition + schwa-deletion logic.

- Devanāgarī is largely phonemic — each akshara maps to one consonant + one vowel.
- The **one** wrinkle is **schwa deletion** (word-final and certain medial positions). A rule-based deleter (Choudhury 2003) hits ~97 % accuracy.
- Reference: `epitran` `hin-Deva`.
- Estimated effort: **3-5 days**.

### 7. OOV Handling

OOV = "out of vocabulary": a word not in the dictionary. Options, ranked by cost:

| Option | Pros | Cons |
|---|---|---|
| (a) Ship a larger lexicon (CMUDict + Wiktionary + Wikipedia titles, ~500k entries) | Best accuracy on names; zero inference cost | Storage bloat (~20 MB); never covers everything |
| (b) Small neural seq2seq G2P (~5M params) running through HartsyInference | Generalizes; handles names; reusable across languages (ByT5) | ~10-20 ms per OOV word; needs HartsyInference inference path |
| (c) Per-language rule-based fallback | No model, no extra MB | Bad on proper nouns and loanwords |
| (d) Return placeholder phoneme (e.g. just spell the letters phonetically) | Simplest | Poor quality; some models choke on unknown tokens |

**Per-TTS-model OOV expectations** (from reading inference code):

- **Kokoro**: unknown chars are silently dropped; an all-unknown input produces silence. We need to never emit chars outside the 178-token vocab.
- **StyleTTS 2**: same — token IDs not in vocab raise an exception. Must produce valid IPA only.
- **MeloTTS**: has its own `cleaned_text_to_sequence` that hard-fails on unknown symbols.
- **F5-TTS / XTTS-v2 / Bark / IndexTTS-2**: not affected — they don't take phonemes.

**Recommendation**: option (a) + (b) — ship a large lexicon (covers 99 % of normal text) and a small ByT5-tiny model for the remaining 1 %. The model runs through our own runtime, so no extra deps.

### 8. Neural G2P Models We Could Run In-Engine

Small transformer G2P models that fit our "run it ourselves" criterion:

| Model | Params | Languages | Source | License |
|---|---|---|---|---|
| **PhonoGlyphe** | ~6 M | English only | [Picus303/PhonoGlyphe](https://github.com/Picus303/PhonoGlyphe) | Apache (intended misaki fallback) |
| **CharsiuG2P byT5-tiny** (8 layers) | ~30 M | 100 languages | [lingjzhu/CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P) | Apache |
| **CharsiuG2P byT5-small** | ~300 M | 100 languages | same | Apache |
| **DeepPhonemizer transformer** | ~10 M | English | [as3eem/DeepPhonemizer](https://github.com/as3eem/DeepPhonemizer) | MIT |
| **g2p-en LSTM** | ~3 M | English | [Kyubyong/g2p](https://github.com/Kyubyong/g2p) | Apache |

All are small enough that running them through HartsyInference itself is fine — we already have transformer kernels for text encoders. Recommended: **PhonoGlyphe (6M, English)** for production, **CharsiuG2P byT5-tiny (30M, 100 langs)** as the universal fallback.

ByT5 is convenient because it's **byte-level** — no tokenizer to ship, the input IS UTF-8 bytes. This sidesteps the SentencePiece-vs-pure-C# headache.

### 9. Per-TTS-Model G2P Requirements

| Model | G2P required? | What it expects | How we satisfy it |
|---|---|---|---|
| **Kokoro 82M** | YES | misaki-style IPA, mapped to its 178-token vocab (pad=0, then punct/letters/IPA, see KOKORO_ARCHITECTURE.md §"Phoneme Set and Encoding") | CMUDict + ARPABET→IPA + heteronym table + PhonoGlyphe fallback |
| **StyleTTS 2 (LJSpeech)** | YES | espeak-style IPA, ~178 symbols (their `text_utils.py`) | Same pipeline as Kokoro; the IPA inventory overlaps ~95 % |
| **MeloTTS** | YES | per-language phoneme set + stress + BERT-base-multilingual embeddings | Per-language G2P + ship `bert-base-multilingual-uncased` as a side model; existing `mini-bart-g2p` checkpoint for English |
| **GPT-SoVITS v2** | YES | Mixed: CMUDict ARPABET (NOT IPA) for English, pinyin for Chinese, romaji-without-accents for Japanese, jamo for Korean, plus punctuation; mapped via their `symbols.py` | Reuse CMUDict directly (no IPA conversion!); pinyin pipeline as above; romaji is trivial; jamo decomposition |
| **F5-TTS** | **NO** | Raw character sequence padded with filler tokens to mel-frame length | Just UTF-8 normalize the text; **no G2P needed** |
| **XTTS-v2 (Coqui)** | **NO** | Custom 6,681-token BPE; CJK is romanized first | Ship the BPE tokenizer JSON (already in their HF repo) |
| **Bark** | **NO** | Hubert-style semantic tokens from text via their built-in model | No G2P at all |
| **CosyVoice 1/2/3** | OPTIONAL | Raw text by default; can take pinyin or CMU phonemes for "pronunciation inpainting" | Default = no G2P. Optional pinyin/CMU pass for control |
| **IndexTTS-2** | OPTIONAL | Hybrid: characters + pinyin (Chinese), characters (English) for controllability | Optional pinyin via jieba.NET pipeline |

**Conclusion**: G2P is only strictly required for Kokoro, StyleTTS 2, MeloTTS, and GPT-SoVITS (which uniquely wants ARPABET, not IPA). The shiny new wave (F5-TTS, XTTS-v2, Bark, CosyVoice, IndexTTS-2) needs none. If we ship those first, we can defer G2P entirely.

### 10. Recommendation — "If We Had to Ship This Monday"

**Phase 1 — Minimal, no G2P at all** (≤ 1 week):

Ship F5-TTS and/or XTTS-v2 as our first TTS models. Both are character-level. They cover the most common user request ("read this sentence in this voice") with reference quality, and require zero phonemization work.

**Phase 2 — English G2P (for Kokoro)** (~2 weeks):

1. Pack CMUDict (134k entries) into a compact binary lookup (`HartsyInference.Audio.Phonemes.CmuDict`).
2. Hard-code the ARPABET → IPA table (single switch statement, ~40 cases).
3. Port misaki's English heteronym table (~1000 entries, Apache-licensed) into a `(word, POS) → IPA` lookup.
4. Port a pure-C# averaged-perceptron POS tagger (model ~3 MB).
5. Implement English number expansion (port `num2words.en`, ~600 LOC).
6. Convert PhonoGlyphe checkpoint to `.safetensors`, wire it as the OOV fallback through HartsyInference's existing transformer runtime.
7. Wire up to Kokoro and StyleTTS 2's expected vocabularies.

Acceptance test: phonemize 10,000 sentences from LibriTTS, compare to misaki+espeak output, target ≥98 % phoneme-level agreement (excluding heteronym corner cases).

**Phase 3 — Mandarin** (~1 week):

1. Take jieba.NET from NuGet.
2. Pack a CC-CEDICT char→pinyin map.
3. Hard-code pinyin → IPA table (~400 syllables).
4. Tone-3 sandhi rule.
5. Wire to GPT-SoVITS, CosyVoice, IndexTTS-2.

**Phase 4 — Korean** (~3 days):

Pure rule-based jamo decomposition + 12 phonological rules. Trivial.

**Phase 5 — Romance + Germanic** (~2-3 weeks):

Port epitran's `spa-Latn`, `ita-Latn`, `por-Latn`, `fra-Latn`, `deu-Latn` rule tables to a C# rule engine. Add gruut's lexicon overrides.

**Punt-list** (acknowledge we can't ship these on day one):

- **Japanese** — needs MeCab + UniDic port; deferred to its own multi-week project. Until then, users get F5-TTS / XTTS-v2 for Japanese.
- **Russian** — needs stress dictionary curation; defer.
- **Arabic, Persian, Thai, Vietnamese, Hebrew** — defer; route through character-level models if possible.
- **Hindi** and other Devanāgarī scripts — easy but lower demand; defer to Phase 6.

## Key Numbers / Constants

- **CMUDict entries**: 134,000+
- **CMUDict file size**: ~3.5 MB raw text, ~1.5 MB packed
- **CMU ARPABET phoneme count**: exactly 39 (15 vowels + 24 consonants)
- **misaki English phoneme count**: 49 (41 shared + 4 US-only + 4 UK-only)
- **Kokoro vocabulary size**: exactly 178 tokens, indices 0-177, with `0` as pad
- **XTTS-v2 BPE vocabulary**: exactly 6,681 tokens (NOT phoneme-based)
- **Standard heteronym table**: ~1,000 English entries
- **ARPABET stress markers**: digits `0` (unstressed), `1` (primary), `2` (secondary), appended to vowel symbol
- **IPA stress markers**: `ˈ` (U+02C8) primary, `ˌ` (U+02CC) secondary, prefix before the stressed syllable
- **IPA length marker**: `ː` (U+02D0)
- **IPA Latin small letter script g**: `ɡ` (U+0261) — NOT ASCII g (U+0067)
- **NLTK averaged-perceptron POS tagger model size**: ~3 MB
- **PhonoGlyphe G2P fallback model**: ~6 M params
- **CharsiuG2P byT5-tiny (8 layers, 100 languages)**: ~30 M params
- **OpenPhonemizer bundled dictionary**: ~100 MB (too big for us; we'd use ~5-10 MB equivalent)
- **Mandarin lexical tones**: 4 + neutral = 5 categories; Chao pitch numbers 55, 35, 214, 51, (neutral) for T1-T4 + T0
- **Hangul Unicode range**: U+AC00 to U+D7A3 (11,172 syllables), decomposable arithmetically into (19 initial × 21 medial × 28 final) jamo
- **CC-CEDICT raw size**: ~14 MB (compressed: ~3 MB)
- **MeCab Naist-JDIC for Japanese**: ~50 MB; UniDic CSJ: ~750 MB
- **espeak-ng languages supported**: ~100 (reference benchmark)
- **gruut languages supported**: 13 (ar, cs-cz, de-de, en-us, en-gb, es-es, fa, it-it, lb-lb, nl, pt, ru-ru, sw)
- **epitran languages supported**: 61

## Data Layouts / Formats

### CMUDict text format

```
;;; comment line, starts with three semicolons
WORD  PH1 PH2 PH3 ...        # two spaces between word and phones
WORD(2)  PH1 PH2 ...         # alternate pronunciation, suffix (2), (3), ...
WORD(1)                       # is NOT used; the primary entry has no parenthetical
```

- Words: uppercased ASCII; can contain apostrophe (`DON'T`).
- Phones: space-separated ARPABET; vowels carry a final digit 0/1/2.
- Lines starting with `;;;` are comments (one big license/header at the top).

### CMUDict packed runtime layout (recommended for us)

- Sort entries alphabetically; emit two arrays:
  - `string[] words` — sorted headwords (variants encoded as `WORD\x00WORD(2)\x00...`)
  - `byte[][] phones` — one byte per phoneme, indexed into the 39-symbol ARPABET alphabet
- Binary search on the words array; load both arrays from a single packed `.bin` resource.
- 134k entries × (10 bytes word avg + 10 bytes phones avg) ≈ 2.7 MB unpacked, ~1.2 MB LZ4.

### ARPABET → IPA table (single struct)

```csharp
public readonly struct ArpabetEntry
{
    public readonly string ArpaSymbol;     // e.g. "AH"
    public readonly string IpaStressed;    // e.g. "ʌ"
    public readonly string IpaUnstressed;  // e.g. "ə" — same as Stressed for consonants
}
```

39 rows. Compile-time constant. The two-form representation handles the AH→ʌ/ə and ER→ɝ/ɚ stress-dependent cases.

### Kokoro vocab.json layout

Already documented in [KOKORO_ARCHITECTURE.md](KOKORO_ARCHITECTURE.md). Single JSON dict: `{"\x00": 0, "a": 16, "ˈ": 156, ...}`. 178 entries. Our IPA output gets mapped char-by-char through this dict; unknown chars are dropped.

### Per-language phoneme rule table format (for rule-based languages)

Recommended TSV format (matches epitran):

```
# spa-Latn ruleset
# format: pattern <tab> replacement
# applied top-to-bottom; longest pattern wins per position
ch	tʃ
ll	ʎ
ñ	ɲ
qu	k
gu(?=e|i)	ɡ
...
```

Compile these into a single trie at startup; runtime apply with longest-match-first.

## Algorithm Steps

### English phonemize (text → IPA string)

```
function phonemize_en(text):
    sentences = sentence_split(text)
    out_ipa = []
    for sent in sentences:
        words = word_tokenize(sent)              # whitespace + Penn-Treebank style
        pos_tags = perceptron_tagger.tag(words)
        for (word, pos) in pos_tags:
            normalized = lowercase(strip_punct(word))
            if normalized is empty:
                out_ipa.append(map_punctuation(word))
                continue
            if normalized.is_numeric():
                normalized = num2words_en(normalized)
                # recurse on the expanded number
            # 1. heteronym table
            if (normalized, pos) in heteronyms:
                ipa = heteronyms[(normalized, pos)]
            # 2. CMUDict primary entry
            elif normalized in cmudict:
                arpa = cmudict[normalized]        # first variant
                ipa = arpabet_to_ipa(arpa)
            # 3. neural fallback
            else:
                arpa_or_ipa = phonoglyphe.predict(normalized)
                ipa = ensure_ipa(arpa_or_ipa)
            out_ipa.append(ipa)
        out_ipa.append(' ')                       # word separator
    return ''.join(out_ipa)
```

### ARPABET → IPA (one phoneme)

```
function arpabet_to_ipa(arpa_symbol):
    stress_digit = last_char_if_digit(arpa_symbol)
    base = arpa_symbol[:-1] if stress_digit else arpa_symbol
    if base in vowels:
        if base == "AH":
            ipa = "ʌ" if stress_digit in {"1","2"} else "ə"
        elif base == "ER":
            ipa = "ɝ" if stress_digit in {"1","2"} else "ɚ"
        else:
            ipa = vowel_table[base]
    else:
        ipa = consonant_table[base]
        stress_digit = None                       # consonants don't carry stress
    if stress_digit == "1":
        return "ˈ" + ipa                          # NB: prepended; syllabification implicit
    elif stress_digit == "2":
        return "ˌ" + ipa
    else:
        return ipa
```

For full syllable-aware stress placement (`ˈ` before the syllable onset, not before the vowel itself), apply a maximal-onset post-process. Most current TTS models tolerate vowel-anchored stress, so the simple version is acceptable as v1.

### Hangul → IPA decomposition

```
function hangul_to_ipa(char):
    code = ord(char)
    if code < 0xAC00 or code > 0xD7A3:
        return char                               # punctuation / Latin pass-through
    s = code - 0xAC00
    initial_idx = s / 588                         # integer division
    medial_idx = (s / 28) % 21
    final_idx = s % 28
    initial = INITIAL_IPA[initial_idx]            # 19 entries, e.g. "k", "n", ...
    medial = MEDIAL_IPA[medial_idx]               # 21 entries
    final = FINAL_IPA[final_idx]                  # 28 entries (index 0 = empty string)
    return initial + medial + final

function phonemize_ko(text):
    syllables = [hangul_to_ipa(c) for c in text]
    return apply_phonological_rules(syllables)    # nasal assim, palatalization, etc.
```

### Mandarin phonemize

```
function phonemize_zh(text):
    words = jieba.cut(text)                       # via jieba.NET
    out = []
    for word in words:
        if word in cedict:
            pinyin_seq = cedict[word]             # e.g. "你好" -> ["ni3", "hao3"]
        else:
            pinyin_seq = [char_to_pinyin[c] for c in word]
        out.extend(pinyin_seq)
    out = apply_tone_sandhi(out)                  # T3+T3 -> T2+T3, etc.
    ipa = [pinyin_to_ipa(p) for p in out]
    return ' '.join(ipa)
```

## Reference Implementations

- **espeak-ng** — https://github.com/espeak-ng/espeak-ng — GPL; consult source for the canonical correct output but do NOT copy code or data.
- **phonemizer** (espeak wrapper) — https://github.com/bootphon/phonemizer — most TTS training pipelines use this.
- **misaki** (Kokoro G2P) — https://github.com/hexgrad/misaki — Apache 2.0; misaki's English dictionaries (`misaki/data/us_gold.json`, `us_silver.json`) and heteronym table are directly reusable.
- **gruut** — https://github.com/rhasspy/gruut — MIT; the closest analogue to what we want to build in C#. Use it as the design template.
- **gruut-ipa** — https://github.com/rhasspy/gruut-ipa — MIT; IPA utilities (phoneme normalization, accent/diacritic handling).
- **epitran** — https://github.com/dmort27/epitran — MIT; rule tables for 61 languages. The TSV rule format is portable.
- **CMUdict** — https://github.com/cmusphinx/cmudict — public domain.
- **g2p-en** — https://github.com/Kyubyong/g2p — Apache 2.0; reference for the CMUDict + heteronym + NLTK pipeline.
- **PhonoGlyphe** — https://github.com/Picus303/PhonoGlyphe — small (6M) transformer English G2P, intended as misaki fallback.
- **CharsiuG2P** — https://github.com/lingjzhu/CharsiuG2P — multilingual ByT5; checkpoints on Hugging Face under `charsiu/g2p_multilingual_byT5_*`.
- **OpenPhonemizer** — https://github.com/NeuralVox/OpenPhonemizer — DeepPhonemizer-based; unmaintained but instructive.
- **Phonikud** — https://github.com/thewh1teagle/phonikud — Hebrew; relevant for the FST-runtime idea.
- **MisakiSwift** — https://github.com/mlalma/MisakiSwift — Swift port of misaki; useful as a precedent for how to do this in a non-Python language.
- **misaki-rs** — https://lib.rs/crates/misaki-rs — Rust port of misaki; useful precedent.
- **NMeCab** — https://github.com/komutan/NMeCab — pure-C# MeCab port (for Japanese, if and when we tackle it).
- **jieba.NET** — https://github.com/anderscui/jieba.NET — pure-C# jieba port (for Mandarin).
- **OpenCCNET** — https://www.nuget.org/packages/OpenCCNET — pure-C# Traditional↔Simplified Chinese conversion (useful preproc for zh-Hant inputs).

## Differences Between Implementations

- **espeak-ng vs misaki vs gruut** all produce slightly different IPA strings for the same input. The differences are systematic: espeak uses `ɹ` for English /r/, misaki uses `ɹ`, but some older models trained on espeak earlier output used `r`. Stress placement (`ˈ` exactly where) varies. Models are typically robust to ±1 character variation in IPA, but only if they were trained that way.
- **misaki vs CMUDict**: misaki returns 49 IPA symbols including custom diacritics (`ᵊ`, `ᵻ`); a pure CMUDict→IPA conversion does NOT use these. Kokoro was trained on misaki output specifically, so for Kokoro we must match misaki's exact IPA inventory or live with degraded quality.
- **CMUDict vs gruut English lexicon**: gruut's en-us lexicon is derived from espeak output (re-licensed), so it differs from CMUDict on ~5 % of common words. CMUDict tends to be more conservative on schwa reduction.
- **GPT-SoVITS English** uses CMUDict directly with ARPABET symbols (NOT IPA). Its `symbols.py` has fixed token IDs for ARPABET `AA0 AA1 AA2 ... ZH`. So for GPT-SoVITS we skip ARPABET→IPA conversion entirely.
- **MeloTTS English** uses a different phoneme set again (with stress encoded as separate `tone` tokens). Their `english.py` and `english_bert.py` define the exact set. We'd need a MeloTTS-specific output path.
- **OpenJTalk vs MeCab+UniDic for Japanese**: different dictionaries give different readings for kanji. pyopenjtalk uses Naist-JDIC. Anything else gives slightly different phoneme sequences.

## Open Questions

- Do we want a **single unified IPA inventory** internally, with per-model post-processing to map to that model's expected vocab? Or do we keep per-model G2P paths? (Recommendation: unified misaki-49 inventory + per-model mapping table.)
- For OOV neural G2P, do we host **PhonoGlyphe (6M, English-only)** or **CharsiuG2P byT5-tiny (30M, 100 languages)** as our default? (Recommendation: PhonoGlyphe for English-first ship, CharsiuG2P later for breadth.)
- Should we ship a **stress-aware syllabifier** (~100 LOC, maximal-onset rule) so that `ˈ` is placed before the syllable onset instead of right before the vowel? Some TTS models care, some don't.
- Should we attempt to ship Japanese G2P at all, or **explicitly mark Kokoro-Japanese as out-of-scope until NMeCab + Naist-JDIC port is done**?
- License audit needed: misaki's `us_gold.json` is Apache 2.0 — confirm we can redistribute. epitran's rule tables are Apache — confirm. gruut's lexicons are MIT but partially espeak-derived — confirm.
- Do we attach a **TTS-time text normalizer** (handling "$5.99" → "five dollars and ninety-nine cents", dates, abbreviations, URLs) before the G2P stage? This is technically separate from G2P but every real TTS shipping product has one. (Recommendation: yes, ship a small one — port num2words + a handful of regex rules.)
- For Mandarin polyphone disambiguation beyond what jieba+CC-CEDICT covers, is a small BERT-based disambiguator worth it? Or accept ~1 % wrong tones?

## Implementation Notes for HartsyInference

### Package layout

Per the file structure rules (CLAUDE.md), this belongs in its own assembly inside the audio package:

```
src/HartsyInference.Audio.Phonemes/
    CmuDictionary.cs              # public CMUDict loader/lookup
    ArpabetToIpa.cs               # 39-row static conversion table
    Heteronyms.cs                 # (word, POS) -> IPA table
    PosTagger.cs                  # averaged-perceptron POS tagger
    EnglishPhonemizer.cs          # full English pipeline (entry point)
    MandarinPhonemizer.cs         # entry point; uses jieba.NET + CC-CEDICT
    KoreanPhonemizer.cs           # entry point; pure rule-based
    RulePhonemizer.cs             # generic epitran-style rule engine
    Rules/                        # per-language TSV resources
        spa-Latn.tsv
        ita-Latn.tsv
        deu-Latn.tsv
        fra-Latn.tsv
        ...
    Data/                         # embedded resources (packed binaries)
        cmudict.bin               # ~1.2 MB
        heteronyms.bin
        pos_tagger.bin            # ~3 MB
        cc_cedict.bin             # ~3 MB
    OovFallback/
        IOovFallback.cs           # interface so users can plug in
        PhonoGlypheFallback.cs    # default: runs PhonoGlyphe through HartsyInference
        CharsiuG2PFallback.cs     # alternative: multilingual ByT5
```

### Hot-path discipline (per CLAUDE.md)

- CMUDict and heteronym lookups: pre-loaded into `NativeMemory.AlignedAlloc` blobs; use `Span<byte>` + binary search. No string allocation per lookup.
- ARPABET→IPA conversion: stack-allocated builder (`Span<char>` or `ValueStringBuilder` pattern), no `StringBuilder`. The output IPA string is allocated once at the end per phonemize call.
- POS tagger: weights in unmanaged memory, dot product via `Tensor<float>` SIMD (we already have it).
- For Mandarin, jieba.NET allocates managed strings — wrap it in an adapter that returns spans where possible. May need a small re-port to avoid the GC pressure.

### Validation strategy

For each language, hold out a 10k-sentence reference set, phonemize with the gold reference (misaki + espeak via Python subprocess at test time only — not at runtime), and require ≥98 % token-level agreement (where "token" = one IPA character) for in-vocab text, ≥92 % for OOV text. Document tolerances per language in `tests/HartsyInference.Audio.Tests/Phonemes/`.

### Encoding / Unicode

- Always emit IPA as **NFC-normalized UTF-16 strings** (C# `string` default). Some IPA characters have composed and decomposed forms — pick NFC consistently to avoid token-lookup misses.
- Be explicit about U+0261 (`ɡ`) vs U+0067 (ASCII `g`). Kokoro's vocab uses U+0261; ARPABET converters that emit ASCII `g` will silently fail.
- Stress marks `ˈ`/`ˌ` and length `ː` are all single Unicode code points (not combining); no surrogate pair issues.

### Day-one minimum viable subset

To ship Kokoro with English voices on Monday:

1. `HartsyInference.Audio.Phonemes.CmuDictionary` (loader + lookup) — 1 day
2. `ArpabetToIpa` static table + converter — 0.5 day
3. `Heteronyms` table (port misaki's gold heteronyms) — 0.5 day
4. `PosTagger` (averaged perceptron, port from existing C# implementations) — 1 day
5. `EnglishPhonemizer` (orchestrator) — 0.5 day
6. `KokoroVocabMapper` (IPA string → 178-token int sequence) — 0.5 day
7. Tests against 100 hand-picked sentences — 1 day

That's ~5 days of focused work for production-quality English. OOV neural fallback is a nice-to-have that can ship in v1.1.

Sources:
- [hexgrad/misaki](https://github.com/hexgrad/misaki)
- [misaki EN_PHONES.md](https://github.com/hexgrad/misaki/blob/main/EN_PHONES.md)
- [espeak-ng dictionary docs](https://github.com/espeak-ng/espeak-ng/blob/master/docs/dictionary.md)
- [CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict)
- [ARPABET Wikipedia](https://en.wikipedia.org/wiki/ARPABET)
- [rhasspy/gruut](https://github.com/rhasspy/gruut)
- [rhasspy/gruut-ipa](https://github.com/rhasspy/gruut-ipa)
- [Kyubyong/g2p (g2p-en)](https://github.com/Kyubyong/g2p)
- [Picus303/PhonoGlyphe](https://github.com/Picus303/PhonoGlyphe)
- [lingjzhu/CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P)
- [NeuralVox/OpenPhonemizer](https://github.com/NeuralVox/OpenPhonemizer)
- [thewh1teagle/phonikud](https://github.com/thewh1teagle/phonikud)
- [dmort27/epitran](https://github.com/dmort27/epitran)
- [r9y9/pyopenjtalk](https://github.com/r9y9/pyopenjtalk)
- [komutan/NMeCab](https://github.com/komutan/NMeCab)
- [anderscui/jieba.NET](https://github.com/anderscui/jieba.NET)
- [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)
- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS)
- [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2)
- [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- [ByT5 G2P paper](https://arxiv.org/abs/2204.03067)
- [Phonikud paper](https://arxiv.org/abs/2506.12311)
