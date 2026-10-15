# ZipVoice against upstream, 2026-10-15

Upstream is k2-fsa/ZipVoice at 2f7326f (`infer_zipvoice.generate_sentence`, the default non-raw path) with
piper_phonemize 1.5.0, torch 2.9.1, lhotse 1.33.0, the engine's repacked checkpoint (k2-fsa/ZipVoice@4ed45fb6) and its
vocoder (`lucasnewman/vocos-mel-24khz`). Sampler: 16 steps, guidance 1.0, speed 1.0, t_shift 0.5, `fix_random_seed(seed)`
before each row, run under `torch.inference_mode()` on the 3060. The engine ran through the live SwarmUI
(`ProcessAudio`, provider `zipvoice_tts`, model `base`) on a local build of PR #426 merged with main, with the sweep's
reference clip and transcript. Both sides were transcribed through `ProcessSTT` with Whisper `medium`.

## Results

Mean WER over Whisper-normalized tokens (`ok` and `okay` equal), n = takes. "Engine before" is the build the audit
measured (seeds 0-3). Upstream ran seeds 0-11; the engine ran seeds 0-7 on the short rows and fewer on the long ones,
since it takes minutes per row (see Open).

| row | text | engine before | engine after | upstream | mean duration, engine after / upstream |
|---|---|---|---|---|---|
| 01 | `Yes.` | 0.500 (n=4) | 1.000 (n=8) | 0.917 (n=12) | 0.19 s / 0.20 s |
| 02 | `Okay, see you.` | 0.000 (n=4) | 0.000 (n=8) | 0.000 (n=12) | 0.83 s / 0.84 s |
| 03 | eight words | 0.000 (n=4) | 0.000 (n=8) | 0.000 (n=12) | 2.71 s / 2.71 s |
| 04 | twenty words | 0.067 (n=4) | 0.000 (n=4) | 0.000 (n=12) | 7.81 s / 7.82 s |
| 05 | paragraph | 0.091 (n=4) | 0.000 (n=3) | 0.000 (n=12) | 21.75 s / 21.78 s |
| 06 | numbers and dates (diagnostic) | not run | 0.442 (n=4) | 0.436 (n=12) | 7.38 s / 7.64 s |
| 07 | punctuation (diagnostic) | not run | 0.429 (n=4) | 0.310 (n=12) | 2.88 s / 3.05 s |

- Row 01 is a 0.2 s output; Whisper hears `Yeah` from both sides (8 of 8 engine takes, 11 of 12 upstream, one `Yes.`).
- Rows 02-05 transcribe word-perfect on every take, engine and upstream.
- Paired gate durations (same seed): median absolute relative difference 3.5%, largest 19% (row 01).
- Row 06: both sides misread `$99.95` (`99.95s`, `99 Y95s`). The engine reads `PM` as `pəm` where piper spells the
  letters, so `4:30 PM` comes out as `4.30 pm`, `430 Pum` or `4.30pm`.
- Row 07: the only difference is the last word. The engine's espeak reads `ok` as `ˈɑːk` (heard as `ox`, `ock`) where
  piper says `oʊkˈeɪ`.



## What was wrong

The engine differed from upstream at every layer between the text and the waveform; the table above is after all of
them were fixed.

| layer | before | upstream |
|---|---|---|
| espeak voice | British `en` (`əʊ`, non-rhotic) | `en-us` |
| clauses and marks | regex over the engine's punctuation-preserving output | piper_phonemize: per clause, bare `. ? !`, `, ; :` plus a space, ellipsis and spaced dash rules |
| symbols | grapheme clusters (`n̩`, `ɑ̃` dropped) | code points after NFD |
| text normalization | none (espeak read digits itself) | `EnglishTextNormalizer` |
| reference | RMS-lift, then a windowed-RMS trim | pydub `remove_silence`, RMS-lift, frames cut to lhotse's count, log floor 1e-7 |
| target length | double arithmetic | float32 (differs on 0.75% of lengths) |
| long text | one sample | chunks that fill 25 s with the reference, own seed each, 0.1 s cross-fade, edge trim |

## Front-end evidence

- Tokens on a 137-line corpus (the GPT-SoVITS frontend corpus plus ZipVoice extras): word agreement 57.6% (British) to
  93.7% with the cached espeak data and 95.5% with piper's own data. A prose sample of 75 GPL sentences: 93.5% and 95.1%.
- The remaining differences are in the espeak port, not this front end: all-caps acronyms (`AI`, `PC`, `GNU`),
  vowel-less abbreviations (`km`, `pm`), camel-case brands, `+` and the degree sign, intra-word dots, `con-` reduction,
  a final unstressed `y`, a clause-final `one`.
- Clause structure: 3,000 random strings of spaced marks, quotes, parentheses, dashes and ellipses between lexicon
  words are identical to piper_phonemize.
- Prompt layer on the sweep's reference clip: 97,078 samples after `remove_silence` and the RMS lift (upstream 97,078,
  RMS 0.05038108676671982 on both), 379 mel frames (upstream 379), mel relative L2 1.0e-6 (max abs 1.1e-4).
- Normalizer: 4,000 random strings of numbers, money, fractions, percentages, ordinals and abbreviations are identical
  to upstream's `EnglishTextNormalizer`.
- pydub silence removal: 18 synthetic signals give identical SHA-1 on the output samples, including the near-threshold
  integer-RMS case and the 11.025 kHz trailing silence. Cross-fade, chunker (112 budgets), normalizer (133 lines) and
  target length (400 cases) match.

## Open

- Generation takes about 8 times real time on the 4090 (180 s for the 21.7 s paragraph); upstream takes about 0.5 times
  real time on the 3060 (10.9 s for the same row). In one sample during a generation SwarmUI used about one core and
  the 4090 was 17% utilized. Unchanged by this work.
- Chinese needs jieba and pypinyin and is not ported.
