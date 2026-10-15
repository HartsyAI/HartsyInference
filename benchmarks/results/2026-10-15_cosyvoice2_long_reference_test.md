# CosyVoice2 voice cloning with long references, 2026-10-15

Engine: live Swarm, `cosyvoice_tts:2-0.5b`, alpha.369 integration build. Oracle: Whisper medium through Swarm `ProcessSTT`, Whisper English normalizer, word-level alignment. Upstream: FunAudioLLM/CosyVoice at 074ca6d, `CosyVoice2.inference_zero_shot`, CPU, the engine's repacked weights loaded through safetensors. Cells show output seconds / word error rate.

### Pass 1: eight references (4.4 to 30 s), four reporter-style texts, seed 0

| reference | audio s | transcript chars | narr s0 | info s0 | dial s0 | emot s0 |
|---|---|---|---|---|---|---|
| sweep_ref | 4.4 | 64 | 12.4 s / 0.05 | 12.6 s / 0.00 | 7.0 s / 0.00 | 9.0 s / 0.00 |
| read_speech_c | 8.04 | 103 | 14.4 s / 0.05 | 13.5 s / 0.00 | 8.8 s / 0.00 | 10.5 s / 0.00 |
| read_speech_d | 9.08 | 124 | 15.0 s / 0.00 | 16.0 s / 0.00 | 8.5 s / 0.00 | 8.8 s / 0.00 |
| jfk | 11.0 | 108 | 13.3 s / 0.03 | 17.0 s / 0.03 | 10.2 s / 0.00 | 8.9 s / 0.00 |
| read_speech_b | 12.0 | 151 | 13.7 s / 0.00 | 17.5 s / 0.03 | 8.0 s / 0.00 | 7.9 s / 0.00 |
| read_speech_a | 17.32 | 235 | 13.4 s / 0.05 | 14.7 s / 0.06 | 7.9 s / 0.00 | 7.6 s / 0.00 |
| conversational_a | 30.0 | 513 | 4.0 s / 1.00 **FAIL** | 3.6 s / 0.97 **FAIL** | 6.0 s / 0.00 | 5.2 s / 0.04 |
| conversational_b | 30.0 | 489 | 4.2 s / 1.00 **FAIL** | 4.3 s / 1.00 **FAIL** | 2.8 s / 0.59 **FAIL** | 3.5 s / 0.39 **FAIL** |

6 of 32 takes fail (word error rate above 0.25).

### Pass 3: prefix cuts of the two 30 s clips (length sweep, same speaker)

| reference | audio s | transcript chars | narr s0 | info s0 |
|---|---|---|---|---|
| conv_a_7.5s | 7.5 | 157 | 10.1 s / 0.00 | 11.4 s / 0.03 |
| conv_a_11.1s | 11.1 | 229 | 10.9 s / 0.03 | 11.9 s / 0.06 |
| conv_a_15.0s | 15.0 | 290 | 9.4 s / 0.05 | 11.8 s / 0.00 |
| conv_a_20.8s | 20.8 | 339 | 8.2 s / 0.00 | 8.4 s / 0.14 |
| conv_a_24.1s | 24.1 | 407 | 8.7 s / 0.00 | 7.6 s / 0.51 **FAIL** |
| conv_b_7.5s | 7.5 | 93 | 11.6 s / 0.00 | 13.5 s / 0.00 |
| conv_b_9.5s | 9.5 | 130 | 10.3 s / 0.00 | 13.5 s / 0.03 |
| conv_b_13.9s | 13.9 | 183 | 11.8 s / 0.03 | 14.2 s / 0.09 |
| conv_b_19.1s | 19.1 | 289 | 10.3 s / 0.00 | 11.2 s / 0.17 |
| conv_b_26.4s | 26.4 | 415 | 3.5 s / 0.95 **FAIL** | 3.7 s / 0.91 **FAIL** |

3 of 20 takes fail (word error rate above 0.25).

### Pass 2: sample-rate variants (30 s clips at 24 and 16 kHz; 8 and 9 s clips upsampled to 44.1 kHz)

| reference | audio s | transcript chars | narr s0 | narr s1 | info s0 | info s1 |
|---|---|---|---|---|---|---|
| conversational_a_at24k | 30.0 | 513 | 8.6 s / 0.97 **FAIL** | 3.6 s / 0.97 **FAIL** | 4.2 s / 0.97 **FAIL** | 5.0 s / 0.97 **FAIL** |
| conversational_a_at16k | 30.0 | 513 | 3.4 s / 1.00 **FAIL** | 4.7 s / 0.97 **FAIL** | 5.1 s / 0.97 **FAIL** | 4.3 s / 1.00 **FAIL** |
| conversational_b_at24k | 30.0 | 489 | 5.9 s / 0.95 **FAIL** | 3.6 s / 0.97 **FAIL** | 12.8 s / 0.94 **FAIL** | 6.6 s / 0.97 **FAIL** |
| conversational_b_at16k | 30.0 | 489 | 3.7 s / 0.92 **FAIL** | 5.0 s / 0.97 **FAIL** | 14.3 s / 1.00 **FAIL** | 5.5 s / 0.97 **FAIL** |
| read_speech_c_at44k | 8.04 | 103 | 13.7 s / 0.00 | 12.8 s / 0.00 | 14.9 s / 0.00 | 14.8 s / 0.00 |
| read_speech_d_at44k | 9.08 | 124 | 14.4 s / 0.00 | 14.8 s / 0.00 | 16.2 s / 0.00 | 14.2 s / 0.00 |

16 of 24 takes fail (word error rate above 0.25).

### Pass 4: the 30 s clips with long texts (26 and 77 words)

| reference | audio s | transcript chars | gate04 s0 | gate04 s1 | gate05 s0 | gate05 s1 |
|---|---|---|---|---|---|---|
| conversational_a | 30.0 | 513 | 6.1 s / 0.00 | 6.9 s / 0.00 | 10.2 s / 0.95 **FAIL** | 10.0 s / 0.92 **FAIL** |
| conversational_b | 30.0 | 489 | 4.6 s / 0.39 **FAIL** | 5.0 s / 0.42 **FAIL** | 10.9 s / 0.93 **FAIL** | 7.8 s / 0.95 **FAIL** |

6 of 8 takes fail (word error rate above 0.25).

### Pass 5: the 30 s clips with the verbatim large-v3 transcript

| reference | audio s | transcript chars | narr s0 | narr s1 | info s0 | info s1 |
|---|---|---|---|---|---|---|
| conversational_a_v3text | 30.0 | 495 | 3.6 s / 1.00 **FAIL** | 3.4 s / 0.97 **FAIL** | 3.9 s / 0.97 **FAIL** | 3.8 s / 0.97 **FAIL** |
| conversational_b_v3text | 30.0 | 487 | 6.1 s / 0.95 **FAIL** | 3.8 s / 0.97 **FAIL** | 6.2 s / 0.94 **FAIL** | 3.4 s / 0.97 **FAIL** |

8 of 8 takes fail (word error rate above 0.25).

### Official Python against the engine, same references, texts and seed 0

| reference | text | engine s / WER / words dropped | official Python s / WER / words dropped |
|---|---|---|---|
| conversational_a | dial | 6.0 / 0.00 / 0 | 5.2 / 0.04 / 1 |
| conversational_a | emot | 5.2 / 0.04 / 0 | 5.6 / 0.00 / 0 |
| conversational_a | info | 3.6 / 0.97 / 26 | 3.9 / 1.00 / 29 |
| conversational_a | narr | 4.0 / 1.00 / 34 | 3.8 / 1.00 / 28 |
| conversational_b | dial | 2.8 / 0.59 / 13 | 2.9 / 0.59 / 13 |
| conversational_b | emot | 3.5 / 0.39 / 10 | 3.8 / 0.39 / 10 |
| conversational_b | info | 4.3 / 1.00 / 29 | 3.5 / 0.97 / 28 |
| conversational_b | narr | 4.2 / 1.00 / 31 | 3.4 / 0.95 / 29 |
| conv_a_20.8s | info | 8.4 / 0.14 / 5 | 7.8 / 0.26 / 8 |
| conv_a_20.8s | narr | 8.2 / 0.00 / 0 | 8.2 / 0.00 / 0 |
| conv_a_24.1s | info | 7.6 / 0.51 / 18 | 7.6 / 0.57 / 16 |
| conv_a_24.1s | narr | 8.7 / 0.00 / 0 | 9.1 / 0.19 / 0 |

### Official Python cross-lingual mode on the failing 30 s clips (no prompt text or prompt speech in the LM)

| reference | text | zero-shot s / WER | cross-lingual s / WER |
|---|---|---|---|
| conversational_a | info | 3.9 / 1.00 | 14.7 / 0.00 |
| conversational_a | narr | 3.8 / 1.00 | 13.8 / 0.00 |
| conversational_b | info | 3.5 / 0.97 | 14.7 / 0.00 |
| conversational_b | narr | 3.4 / 0.95 | 13.8 / 0.00 |
