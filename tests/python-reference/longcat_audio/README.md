# LongCat audio features: reference dumpers

References for phase B0 (`LongCatAudioFeatures`, the Whisper all-hidden-states overload, the 128-bin mel, the Avatar 1.0 wav2vec2
path). Reference code: `meituan-longcat/LongCat-Video` @ `58eb336edc5c8c63389838727e40b8c18434e54e`. Run on WSL or a pod, never in
the Windows venv. The dumps are raw little-endian F32 plus `meta.json`, written to `tests/python-reference/longcat_audio_reference_tensors/`
(gitignored); point `HARTSY_LONGCAT_AUDIO_DIR` at it for `LongCatAudioReferenceTests`.

| Script | Needs | Writes | C# test |
|---|---|---|---|
| `dump_host_math.py` | torch, numpy (no weights) | pooled/resampled features, windows, first/later groups for both layouts | `HostMath_*` |
| `dump_mel_128.py --wav clip.wav --whisper-dir D` | transformers, soundfile, `D/preprocessor_config.json` (no weights) | `speech16k`, `mel128_chunks` | `Mel128_MatchesHfWhisperFeatureExtractor` |
| `dump_whisper_features.py --whisper-dir D` | `D/model.safetensors` (3.09 GB), the files above | `ws_*` | `WhisperLargeV3Features_MatchTheReference` (also set `HARTSY_LONGCAT_WHISPER_SAFETENSORS`) |
| `dump_wav2vec2_v10.py --longcat-repo R --w2v-dir D` | `D/pytorch_model.bin` (380 MB) and `config.json`, a checkout of the pinned repo | `w2v_*`, `w2vr_*` | `Wav2Vec2V10Features_MatchTheReference` (also set `HARTSY_LONGCAT_W2V_CHECKPOINT` to the `.bin`; the engine reads the pickle itself) |

Run order: `dump_host_math.py`, `dump_mel_128.py` (writes the saved clip), then the two real-weight scripts. The saved clip is
pre-separated 16 kHz mono vocals; the separator is not part of this comparison. The Avatar 1.0 noise floor is unseeded upstream, so
the dumper seeds it and saves the preprocessed waveform and the extractor's `input_values`.

Environment: pin `transformers==4.41.0` (the reference's wrapper calls `_autoset_attn_implementation`, which later releases changed) and torch >= 2.6.
`dump_mel_128.py` runs on numpy alone, but with torch installed HF computes the STFT in torch F32; `meta.json` records which. None of these scripts has been run yet.
