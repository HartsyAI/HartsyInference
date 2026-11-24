"""Reference dump for the Kim_Vocal_2 vocal separator (UVR MDX-Net, run through onnxruntime).

Runs LongCat-Video's own ``separate_vocals`` (audio-separator 0.30.2 + onnxruntime) on one clip and saves what the C# port is
compared against: the normalized mix, the first chunks' STFT input and ONNX output, every TFC-TDF block output of the network
for chunk 0, the stereo stem, and the 16 kHz mono vocals that LongCat feeds to Whisper.

Environment (WSL or a pod; never the Windows venv, and onnxruntime never enters the engine):
    pip install "audio-separator[cpu]==0.30.2" onnx numpy librosa
    git clone https://github.com/meituan-longcat/LongCat-Video && git -C LongCat-Video checkout 58eb336edc5c8c63389838727e40b8c18434e54e
    PYTHONPATH=LongCat-Video python dump_kim_vocal2_reference.py --onnx Kim_Vocal_2.onnx --wav clip.wav --out ref_dir

The vocal_separator folder next to the .onnx may hold only the .onnx; audio-separator finds the model by file name and
reads the n_fft / dim_f / compensate row for it from its own mdx_model_data.json (md5 970b3f9492014d18fefeedfe4773cb42).
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

EXPECTED_SHA256 = "ce74ef3b6a6024ce44211a07be9cf8bc6d87728cc852a68ab34eb8e58cde9c8b"
CHUNKS_TO_SAVE = 1  # run_model sees a batch of 2 chunks; only the first chunk of the first batch is saved


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(out, name, array):
    np.save(os.path.join(out, name + ".npy"), np.ascontiguousarray(array, dtype=np.float32))


def stage_outputs(onnx_path):
    """Names of the tensors every Add node produces: the 5 encoder blocks, the bottleneck, then the 5 decoder blocks."""
    import onnx

    model = onnx.load(onnx_path)
    return model, [n.output[0] for n in model.graph.node if n.op_type == "Add"]


def stage_session(onnx_path, names):
    import onnx
    import onnxruntime as ort

    model, _ = stage_outputs(onnx_path)
    for name in names:
        model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 0
    return ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--duration", type=float, default=None, help="seconds, as LongCat's audio_process_duration; default whole clip")
    args = parser.parse_args()

    digest = sha256(args.onnx)
    if digest != EXPECTED_SHA256:
        sys.exit(f"Kim_Vocal_2.onnx sha256 {digest} is not the pinned {EXPECTED_SHA256}")
    os.makedirs(args.out, exist_ok=True)

    import audio_separator
    import librosa
    import onnxruntime
    import torch
    from audio_separator.separator import Separator
    from audio_separator.separator.uvr_lib_v5 import spec_utils
    from longcat_video.audio_process.vocal_separator import _demix_batched, separate_vocals

    separator = Separator(
        output_dir=os.path.join(args.out, "separator_tmp"),
        output_single_stem="vocals",
        model_file_dir=os.path.dirname(os.path.abspath(args.onnx)),
    )
    separator.onnx_execution_provider = ["CPUExecutionProvider"]
    separator.load_model(os.path.basename(args.onnx))
    model = separator.model_instance
    settings = dict(n_fft=model.n_fft, dim_f=model.dim_f, dim_t=model.dim_t, compensate=model.compensate,
                    hop=model.hop_length, segment=model.segment_size, overlap=model.overlap,
                    sample_rate=model.sample_rate, normalization_threshold=model.normalization_threshold)

    # Per-chunk records: wrap run_model so each call saves its STFT, the raw ONNX output and the inverse-STFT result.
    records = []
    original_run_model = model.run_model

    def recording_run_model(mix_wave, is_match_mix=False):
        out = original_run_model(mix_wave, is_match_mix=is_match_mix)
        if len(records) < CHUNKS_TO_SAVE:
            spek = model.stft(mix_wave.to(model.torch_device))
            spek[:, :, :3, :] *= 0
            records.append({"wave": mix_wave.cpu().numpy()[:1], "spek": spek.cpu().numpy()[:1],
                            "pred": np.array(model.model_run(spek))[:1], "istft": np.array(out)[:1]})
        return out

    model.run_model = recording_run_model

    # The same steps as LongCat's separate_vocals, with the intermediates kept.
    mix = model.prepare_mix(args.wav)
    mix = spec_utils.normalize(wave=mix, max_peak=model.normalization_threshold, min_peak=-model.normalization_threshold)
    source_rate = model.sample_rate
    if args.duration is not None:
        model.initialize_model_settings()
        target_samples = min(math.ceil(args.duration * source_rate), mix.shape[-1])
        mix = mix[..., : min(target_samples + model.chunk_size, mix.shape[-1])]
    else:
        target_samples = None
    save(args.out, "mix_normalized", mix)

    source = _demix_batched(model, mix, 2)
    save(args.out, "stem_compensated", source)
    source = spec_utils.normalize(wave=source, max_peak=model.normalization_threshold, min_peak=-model.normalization_threshold)
    save(args.out, "stem_normalized", source)
    mono = librosa.to_mono(source)
    save(args.out, "mono_44k", mono)
    mono16 = librosa.resample(mono, orig_sr=source_rate, target_sr=16000)
    if target_samples is not None:
        mono16 = mono16[: math.ceil(target_samples * 16000 / source_rate)]
    save(args.out, "vocals_16k", mono16)

    model.run_model = original_run_model
    official, official_rate = separate_vocals(separator, args.wav, duration=args.duration)
    same = official.shape == mono16.shape and float(np.max(np.abs(official - mono16))) < 1e-6
    if not same:
        sys.exit("the dumper's replay of separate_vocals differs from separate_vocals itself; fix the replay before using the dump")

    for index, record in enumerate(records):
        save(args.out, f"chunk{index}_wave", record["wave"])
        save(args.out, f"chunk{index}_spek", record["spek"])
        save(args.out, f"chunk{index}_pred", record["pred"])
        save(args.out, f"chunk{index}_istft", record["istft"])

    # Block outputs of chunk 0 straight from the graph (layout [1, C, T, F] as the ONNX nodes see it).
    _, add_names = stage_outputs(args.onnx)
    session = stage_session(args.onnx, add_names)
    results = session.run(add_names, {"input": records[0]["spek"].astype(np.float32)})
    stage_names = [f"encoding_blocks.{i}" for i in range(5)] + ["bottleneck_block"] + [f"decoding_blocks.{i}" for i in range(5)]
    for name, value in zip(stage_names, results):
        save(args.out, "stage_" + name, value)

    meta = {
        "onnx_sha256": digest,
        "settings": settings,
        "chunks_saved": len(records),
        "stage_names": stage_names,
        "samples_44k": int(mix.shape[-1]),
        "target_samples_44k": target_samples,
        "versions": {"audio_separator": getattr(audio_separator, "__version__", "?"), "onnxruntime": onnxruntime.__version__,
                     "torch": torch.__version__, "numpy": np.__version__, "librosa": librosa.__version__},
        "official_matches_replay": bool(same),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
