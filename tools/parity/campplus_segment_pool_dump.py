"""CAM++ speaker embedding as each upstream runs it: Chatterbox (PyTorch CAMPPlus) and CosyVoice 2 (campplus.onnx).

The CAM context pool averages 100-frame segments with ceil_mode, so the trailing partial segment's divisor depends on
the runtime: PyTorch avg_pool1d divides by the frames the window covers; campplus.onnx (opset 14, AveragePool with
ceil_mode 1) under the ORT_ENABLE_ALL session CosyVoice builds divides by the kernel size, while the same graph with
graph optimizations disabled matches PyTorch. This records, for one 16 kHz clip, the fbank both upstreams feed CAM++ and the embedding each produces:

    audio16k         the clip                                                             [samples]
    fbank            Kaldi.fbank(num_mel_bins=80, dither 0) minus the per-bin mean        [T, 80]
    emb_chatterbox   chatterbox/models/s3gen/xvector.py CAMPPlus on s3gen.safetensors     [192]
                     (speaker_encoder.*), exactly as S3Gen.embed_ref calls it
    emb_torch_div100 the same module with seg_pooling dividing every segment by 100      [192]
    emb_cosyvoice    CosyVoice 2's campplus.onnx through onnxruntime (CPU, ORT_ENABLE_ALL,  [192]
                     one intra-op thread), as cosyvoice/cli/frontend.py _extract_spk_embedding calls it
    emb_onnx_noopt   the same session with ORT_DISABLE_ALL                                  [192]

    cosy-full-venv/bin/python tools/parity/campplus_segment_pool_dump.py <chatterbox checkout> \
        <ResembleAI--chatterbox dir> <FunAudioLLM--CosyVoice2-0.5B dir> <clip.wav (16 kHz mono)> <out.safetensors>
"""
import importlib.util
import os
import sys

import numpy as np
import onnxruntime
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
from safetensors.torch import load_file, save_file


def load_xvector(chatterbox_root):
    path = os.path.join(chatterbox_root, "src", "chatterbox", "models", "s3gen", "xvector.py")
    spec = importlib.util.spec_from_file_location("chatterbox_xvector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    chatterbox_root, chatterbox_dir, cosy_dir, wav_path, out_path = sys.argv[1:6]
    torch.set_num_threads(8)
    audio, rate = sf.read(wav_path, dtype="float32")
    if rate != 16000 or audio.ndim != 1:
        raise SystemExit(f"{wav_path}: need 16 kHz mono, got {rate} Hz with shape {audio.shape}")
    wav = torch.from_numpy(audio)

    xvector = load_xvector(chatterbox_root)
    model = xvector.CAMPPlus(memory_efficient=False)
    state = load_file(os.path.join(chatterbox_dir, "s3gen.safetensors"))
    prefix = "speaker_encoder."
    model.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
    model.eval()

    # S3Gen.embed_ref: self.speaker_encoder.inference(ref_wav_16), ref_wav_16 shaped [1, samples].
    with torch.no_grad():
        emb_chatterbox = model.inference(wav.unsqueeze(0)).reshape(-1)
        feats, _, _ = xvector.extract_feature(wav.unsqueeze(0))
        fbank = feats[0]

        def seg_pooling_div100(self, x, seg_len=100, stype="avg"):
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
            tail = x.shape[-1] % seg_len
            if tail:
                seg[..., -1] = seg[..., -1] * tail / seg_len
            shape = seg.shape
            seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
            return seg[..., : x.shape[-1]]

        xvector.CAMLayer.seg_pooling = seg_pooling_div100
        emb_torch_div100 = model(fbank.unsqueeze(0)).reshape(-1)

    # CosyVoiceFrontEnd._extract_spk_embedding: kaldi.fbank(num_mel_bins=80, dither=0, sample_frequency=16000), CMN,
    # then the ONNX session on CPU.
    cosy_feat = kaldi.fbank(wav.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
    cosy_feat = cosy_feat - cosy_feat.mean(dim=0, keepdim=True)
    def run_onnx(level):
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = level
        option.intra_op_num_threads = 1
        session = onnxruntime.InferenceSession(os.path.join(cosy_dir, "campplus.onnx"), sess_options=option,
                                               providers=["CPUExecutionProvider"])
        return session.run(None, {session.get_inputs()[0].name: cosy_feat.unsqueeze(0).numpy()})[0].reshape(-1)

    emb_cosyvoice = run_onnx(onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL)
    emb_onnx_noopt = run_onnx(onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL)

    def rel(a, b):
        a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
        return ((a - b).norm() / b.norm()).item()

    print(f"clip {wav.numel()} samples, fbank {tuple(fbank.shape)}, CAM time steps {(fbank.shape[0] + 1) // 2}")
    print(f"fbank chatterbox vs cosyvoice relL2 {rel(fbank, cosy_feat):.3e}")
    print(f"chatterbox(torch) vs torch/100  relL2 {rel(emb_torch_div100, emb_chatterbox):.3e}")
    print(f"cosyvoice(onnx)   vs torch/100  relL2 {rel(emb_cosyvoice, emb_torch_div100):.3e}")
    print(f"cosyvoice(onnx)   vs chatterbox relL2 {rel(emb_cosyvoice, emb_chatterbox):.3e}")
    print(f"onnx unoptimized  vs chatterbox relL2 {rel(emb_onnx_noopt, emb_chatterbox):.3e}")
    save_file({
        "audio16k": wav.contiguous(),
        "fbank": fbank.contiguous(),
        "emb_chatterbox": emb_chatterbox.contiguous(),
        "emb_torch_div100": emb_torch_div100.contiguous(),
        "emb_cosyvoice": torch.from_numpy(emb_cosyvoice).contiguous(),
        "emb_onnx_noopt": torch.from_numpy(emb_onnx_noopt).contiguous(),
    }, out_path)


if __name__ == "__main__":
    main()
