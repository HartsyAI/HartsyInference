"""Generates a tiny AudioGen-style PyTorch .bin (pickle) checkpoint for AudioGenLoaderTests.
Run:  python3 audiogen_tiny_bin_ref.py /tmp/audiogen_tiny.bin
Then: HARTSYINFERENCE_AUDIOGEN_BIN=/tmp/audiogen_tiny.bin dotnet test ...AudioGenLoaderTests"""
import sys
sys.modules['torchvision'] = None  # avoid a broken torchvision poisoning the import
import torch

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/audiogen_tiny.bin"
sd = {
    "decoder.model.decoder.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
    "decoder.model.decoder.embed_positions.weights": torch.randn(8, 4),
    "decoder.lm_heads.0.weight": torch.randn(4, 4),
    "enc_to_dec_proj.weight": torch.randn(4, 4),
    "text_encoder.shared.weight": torch.randn(6, 4),
    "text_encoder.encoder.final_layer_norm.weight": torch.randn(4),
    "audio_encoder.encoder.layers.0.conv.weight_g": torch.randn(4, 1, 1),
    "audio_encoder.encoder.layers.0.conv.weight_v": torch.randn(4, 2, 3),
}
torch.save(sd, out)
print("saved", len(sd), "tensors to", out)
