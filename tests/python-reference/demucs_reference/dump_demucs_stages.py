"""HTDemucs per-stage dump for C# parity debugging. Hooks every encoder/decoder/tencoder/tdecoder layer +
the cross-transformer, and captures the spec/norm front matter, so the C# can be diffed stage-by-stage."""
import os, torch
from demucs.pretrained import get_model
from safetensors.torch import save_file, load_file
from einops import rearrange

OUT = os.path.dirname(os.path.abspath(__file__))
m = get_model("htdemucs").models[0].eval()
m.use_train_segment = False

ref = load_file(os.path.join(OUT, "demucs_ref_io.safetensors"))
wav = ref["wav"]                      # [1, C, L]
d = {}

cap = {}
def hook(name):
    def fn(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        cap[name] = o.detach().float().contiguous()
    return fn
for i in range(len(m.encoder)): m.encoder[i].register_forward_hook(hook(f"enc{i}"))
for i in range(len(m.tencoder)): m.tencoder[i].register_forward_hook(hook(f"tenc{i}"))
for i in range(len(m.decoder)): m.decoder[i].register_forward_hook(hook(f"dec{i}"))
for i in range(len(m.tdecoder)): m.tdecoder[i].register_forward_hook(hook(f"tdec{i}"))

def ct_hook(mod, inp, out):
    cap["ct_in_x"] = inp[0].detach().float().contiguous()
    cap["ct_in_xt"] = inp[1].detach().float().contiguous()
    cap["ct_out_x"] = out[0].detach().float().contiguous()
    cap["ct_out_xt"] = out[1].detach().float().contiguous()
m.crosstransformer.register_forward_hook(ct_hook)
def l0_hook(name):
    def fn(mod, inp, out):
        cap[name + "_in"] = inp[0].detach().float().contiguous()
        o = out[0] if isinstance(out, tuple) else out
        cap[name + "_out"] = o.detach().float().contiguous()
    return fn
m.crosstransformer.layers[0].self_attn.register_forward_hook(lambda mod, inp, out: cap.__setitem__("ct_l0_attn", (out[0] if isinstance(out, tuple) else out).detach().float().contiguous()))
m.crosstransformer.layers[0].norm_out.register_forward_hook(lambda mod, inp, out: cap.__setitem__("ct_l0_out_true", inp[0].detach().float().contiguous()))
m.crosstransformer.layers[0].register_forward_hook(l0_hook("ct_l0x"))
m.crosstransformer.layers_t[0].register_forward_hook(l0_hook("ct_l0xt"))

m.encoder[0].conv.register_forward_hook(hook("enc0_conv"))
m.encoder[0].dconv.register_forward_hook(hook("enc0_postdconv"))
m.encoder[1].conv.register_forward_hook(hook("enc1_conv"))
m.encoder[1].dconv.register_forward_hook(hook("enc1_postdconv"))
with torch.no_grad():
    z = m._spec(wav)
    mag = m._magnitude(z)
    d["spec_cac"] = mag.detach().float().contiguous()
    mn = mag.mean(dim=(1, 2, 3), keepdim=True); sd_ = mag.std(dim=(1, 2, 3), keepdim=True)
    d["spec_norm"] = ((mag - mn) / (1e-5 + sd_)).detach().float().contiguous()
    out = m(wav)
d["stems"] = out.detach().float().contiguous()
# enc0 dconv output is captured in the per-freq-row [Fr, C, T] layout; permute to [1, C, Fr, T] to match the C#.
for kk in ("enc0_postdconv", "enc1_postdconv"):
    if kk in cap and cap[kk].dim() == 3:
        cap[kk] = cap[kk].permute(1, 0, 2).unsqueeze(0).contiguous()
for k, v in cap.items(): d[k] = v
d["enc0_prefreq"] = cap["enc0"].clone()   # encoder[0] output is pre-freq-emb (freq_emb added later in forward)
# aliases matching the C# transformer probe names
d["ct_pos_x"] = cap["ct_l0x_in"].clone(); d["ct_l0_x"] = cap["ct_l0x_out"].clone()
d["ct_pos_xt"] = cap["ct_l0xt_in"].clone(); d["ct_l0_xt"] = cap["ct_l0xt_out"].clone()
# afterAttn ref = layer0 input + gamma_1 * attn_out (channel-last scale over C)
_g1 = m.crosstransformer.layers[0].gamma_1.scale.detach().float()
d["ct_l0_afterattn"] = (cap["ct_l0x_in"] + cap["ct_l0_attn"] * _g1).contiguous()
_l0 = m.crosstransformer.layers[0]
_aa = cap["ct_l0x_in"] + cap["ct_l0_attn"] * _g1
_fn = torch.nn.functional.layer_norm(_aa, (512,), _l0.norm2.weight, _l0.norm2.bias, _l0.norm2.eps)
_h = _l0.linear2(torch.nn.functional.gelu(_l0.linear1(_fn)))
d["ct_l0_out"] = cap["ct_l0_out_true"].clone()   # TRUE demucs pre-norm_out (norm_out module input)
for k in sorted(d): print(f"{k:10s} {tuple(d[k].shape)}")
save_file(d, os.path.join(OUT, "demucs_stages.safetensors"))
print("wrote demucs_stages.safetensors")
