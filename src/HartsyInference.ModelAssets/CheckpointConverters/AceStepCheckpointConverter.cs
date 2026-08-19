using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Loads ACE-Step v1 checkpoints (<c>ACE-Step/ACE-Step-v1-3.5B</c>, diffusers folder layout) for the
/// <c>AceStepDit</c> / <c>MusicDcaeDecoder</c> / <c>AdaMosHiFiGanV1</c> classes. The HF repo splits components into
/// <c>ace_step_transformer/</c>, <c>music_dcae_f8c8/</c>, <c>music_vocoder/</c>, and <c>umt5-base/</c> folders —
/// load each safetensors with the matching method here. DiT: passthrough that drops the training-only SSL alignment
/// heads (<c>projectors.*</c>). DCAE: passthrough (already <c>encoder./decoder.</c> named). Vocoder: fuses PyTorch
/// weight-norm pairs (<c>weight_g/weight_v → weight = g·v/‖v‖</c>, per-output-channel norm) and strips an optional
/// <c>generator.</c> prefix. The lyric tokenizer needs a one-time <c>vocab.json</c> + <c>merges.txt</c> export from
/// the repo's HF <c>tokenizer.json</c> (<c>python -c "import json; t=json.load(open('tokenizer.json'));
/// json.dump(t['model']['vocab'], open('vocab.json','w')); open('merges.txt','w').write('\n'.join(' '.join(m) if
/// isinstance(m, list) else m for m in t['model']['merges']))"</c>).</summary>
public sealed class AceStepCheckpointConverter
{
    /// <summary>Loads the DiT safetensors, dropping training-only keys. With <paramref name="castToF32"/> every
    /// tensor is materialized as F32 (required for CPU inference — the BF16 checkpoint is ~13 GB in F32; without it
    /// tensors borrow the mmap and stay BF16, which is fine for key/shape validation and GPU paths with F16 kernels).
    /// Caller owns the loader.</summary>
    public static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) LoadTransformer(string path, bool castToF32 = false)
    {
        SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> weights = new();
        foreach (string key in loader.Descriptors.Keys)
        {
            if (key.StartsWith("projectors.", StringComparison.Ordinal)) continue;   // SSL heads (training only)
            weights[Strip(key, "model.")] = MaybeCast(loader.GetTensor(key), castToF32);
        }
        return (weights, loader);
    }

    /// <summary>Loads the Music-DCAE safetensors (passthrough; strips an optional <c>autoencoder.</c> prefix).</summary>
    public static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) LoadDcae(string path, bool castToF32 = false)
    {
        SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> weights = new();
        foreach (string key in loader.Descriptors.Keys)
            weights[Strip(Strip(key, "autoencoder."), "dcae.")] = MaybeCast(loader.GetTensor(key), castToF32);
        return (weights, loader);
    }

    /// <summary>Loads the ADaMoS vocoder safetensors with weight-norm fusion. Fused tensors are freshly allocated;
    /// the caller owns the loader for the rest.</summary>
    public static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) LoadVocoder(string path, bool castToF32 = false)
    {
        SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> raw = new();
        foreach (string key in loader.Descriptors.Keys)
            raw[Strip(Strip(key, "generator."), "model.")] = MaybeCast(loader.GetTensor(key), castToF32);
        return (FuseWeightNorm(raw), loader);
    }

    /// <summary>Loads the ACE-Step <b>v1.5</b> turbo main safetensors (<c>ACE-Step/Ace-Step1.5</c>,
    /// <c>acestep-v15-turbo/model.safetensors</c>, 677 BF16 keys) — one file holds the DiT (<c>decoder.*</c>, for
    /// <c>AceStep15Dit</c>) and the condition encoders (<c>encoder.*</c>, for <c>AceStep15ConditionEncoder</c>);
    /// both classes pick their prefixes from this single dict. Passthrough: the FSQ <c>tokenizer.*</c> /
    /// <c>detokenizer.*</c> hint path and <c>null_condition_emb</c> (phase 2 / CFG-only) are kept but unused by the
    /// turbo text-to-music path. The Oobleck VAE is a separate file loaded straight into <c>OobleckVae</c>
    /// (it fuses its own weight norm). Caller owns the loader.</summary>
    public static (Dictionary<string, Tensor> Weights, SafeTensorsLoader Loader) LoadModel15(string path, bool castToF32 = false)
    {
        SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> weights = new();
        foreach (string key in loader.Descriptors.Keys)
            weights[Strip(key, "model.")] = MaybeCast(loader.GetTensor(key), castToF32);
        return (weights, loader);
    }

    /// <summary>True when the header is a Comfy-repackaged ACE-Step v1 all-in-one
    /// (<c>Comfy-Org/ACE-Step_ComfyUI_repackaged</c> <c>all_in_one/ace_step_v1_3.5b.safetensors</c>).</summary>
    public static bool IsV1AllInOne(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors) =>
        descriptors.ContainsKey("model.diffusion_model.lyric_embs.weight");

    /// <summary>Splits the v1 all-in-one file into the four component dictionaries the existing classes consume:
    /// the DiT (bare keys, same as <see cref="LoadTransformer"/>, <c>projectors.*</c> skipped), the Music-DCAE,
    /// the ADaMoS vocoder (weight-norm fused only when the file still ships <c>_g/_v</c> pairs), and the UMT5-base
    /// text encoder (<c>encoder.block…</c>/<c>shared.weight</c>; <c>logit_scale</c> dropped). Caller owns the loader.</summary>
    public static (Dictionary<string, Tensor> Transformer, Dictionary<string, Tensor> Dcae,
        Dictionary<string, Tensor> Vocoder, Dictionary<string, Tensor> TextEncoder, SafeTensorsLoader Loader)
        LoadV1AllInOne(string path, bool castToF32 = false)
    {
        SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> transformer = new();
        Dictionary<string, Tensor> dcae = new();
        Dictionary<string, Tensor> vocoder = new();
        Dictionary<string, Tensor> text = new();
        foreach (string key in loader.Descriptors.Keys)
        {
            if (key.StartsWith("model.diffusion_model.", StringComparison.Ordinal))
            {
                string bare = key["model.diffusion_model.".Length..];
                if (bare.StartsWith("projectors.", StringComparison.Ordinal)) continue;   // SSL heads (training only)
                transformer[bare] = MaybeCast(loader.GetTensor(key), castToF32);
            }
            else if (key.StartsWith("vae.dcae.", StringComparison.Ordinal))
            {
                dcae[key["vae.dcae.".Length..]] = MaybeCast(loader.GetTensor(key), castToF32);
            }
            else if (key.StartsWith("vae.vocoder.", StringComparison.Ordinal))
            {
                vocoder[key["vae.vocoder.".Length..]] = MaybeCast(loader.GetTensor(key), castToF32);
            }
            else if (key.StartsWith("text_encoders.umt5base.transformer.", StringComparison.Ordinal))
            {
                text[key["text_encoders.umt5base.transformer.".Length..]] = MaybeCast(loader.GetTensor(key), castToF32);
            }
            else if (key == "text_encoders.umt5base.logit_scale" || key == "text_encoders.spiece_model")
            {
                // logit_scale: CLAP-style scale the encoder never uses. spiece_model: Comfy's embedded
                // SentencePiece blob — the engine ships its own umT5 spiece resource.
            }
            else
            {
                throw new HartsyInferenceException($"Unrecognized ACE-Step v1 all-in-one key '{key}' — refusing a partial load.");
            }
        }
        if (vocoder.Keys.Any(k => k.EndsWith(".weight_g", StringComparison.Ordinal)))
        {
            vocoder = FuseWeightNorm(vocoder);
        }
        return (transformer, dcae, vocoder, text, loader);
    }

    private static Tensor MaybeCast(Tensor t, bool castToF32) =>
        castToF32 && t.DType != DType.F32 ? t.CastTo(DType.F32) : t;

    /// <summary>Fuses <c>*.weight_g/*.weight_v</c> pairs into plain <c>*.weight</c> (PyTorch <c>weight_norm</c>,
    /// dim=0: per-output-channel L2 over the remaining dims). Non-paired keys pass through.</summary>
    public static unsafe Dictionary<string, Tensor> FuseWeightNorm(Dictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> result = new();
        foreach ((string key, Tensor value) in raw)
        {
            if (key.EndsWith(".weight_g", StringComparison.Ordinal))
            {
                string baseKey = key[..^".weight_g".Length];
                Tensor g = value.DType == DType.F32 ? value : value.CastTo(DType.F32);
                Tensor vRaw = raw[$"{baseKey}.weight_v"];
                Tensor v = vRaw.DType == DType.F32 ? vRaw : vRaw.CastTo(DType.F32);
                long outC = v.Shape[0];
                long per = v.Shape.ElementCount / outC;
                Tensor fused = new Tensor(v.Shape, DType.F32);
                float* gp = (float*)g.DataPointer;
                float* vp = (float*)v.DataPointer;
                float* fp = (float*)fused.DataPointer;
                for (long o = 0; o < outC; o++)
                {
                    double normSq = 0;
                    for (long i = 0; i < per; i++)
                    {
                        float val = vp[o * per + i];
                        normSq += (double)val * val;
                    }
                    float scale = gp[o] / (float)Math.Max(Math.Sqrt(normSq), 1e-12);
                    for (long i = 0; i < per; i++) fp[o * per + i] = vp[o * per + i] * scale;
                }
                result[$"{baseKey}.weight"] = fused;
            }
            else if (!key.EndsWith(".weight_v", StringComparison.Ordinal))
            {
                result[key] = value;
            }
        }
        return result;
    }

    private static string Strip(string key, string prefix) =>
        key.StartsWith(prefix, StringComparison.Ordinal) ? key[prefix.Length..] : key;
}
