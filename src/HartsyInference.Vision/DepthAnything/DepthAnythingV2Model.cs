using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.Vision.DepthAnything;

/// <summary>Depth-Anything-V2 relative (inverse) monocular depth estimator: a DINOv2 backbone whose four
/// tapped block outputs feed a <see cref="DptHead"/>. Loads the official checkpoints
/// (<c>depth_anything_v2_vit{s,b,l}.pth</c> / their safetensors mirrors — torch.hub-style keys with fused
/// <c>attn.qkv</c>, remapped at load). Output is non-metric: larger values = nearer; min-max normalize
/// (<see cref="DepthAnythingPreprocessor.NormalizeToUnit"/>) for ControlNet/Flux-Depth conditioning.</summary>
public sealed unsafe class DepthAnythingV2Model
{
    private readonly DepthAnythingPreset _preset;
    private readonly Dinov2VisionEncoder _backbone;
    private readonly DptHead _head;

    public DepthAnythingPreset Preset => _preset;

    public DepthAnythingV2Model(DepthAnythingPreset preset)
    {
        _preset = preset;
        _backbone = new Dinov2VisionEncoder(preset.Backbone);
        _head = new DptHead(preset.Backbone.HiddenSize, preset.Features, preset.OutChannels);
    }

    /// <summary>Loads an official checkpoint state dict: <c>pretrained.*</c> (torch.hub DINOv2 keys) is
    /// remapped to the HF layout <see cref="Dinov2VisionEncoder"/> expects (fused qkv split into q/k/v,
    /// <c>ls·.gamma</c> → <c>layer_scale·.lambda1</c>); <c>depth_head.*</c> loads directly.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> backbone = RemapBackboneKeys(weights, _preset.Backbone.HiddenSize);
        _backbone.LoadWeights(backbone);
        _head.LoadWeights(weights);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _backbone.EnumerateWeights()) yield return t;
        foreach (Tensor t in _head.EnumerateWeights()) yield return t;
    }

    /// <summary>Predicts relative depth for preprocessed pixels <c>[B,3,H,W]</c> (H, W multiples of 14,
    /// ImageNet-normalized — see <see cref="DepthAnythingPreprocessor"/>). Returns <c>[B,1,H,W]</c>, ≥ 0.
    /// <paramref name="tap"/> receives named intermediates for the parity harness.</summary>
    public Tensor Forward(IBackend backend, Tensor pixelValues, Action<string, Tensor>? tap = null)
    {
        int gridH = (int)pixelValues.Shape[2] / _preset.Backbone.PatchSize;
        int gridW = (int)pixelValues.Shape[3] / _preset.Backbone.PatchSize;

        Tensor[] taps = _backbone.EncodeIntermediates(backend, pixelValues, _preset.LayerIndices);
        Tensor[] patchFeatures = new Tensor[4];
        for (int i = 0; i < 4; i++)
        {
            patchFeatures[i] = SlicePatchTokens(taps[i], 1 + _preset.Backbone.NumRegisterTokens);
            taps[i].Dispose();
            tap?.Invoke($"feat_{i}", patchFeatures[i]);
        }

        Tensor depth = _head.Forward(backend, patchFeatures, gridH, gridW, tap);
        foreach (Tensor f in patchFeatures) f.Dispose();
        return depth;
    }

    /// <summary>Drops the CLS (+ register) rows: <c>[B, prefix+N, C]</c> → <c>[B, N, C]</c>. Host memcpy —
    /// once per tap, outside the per-block compute path.</summary>
    private static Tensor SlicePatchTokens(Tensor sequence, int prefixTokens)
    {
        int batch = (int)sequence.Shape[0];
        int seqLen = (int)sequence.Shape[1];
        int hidden = (int)sequence.Shape[2];
        int numPatches = seqLen - prefixTokens;
        Tensor patches = new(new TensorShape(batch, numPatches, hidden), DType.F32);
        float* src = (float*)sequence.DataPointer;
        float* dst = (float*)patches.DataPointer;
        long rowBytes = (long)numPatches * hidden * sizeof(float);
        for (int b = 0; b < batch; b++)
            Buffer.MemoryCopy(src + ((long)b * seqLen + prefixTokens) * hidden, dst + (long)b * numPatches * hidden, rowBytes, rowBytes);
        return patches;
    }

    /// <summary>torch.hub DINOv2 keys → the HF-style keys <see cref="Dinov2VisionEncoder"/> loads.</summary>
    private static Dictionary<string, Tensor> RemapBackboneKeys(IReadOnlyDictionary<string, Tensor> w, int hidden)
    {
        const string p = "pretrained.";
        Dictionary<string, Tensor> mapped = new()
        {
            ["embeddings.cls_token"] = w[$"{p}cls_token"],
            ["embeddings.position_embeddings"] = w[$"{p}pos_embed"],
            ["embeddings.patch_embeddings.projection.weight"] = w[$"{p}patch_embed.proj.weight"],
            ["embeddings.patch_embeddings.projection.bias"] = w[$"{p}patch_embed.proj.bias"],
            ["layernorm.weight"] = w[$"{p}norm.weight"],
            ["layernorm.bias"] = w[$"{p}norm.bias"],
        };
        if (w.TryGetValue($"{p}register_tokens", out Tensor? reg))
            mapped["embeddings.register_tokens"] = reg;

        for (int i = 0; w.ContainsKey($"{p}blocks.{i}.norm1.weight"); i++)
        {
            string src = $"{p}blocks.{i}";
            string dst = $"encoder.layer.{i}";
            mapped[$"{dst}.norm1.weight"] = w[$"{src}.norm1.weight"];
            mapped[$"{dst}.norm1.bias"] = w[$"{src}.norm1.bias"];
            mapped[$"{dst}.norm2.weight"] = w[$"{src}.norm2.weight"];
            mapped[$"{dst}.norm2.bias"] = w[$"{src}.norm2.bias"];
            mapped[$"{dst}.attention.output.dense.weight"] = w[$"{src}.attn.proj.weight"];
            mapped[$"{dst}.attention.output.dense.bias"] = w[$"{src}.attn.proj.bias"];
            mapped[$"{dst}.layer_scale1.lambda1"] = w[$"{src}.ls1.gamma"];
            mapped[$"{dst}.layer_scale2.lambda1"] = w[$"{src}.ls2.gamma"];
            mapped[$"{dst}.mlp.fc1.weight"] = w[$"{src}.mlp.fc1.weight"];
            mapped[$"{dst}.mlp.fc1.bias"] = w[$"{src}.mlp.fc1.bias"];
            mapped[$"{dst}.mlp.fc2.weight"] = w[$"{src}.mlp.fc2.weight"];
            mapped[$"{dst}.mlp.fc2.bias"] = w[$"{src}.mlp.fc2.bias"];

            Tensor qkvW = Dinov2VisionEncoder.F32(w[$"{src}.attn.qkv.weight"]);
            Tensor qkvB = Dinov2VisionEncoder.F32(w[$"{src}.attn.qkv.bias"]);
            for (int part = 0; part < 3; part++)
            {
                string name = part == 0 ? "query" : part == 1 ? "key" : "value";
                Tensor pw = new(new TensorShape(hidden, hidden), DType.F32);
                Tensor pb = new(new TensorShape(hidden), DType.F32);
                long wBytes = (long)hidden * hidden * sizeof(float);
                Buffer.MemoryCopy((float*)qkvW.DataPointer + (long)part * hidden * hidden, (float*)pw.DataPointer, wBytes, wBytes);
                Buffer.MemoryCopy((float*)qkvB.DataPointer + (long)part * hidden, (float*)pb.DataPointer, (long)hidden * sizeof(float), (long)hidden * sizeof(float));
                mapped[$"{dst}.attention.attention.{name}.weight"] = pw;
                mapped[$"{dst}.attention.attention.{name}.bias"] = pb;
            }
        }
        return mapped;
    }
}
