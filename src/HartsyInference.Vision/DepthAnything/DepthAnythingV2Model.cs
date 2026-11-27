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
        Dictionary<string, Tensor> backbone = Dinov2HubKeys.Remap(weights, "pretrained.", _preset.Backbone);
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
}
