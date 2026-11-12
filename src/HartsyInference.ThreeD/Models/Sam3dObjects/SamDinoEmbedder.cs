using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// The DINOv2-L/14 with registers that both SAM 3D condition embedders wrap (upstream <c>Dino</c>, <c>embedder/dino.py</c>, <c>dinov2_vitl14_reg</c>, input 518,
/// ImageNet normalisation). Weights come with <c>torch.hub</c> names (<c>patch_embed.proj</c>, <c>blocks.i.attn.qkv</c>, <c>ls1.gamma</c>, <c>norm</c>); they are
/// re-keyed to the names <see cref="Dinov2VisionEncoder"/> reads, with the fused QKV split into three copies this object owns. Two outputs:
/// <see cref="StructureTokens"/> (<c>forward_features</c>: final norm, class token then patch tokens, the four register tokens dropped, <c>[1, 1370, 1024]</c>) feeds the
/// structure stage's fuser, and <see cref="SlatTokens"/> (<c>prenorm_features</c>: the block output before the final norm, registers kept, then a non-affine LayerNorm,
/// <c>[1, 1374, 1024]</c>) feeds the SLAT flow. The mask tower is the same network with the mask repeated over three channels (<c>Dino</c> is called on the mask).
/// </summary>
public sealed class SamDinoEmbedder : IDisposable
{
    /// <summary>Side of the square input (<c>input_size</c>); inputs must already have this size, as the upstream preprocessor resizes first.</summary>
    public const int InputSize = 518;

    /// <summary>Token width of the released tower (the fuser's DINO <c>embed_dim</c>).</summary>
    public const int Width = 1024;

    private const float NormEps = 1e-5f;
    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    private readonly Dinov2Preset _preset;
    private readonly Dinov2VisionEncoder _encoder;
    private readonly List<Tensor> _owned = [];

    /// <param name="weights">The DINO tensors by hub name under <paramref name="prefix"/>, F32; they stay owned by the caller and must outlive this object.</param>
    /// <param name="prefix">Everything before <c>patch_embed.proj.weight</c> and the other keys, such as <c>condition_embedder.module_list.0.backbone.</c>.</param>
    /// <exception cref="InvalidOperationException">A weight is missing, not F32 or of the wrong shape.</exception>
    public SamDinoEmbedder(IReadOnlyDictionary<string, Tensor> weights, string prefix = "")
        : this(Dinov2Preset.LargeReg, weights, prefix)
    {
    }

    /// <summary>The same network at another size, for tests on a tiny model; the released one is <see cref="Dinov2Preset.LargeReg"/>.</summary>
    internal SamDinoEmbedder(Dinov2Preset preset, IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        ArgumentNullException.ThrowIfNull(preset);
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(prefix);
        _preset = preset;
        _encoder = new Dinov2VisionEncoder(preset);
        try
        {
            _encoder.LoadWeights(ToEncoderNames(weights, prefix));
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>The encoder's weights under its own names: renamed ones are the caller's tensors, the split query, key and value projections are owned copies.</summary>
    private Dictionary<string, Tensor> ToEncoderNames(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        int d = _preset.HiddenSize, layers = _preset.NumLayers, patch = _preset.PatchSize, registers = _preset.NumRegisterTokens;
        Dictionary<string, Tensor> result = new(StringComparer.Ordinal)
        {
            ["embeddings.patch_embeddings.projection.weight"] = SamWeights.Require(weights, prefix + "patch_embed.proj.weight", d, 3, patch, patch),
            ["embeddings.patch_embeddings.projection.bias"] = SamWeights.Require(weights, prefix + "patch_embed.proj.bias", d),
            ["embeddings.cls_token"] = SamWeights.Require(weights, prefix + "cls_token", 1, 1, d),
            ["embeddings.position_embeddings"] = SamWeights.Require(weights, prefix + "pos_embed", 1, 1 + _preset.NumPatches, d),
            ["embeddings.register_tokens"] = SamWeights.Require(weights, prefix + "register_tokens", 1, registers, d),
            ["layernorm.weight"] = SamWeights.Require(weights, prefix + "norm.weight", d),
            ["layernorm.bias"] = SamWeights.Require(weights, prefix + "norm.bias", d),
        };
        for (int i = 0; i < layers; i++)
        {
            string hub = $"{prefix}blocks.{i}.", hf = $"encoder.layer.{i}.";
            result[hf + "norm1.weight"] = SamWeights.Require(weights, hub + "norm1.weight", d);
            result[hf + "norm1.bias"] = SamWeights.Require(weights, hub + "norm1.bias", d);
            result[hf + "norm2.weight"] = SamWeights.Require(weights, hub + "norm2.weight", d);
            result[hf + "norm2.bias"] = SamWeights.Require(weights, hub + "norm2.bias", d);
            result[hf + "attention.output.dense.weight"] = SamWeights.Require(weights, hub + "attn.proj.weight", d, d);
            result[hf + "attention.output.dense.bias"] = SamWeights.Require(weights, hub + "attn.proj.bias", d);
            result[hf + "layer_scale1.lambda1"] = SamWeights.Require(weights, hub + "ls1.gamma", d);
            result[hf + "layer_scale2.lambda1"] = SamWeights.Require(weights, hub + "ls2.gamma", d);
            result[hf + "mlp.fc1.weight"] = SamWeights.Require(weights, hub + "mlp.fc1.weight", _preset.IntermediateSize, d);
            result[hf + "mlp.fc1.bias"] = SamWeights.Require(weights, hub + "mlp.fc1.bias", _preset.IntermediateSize);
            result[hf + "mlp.fc2.weight"] = SamWeights.Require(weights, hub + "mlp.fc2.weight", d, _preset.IntermediateSize);
            result[hf + "mlp.fc2.bias"] = SamWeights.Require(weights, hub + "mlp.fc2.bias", d);
            Tensor qkvWeight = SamWeights.Require(weights, hub + "attn.qkv.weight", 3 * d, d);
            Tensor qkvBias = SamWeights.Require(weights, hub + "attn.qkv.bias", 3 * d);
            string[] names = ["query", "key", "value"];
            for (int part = 0; part < 3; part++)
            {
                result[$"{hf}attention.attention.{names[part]}.weight"] = Own(qkvWeight, part * d * d, d, d);
                result[$"{hf}attention.attention.{names[part]}.bias"] = Own(qkvBias, part * d, d);
            }
        }

        return result;
    }

    /// <summary>An owned copy of <paramref name="count"/> values of <paramref name="source"/> from <paramref name="start"/>, shaped <paramref name="shape"/>.</summary>
    private Tensor Own(Tensor source, int start, params long[] shape)
    {
        Tensor copy = new(new TensorShape(shape), DType.F32);
        _owned.Add(copy);
        source.AsReadOnlySpan<float>().Slice(start, (int)copy.Shape.ElementCount).CopyTo(copy.AsSpan<float>());
        return copy;
    }

    /// <summary>The structure-stage tokens of an RGB image: planar F32 <c>[3, 518, 518]</c> in 0 to 1. Returns F32 <c>[1, 1370, 1024]</c>, owned by the caller.</summary>
    public Tensor StructureTokens(IBackend backend, ReadOnlySpan<float> rgb)
    {
        ArgumentNullException.ThrowIfNull(backend);
        using Tensor pixels = Normalize(rgb, 3);
        using Tensor all = _encoder.Encode(backend, pixels);
        return DropRegisters(backend, all);
    }

    /// <summary>The structure-stage tokens of a mask: F32 <c>[518, 518]</c> (one channel, repeated over three before the normalisation). Returns F32 <c>[1, 1370, 1024]</c>.</summary>
    public Tensor StructureTokensOfMask(IBackend backend, ReadOnlySpan<float> mask)
    {
        ArgumentNullException.ThrowIfNull(backend);
        using Tensor pixels = Normalize(mask, 1);
        using Tensor all = _encoder.Encode(backend, pixels);
        return DropRegisters(backend, all);
    }

    /// <summary>The SLAT-stage tokens of an RGB image (see <see cref="StructureTokens"/> for the input): the pre-norm block output, then <c>F.layer_norm</c>. F32 <c>[1, 1374, 1024]</c>.</summary>
    public Tensor SlatTokens(IBackend backend, ReadOnlySpan<float> rgb)
    {
        ArgumentNullException.ThrowIfNull(backend);
        using Tensor pixels = Normalize(rgb, 3);
        return PrenormTokens(backend, pixels);
    }

    /// <summary>The SLAT-stage tokens of a mask (see <see cref="StructureTokensOfMask"/> for the input).</summary>
    public Tensor SlatTokensOfMask(IBackend backend, ReadOnlySpan<float> mask)
    {
        ArgumentNullException.ThrowIfNull(backend);
        using Tensor pixels = Normalize(mask, 1);
        return PrenormTokens(backend, pixels);
    }

    private Tensor PrenormTokens(IBackend backend, Tensor pixels)
    {
        using Tensor prenorm = _encoder.Encode(backend, pixels, applyFinalNorm: false);
        Tensor result = new(prenorm.Shape, DType.F32);
        backend.LayerNormNoAffine(result, prenorm, NormEps);
        return result;
    }

    /// <summary>Class token then patch tokens, the register rows (1 to 4) dropped: <c>forward_features</c>'s <c>x_norm_clstoken</c> and <c>x_norm_patchtokens</c>.</summary>
    private Tensor DropRegisters(IBackend backend, Tensor all)
    {
        using Tensor cls = SamMotOps.Tokens(backend, all, 0, 1);
        using Tensor patches = SamMotOps.Tokens(backend, all, 1 + _preset.NumRegisterTokens, _preset.NumPatches);
        return SamMotOps.Concat(backend, [cls, patches]);
    }

    /// <summary>ImageNet normalisation into a <c>[1, 3, 518, 518]</c> tensor; a one-channel input is repeated over the three channels first (<c>x.repeat(1, 3, 1, 1)</c>).</summary>
    private Tensor Normalize(ReadOnlySpan<float> pixels, int channels)
    {
        int size = _preset.ImageSize, plane = size * size;
        if (pixels.Length != channels * plane)
            throw new ArgumentException($"DINO takes {channels} x {size} x {size} values; got {pixels.Length}.", nameof(pixels));
        Tensor result = new(new TensorShape(1, 3, size, size), DType.F32);
        Span<float> target = result.AsSpan<float>();
        for (int c = 0; c < 3; c++)
        {
            ReadOnlySpan<float> source = pixels.Slice(channels == 1 ? 0 : c * plane, plane);
            float mean = Mean[c], inverse = 1f / Std[c];
            Span<float> row = target.Slice(c * plane, plane);
            for (int i = 0; i < plane; i++)
                row[i] = (source[i] - mean) * inverse;
        }

        return result;
    }

    public void Dispose()
    {
        foreach (Tensor tensor in _owned)
            tensor.Dispose();
        _owned.Clear();
    }
}
