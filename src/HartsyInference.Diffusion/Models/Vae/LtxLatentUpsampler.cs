using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>LTX-2.5's learned x2 spatial latent upsampler (<c>LatentUpsampler</c>), the module the shipped two-stage
/// pipeline runs between its half-resolution denoise and its full-resolution refine. Consumes and produces
/// UN-normalized VAE latents <c>[1,128,F,H,W] → [1,128,F,2H,2W]</c>.
///
/// <para>Structure: <c>initial_conv</c> → GroupNorm+SiLU → 4 res blocks → a per-frame 3x3 Conv2d to 4x the channels
/// → 2x2 pixel-shuffle → 4 res blocks → <c>final_conv</c>. Every conv is symmetric zero-padded and NON-causal, so
/// this must not reuse the LTX VAE's <see cref="CausalConv3d"/> path.</para></summary>
public sealed unsafe class LtxLatentUpsampler
{
    private const int NormGroups = 32;
    private const float NormEps = 1e-5f;                // torch nn.GroupNorm default

    private Tensor _initialConvW = null!, _initialConvB = null!, _initialNormW = null!, _initialNormB = null!;
    private ResBlock[] _resBlocks = [];
    private ResBlock[] _postResBlocks = [];
    // The checkpoint's per-frame Conv2d, materialized as a [C_out, C_in, 1, 3, 3] Conv3d weight at load. A
    // Reshape view over the source tensor would alias its buffer; this repo has lost days to that twice.
    private Tensor _upsamplerW = null!, _upsamplerB = null!;
    private Tensor _finalConvW = null!, _finalConvB = null!;

    /// <summary>Latent channel count in and out (checkpoint <c>in_channels</c>).</summary>
    public int InChannels { get; private set; } = 128;

    /// <summary>Working width of the trunk (checkpoint <c>mid_channels</c>).</summary>
    public int MidChannels { get; private set; } = 1024;

    /// <summary>Res blocks per stage, before and after the shuffle (checkpoint <c>num_blocks_per_stage</c>).</summary>
    public int NumBlocksPerStage { get; private set; } = 4;

    /// <summary>Spatial upscale factor; the only shipped value is 2.</summary>
    public int SpatialScale { get; private set; } = 2;

    /// <summary>Per-stage output hook for the reference layer diff; leave null in production.</summary>
    public Action<string, Tensor>? Tap { get; set; }

    /// <summary>Binds the checkpoint's tensors (F32) and validates <paramref name="configJson"/> — the
    /// <c>__metadata__.config</c> blob — against what this forward pass implements. The config is authoritative:
    /// the reference class defaults <c>mid_channels</c> to 512 where this checkpoint is 1024, so a build that
    /// ignored the metadata would load partially and be confidently wrong.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string? configJson)
    {
        ApplyConfig(configJson);
        _initialConvW = weights["initial_conv.weight"];
        _initialConvB = weights["initial_conv.bias"];
        _initialNormW = weights["initial_norm.weight"];
        _initialNormB = weights["initial_norm.bias"];
        _resBlocks = new ResBlock[NumBlocksPerStage];
        _postResBlocks = new ResBlock[NumBlocksPerStage];
        for (int i = 0; i < NumBlocksPerStage; i++)
        {
            _resBlocks[i] = ResBlock.Bind(weights, $"res_blocks.{i}");
            _postResBlocks[i] = ResBlock.Bind(weights, $"post_upsample_res_blocks.{i}");
        }
        _upsamplerW = ToConv3dWeight(weights["upsampler.0.weight"]);
        _upsamplerB = weights["upsampler.0.bias"];
        _finalConvW = weights["final_conv.weight"];
        _finalConvB = weights["final_conv.bias"];

        int r2 = SpatialScale * SpatialScale;
        if (_initialConvW.Shape[0] != MidChannels || _initialConvW.Shape[1] != InChannels)
            throw new HartsyInferenceException($"initial_conv is {_initialConvW.Shape}, expected [{MidChannels},{InChannels},3,3,3].");
        if (_upsamplerW.Shape[0] != (long)MidChannels * r2)
            throw new HartsyInferenceException($"upsampler.0 is {_upsamplerW.Shape}, expected [{MidChannels * r2},{MidChannels},1,3,3].");
        if (_finalConvW.Shape[0] != InChannels || _finalConvW.Shape[1] != MidChannels)
            throw new HartsyInferenceException($"final_conv is {_finalConvW.Shape}, expected [{InChannels},{MidChannels},3,3,3].");
    }

    /// <summary>Upsamples an UN-normalized latent <c>[1,C,F,H,W] → [1,C,F,2H,2W]</c>. The caller owns the result.</summary>
    public Tensor Forward(IBackend backend, Tensor latent)
    {
        if (latent.Shape.Rank != 5 || latent.Shape[1] != InChannels)
            throw new HartsyInferenceException($"LatentUpsampler expects [1,{InChannels},F,H,W]; got {latent.Shape}.");
        if (latent.DType != DType.F32)
            throw new HartsyInferenceException($"LatentUpsampler is F32-only (Conv3d has no lower-precision path); got {latent.DType}.");
        int b = (int)latent.Shape[0], f = (int)latent.Shape[2], h = (int)latent.Shape[3], w = (int)latent.Shape[4];
        int r = SpatialScale;

        Tensor x = new Tensor(new TensorShape([b, MidChannels, f, h, w]), DType.F32);
        backend.Conv3d(x, latent, _initialConvW, _initialConvB, 1, 1, 1, 1, 1, 1);
        Tap?.Invoke("initial_conv", x);
        Tensor act = new Tensor(x.Shape, DType.F32);
        backend.GroupNormSilu(act, x, _initialNormW, _initialNormB, NormGroups, NormEps);
        x.Dispose();
        x = act;
        Tap?.Invoke("initial_act", x);

        for (int i = 0; i < _resBlocks.Length; i++)
        {
            x = _resBlocks[i].Forward(backend, x, NormGroups, NormEps);
            Tap?.Invoke($"res{i}", x);
        }

        Tensor projected = new Tensor(new TensorShape([b, (long)MidChannels * r * r, f, h, w]), DType.F32);
        backend.Conv3d(projected, x, _upsamplerW, _upsamplerB, 1, 1, 1, 0, 1, 1);
        x.Dispose();
        Tap?.Invoke("up_conv", projected);
        x = new Tensor(new TensorShape([b, MidChannels, f, (long)h * r, (long)w * r]), DType.F32);
        backend.PixelShuffle2d(x, projected, r);
        projected.Dispose();
        Tap?.Invoke("up_shuffle", x);

        for (int i = 0; i < _postResBlocks.Length; i++)
        {
            x = _postResBlocks[i].Forward(backend, x, NormGroups, NormEps);
            Tap?.Invoke($"post{i}", x);
        }

        Tensor output = new Tensor(new TensorShape([b, InChannels, f, (long)h * r, (long)w * r]), DType.F32);
        backend.Conv3d(output, x, _finalConvW, _finalConvB, 1, 1, 1, 1, 1, 1);
        x.Dispose();
        Tap?.Invoke("output", output);
        return output;
    }

    /// <summary>Weights for backend preload/free.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        yield return _initialConvW;
        yield return _initialConvB;
        yield return _initialNormW;
        yield return _initialNormB;
        foreach (ResBlock block in _resBlocks)
            foreach (Tensor t in block.EnumerateWeights()) yield return t;
        yield return _upsamplerW;
        yield return _upsamplerB;
        foreach (ResBlock block in _postResBlocks)
            foreach (Tensor t in block.EnumerateWeights()) yield return t;
        yield return _finalConvW;
        yield return _finalConvB;
    }

    private void ApplyConfig(string? configJson)
    {
        if (string.IsNullOrWhiteSpace(configJson))
            throw new HartsyInferenceException("The latent upsampler checkpoint carries no '__metadata__.config'; refusing to guess its shape.");
        using JsonDocument doc = JsonDocument.Parse(configJson);
        JsonElement root = doc.RootElement;
        string Str(string key) => root.TryGetProperty(key, out JsonElement e) ? e.ToString() : "<absent>";
        int Int(string key) => root.TryGetProperty(key, out JsonElement e) && e.ValueKind == JsonValueKind.Number
            ? e.GetInt32()
            : throw new HartsyInferenceException($"Latent upsampler config has no numeric '{key}': {configJson}");
        bool Bool(string key) => root.TryGetProperty(key, out JsonElement e)
            ? e.ValueKind == JsonValueKind.True
            : throw new HartsyInferenceException($"Latent upsampler config has no '{key}': {configJson}");

        if (Str("_class_name") is not ("LatentUpsampler" or "<absent>"))
            throw new HartsyInferenceException($"Expected a LatentUpsampler checkpoint, got _class_name={Str("_class_name")}.");
        InChannels = Int("in_channels");
        MidChannels = Int("mid_channels");
        NumBlocksPerStage = Int("num_blocks_per_stage");
        int dims = Int("dims");
        double scale = root.GetProperty("spatial_scale").GetDouble();
        SpatialScale = (int)scale;
        // Every one of these selects a different graph in the reference, and only this combination is implemented.
        if (dims != 3)
            throw new HartsyInferenceException($"Latent upsampler dims={dims}; only the 3-D (per-volume Conv3d) trunk is implemented.");
        if (!Bool("spatial_upsample"))
            throw new HartsyInferenceException("Latent upsampler has spatial_upsample=false; nothing to upsample.");
        if (Bool("temporal_upsample"))
            throw new HartsyInferenceException("Latent upsampler has temporal_upsample=true; the temporal shuffle (and its frame-1 drop) is not implemented.");
        if (Bool("rational_resampler"))
            throw new HartsyInferenceException("Latent upsampler has rational_resampler=true; the blur-downsample path is not implemented.");
        if (scale != SpatialScale || SpatialScale != 2)
            throw new HartsyInferenceException($"Latent upsampler spatial_scale={scale}; only the integer x2 shuffle is implemented.");
        if (MidChannels % NormGroups != 0)
            throw new HartsyInferenceException($"Latent upsampler mid_channels={MidChannels} is not divisible by {NormGroups} GroupNorm groups.");
    }

    /// <summary>Copies a 4-D <c>[C_out,C_in,kH,kW]</c> Conv2d weight into a 5-D <c>[C_out,C_in,1,kH,kW]</c> Conv3d
    /// weight — the per-frame Conv2d is exactly a depth-1 Conv3d with no depth padding.</summary>
    private static Tensor ToConv3dWeight(Tensor conv2d)
    {
        if (conv2d.Shape.Rank != 4)
            throw new HartsyInferenceException($"upsampler.0.weight must be a 4-D Conv2d weight; got {conv2d.Shape}.");
        if (conv2d.DType != DType.F32)
            throw new HartsyInferenceException($"upsampler.0.weight must be F32; got {conv2d.DType}.");
        Tensor outT = new Tensor(new TensorShape([conv2d.Shape[0], conv2d.Shape[1], 1, conv2d.Shape[2], conv2d.Shape[3]]), DType.F32);
        long bytes = conv2d.ElementCount * sizeof(float);
        Buffer.MemoryCopy((void*)conv2d.DataPointer, (void*)outT.DataPointer, bytes, bytes);
        return outT;
    }

    /// <summary>The reference <c>ResBlock</c>: conv→norm→SiLU→conv→norm, then SiLU applied AFTER the residual add.</summary>
    private sealed class ResBlock
    {
        private Tensor _conv1W = null!, _conv1B = null!, _conv2W = null!, _conv2B = null!;
        private Tensor _norm1W = null!, _norm1B = null!, _norm2W = null!, _norm2B = null!;

        public static ResBlock Bind(IReadOnlyDictionary<string, Tensor> w, string prefix) => new()
        {
            _conv1W = w[$"{prefix}.conv1.weight"],
            _conv1B = w[$"{prefix}.conv1.bias"],
            _conv2W = w[$"{prefix}.conv2.weight"],
            _conv2B = w[$"{prefix}.conv2.bias"],
            _norm1W = w[$"{prefix}.norm1.weight"],
            _norm1B = w[$"{prefix}.norm1.bias"],
            _norm2W = w[$"{prefix}.norm2.weight"],
            _norm2B = w[$"{prefix}.norm2.bias"],
        };

        /// <summary>Consumes <paramref name="x"/> and returns a fresh tensor of the same shape.</summary>
        public Tensor Forward(IBackend backend, Tensor x, int groups, float eps)
        {
            TensorShape shape = x.Shape;
            Tensor c1 = new Tensor(shape, DType.F32);
            backend.Conv3d(c1, x, _conv1W, _conv1B, 1, 1, 1, 1, 1, 1);
            Tensor n1 = new Tensor(shape, DType.F32);
            backend.GroupNormSilu(n1, c1, _norm1W, _norm1B, groups, eps);
            c1.Dispose();
            Tensor c2 = new Tensor(shape, DType.F32);
            backend.Conv3d(c2, n1, _conv2W, _conv2B, 1, 1, 1, 1, 1, 1);
            n1.Dispose();
            Tensor n2 = new Tensor(shape, DType.F32);
            backend.GroupNorm(n2, c2, _norm2W, _norm2B, groups, eps);
            c2.Dispose();
            Tensor sum = new Tensor(shape, DType.F32);
            backend.Add(sum, n2, x);
            n2.Dispose();
            x.Dispose();
            Tensor output = new Tensor(shape, DType.F32);
            backend.Silu(output, sum);
            sum.Dispose();
            return output;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            yield return _conv1W; yield return _conv1B;
            yield return _conv2W; yield return _conv2B;
            yield return _norm1W; yield return _norm1B;
            yield return _norm2W; yield return _norm2B;
        }
    }
}
