using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>The image-prompt projection used by IP-Adapter FaceID-<b>Plus</b> / Plus-<b>v2</b> (h94's <c>ProjPlusModel</c>): the FaceID MLP (<see cref="IpAdapterFaceIdProjection"/>, ArcFace 512-d → <c>num_tokens × cross_attention_dim</c> tokens) whose output tokens become the <i>latents</i> of a <c>FacePerceiverResampler</c> that cross-attends over the CLIP-Vision penultimate hidden states of the aligned face crop. The resampler is the same PerceiverAttention + FeedForward stack IP-Adapter Plus uses (<see cref="ResamplerLayer"/>), at <c>dim = cross_attention_dim, depth = 4, dim_head = 64, heads = cross_attention_dim / 64, ff_mult = 4</c>, with <c>proj_in: clipDim → dim</c>, <c>proj_out: dim → dim</c> and a final LayerNorm. <b>v2 (shortcut):</b> Plus-v2 checkpoints were trained with the shortcut path enabled — <c>out = mlp_tokens + scale × resampler_out</c>, where <c>scale</c> is the user-facing "FaceID V2 weight" (official default 1.0). v1 checkpoints use the resampler output directly. The flag is a constructor argument because v1/v2 files are structurally identical (filename-detected). Checkpoint keys (h94/IP-Adapter-FaceID <c>.bin</c>, flattened): the FaceID MLP under <c>image_proj.proj.{0,2}.* / image_proj.norm.*</c> plus the resampler under <c>image_proj.perceiver_resampler.{proj_in,proj_out,norm_out}.*</c> and <c>image_proj.perceiver_resampler.layers.{i}.{0,1}.*</c>.</summary>
public sealed unsafe class IpAdapterFaceIdPlusProjection : IIpAdapterImageProjection
{
    private const int Depth = 4;
    private const int HeadDim = 64;

    private readonly int _crossAttnDim;
    private readonly int _numTokens;
    private readonly int _clipEmbeddingDim;
    private readonly bool _useShortcut;

    private readonly IpAdapterFaceIdProjection _mlp;
    private readonly ResamplerLayer[] _layers;

    private Tensor? _projInWeight;  // [crossAttnDim, clipEmbeddingDim]
    private Tensor? _projInBias;    // [crossAttnDim]
    private Tensor? _projOutWeight; // [crossAttnDim, crossAttnDim]
    private Tensor? _projOutBias;   // [crossAttnDim]
    private Tensor? _normOutWeight; // [crossAttnDim]
    private Tensor? _normOutBias;   // [crossAttnDim]

    public int NumTokens => _numTokens;

    /// <summary>True for Plus-v2 checkpoints: the forward adds the MLP tokens back as a scaled shortcut.</summary>
    public bool UseShortcut => _useShortcut;

    public IpAdapterFaceIdPlusProjection(int crossAttnDim, int numTokens, int clipEmbeddingDim, bool useShortcut)
    {
        if (crossAttnDim % HeadDim != 0)
            throw new ArgumentException($"crossAttnDim {crossAttnDim} must be a multiple of the resampler head dim {HeadDim}.", nameof(crossAttnDim));
        _crossAttnDim = crossAttnDim;
        _numTokens = numTokens;
        _clipEmbeddingDim = clipEmbeddingDim;
        _useShortcut = useShortcut;
        _mlp = new IpAdapterFaceIdProjection(crossAttnDim, numTokens);
        int numHeads = crossAttnDim / HeadDim;
        _layers = new ResamplerLayer[Depth];
        for (int i = 0; i < Depth; i++)
        {
            _layers[i] = new ResamplerLayer(crossAttnDim, numHeads, HeadDim, innerDim: crossAttnDim, ffMultiplier: 4);
        }
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "image_proj")
    {
        _mlp.LoadWeights(weights, prefix);
        string resampler = $"{prefix}.perceiver_resampler";
        _projInWeight = EnsureF32(weights[$"{resampler}.proj_in.weight"]);
        _projInBias = EnsureF32(weights[$"{resampler}.proj_in.bias"]);
        _projOutWeight = EnsureF32(weights[$"{resampler}.proj_out.weight"]);
        _projOutBias = EnsureF32(weights[$"{resampler}.proj_out.bias"]);
        _normOutWeight = EnsureF32(weights[$"{resampler}.norm_out.weight"]);
        _normOutBias = EnsureF32(weights[$"{resampler}.norm_out.bias"]);
        if (_projInWeight.Shape[0] != _crossAttnDim || _projInWeight.Shape[1] != _clipEmbeddingDim)
        {
            throw new InvalidOperationException(
                $"FaceID-Plus resampler proj_in shape {_projInWeight.Shape} does not match [crossAttnDim={_crossAttnDim}, clipDim={_clipEmbeddingDim}].");
        }
        for (int i = 0; i < Depth; i++)
        {
            _layers[i].LoadWeights(weights, $"{resampler}.layers.{i}");
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor w in _mlp.EnumerateWeights()) yield return w;
        if (_projInWeight is not null) yield return _projInWeight;
        if (_projInBias is not null) yield return _projInBias;
        for (int i = 0; i < Depth; i++)
        {
            foreach (Tensor w in _layers[i].EnumerateWeights()) yield return w;
        }
        if (_projOutWeight is not null) yield return _projOutWeight;
        if (_projOutBias is not null) yield return _projOutBias;
        if (_normOutWeight is not null) yield return _normOutWeight;
        if (_normOutBias is not null) yield return _normOutBias;
    }

    /// <summary>Not supported — FaceID-Plus needs both the ArcFace identity embedding and the CLIP-Vision hidden states of the aligned face crop. Use <see cref="Forward(IBackend, Tensor, Tensor, float)"/> (or <see cref="IpAdapter.ProjectImage(IBackend, Tensor, Tensor, float)"/>).</summary>
    public Tensor Forward(IBackend backend, Tensor visionInput)
    {
        throw new InvalidOperationException(
            "FaceID-Plus projects from TWO inputs (ArcFace face embedding + CLIP-Vision hidden states of the aligned face crop). " +
            "Call the (faceEmbeds, clipEmbeds) overload instead.");
    }

    /// <summary>Projects the L2-normalized ArcFace embedding (<c>[B, 512]</c>) mixed with the CLIP-Vision penultimate hidden states of the aligned 224×224 face crop (<c>[B, seqLen, clipDim]</c>) into <c>[B, numTokens, crossAttnDim]</c> image-prompt tokens. <paramref name="shortcutScale"/> is the v2 "FaceID V2 weight" (ignored for v1 checkpoints).</summary>
    public Tensor Forward(IBackend backend, Tensor faceEmbeds, Tensor clipEmbeds, float shortcutScale = 1.0f)
    {
        if (clipEmbeds.Shape.Rank != 3 || clipEmbeds.Shape[2] != _clipEmbeddingDim)
        {
            throw new ArgumentException(
                $"clipEmbeds must be [B, seqLen, {_clipEmbeddingDim}] penultimate CLIP-Vision hidden states; got {clipEmbeds.Shape}.", nameof(clipEmbeds));
        }
        if (faceEmbeds.Shape.Rank != 2 || faceEmbeds.Shape[0] != clipEmbeds.Shape[0])
        {
            throw new ArgumentException(
                $"faceEmbeds must be [B, idDim] with the same batch as clipEmbeds; got {faceEmbeds.Shape} vs {clipEmbeds.Shape}.", nameof(faceEmbeds));
        }
        int batch = (int)clipEmbeds.Shape[0];
        int seqLen = (int)clipEmbeds.Shape[1];

        // 1. FaceID MLP → the resampler's initial latents: [B, numTokens, crossAttnDim].
        Tensor mlpTokens = _mlp.Forward(backend, faceEmbeds);

        // 2. proj_in over the CLIP hidden states: [B, seqLen, clipDim] → [B, seqLen, crossAttnDim].
        Tensor x = IpaLinear.Apply(backend, clipEmbeds, _projInWeight!, _projInBias, batch, seqLen, _clipEmbeddingDim, _crossAttnDim);

        // 3. Perceiver layers refine a copy of the MLP tokens (the original is kept for the v2 shortcut).
        Tensor latents = CloneF32(mlpTokens);
        for (int i = 0; i < Depth; i++)
        {
            Tensor next = _layers[i].Forward(backend, x, latents, batch, seqLen);
            latents.Dispose();
            latents = next;
        }
        x.Dispose();

        // 4. proj_out + norm_out.
        Tensor output = IpaLinear.Apply(backend, latents, _projOutWeight!, _projOutBias, batch, _numTokens, _crossAttnDim, _crossAttnDim);
        latents.Dispose();
        IpAdapterMath.LayerNormInPlace(output, _normOutWeight!, _normOutBias!,
            rows: batch * _numTokens, dim: _crossAttnDim, eps: 1e-5f);

        // 5. v2 shortcut: out = mlp_tokens + scale × out.
        if (_useShortcut)
        {
            float* op = (float*)output.DataPointer;
            float* mp = (float*)mlpTokens.DataPointer;
            long count = output.ElementCount;
            for (long i = 0; i < count; i++)
            {
                op[i] = mp[i] + shortcutScale * op[i];
            }
        }
        mlpTokens.Dispose();
        return output;
    }

    private static Tensor CloneF32(Tensor source)
    {
        Tensor clone = new Tensor(source.Shape, DType.F32);
        Buffer.MemoryCopy((void*)source.DataPointer, (void*)clone.DataPointer,
            clone.ElementCount * sizeof(float), source.ElementCount * sizeof(float));
        return clone;
    }

    private static Tensor EnsureF32(Tensor t) => t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}
