using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.StyleTts2;

/// <summary>The KDiffusion (EDM) denoiser <c>D(x, σ)</c> for StyleTTS 2's style sampler. Applies the
/// Karras preconditioning around the <see cref="StyleTransformer1d"/> network and classifier-free guidance:
///
/// <code>
///   c_in   = 1/√(σ² + σ_data²)
///   c_out  = σ·σ_data/√(σ² + σ_data²)
///   c_skip = σ_data²/(σ² + σ_data²)
///   c_noise= ln(σ)/4
///   F      = (1−scale)·net(c_in·x, c_noise, ∅)   +   scale·net(c_in·x, c_noise, textEmb, refStyle)
///   D(x,σ) = c_skip·x + c_out·F
/// </code>
///
/// <para>The preconditioning + CFG combine are exact (unit-tested via a stub network); the
/// <see cref="StyleTransformer1d"/> network forward is the real 3-layer 1-D transformer over the length-1
/// 256-d latent with timestep embedding + cross-attention to the PLBERT features. When no transformer
/// weights are loaded the network returns a zero velocity, so the sampler still runs deterministically.</para></summary>
public sealed unsafe class StyleDenoiser : IStyleDenoiser
{
    private readonly StyleTts2Config _cfg;
    private readonly int _dim;
    private Tensor? _textEmb;          // [1, seq, 768] PLBERT features (conditioning)
    private Tensor? _refStyle;         // [1, 256] reference style (optional)

    // The diffusion network. Null until LoadWeights / SetNetwork; Network() then zero-falls-back.
    private StyleTransformer1d? _net;

    public StyleDenoiser(StyleTts2Config cfg)
    {
        _cfg = cfg;
        _dim = cfg.StyleDim;
    }

    /// <summary>Builds the <see cref="StyleTransformer1d"/> from a converted <c>diffusion.*</c> state dict
    /// (archinetai <c>audio-diffusion-pytorch</c> naming). Missing keys leave the corresponding sub-block as a
    /// no-op so partial/pre-reconciliation checkpoints still load. Weights are pre-cast to F32.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "diffusion")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        string b = $"{p}unet.";

        Tensor? timeW1 = TryGet(w, $"{p}to_time.0.weight"), timeB1 = TryGet(w, $"{p}to_time.0.bias");
        Tensor? timeW2 = TryGet(w, $"{p}to_time.2.weight"), timeB2 = TryGet(w, $"{p}to_time.2.bias");
        Tensor? inW = TryGet(w, $"{p}to_in.weight") ?? TryGet(w, $"{b}to_in.weight");
        Tensor? inB = TryGet(w, $"{p}to_in.bias") ?? TryGet(w, $"{b}to_in.bias");
        Tensor? outW = TryGet(w, $"{p}to_out.weight") ?? TryGet(w, $"{b}to_out.weight");
        Tensor? outB = TryGet(w, $"{p}to_out.bias") ?? TryGet(w, $"{b}to_out.bias");

        int numLayers = 3;
        int numHeads = 8;
        int headFeatures = 64;
        int multiplier = 2;

        StyleTransformerLayer[] layers = new StyleTransformerLayer[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            string lp = $"{b}blocks.{i}.";
            layers[i] = new StyleTransformerLayer
            {
                AdaLnW = TryGet(w, $"{lp}attention.norm.to_scale_shift.weight") ?? TryGet(w, $"{lp}norm1.to_scale_shift.weight"),
                AdaLnB = TryGet(w, $"{lp}attention.norm.to_scale_shift.bias") ?? TryGet(w, $"{lp}norm1.to_scale_shift.bias"),
                SelfNormW = TryGet(w, $"{lp}attention.norm.weight") ?? TryGet(w, $"{lp}norm1.weight"),
                SelfNormB = TryGet(w, $"{lp}attention.norm.bias") ?? TryGet(w, $"{lp}norm1.bias"),
                SelfQW = TryGet(w, $"{lp}attention.to_q.weight"), SelfQB = TryGet(w, $"{lp}attention.to_q.bias"),
                SelfKW = TryGet(w, $"{lp}attention.to_k.weight"), SelfKB = TryGet(w, $"{lp}attention.to_k.bias"),
                SelfVW = TryGet(w, $"{lp}attention.to_v.weight"), SelfVB = TryGet(w, $"{lp}attention.to_v.bias"),
                SelfOW = TryGet(w, $"{lp}attention.to_out.weight"), SelfOB = TryGet(w, $"{lp}attention.to_out.bias"),
                CrossNormW = TryGet(w, $"{lp}cross_attention.norm.weight") ?? TryGet(w, $"{lp}norm2.weight"),
                CrossNormB = TryGet(w, $"{lp}cross_attention.norm.bias") ?? TryGet(w, $"{lp}norm2.bias"),
                CrossQW = TryGet(w, $"{lp}cross_attention.to_q.weight"), CrossQB = TryGet(w, $"{lp}cross_attention.to_q.bias"),
                CrossKW = TryGet(w, $"{lp}cross_attention.to_k.weight"), CrossKB = TryGet(w, $"{lp}cross_attention.to_k.bias"),
                CrossVW = TryGet(w, $"{lp}cross_attention.to_v.weight"), CrossVB = TryGet(w, $"{lp}cross_attention.to_v.bias"),
                CrossOW = TryGet(w, $"{lp}cross_attention.to_out.weight"), CrossOB = TryGet(w, $"{lp}cross_attention.to_out.bias"),
                FfNormW = TryGet(w, $"{lp}feed_forward.norm.weight") ?? TryGet(w, $"{lp}norm3.weight"),
                FfNormB = TryGet(w, $"{lp}feed_forward.norm.bias") ?? TryGet(w, $"{lp}norm3.bias"),
                FfW1 = TryGet(w, $"{lp}feed_forward.block.0.weight"), FfB1 = TryGet(w, $"{lp}feed_forward.block.0.bias"),
                FfW2 = TryGet(w, $"{lp}feed_forward.block.2.weight"), FfB2 = TryGet(w, $"{lp}feed_forward.block.2.bias"),
            };
        }

        _net = new StyleTransformer1d(
            _dim, numHeads, headFeatures, _dim, multiplier, layers,
            timeW1, timeB1, timeW2, timeB2, inW, inB, outW, outB);
    }

    /// <summary>Injects an already-constructed network (used by tests with synthetic weights, and by a
    /// converter that builds the <see cref="StyleTransformer1d"/> itself).</summary>
    public void SetNetwork(StyleTransformer1d net) => _net = net;

    private static Tensor? TryGet(IReadOnlyDictionary<string, Tensor> w, string key)
        => w.TryGetValue(key, out Tensor? t) ? WhisperOps.EnsureF32(t) : null;

    /// <summary>Sets the per-call conditioning (PLBERT text features + optional reference style).</summary>
    public void Prepare(Tensor textEmb, Tensor? refStyle)
    {
        _textEmb = textEmb;
        _refStyle = refStyle;
        if (_net is not null)
            _net.SetReferenceStyle(refStyle is not null ? refStyle.Reshape(new TensorShape(1, 1, _dim)) : null);
    }

    public Tensor Denoise(IBackend backend, Tensor x, float sigma)
    {
        float sd = _cfg.SigmaData;
        float denom = MathF.Sqrt(sigma * sigma + sd * sd);
        float cIn = 1f / denom;
        float cOut = sigma * sd / denom;
        float cSkip = sd * sd / (sigma * sigma + sd * sd);
        float cNoise = sigma > 0f ? MathF.Log(sigma) * 0.25f : 0f;

        // scaled input.
        Tensor scaled = new(x.Shape, DType.F32);
        float* xp = (float*)x.DataPointer;
        float* scp = (float*)scaled.DataPointer;
        for (long i = 0; i < x.ElementCount; i++) scp[i] = cIn * xp[i];

        Tensor condF = Network(backend, scaled, cNoise, conditioned: true);
        Tensor f;
        if (MathF.Abs(_cfg.EmbeddingScale - 1f) > 1e-6f)
        {
            Tensor uncondF = Network(backend, scaled, cNoise, conditioned: false);
            f = new(x.Shape, DType.F32);
            float* cp = (float*)condF.DataPointer;
            float* up = (float*)uncondF.DataPointer;
            float* fp = (float*)f.DataPointer;
            float s = _cfg.EmbeddingScale;
            for (long i = 0; i < f.ElementCount; i++) fp[i] = (1f - s) * up[i] + s * cp[i];
            uncondF.Dispose();
            condF.Dispose();
        }
        else
        {
            f = condF;
        }
        scaled.Dispose();

        // D = c_skip·x + c_out·F.
        Tensor outT = new(x.Shape, DType.F32);
        float* op = (float*)outT.DataPointer;
        float* ffp = (float*)f.DataPointer;
        for (long i = 0; i < outT.ElementCount; i++) op[i] = cSkip * xp[i] + cOut * ffp[i];
        f.Dispose();
        return outT;
    }

    /// <summary>One forward of the <see cref="StyleTransformer1d"/>. The conditional call passes the PLBERT
    /// text features as cross-attention context; the unconditional (CFG) call passes null context (the
    /// learned <c>FixedEmbedding</c> path collapses to the no-context branch here).</summary>
    private Tensor Network(IBackend backend, Tensor scaledX, float cNoise, bool conditioned)
    {
        if (_net is null)
            return new Tensor(scaledX.Shape, DType.F32);
        Tensor? context = conditioned ? _textEmb : null;
        return _net.Forward(backend, scaledX, cNoise, context);
    }

    public IEnumerable<Tensor> EnumerateWeights()
        => _net is not null ? _net.EnumerateWeights() : Array.Empty<Tensor>();
}
