using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Wan2.2-S2V causal audio encoder (the <c>CausalAudioEncoder</c> in the original Wan repo's
/// <c>wan/modules/s2v/audio_encoder.py</c>): combines the stacked Wav2Vec2 hidden layers into per-frame features,
/// then a causal Conv1d stack maps them to the DiT inner dim, emitting <c>tokensPerFrame</c> audio tokens per video
/// frame that the audio injector cross-attends to.
///
/// <para><b>Reconstructed (not from diffusers — S2V has no diffusers reference): weight names + the exact conv stack
/// are provisional and validation-gated.</b> Reuses the causal replicate-padded Conv1d pattern from
/// <see cref="WanAnimateFaceEncoder"/>. B=1.</para></summary>
public sealed unsafe class WanS2VAudioEncoder
{
    private readonly int _numLayers, _audioDim, _dim, _tokensPerFrame, _kernel;
    private readonly float _eps;

    private Tensor? _layerWeights;     // [numLayers] — learnable softmax over the stacked Wav2Vec2 layers
    private Tensor? _conv1W, _conv1B, _conv2W, _conv2B;

    public WanS2VAudioEncoder(int numLayers, int audioDim, int dim, int tokensPerFrame = 1, int kernel = 3, float eps = 1e-6f)
    {
        _numLayers = numLayers;
        _audioDim = audioDim;
        _dim = dim;
        _tokensPerFrame = tokensPerFrame;
        _kernel = kernel;
        _eps = eps;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _layerWeights = LoadF32(w, $"{p}.layer_weights");
        _conv1W = LoadF32(w, $"{p}.conv1.weight"); w.TryGetValue($"{p}.conv1.bias", out _conv1B);
        _conv2W = LoadF32(w, $"{p}.conv2.weight"); w.TryGetValue($"{p}.conv2.bias", out _conv2B);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _layerWeights, _conv1W, _conv1B, _conv2W, _conv2B }) if (t is not null) yield return t;
    }

    /// <summary>Encodes per-frame stacked audio features <c>[T, numLayers, audioDim]</c> → audio tokens
    /// <c>[T, tokensPerFrame, dim]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor audio)
    {
        int t = (int)audio.Shape[0];

        // 1. Softmax-weighted combination across the stacked Wav2Vec2 layers → [T, audioDim].
        Tensor combined = new Tensor(new TensorShape(t, _audioDim), DType.F32);
        float* ap = (float*)audio.DataPointer, cp = (float*)combined.DataPointer, lw = (float*)_layerWeights!.DataPointer;
        Span<float> sm = stackalloc float[_numLayers];
        float maxW = float.NegativeInfinity;
        for (int l = 0; l < _numLayers; l++) if (lw[l] > maxW) maxW = lw[l];
        float sum = 0;
        for (int l = 0; l < _numLayers; l++) { sm[l] = MathF.Exp(lw[l] - maxW); sum += sm[l]; }
        for (int l = 0; l < _numLayers; l++) sm[l] /= sum;
        for (int ti = 0; ti < t; ti++)
            for (int d = 0; d < _audioDim; d++)
            {
                float acc = 0;
                for (int l = 0; l < _numLayers; l++) acc += sm[l] * ap[((long)ti * _numLayers + l) * _audioDim + d];
                cp[(long)ti * _audioDim + d] = acc;
            }

        // 2. conv1 (audioDim → dim) + SiLU, conv2 (dim → dim·tokens) — both causal over the frame axis.
        Tensor h1 = Conv1dCausal(backend, combined, _audioDim, _dim, _conv1W!, _conv1B, t);   // [dim, T]
        combined.Dispose();
        Tensor x1 = ChannelsToTokens(h1, _dim, t); h1.Dispose();   // [T, dim]
        Silu(x1);
        Tensor h2 = Conv1dCausal(backend, Transpose(x1, t, _dim), _dim, _dim * _tokensPerFrame, _conv2W!, _conv2B, t); x1.Dispose();
        Tensor x2 = ChannelsToTokens(h2, _dim * _tokensPerFrame, t); h2.Dispose();   // [T, dim·tokens]

        // 3. Reshape → [T, tokensPerFrame, dim].
        Tensor o = new Tensor(new TensorShape(t, _tokensPerFrame, _dim), DType.F32);
        Buffer.MemoryCopy((float*)x2.DataPointer, (float*)o.DataPointer, (long)t * _tokensPerFrame * _dim * 4, (long)t * _tokensPerFrame * _dim * 4);
        x2.Dispose();
        return o;
    }

    /// <summary>Causal Conv1d over the frame axis (replicate-pad left by <c>kernel-1</c>). <paramref name="x"/> is
    /// <c>[Cin, T]</c> (channels-first), returns <c>[Cout, T]</c> (stride 1).</summary>
    private Tensor Conv1dCausal(IBackend backend, Tensor x, int cin, int cout, Tensor weight, Tensor? bias, int t)
    {
        // x may be [T, Cin] (combined) or already [1, Cin, T]; normalize to [1, Cin, T].
        Tensor chFirst = x.Shape.Rank == 2 && (int)x.Shape[0] == t ? Transpose(x, t, cin) : x;
        bool ownChFirst = !ReferenceEquals(chFirst, x);
        int pad = _kernel - 1;
        Tensor padded = new Tensor(new TensorShape(1, cin, t + pad), DType.F32);
        float* pp = (float*)padded.DataPointer, ip = (float*)chFirst.DataPointer;
        for (int c = 0; c < cin; c++)
        {
            float first = ip[(long)c * t];
            for (int k = 0; k < pad; k++) pp[(long)c * (t + pad) + k] = first;
            Buffer.MemoryCopy(ip + (long)c * t, pp + (long)c * (t + pad) + pad, (long)t * 4, (long)t * 4);
        }
        if (ownChFirst) chFirst.Dispose();
        Tensor o = new Tensor(new TensorShape(1, cout, t), DType.F32);
        backend.Conv1d(o, padded, weight, bias, 1, 0, 0, 1, 1);
        padded.Dispose();
        Tensor flat = new Tensor(new TensorShape(cout, t), DType.F32);
        Buffer.MemoryCopy((float*)o.DataPointer, (float*)flat.DataPointer, (long)cout * t * 4, (long)cout * t * 4);
        o.Dispose();
        return flat;
    }

    private static Tensor Transpose(Tensor x, int t, int c)
    {
        Tensor o = new Tensor(new TensorShape(1, c, t), DType.F32);
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        for (int ti = 0; ti < t; ti++)
            for (int ci = 0; ci < c; ci++) op[(long)ci * t + ti] = xp[(long)ti * c + ci];
        return o;
    }

    private static Tensor ChannelsToTokens(Tensor x, int c, int t)
    {
        Tensor o = new Tensor(new TensorShape(t, c), DType.F32);
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        for (int ci = 0; ci < c; ci++)
            for (int ti = 0; ti < t; ti++) op[(long)ti * c + ci] = xp[(long)ci * t + ti];
        return o;
    }

    private static void Silu(Tensor x)
    {
        long n = x.Shape.ElementCount;
        float* p = (float*)x.DataPointer;
        for (long i = 0; i < n; i++) p[i] = p[i] / (1f + MathF.Exp(-p[i]));
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key) { Tensor t = w[key]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }
}
