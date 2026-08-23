using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Wan2.2-S2V causal audio encoder — a faithful port of ComfyUI's <c>CausalAudioEncoder</c> +
/// <c>MotionEncoder_tc</c> (<c>comfy/ldm/wan/model.py</c>). SiLU-activated learnable layer weights <c>[1,25,1,1]</c>
/// take a weighted MEAN over the stacked Wav2Vec2 hidden states (NOT a softmax), then a causal conv stack downsamples
/// the per-video-frame features 4× temporally to the latent frame rate: <c>conv1_local</c> fans out to
/// <c>AudioTokens</c> head streams that each run the shared <c>conv2</c>/<c>conv3</c> (stride 2 each) chain with
/// no-affine LayerNorm + SiLU between stages, yielding <c>AudioTokens</c> local tokens per latent frame (+ a learnable
/// padding token); a parallel single-stream global path (<c>conv1_global</c> → shared chain → <c>final_linear</c>)
/// emits the per-frame AdaIN token the injector's AdaLayerNorm is modulated by. B=1.</summary>
public sealed unsafe class WanS2VAudioEncoder
{
    private readonly int _numLayers, _audioDim, _dim, _numTokens, _quarter, _half;
    private readonly float _eps;

    private Tensor? _layerWeights;                                    // [1, numLayers, 1, 1]
    private Tensor? _conv1LocalW, _conv1LocalB, _conv1GlobalW, _conv1GlobalB;
    private Tensor? _conv2W, _conv2B, _conv3W, _conv3B;               // shared by all streams
    private Tensor? _finalLinearW, _finalLinearB;                     // global path only
    private Tensor? _paddingTokens;                                   // [1, 1, 1, dim]

    public WanS2VAudioEncoder(int numLayers, int audioDim, int dim, int numTokens = 4, float eps = 1e-6f)
    {
        if (dim % 4 != 0)
            throw new ArgumentException($"S2V audio encoder dim must be divisible by 4; got {dim}.", nameof(dim));
        _numLayers = numLayers;
        _audioDim = audioDim;
        _dim = dim;
        _numTokens = numTokens;
        _quarter = dim / 4;
        _half = dim / 2;
        _eps = eps;
    }

    /// <summary>Loads from the converted checkpoint dict; <paramref name="p"/> is the checkpoint prefix
    /// (<c>casual_audio_encoder</c> — the original Wan repo's spelling, passed through the converter unchanged).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p = "casual_audio_encoder")
    {
        _layerWeights = TensorCasts.LoadF32(w, $"{p}.weights");
        _conv1LocalW = TensorCasts.LoadF32(w, $"{p}.encoder.conv1_local.conv.weight");
        _conv1LocalB = LoadF32Opt(w, $"{p}.encoder.conv1_local.conv.bias");
        _conv1GlobalW = TensorCasts.LoadF32(w, $"{p}.encoder.conv1_global.conv.weight");
        _conv1GlobalB = LoadF32Opt(w, $"{p}.encoder.conv1_global.conv.bias");
        _conv2W = TensorCasts.LoadF32(w, $"{p}.encoder.conv2.conv.weight"); _conv2B = LoadF32Opt(w, $"{p}.encoder.conv2.conv.bias");
        _conv3W = TensorCasts.LoadF32(w, $"{p}.encoder.conv3.conv.weight"); _conv3B = LoadF32Opt(w, $"{p}.encoder.conv3.conv.bias");
        _finalLinearW = w[$"{p}.encoder.final_linear.weight"]; w.TryGetValue($"{p}.encoder.final_linear.bias", out _finalLinearB);
        _paddingTokens = TensorCasts.LoadF32(w, $"{p}.encoder.padding_tokens");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _layerWeights, _conv1LocalW, _conv1LocalB, _conv1GlobalW, _conv1GlobalB,
            _conv2W, _conv2B, _conv3W, _conv3B, _finalLinearW, _finalLinearB, _paddingTokens })
            if (t is not null) yield return t;
    }

    /// <summary>Encodes stacked per-video-frame Wav2Vec2 features <c>[T, numLayers, audioDim]</c> (T = 4 × latent
    /// frames) → (<c>Global [T/4, 1, dim]</c> AdaIN tokens, <c>Local [T/4, numTokens+1, dim]</c> attention tokens).</summary>
    public (Tensor Global, Tensor Local) Forward(IBackend backend, Tensor features)
    {
        int t = (int)features.Shape[0];
        if ((int)features.Shape[1] != _numLayers || (int)features.Shape[2] != _audioDim)
            throw new ArgumentException($"audio features must be [T, {_numLayers}, {_audioDim}]; got {features.Shape}.", nameof(features));

        // 1. Weighted MEAN over the layer axis: (features · silu(w)) / silu(w).sum() — NOT a softmax.
        Span<float> lw = stackalloc float[_numLayers];
        float* wp = (float*)_layerWeights!.DataPointer;
        float wSum = 0;
        for (int l = 0; l < _numLayers; l++) { lw[l] = wp[l] / (1f + MathF.Exp(-wp[l])); wSum += lw[l]; }
        Tensor combined = new Tensor(new TensorShape(1, _audioDim, t), DType.F32);   // channels-first for the convs
        float* fp = (float*)features.DataPointer, cp = (float*)combined.DataPointer;
        for (int ti = 0; ti < t; ti++)
            for (int d = 0; d < _audioDim; d++)
            {
                float acc = 0;
                for (int l = 0; l < _numLayers; l++) acc += lw[l] * fp[((long)ti * _numLayers + l) * _audioDim + d];
                cp[(long)d * t + ti] = acc / wSum;
            }

        // 2. Local path: conv1_local fans out to numTokens streams of dim/4 channels; each runs the shared chain.
        Tensor c1Local = CausalConv1d(backend, combined, _audioDim, _quarter * _numTokens, _conv1LocalW!, _conv1LocalB, 1, t);
        int tOut = -1;
        Tensor[] localHeads = new Tensor[_numTokens];
        for (int n = 0; n < _numTokens; n++)
        {
            Tensor stream = SliceChannelBlock(c1Local, n * _quarter, _quarter, t);
            localHeads[n] = StreamChain(backend, stream, t, out tOut);   // [tOut, dim], disposes stream
        }
        c1Local.Dispose();

        Tensor local = new Tensor(new TensorShape(tOut, _numTokens + 1, _dim), DType.F32);
        float* lp = (float*)local.DataPointer, padp = (float*)_paddingTokens!.DataPointer;
        for (int ti = 0; ti < tOut; ti++)
        {
            for (int n = 0; n < _numTokens; n++)
                Buffer.MemoryCopy((float*)localHeads[n].DataPointer + (long)ti * _dim,
                    lp + ((long)ti * (_numTokens + 1) + n) * _dim, (long)_dim * 4, (long)_dim * 4);
            Buffer.MemoryCopy(padp, lp + ((long)ti * (_numTokens + 1) + _numTokens) * _dim, (long)_dim * 4, (long)_dim * 4);
        }
        foreach (Tensor h in localHeads) h.Dispose();

        // 3. Global path: single dim/4 stream through the same chain, then final_linear.
        Tensor c1Global = CausalConv1d(backend, combined, _audioDim, _quarter, _conv1GlobalW!, _conv1GlobalB, 1, t);
        combined.Dispose();
        Tensor gChain = StreamChain(backend, c1Global, t, out _);        // [tOut, dim], disposes c1Global
        Tensor global = new Tensor(new TensorShape(tOut, 1, _dim), DType.F32);
        backend.Linear(global, gChain, _finalLinearW!, _finalLinearB);
        gChain.Dispose();

        return (global, local);
    }

    /// <summary>The shared per-stream chain: LN+SiLU → conv2 (dim/4→dim/2, stride 2) → LN+SiLU → conv3
    /// (dim/2→dim, stride 2) → LN+SiLU. Input <c>[1, dim/4, T]</c> (consumed), output tokens <c>[T/4, dim]</c>.</summary>
    private Tensor StreamChain(IBackend backend, Tensor stream, int t, out int tOut)
    {
        Tensor x1 = WanEncoderOps.ChannelsToTokens(stream, _quarter, t); stream.Dispose();
        WanEncoderOps.LayerNormSilu(x1, t, _quarter, _eps);
        Tensor c2 = CausalConv1d(backend, WanEncoderOps.TokensToChannels(x1, _quarter, t), _quarter, _half, _conv2W!, _conv2B, 2, t, disposeInput: true);
        x1.Dispose();
        int t2 = (int)c2.Shape[2];
        Tensor x2 = WanEncoderOps.ChannelsToTokens(c2, _half, t2); c2.Dispose();
        WanEncoderOps.LayerNormSilu(x2, t2, _half, _eps);
        Tensor c3 = CausalConv1d(backend, WanEncoderOps.TokensToChannels(x2, _half, t2), _half, _dim, _conv3W!, _conv3B, 2, t2, disposeInput: true);
        x2.Dispose();
        tOut = (int)c3.Shape[2];
        Tensor x3 = WanEncoderOps.ChannelsToTokens(c3, _dim, tOut); c3.Dispose();
        WanEncoderOps.LayerNormSilu(x3, tOut, _dim, _eps);
        return x3;
    }

    /// <summary>Causal Conv1d: replicate-pads the left by <c>kernel-1</c> then strides. <paramref name="x"/> is
    /// <c>[1, cin, T]</c>, returns <c>[1, cout, Tout]</c> with <c>Tout = (T-1)/stride + 1</c>.</summary>
    private static Tensor CausalConv1d(IBackend backend, Tensor x, int cin, int cout, Tensor weight, Tensor? bias,
        int stride, int t, bool disposeInput = false)
    {
        int kernel = (int)weight.Shape[2];
        int pad = kernel - 1;
        Tensor padded = new Tensor(new TensorShape(1, cin, t + pad), DType.F32);
        float* pp = (float*)padded.DataPointer, ip = (float*)x.DataPointer;
        for (int c = 0; c < cin; c++)
        {
            float first = ip[(long)c * t];
            for (int k = 0; k < pad; k++) pp[(long)c * (t + pad) + k] = first;
            Buffer.MemoryCopy(ip + (long)c * t, pp + (long)c * (t + pad) + pad, (long)t * 4, (long)t * 4);
        }
        if (disposeInput) x.Dispose();
        int tOut = (t + pad - kernel) / stride + 1;
        Tensor o = new Tensor(new TensorShape(1, cout, tOut), DType.F32);
        backend.Conv1d(o, padded, weight, bias, stride, 0, 0, 1, 1);
        padded.Dispose();
        return o;
    }

    private static Tensor SliceChannelBlock(Tensor x, int startCh, int channels, int t)
    {
        Tensor o = new Tensor(new TensorShape(1, channels, t), DType.F32);
        Buffer.MemoryCopy((float*)x.DataPointer + (long)startCh * t, (float*)o.DataPointer,
            (long)channels * t * 4, (long)channels * t * 4);
        return o;
    }

    private static Tensor? LoadF32Opt(IReadOnlyDictionary<string, Tensor> w, string key) => w.TryGetValue(key, out Tensor? t) ? (t.DType == DType.F32 ? t : t.CastTo(DType.F32)) : null;
}
