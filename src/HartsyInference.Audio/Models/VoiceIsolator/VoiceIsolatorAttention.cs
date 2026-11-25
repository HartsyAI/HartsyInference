using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Layers;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.VoiceIsolator;

/// <summary>A dual-path attention block: a pre-norm transformer along time within each band (rotary positions, optional causal
/// window, exclusive attention), then one across the bands of each frame.</summary>
internal sealed class VoiceIsolatorAttentionBlock
{
    private const float MaskedScore = -1e9f;

    private readonly VoiceIsolatorConfig _config;
    private readonly Tensor _tNorm1W, _tNorm1B, _tNorm2W, _tNorm2B;
    private readonly Tensor _qkvW, _qkvB, _projW, _projB, _tFfn1W, _tFfn1B, _tFfn2W, _tFfn2B;
    private readonly Tensor _bNorm1W, _bNorm1B, _bNorm2W, _bNorm2B;
    private readonly Tensor _inW, _inB, _outW, _outB, _bFfn1W, _bFfn1B, _bFfn2W, _bFfn2B;

    public VoiceIsolatorAttentionBlock(VoiceIsolatorConfig config, IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _config = config;
        _tNorm1W = w[$"{p}.time_transformer.norm1.weight"];
        _tNorm1B = w[$"{p}.time_transformer.norm1.bias"];
        _tNorm2W = w[$"{p}.time_transformer.norm2.weight"];
        _tNorm2B = w[$"{p}.time_transformer.norm2.bias"];
        _qkvW = w[$"{p}.time_transformer.attn.qkv.weight"];
        _qkvB = w[$"{p}.time_transformer.attn.qkv.bias"];
        _projW = w[$"{p}.time_transformer.attn.proj.weight"];
        _projB = w[$"{p}.time_transformer.attn.proj.bias"];
        _tFfn1W = w[$"{p}.time_transformer.ffn.0.weight"];
        _tFfn1B = w[$"{p}.time_transformer.ffn.0.bias"];
        _tFfn2W = w[$"{p}.time_transformer.ffn.3.weight"];
        _tFfn2B = w[$"{p}.time_transformer.ffn.3.bias"];
        _bNorm1W = w[$"{p}.band_transformer.norm1.weight"];
        _bNorm1B = w[$"{p}.band_transformer.norm1.bias"];
        _bNorm2W = w[$"{p}.band_transformer.norm2.weight"];
        _bNorm2B = w[$"{p}.band_transformer.norm2.bias"];
        _inW = w[$"{p}.band_transformer.self_attn.in_proj_weight"];
        _inB = w[$"{p}.band_transformer.self_attn.in_proj_bias"];
        _outW = w[$"{p}.band_transformer.self_attn.out_proj.weight"];
        _outB = w[$"{p}.band_transformer.self_attn.out_proj.bias"];
        _bFfn1W = w[$"{p}.band_transformer.linear1.weight"];
        _bFfn1B = w[$"{p}.band_transformer.linear1.bias"];
        _bFfn2W = w[$"{p}.band_transformer.linear2.weight"];
        _bFfn2B = w[$"{p}.band_transformer.linear2.bias"];
    }

    /// <summary>Runs the block over band-major tokens <c>[bands * frames, width]</c>; returns a new tensor of the same layout.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int bands, int frames, bool causal, float[] ropeCos, float[] ropeSin,
        VoiceIsolatorAttentionLane? lane = null, VoiceIsolatorStreamAttention? stream = null)
    {
        Tensor afterTime = TimeTransformer(backend, x, bands, frames, causal, ropeCos, ropeSin, lane, stream);
        int width = _config.Width;
        Tensor frameMajor = new(new TensorShape(frames, bands, width), DType.F32);
        VoiceIsolatorOps.SwapLeading(backend, frameMajor, afterTime, bands, frames, width);
        afterTime.Dispose();
        Tensor afterBand = BandTransformer(backend, frameMajor, bands, frames);
        frameMajor.Dispose();
        Tensor bandMajor = new(new TensorShape(1, (long)bands * frames, width), DType.F32);
        VoiceIsolatorOps.SwapLeading(backend, bandMajor, afterBand, frames, bands, width);
        afterBand.Dispose();
        return bandMajor;
    }

    private Tensor TimeTransformer(IBackend backend, Tensor x, int bands, int frames, bool causal, float[] ropeCos, float[] ropeSin,
        VoiceIsolatorAttentionLane? lane, VoiceIsolatorStreamAttention? stream)
    {
        int width = _config.Width;
        int heads = _config.NumHeads;
        int hd = _config.HeadDim;
        int n = bands * frames;
        Tensor normed = LayerNorm(backend, x, _tNorm1W, _tNorm1B, x.Shape);
        Tensor qkv = WhisperOps.ProjectLinear(backend, normed, _qkvW, _qkvB, 1, n, width, 3 * width);
        normed.Dispose();
        TensorShape headShape = new(bands, heads, frames, hd);
        Tensor q = new(headShape, DType.F32);
        Tensor k = new(headShape, DType.F32);
        Tensor v = new(headShape, DType.F32);
        VoiceIsolatorOps.SplitQkvToHeads(backend, qkv, q, k, v, bands, frames, heads, hd);
        qkv.Dispose();
        Tensor attended;
        if (lane is not null && stream is not null)
        {
            attended = stream.Attend(backend, lane, q, k, v, bands, heads, frames, hd);
        }
        else
        {
            VoiceIsolatorOps.Rope(backend, q, frames, hd, 0, ropeCos, ropeSin);
            VoiceIsolatorOps.Rope(backend, k, frames, hd, 0, ropeCos, ropeSin);
            Tensor? mask = causal ? CausalMask(frames, _config.AttentionWindowFrames) : null;
            attended = new Tensor(headShape, DType.F32);
            backend.ScaledDotProductAttention(attended, q, k, v, mask, 1f / MathF.Sqrt(hd));
            mask?.Dispose();
        }
        q.Dispose();
        k.Dispose();
        if (_config.ExclusiveSelfAttention)
        {
            VoiceIsolatorOps.ExcludeValue(attended, v, (long)bands * heads * frames, hd);
        }
        v.Dispose();
        Tensor tokens = new(new TensorShape(1, n, width), DType.F32);
        VoiceIsolatorOps.HeadsToTokens(backend, tokens, attended, bands, frames, heads, hd);
        attended.Dispose();
        Tensor projected = WhisperOps.ProjectLinear(backend, tokens, _projW, _projB, 1, n, width, width);
        tokens.Dispose();
        Tensor residual = new(new TensorShape(1, n, width), DType.F32);
        backend.Add(residual, x, projected);
        projected.Dispose();
        return FeedForward(backend, residual, n, _tNorm2W, _tNorm2B, _tFfn1W, _tFfn1B, _tFfn2W, _tFfn2B);
    }

    private Tensor BandTransformer(IBackend backend, Tensor x, int bands, int frames)
    {
        int width = _config.Width;
        int heads = _config.NumHeads;
        int hd = _config.HeadDim;
        int n = bands * frames;
        Tensor normed = LayerNorm(backend, x, _bNorm1W, _bNorm1B, x.Shape);
        Tensor qkv = WhisperOps.ProjectLinear(backend, normed, _inW, _inB, 1, n, width, 3 * width);
        normed.Dispose();
        TensorShape headShape = new(frames, heads, bands, hd);
        Tensor q = new(headShape, DType.F32);
        Tensor k = new(headShape, DType.F32);
        Tensor v = new(headShape, DType.F32);
        VoiceIsolatorOps.SplitQkvToHeads(backend, qkv, q, k, v, frames, bands, heads, hd);
        qkv.Dispose();
        Tensor attended = new(headShape, DType.F32);
        backend.ScaledDotProductAttention(attended, q, k, v, null, 1f / MathF.Sqrt(hd));
        q.Dispose();
        k.Dispose();
        v.Dispose();
        Tensor tokens = new(new TensorShape(1, n, width), DType.F32);
        VoiceIsolatorOps.HeadsToTokens(backend, tokens, attended, frames, bands, heads, hd);
        attended.Dispose();
        Tensor projected = WhisperOps.ProjectLinear(backend, tokens, _outW, _outB, 1, n, width, width);
        tokens.Dispose();
        Tensor residual = new(new TensorShape(1, n, width), DType.F32);
        backend.Add(residual, x, projected);
        projected.Dispose();
        return FeedForward(backend, residual, n, _bNorm2W, _bNorm2B, _bFfn1W, _bFfn1B, _bFfn2W, _bFfn2B);
    }

    private Tensor FeedForward(IBackend backend, Tensor x, int n, Tensor normW, Tensor normB, Tensor w1, Tensor b1, Tensor w2, Tensor b2)
    {
        int width = _config.Width;
        Tensor normed = LayerNorm(backend, x, normW, normB, x.Shape);
        Tensor hidden = WhisperOps.ProjectLinear(backend, normed, w1, b1, 1, n, width, 4 * width);
        normed.Dispose();
        Activations.ErfGelu(hidden);
        Tensor down = WhisperOps.ProjectLinear(backend, hidden, w2, b2, 1, n, 4 * width, width);
        hidden.Dispose();
        Tensor output = new(new TensorShape(1, n, width), DType.F32);
        backend.Add(output, x, down);
        down.Dispose();
        x.Dispose();
        return output;
    }

    private static Tensor LayerNorm(IBackend backend, Tensor x, Tensor weight, Tensor bias, TensorShape shape)
    {
        Tensor output = new(shape, DType.F32);
        backend.LayerNorm(output, x, weight, bias, 1e-5f);
        return output;
    }

    /// <summary>Additive <c>[1, 1, frames, frames]</c> mask: each query sees itself and the earlier frames within <paramref name="window"/> (0 = all).</summary>
    internal static unsafe Tensor CausalMask(int frames, int window)
    {
        Tensor mask = new(new TensorShape(1, 1, frames, frames), DType.F32);
        float* p = (float*)mask.DataPointer;
        for (int q = 0; q < frames; q++)
        {
            for (int k = 0; k < frames; k++)
            {
                int lag = q - k;
                bool visible = lag >= 0 && (window == 0 || lag < window);
                p[(long)q * frames + k] = visible ? 0f : MaskedScore;
            }
        }
        return mask;
    }
}
