using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Wan2.1 3D causal VAE encoder (<c>Encoder3d</c> in <c>Wan-Video/Wan2.1 vae.py</c>) — RGB → 16-channel
/// normalized latent at 8× spatial / 4× temporal. Mirror of <see cref="Wan21VaeDecoder"/> from the same shared blocks;
/// vs the Wan2.2 encoder there is no pixel patchify, no AvgDown shortcuts, a flat <c>encoder.downsamples.{idx}</c>
/// list, and same-channel stride-2 downsample convs.
///
/// <para><b>Scope:</b> the single-frame / first-chunk path (one RGB frame → one latent frame) — what Matrix-Game 2.0
/// needs for the conditioning image. The temporal stride-2 <c>time_conv</c>s are skipped on the first chunk;
/// multi-frame streaming encode is the shared Wan-family follow-up. Numerics validation-pending.</para></summary>
public sealed unsafe class Wan21VaeEncoder : IWanVaeEncoder
{
    private readonly int _zDim;
    private readonly int _dim;
    private readonly int[] _dimMult;
    private readonly int _numResBlocks;
    private readonly bool[] _temperalDownsample;

    private CausalConv3d? _convIn;      // encoder.conv1 (3 → dims[0], 3)
    private DownStage[] _stages = [];
    private Wan22ResidualBlock? _midRes0, _midRes2;
    private Wan22AttentionBlock? _midAttn;
    private WanRmsNorm? _headNorm;
    private CausalConv3d? _headConv;    // out → 2·z, 3
    private CausalConv3d? _quantConv;   // top-level conv1 (2·z → 2·z, 1×1×1)

    private sealed class DownStage
    {
        public required Wan22ResidualBlock[] Res;
        public Wan22Resample? Resample;
    }

    /// <summary>Creates the Wan2.1 encoder (defaults match <c>WanVAE(dim=96, z_dim=16, dim_mult=[1,2,4,4],
    /// num_res_blocks=2, temperal_downsample=[F,T,T])</c>).</summary>
    public Wan21VaeEncoder(int dim = 96, int zDim = 16, int[]? dimMult = null, int numResBlocks = 2, bool[]? temperalDownsample = null)
    {
        _dim = dim;
        _zDim = zDim;
        _dimMult = dimMult ?? [1, 2, 4, 4];
        _numResBlocks = numResBlocks;
        _temperalDownsample = temperalDownsample ?? [false, true, true];
    }

    private int[] BuildDims()
    {
        // dims = [dim·u for u in [1] + dim_mult]
        int[] dims = new int[_dimMult.Length + 1];
        dims[0] = _dim;
        for (int i = 0; i < _dimMult.Length; i++) dims[i + 1] = _dim * _dimMult[i];
        return dims;
    }

    /// <summary>Loads weights keyed <c>encoder.*</c> (flat <c>encoder.downsamples.{idx}</c>: 2 residuals per stage,
    /// then a resample for non-final stages) plus the top-level quant conv <c>conv1.*</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        int[] dims = BuildDims();
        _convIn = new CausalConv3d(w["encoder.conv1.weight"], Bias(w, "encoder.conv1.bias"), padT: 1, padH: 1, padW: 1);

        int numStages = _dimMult.Length;
        int flat = 0;
        _stages = new DownStage[numStages];
        for (int i = 0; i < numStages; i++)
        {
            int inDim = dims[i];
            int outDim = dims[i + 1];
            Wan22ResidualBlock[] res = new Wan22ResidualBlock[_numResBlocks];
            int cur = inDim;
            for (int j = 0; j < res.Length; j++)
            {
                res[j] = new Wan22ResidualBlock(cur, outDim);
                res[j].LoadWeights(w, $"encoder.downsamples.{flat++}");
                cur = outDim;
            }
            Wan22Resample? resample = null;
            if (i != numStages - 1)
            {
                bool tDown = i < _temperalDownsample.Length && _temperalDownsample[i];
                resample = new Wan22Resample(outDim, tDown ? Wan22ResampleMode.Downsample3d : Wan22ResampleMode.Downsample2d);
                resample.LoadWeights(w, $"encoder.downsamples.{flat++}");
            }
            _stages[i] = new DownStage { Res = res, Resample = resample };
        }

        int headDim = dims[^1];
        _midRes0 = new Wan22ResidualBlock(headDim, headDim);
        _midRes0.LoadWeights(w, "encoder.middle.0");
        _midAttn = new Wan22AttentionBlock(headDim);
        _midAttn.LoadWeights(w, "encoder.middle.1");
        _midRes2 = new Wan22ResidualBlock(headDim, headDim);
        _midRes2.LoadWeights(w, "encoder.middle.2");

        _headNorm = new WanRmsNorm(headDim);
        _headNorm.LoadWeights(w["encoder.head.0.gamma"]);
        _headConv = new CausalConv3d(w["encoder.head.2.weight"], Bias(w, "encoder.head.2.bias"), padT: 1, padH: 1, padW: 1);
        _quantConv = new CausalConv3d(w["conv1.weight"], Bias(w, "conv1.bias"), padT: 0, padH: 0, padW: 0);
    }

    /// <summary>Enumerates all weights for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_convIn is not null) foreach (Tensor t in _convIn.EnumerateWeights()) yield return t;
        foreach (DownStage s in _stages)
        {
            foreach (Wan22ResidualBlock r in s.Res) foreach (Tensor t in r.EnumerateWeights()) yield return t;
            if (s.Resample is not null) foreach (Tensor t in s.Resample.EnumerateWeights()) yield return t;
        }
        foreach (Wan22ResidualBlock? r in new[] { _midRes0, _midRes2 })
            if (r is not null) foreach (Tensor t in r.EnumerateWeights()) yield return t;
        if (_midAttn is not null) foreach (Tensor t in _midAttn.EnumerateWeights()) yield return t;
        if (_headNorm is not null) foreach (Tensor t in _headNorm.EnumerateWeights()) yield return t;
        if (_headConv is not null) foreach (Tensor t in _headConv.EnumerateWeights()) yield return t;
        if (_quantConv is not null) foreach (Tensor t in _quantConv.EnumerateWeights()) yield return t;
    }

    /// <summary>Encodes a single RGB frame <c>[1, 3, 1, H, W]</c> in [-1, 1] to the <b>normalized</b> latent
    /// <c>[1, 16, 1, H/8, W/8]</c> (μ of the posterior + <see cref="Wan21VaeLatentNorm.Normalize"/>).</summary>
    public Tensor EncodeFrame(IBackend backend, Tensor rgb)
    {
        if (rgb.Shape.Rank != 5 || rgb.Shape[1] != 3 || rgb.Shape[2] != 1)
            throw new ArgumentException($"expected RGB [1,3,1,H,W]; got {rgb.Shape}.", nameof(rgb));
        return EncodeCore(backend, rgb);
    }

    /// <summary>Encodes a full RGB clip <c>[1, 3, T, H, W]</c> in [-1, 1] to the normalized latent
    /// <c>[1, 16, (T−1)/4+1, H/8, W/8]</c> (whole-clip causal temporal compression). For VACE / Video2Video.</summary>
    public Tensor Encode(IBackend backend, Tensor rgb)
    {
        if (rgb.Shape.Rank != 5 || rgb.Shape[1] != 3)
            throw new ArgumentException($"expected RGB [1,3,T,H,W]; got {rgb.Shape}.", nameof(rgb));
        return EncodeCore(backend, rgb);
    }

    private Tensor EncodeCore(IBackend backend, Tensor rgb)
    {
        if (rgb.Shape[3] % 8 != 0 || rgb.Shape[4] % 8 != 0)
            throw new ArgumentException("H and W must be divisible by 8.", nameof(rgb));

        Tensor h = _convIn!.Forward(backend, rgb);
        foreach (DownStage s in _stages)
        {
            foreach (Wan22ResidualBlock r in s.Res)
            {
                Tensor next = r.Forward(backend, h);
                h.Dispose();
                h = next;
            }
            if (s.Resample is not null)
            {
                Tensor down = s.Resample.Forward(backend, h, applyTemporal: true);
                h.Dispose();
                h = down;
            }
        }

        Tensor m0 = _midRes0!.Forward(backend, h); h.Dispose();
        Tensor m1 = _midAttn!.Forward(backend, m0); m0.Dispose();
        Tensor cur = _midRes2!.Forward(backend, m1); m1.Dispose();

        Tensor hn = _headNorm!.Forward(cur);
        cur.Dispose();
        backend.Silu(hn, hn);
        Tensor doubled = _headConv!.Forward(backend, hn);
        hn.Dispose();
        Tensor quant = _quantConv!.Forward(backend, doubled);
        doubled.Dispose();

        Tensor mu = SliceChannels(quant, 0, _zDim);
        quant.Dispose();
        Wan21VaeLatentNorm.Normalize(mu);
        return mu;
    }

    /// <summary>Convenience: interleaved RGB24 bytes → normalized conditioning latent.</summary>
    public Tensor EncodeRgbFrame(IBackend backend, ReadOnlySpan<byte> rgb24, int width, int height)
    {
        if (rgb24.Length < width * height * 3)
            throw new ArgumentException($"expected {width * height * 3} bytes; got {rgb24.Length}.", nameof(rgb24));
        Tensor rgb = new Tensor(new TensorShape([1L, 3, 1, height, width]), DType.F32);
        float* p = (float*)rgb.DataPointer;
        long frame = (long)height * width;
        for (long pix = 0; pix < frame; pix++)
            for (int c = 0; c < 3; c++)
                p[c * frame + pix] = rgb24[(int)(pix * 3 + c)] / 127.5f - 1f;
        try
        {
            return EncodeFrame(backend, rgb);
        }
        finally
        {
            rgb.Dispose();
        }
    }

    private static Tensor SliceChannels(Tensor x, int start, int count)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        Tensor o = new Tensor(new TensorShape([(long)b, count, t, h, w]), DType.F32);
        long per = (long)t * h * w;
        float* sp = (float*)x.DataPointer;
        float* op = (float*)o.DataPointer;
        for (int bi = 0; bi < b; bi++)
            Buffer.MemoryCopy(
                sp + ((long)bi * c + start) * per,
                op + (long)bi * count * per,
                (long)count * per * 4, (long)count * per * 4);
        return o;
    }

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? b) ? b : null;
}
