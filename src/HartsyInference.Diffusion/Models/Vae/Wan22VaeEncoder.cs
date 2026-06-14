using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Wan2.2 3D causal VAE encoder (<c>WanVAE_</c> encode path + <c>Encoder3d</c> in <c>vae2_2.py</c>) — the
/// mirror of <see cref="Wan22VaeDecoder"/>, assembled from the same verified blocks (<see cref="CausalConv3d"/>,
/// <see cref="Wan22ResidualBlock"/>, <see cref="Wan22AttentionBlock"/>, <see cref="Wan22Resample"/> downsample modes,
/// <see cref="AvgDown3D"/>, <see cref="WanRmsNorm"/>, <see cref="Wan22VaePatch"/>). Unblocks RGB-input I2V for the
/// Wan2.2 TI2V pipeline and seed-image encoding for Matrix-Game 3.0.
///
/// <para><b>Scope:</b> the <b>image / first-chunk</b> encode path (one RGB frame → one latent frame), which is fully
/// stateless — every conv zero-pads and the temporal stride-2 <c>time_conv</c>s are skipped on the first chunk, so a
/// single frame passes straight through the spatial path. Multi-frame video encode (4-frame chunks with the streaming
/// cache) is a documented follow-up. <b>Numerics are validation-pending vs the real checkpoint.</b></para>
///
/// <para>Encode: <c>RGB[-1,1] → patchify(2)→12ch → conv1(12→160) → 4 down-stages (res×2 [+down resample] + AvgDown3D
/// shortcut) → [Res,Attn,Res] → head(RMS,SiLU,conv→96) → quant conv1(96→96, 1×1×1) → μ = first 48 → (μ−mean)/std</c>.
/// Spatial: patchify 2 × three stride-2 stages = 16×. Loads from the same <c>wan22_vae.safetensors</c> as the decoder
/// (keys <c>encoder.*</c> + top-level <c>conv1.*</c>).</para></summary>
public sealed unsafe class Wan22VaeEncoder
{
    private readonly int _zDim;
    private readonly int _dim;
    private readonly int[] _dimMult;
    private readonly int _numResBlocks;
    private readonly bool[] _temperalDownsample;

    private CausalConv3d? _convIn;      // encoder.conv1 (12 → dims[0], 3)
    private DownStage[] _stages = [];
    private Wan22ResidualBlock? _midRes0, _midRes2;
    private Wan22AttentionBlock? _midAttn;
    private WanRmsNorm? _headNorm;
    private CausalConv3d? _headConv;    // out → 2·z, 3
    private CausalConv3d? _quantConv;   // top-level WanVAE_.conv1 (2·z → 2·z, 1×1×1)

    private sealed class DownStage
    {
        public required int InDim;
        public required int OutDim;
        public required int ShortcutFactorT;
        public required int ShortcutFactorS;
        public required Wan22ResidualBlock[] Res;
        public Wan22Resample? Resample;   // present iff down_flag
    }

    /// <summary>Creates the Wan2.2 encoder (defaults match <c>WanVAE_(dim=160, z_dim=48, dim_mult=[1,2,4,4], num_res_blocks=2, temperal_downsample=[T,T,F])</c>).</summary>
    public Wan22VaeEncoder(int dim = 160, int zDim = 48, int[]? dimMult = null, int numResBlocks = 2, bool[]? temperalDownsample = null)
    {
        _dim = dim;
        _zDim = zDim;
        _dimMult = dimMult ?? [1, 2, 4, 4];
        _numResBlocks = numResBlocks;
        _temperalDownsample = temperalDownsample ?? [true, true, false];
    }

    private int[] BuildDims()
    {
        // dims = [dim * m for m in [1] + dim_mult]
        int[] dims = new int[_dimMult.Length + 1];
        dims[0] = _dim;
        for (int i = 0; i < _dimMult.Length; i++) dims[i + 1] = _dim * _dimMult[i];
        return dims;
    }

    /// <summary>Loads weights from the same dict the decoder uses — keys <c>encoder.*</c> (Encoder3d) and the
    /// top-level quant conv <c>conv1.*</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        int[] dims = BuildDims();
        _convIn = new CausalConv3d(w["encoder.conv1.weight"], Bias(w, "encoder.conv1.bias"), padT: 1, padH: 1, padW: 1);

        int numStages = _dimMult.Length;
        _stages = new DownStage[numStages];
        for (int i = 0; i < numStages; i++)
        {
            int inDim = dims[i];
            int outDim = dims[i + 1];
            bool downFlag = i != _dimMult.Length - 1;
            bool tDown = i < _temperalDownsample.Length && _temperalDownsample[i];

            Wan22ResidualBlock[] res = new Wan22ResidualBlock[_numResBlocks];
            int cur = inDim;
            for (int j = 0; j < _numResBlocks; j++)
            {
                res[j] = new Wan22ResidualBlock(cur, outDim);
                res[j].LoadWeights(w, $"encoder.downsamples.{i}.downsamples.{j}");
                cur = outDim;
            }
            Wan22Resample? resample = null;
            if (downFlag)
            {
                resample = new Wan22Resample(outDim, tDown ? Wan22ResampleMode.Downsample3d : Wan22ResampleMode.Downsample2d);
                resample.LoadWeights(w, $"encoder.downsamples.{i}.downsamples.{_numResBlocks}");
            }
            _stages[i] = new DownStage
            {
                InDim = inDim,
                OutDim = outDim,
                ShortcutFactorT = tDown ? 2 : 1,
                ShortcutFactorS = downFlag ? 2 : 1,
                Res = res,
                Resample = resample,
            };
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
    /// <c>[1, 48, 1, H/16, W/16]</c> the Wan pipelines denoise in (μ of the Gaussian posterior, then
    /// <c>(μ−mean)/std</c> — diffusers' <c>sample_mode="argmax"</c> + scale).</summary>
    public Tensor EncodeFrame(IBackend backend, Tensor rgb)
    {
        if (rgb.Shape.Rank != 5 || rgb.Shape[1] != 3 || rgb.Shape[2] != 1)
            throw new ArgumentException($"expected RGB [1,3,1,H,W]; got {rgb.Shape}.", nameof(rgb));
        if (rgb.Shape[3] % 16 != 0 || rgb.Shape[4] % 16 != 0)
            throw new ArgumentException("H and W must be divisible by 16.", nameof(rgb));

        Tensor x = Wan22VaePatch.Patchify(rgb, 2);                       // [1,12,1,H/2,W/2]
        Tensor h = _convIn!.Forward(backend, x);
        x.Dispose();

        foreach (DownStage s in _stages)
        {
            Tensor main = CloneRef(h);
            foreach (Wan22ResidualBlock r in s.Res)
            {
                Tensor next = r.Forward(backend, main);
                main.Dispose();
                main = next;
            }
            if (s.Resample is not null)
            {
                Tensor down = s.Resample.Forward(backend, main);
                main.Dispose();
                main = down;
            }
            Tensor shortcut = AvgDown3D.Forward(h, s.OutDim, s.ShortcutFactorT, s.ShortcutFactorS);
            Tensor sum = new Tensor(main.Shape, DType.F32);
            backend.Add(sum, main, shortcut);
            main.Dispose();
            shortcut.Dispose();
            h.Dispose();
            h = sum;
        }

        Tensor m0 = _midRes0!.Forward(backend, h); h.Dispose();
        Tensor m1 = _midAttn!.Forward(backend, m0); m0.Dispose();
        Tensor cur = _midRes2!.Forward(backend, m1); m1.Dispose();

        Tensor hn = _headNorm!.Forward(cur);
        cur.Dispose();
        backend.Silu(hn, hn);
        Tensor doubled = _headConv!.Forward(backend, hn);                // [1, 2z, 1, h, w]
        hn.Dispose();
        Tensor quant = _quantConv!.Forward(backend, doubled);
        doubled.Dispose();

        Tensor mu = SliceChannels(quant, 0, _zDim);                       // μ — argmax sample
        quant.Dispose();
        Wan22VaeLatentNorm.Normalize(mu);
        return mu;
    }

    /// <summary>Convenience: interleaved RGB24 bytes → normalized conditioning latent (for I2V / seed-image use).</summary>
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

    private static Tensor CloneRef(Tensor x)
    {
        Tensor t = new Tensor(x.Shape, x.DType);
        long n = x.Shape.ElementCount;
        Buffer.MemoryCopy(x.DataPointer, t.DataPointer, n * 4, n * 4);
        return t;
    }
}
