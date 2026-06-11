using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Vae;

/// <summary>Wan2.1 3D causal VAE decoder (<c>Decoder3d</c> in <c>Wan-Video/Wan2.1 vae.py</c>) — 16 latent channels,
/// 8× spatial / 4× temporal. Assembled from the same verified blocks as the Wan2.2 decoder
/// (<see cref="CausalConv3d"/>, <see cref="Wan22ResidualBlock"/>, <see cref="Wan22AttentionBlock"/>,
/// <see cref="Wan22Resample"/>, <see cref="WanRmsNorm"/>, <see cref="Wan22StreamCache"/>); the structural differences
/// from Wan2.2 are: no pixel patchify (the head emits RGB directly at 8×), no dup/avg stage shortcuts, a <b>flat</b>
/// <c>decoder.upsamples.{idx}</c> module list, and channel-<b>halving</b> upsample convs (read from the weights).
/// First user: Matrix-Game 2.0 / SkyReels-V2. Numerics validation-pending.
///
/// <para>Decode: <c>z·std+mean → conv2(16→16) → conv1(16→384) → [Res,Attn,Res] → stage0 res×3+up3d → stage1 res×3+up3d
/// → stage2 res×3+up2d → stage3 res×3 → head(RMS,SiLU,conv→3)</c>. T&gt;1 streams per latent frame via the shared
/// temporal cache (frames = (T−1)·4 + 1).</para></summary>
public sealed unsafe class Wan21VaeDecoder
{
    private readonly int _zDim;
    private readonly int _dim;
    private readonly int[] _dimMult;
    private readonly int _numResBlocks;
    private readonly bool[] _temperalUpsample;

    private CausalConv3d? _conv2;       // top-level post-quant conv (16→16, 1×1×1)
    private CausalConv3d? _conv1;       // decoder.conv1 (16→dims[0], 3)
    private Wan22ResidualBlock? _midRes0, _midRes2;
    private Wan22AttentionBlock? _midAttn;
    private UpStage[] _stages = [];
    private WanRmsNorm? _headNorm;
    private CausalConv3d? _headConv;    // out→3, 3

    private sealed class UpStage
    {
        public required Wan22ResidualBlock[] Res;
        public Wan22Resample? Resample;
    }

    /// <summary>Creates the Wan2.1 decoder (defaults match <c>WanVAE(dim=96, z_dim=16, dim_mult=[1,2,4,4],
    /// num_res_blocks=2, temperal_downsample=[F,T,T])</c> — decoder temporal upsample is the reverse, [T,T,F]).</summary>
    public Wan21VaeDecoder(int dim = 96, int zDim = 16, int[]? dimMult = null, int numResBlocks = 2, bool[]? temperalUpsample = null)
    {
        _dim = dim;
        _zDim = zDim;
        _dimMult = dimMult ?? [1, 2, 4, 4];
        _numResBlocks = numResBlocks;
        _temperalUpsample = temperalUpsample ?? [true, true, false];
    }

    /// <summary>Output RGB frames for a latent frame count (causal temporal 4×).</summary>
    public int OutputFrames(int latentFrames) => latentFrames == 1 ? 1 : (latentFrames - 1) * 4 + 1;

    private int[] BuildDims()
    {
        // dims = [dim·dim_mult[-1]] + [dim·u for u in reversed(dim_mult)]
        int[] dims = new int[_dimMult.Length + 1];
        dims[0] = _dim * _dimMult[^1];
        for (int i = 0; i < _dimMult.Length; i++)
            dims[i + 1] = _dim * _dimMult[_dimMult.Length - 1 - i];
        return dims;
    }

    /// <summary>Loads weights from a dict keyed <c>conv2.*</c> (top-level) and <c>decoder.*</c> with the FLAT
    /// <c>decoder.upsamples.{idx}</c> indexing (3 residuals per stage, then a resample for non-final stages).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _conv2 = new CausalConv3d(w["conv2.weight"], Bias(w, "conv2.bias"), padT: 0, padH: 0, padW: 0);

        int[] dims = BuildDims();
        _conv1 = new CausalConv3d(w["decoder.conv1.weight"], Bias(w, "decoder.conv1.bias"), padT: 1, padH: 1, padW: 1);
        _midRes0 = new Wan22ResidualBlock(dims[0], dims[0]);
        _midRes0.LoadWeights(w, "decoder.middle.0");
        _midAttn = new Wan22AttentionBlock(dims[0]);
        _midAttn.LoadWeights(w, "decoder.middle.1");
        _midRes2 = new Wan22ResidualBlock(dims[0], dims[0]);
        _midRes2.LoadWeights(w, "decoder.middle.2");

        int numStages = _dimMult.Length;
        int flat = 0;
        _stages = new UpStage[numStages];
        for (int i = 0; i < numStages; i++)
        {
            int inDim = dims[i];
            int outDim = dims[i + 1];
            if (i is 1 or 2 or 3) inDim /= 2;   // previous stage's upsample halved the channels

            Wan22ResidualBlock[] res = new Wan22ResidualBlock[_numResBlocks + 1];
            int cur = inDim;
            for (int j = 0; j < res.Length; j++)
            {
                res[j] = new Wan22ResidualBlock(cur, outDim);
                res[j].LoadWeights(w, $"decoder.upsamples.{flat++}");
                cur = outDim;
            }
            Wan22Resample? resample = null;
            if (i != numStages - 1)
            {
                bool tUp = i < _temperalUpsample.Length && _temperalUpsample[i];
                resample = new Wan22Resample(outDim, tUp ? Wan22ResampleMode.Upsample3d : Wan22ResampleMode.Upsample2d);
                resample.LoadWeights(w, $"decoder.upsamples.{flat++}");
            }
            _stages[i] = new UpStage { Res = res, Resample = resample };
        }

        int headDim = dims[^1];
        _headNorm = new WanRmsNorm(headDim);
        _headNorm.LoadWeights(w["decoder.head.0.gamma"]);
        _headConv = new CausalConv3d(w["decoder.head.2.weight"], Bias(w, "decoder.head.2.bias"), padT: 1, padH: 1, padW: 1);
    }

    /// <summary>Enumerates all weights for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_conv2 is not null) foreach (Tensor t in _conv2.EnumerateWeights()) yield return t;
        if (_conv1 is not null) foreach (Tensor t in _conv1.EnumerateWeights()) yield return t;
        foreach (Wan22ResidualBlock? r in new[] { _midRes0, _midRes2 })
            if (r is not null) foreach (Tensor t in r.EnumerateWeights()) yield return t;
        if (_midAttn is not null) foreach (Tensor t in _midAttn.EnumerateWeights()) yield return t;
        foreach (UpStage s in _stages)
        {
            foreach (Wan22ResidualBlock r in s.Res) foreach (Tensor t in r.EnumerateWeights()) yield return t;
            if (s.Resample is not null) foreach (Tensor t in s.Resample.EnumerateWeights()) yield return t;
        }
        if (_headNorm is not null) foreach (Tensor t in _headNorm.EnumerateWeights()) yield return t;
        if (_headConv is not null) foreach (Tensor t in _headConv.EnumerateWeights()) yield return t;
    }

    /// <summary>Decodes a normalized latent <c>[B, 16, T, H, W]</c> → RGB <c>[B, 3, Tout, H·8, W·8]</c> in [-1, 1],
    /// streaming the temporal cache per latent frame for T&gt;1. Applies the fixed latent denorm internally.</summary>
    public Tensor Decode(IBackend backend, Tensor latent)
    {
        if ((int)latent.Shape[1] != _zDim)
            throw new ArgumentException($"latent channels {latent.Shape[1]} != z_dim {_zDim}.", nameof(latent));

        Tensor z = Clone(latent);
        Wan21VaeLatentNorm.Denormalize(z);
        Tensor x = _conv2!.Forward(backend, z);
        z.Dispose();

        int t = (int)x.Shape[2];
        List<Tensor> groups = new();
        if (t == 1)
        {
            groups.Add(DecodeFrame(backend, x, cache: null));
        }
        else
        {
            using Wan22StreamCache cache = new();
            for (int i = 0; i < t; i++)
            {
                cache.NewFrame();
                Tensor frame = Vae3dLayout.SliceFrames(x, i, 1);
                groups.Add(DecodeFrame(backend, frame, cache));
                frame.Dispose();
            }
        }
        x.Dispose();
        Tensor rgb = groups.Count == 1 ? groups[0] : Vae3dLayout.ConcatFrames(groups);
        if (groups.Count != 1) foreach (Tensor g in groups) g.Dispose();
        return rgb;
    }

    private Tensor DecodeFrame(IBackend backend, Tensor x, Wan22StreamCache? cache)
    {
        Tensor? cc = cache?.StepConv(x);
        Tensor h = _conv1!.Forward(backend, x, cc);
        cc?.Dispose();

        Tensor m0 = _midRes0!.Forward(backend, h, cache); h.Dispose();
        Tensor m1 = _midAttn!.Forward(backend, m0); m0.Dispose();
        Tensor cur = _midRes2!.Forward(backend, m1, cache); m1.Dispose();

        foreach (UpStage s in _stages)
        {
            foreach (Wan22ResidualBlock r in s.Res)
            {
                Tensor next = r.Forward(backend, cur, cache);
                cur.Dispose();
                cur = next;
            }
            if (s.Resample is not null)
            {
                Tensor up = s.Resample.Forward(backend, cur, cache);
                cur.Dispose();
                cur = up;
            }
        }

        Tensor hn = _headNorm!.Forward(cur);
        cur.Dispose();
        backend.Silu(hn, hn);
        Tensor? hcc = cache?.StepConv(hn);
        Tensor rgb = _headConv!.Forward(backend, hn, hcc);
        hcc?.Dispose();
        hn.Dispose();
        return rgb;
    }

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? b) ? b : null;

    private static Tensor Clone(Tensor x)
    {
        Tensor t = new Tensor(x.Shape, x.DType);
        long n = x.Shape.ElementCount;
        Buffer.MemoryCopy(x.DataPointer, t.DataPointer, n * 4, n * 4);
        return t;
    }
}
