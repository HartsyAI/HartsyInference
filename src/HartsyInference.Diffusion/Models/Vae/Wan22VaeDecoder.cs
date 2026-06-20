using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Wan2.2 3D causal VAE decoder (<c>WanVAE_</c> decode path + <c>Decoder3d</c> in <c>vae2_2.py</c>), assembled from the verified reusable blocks (<see cref="CausalConv3d"/>, <see cref="Wan22ResidualBlock"/>, <see cref="Wan22AttentionBlock"/>, <see cref="Wan22Resample"/>, <see cref="DupUp3D"/>, <see cref="WanRmsNorm"/>, <see cref="Wan22VaePatch"/>, <see cref="Wan22VaeLatentNorm"/>). Used by Lance (image + video); reusable by any Wan2.2-VAE consumer.
///
/// <para><b>Scope:</b> this implements the <b>image / first-chunk</b> decode path (T=1), which is fully <b>stateless</b> — the upstream driver decodes frame 0 with <c>first_chunk=True</c> and a fresh all-None cache, so every conv zero-pads and the temporal <c>time_conv</c> is skipped. Multi-frame video (chunks 2+, <c>feat_cache</c> streaming) is a documented follow-up. <b>Numeric output is validation-pending vs the real checkpoint</b> (no weights available in this environment); structure + shapes are CPU-tested.</para>
///
/// <para>Decode: <c>z·std+mean → conv2(48→48) → conv1(48→1024) → [Res,Attn,Res] → 4 up-stages → head(RMS,SiLU,conv→12) → unpatchify(2) → RGB</c>. Spatial 8× (3 upsampling stages) × unpatchify 2× = 16×.</para></summary>
public sealed unsafe class Wan22VaeDecoder : IWanVaeDecoder
{
    private readonly int _zDim;
    private readonly int _dim;
    private readonly int[] _dimMult;
    private readonly int _numResBlocks;
    private readonly bool[] _temperalUpsample;

    private CausalConv3d? _conv2;       // top-level WanVAE_.conv2 (48→48, 1×1×1)
    private CausalConv3d? _conv1;       // decoder.conv1 (48→dims[0], 3)
    private Wan22ResidualBlock? _midRes0, _midRes2;
    private Wan22AttentionBlock? _midAttn;
    private UpStage[] _stages = [];
    private WanRmsNorm? _headNorm;
    private CausalConv3d? _headConv;    // out→12, 3

    private sealed class UpStage
    {
        public required int InDim;
        public required int OutDim;
        public required bool UpFlag;
        public required int ShortcutFactorT;   // DupUp3D factor_t (2 if temporal, else 1)
        public required Wan22ResidualBlock[] Res;   // num_res_blocks+1 of them
        public Wan22Resample? Resample;             // present iff UpFlag
    }

    /// <summary>Creates the Lance Wan2.2 decoder (defaults match <c>WanVAE_(dec_dim=256, z_dim=48, dim_mult=[1,2,4,4], num_res_blocks=2, temperal_upsample=[F,T,T])</c>).</summary>
    public Wan22VaeDecoder(int dim = 256, int zDim = 48, int[]? dimMult = null, int numResBlocks = 2, bool[]? temperalUpsample = null)
    {
        _dim = dim;
        _zDim = zDim;
        _dimMult = dimMult ?? [1, 2, 4, 4];
        _numResBlocks = numResBlocks;
        _temperalUpsample = temperalUpsample ?? [false, true, true];
    }

    private int[] BuildDims()
    {
        // dims = [dim*dim_mult[-1]] + [dim*u for u in reversed(dim_mult)]
        int[] dims = new int[_dimMult.Length + 1];
        dims[0] = _dim * _dimMult[^1];
        for (int i = 0; i < _dimMult.Length; i++)
            dims[i + 1] = _dim * _dimMult[_dimMult.Length - 1 - i];
        return dims;
    }

    /// <summary>Loads weights from a dict whose keys are <c>conv2.*</c> (top-level) and <c>decoder.*</c> (Decoder3d). The converter strips any wrapper prefix.</summary>
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
        _stages = new UpStage[numStages];
        for (int i = 0; i < numStages; i++)
        {
            int inDim = dims[i];
            int outDim = dims[i + 1];
            bool upFlag = i != _dimMult.Length - 1;
            bool tUp = i < _temperalUpsample.Length && _temperalUpsample[i];
            int mult = _numResBlocks + 1;

            Wan22ResidualBlock[] res = new Wan22ResidualBlock[mult];
            int cur = inDim;
            for (int j = 0; j < mult; j++)
            {
                res[j] = new Wan22ResidualBlock(cur, outDim);
                res[j].LoadWeights(w, $"decoder.upsamples.{i}.upsamples.{j}");
                cur = outDim;
            }
            Wan22Resample? resample = null;
            if (upFlag)
            {
                resample = new Wan22Resample(outDim, temporal: tUp);
                resample.LoadWeights(w, $"decoder.upsamples.{i}.upsamples.{mult}");
            }
            _stages[i] = new UpStage
            {
                InDim = inDim,
                OutDim = outDim,
                UpFlag = upFlag,
                ShortcutFactorT = tUp ? 2 : 1,
                Res = res,
                Resample = resample,
            };
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

    /// <summary>Decodes a latent <c>[B, 48, T, H, W]</c> → RGB <c>[B, 3, Tout, H·16, W·16]</c>. <c>T=1</c> (image) is a single stateless pass; <c>T&gt;1</c> (video) streams one latent frame at a time threading a temporal <see cref="Wan22StreamCache"/> (frame 0 is the first chunk), so the decoder never materializes the whole clip and the temporal upsample expands <c>T → (T−1)·4 + 1</c> frames. Applies the fixed latent denorm internally.</summary>
    public Tensor Decode(IBackend backend, Tensor latent)
    {
        if ((int)latent.Shape[1] != _zDim)
            throw new ArgumentException($"latent channels {latent.Shape[1]} != z_dim {_zDim}.", nameof(latent));
        int t = (int)latent.Shape[2];

        // z = z·std + mean (decode-side latent norm). conv2 is 1×1×1 (no temporal cache needed).
        Tensor z = CloneRef(latent);
        Wan22VaeLatentNorm.Denormalize(z);
        Tensor x = _conv2!.Forward(backend, z);
        z.Dispose();

        List<Tensor> groups = new();
        foreach (Tensor g in DecodeRgbGroups(backend, x)) groups.Add(g);
        x.Dispose();
        Tensor rgb = groups.Count == 1 ? groups[0] : Vae3dLayout.ConcatFrames(groups);
        if (groups.Count != 1) foreach (Tensor g in groups) g.Dispose();
        return rgb;
    }

    /// <summary>Streams the RGB output one latent-frame group at a time (each <c>[B,3,groupT,H·16,W·16]</c>, already unpatchified), so a video clip's frames can be consumed/encoded without ever holding the whole decoded clip in memory. The caller must dispose each yielded tensor. Pass a latent that already includes the denorm? No — pass the raw latent; this applies denorm + conv2 internally, same as <see cref="Decode"/>.</summary>
    public IEnumerable<Tensor> DecodeStreaming(IBackend backend, Tensor latent)
    {
        if ((int)latent.Shape[1] != _zDim)
            throw new ArgumentException($"latent channels {latent.Shape[1]} != z_dim {_zDim}.", nameof(latent));
        Tensor z = CloneRef(latent);
        Wan22VaeLatentNorm.Denormalize(z);
        Tensor x = _conv2!.Forward(backend, z);
        z.Dispose();
        try
        {
            foreach (Tensor g in DecodeRgbGroups(backend, x)) yield return g;
        }
        finally
        {
            x.Dispose();
        }
    }

    /// <summary>Yields the unpatchified RGB group for each latent frame (1 frame for the first chunk, 4 for subsequent chunks via temporal upsample), threading the streaming cache in order. <paramref name="x"/> is the post-conv2 latent.</summary>
    private IEnumerable<Tensor> DecodeRgbGroups(IBackend backend, Tensor x)
    {
        int t = (int)x.Shape[2];
        if (t == 1)
        {
            Tensor twelve = DecodeFrame(backend, x, cache: null, firstChunk: true);
            Tensor rgb = Wan22VaePatch.Unpatchify(twelve, 2);
            twelve.Dispose();
            yield return rgb;
            yield break;
        }
        using Wan22StreamCache cache = new();
        for (int i = 0; i < t; i++)
        {
            cache.NewFrame();
            Tensor frame = Vae3dLayout.SliceFrames(x, i, 1);
            Tensor twelve = DecodeFrame(backend, frame, cache, firstChunk: i == 0);
            frame.Dispose();
            Tensor rgb = Wan22VaePatch.Unpatchify(twelve, 2);
            twelve.Dispose();
            yield return rgb;
        }
    }

    /// <summary>Decodes a single latent frame <c>[B, 48, 1, H, W]</c> through the Decoder3d graph to the pre-unpatchify 12-channel frame(s). With <paramref name="cache"/> non-null, every CausalConv3d threads its temporal cache (in forward order) and the temporal Resample doubles T from the second chunk on.</summary>
    private Tensor DecodeFrame(IBackend backend, Tensor x, Wan22StreamCache? cache, bool firstChunk)
    {
        Tensor? cc = cache?.StepConv(x);
        Tensor h = _conv1!.Forward(backend, x, cc);
        cc?.Dispose();

        Tensor m0 = _midRes0!.Forward(backend, h, cache); h.Dispose();
        Tensor m1 = _midAttn!.Forward(backend, m0); m0.Dispose();
        Tensor cur = _midRes2!.Forward(backend, m1, cache); m1.Dispose();

        foreach (UpStage s in _stages)
        {
            Tensor main = CloneRef(cur);
            foreach (Wan22ResidualBlock r in s.Res)
            {
                Tensor next = r.Forward(backend, main, cache);
                main.Dispose();
                main = next;
            }
            if (s.Resample is not null)
            {
                Tensor up = s.Resample.Forward(backend, main, cache);
                main.Dispose();
                main = up;
            }
            if (s.UpFlag)
            {
                Tensor shortcut = DupUp3D.Forward(cur, s.OutDim, factorT: s.ShortcutFactorT, factorS: 2, firstChunk: firstChunk);
                Tensor sum = new Tensor(main.Shape, DType.F32);
                backend.Add(sum, main, shortcut);
                main.Dispose();
                shortcut.Dispose();
                cur.Dispose();
                cur = sum;
            }
            else
            {
                cur.Dispose();
                cur = main;
            }
        }

        Tensor hn = _headNorm!.Forward(cur);
        cur.Dispose();
        backend.Silu(hn, hn);
        Tensor? hcc = cache?.StepConv(hn);
        Tensor twelve = _headConv!.Forward(backend, hn, hcc);
        hcc?.Dispose();
        hn.Dispose();
        return twelve;
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
