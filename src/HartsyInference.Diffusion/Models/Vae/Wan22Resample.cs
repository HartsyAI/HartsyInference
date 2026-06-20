using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Resample direction/mode, matching <c>vae2_2.py</c>'s <c>Resample(mode=...)</c>.</summary>
public enum Wan22ResampleMode
{
    /// <summary>Decoder: per-frame nearest 2× + 3×3 Conv2d.</summary>
    Upsample2d,
    /// <summary>Decoder: temporal <c>time_conv</c> (T×2 from chunk 2 on) + spatial upsample.</summary>
    Upsample3d,
    /// <summary>Encoder: ZeroPad(right/bottom 1) + 3×3 Conv2d stride 2.</summary>
    Downsample2d,
    /// <summary>Encoder: spatial downsample + temporal stride-2 <c>time_conv</c> (skipped on the first chunk).</summary>
    Downsample3d,
}

/// <summary>Wan2.2 VAE spatial resample (<c>Resample</c> in <c>vae2_2.py</c>): the decoder's up-sampling modes
/// (per-frame nearest 2× + 3×3 Conv2d) and the encoder's down-sampling modes (asymmetric right/bottom zero-pad +
/// 3×3 Conv2d stride 2). Reuses the backend <see cref="IBackend.UpsampleNearest2D"/>/<see cref="IBackend.Conv2D"/> +
/// <see cref="Vae3dLayout"/>.
///
/// <para><b>Image / first-chunk path only for the temporal branches:</b> both <c>upsample3d</c>'s and
/// <c>downsample3d</c>'s <c>time_conv</c> are skipped on the first chunk (the cache slot is just primed), so a
/// single-frame pass needs only the spatial path. Multi-frame streaming for the encoder is a follow-up — the
/// downsample3d <c>time_conv</c> weights are loaded for enumeration/preload but not yet convolved.</para></summary>
public sealed unsafe class Wan22Resample
{
    private readonly int _dim;
    private readonly Wan22ResampleMode _mode;
    private Tensor? _convW, _convB;
    private CausalConv3d? _timeConv;            // upsample3d temporal branch (dim → 2·dim, kernel (3,1,1))
    private CausalConv3d? _downTimeConv;        // downsample3d temporal branch (dim → dim, kernel (3,1,1), strideT 2, causal)

    public Wan22Resample(int dim, bool temporal = false)
        : this(dim, temporal ? Wan22ResampleMode.Upsample3d : Wan22ResampleMode.Upsample2d)
    {
    }

    public Wan22Resample(int dim, Wan22ResampleMode mode)
    {
        _dim = dim;
        _mode = mode;
    }

    /// <summary>Loads the spatial conv (Sequential index 1: <c>resample.1</c>) and, for the temporal variants, the <c>time_conv</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _convW = weights[$"{prefix}.resample.1.weight"];
        weights.TryGetValue($"{prefix}.resample.1.bias", out _convB);
        if (_mode == Wan22ResampleMode.Upsample3d)
        {
            weights.TryGetValue($"{prefix}.time_conv.bias", out Tensor? tb);
            _timeConv = new CausalConv3d(weights[$"{prefix}.time_conv.weight"], tb, padT: 1, padH: 0, padW: 0);
        }
        else if (_mode == Wan22ResampleMode.Downsample3d)
        {
            weights.TryGetValue($"{prefix}.time_conv.bias", out Tensor? tb);
            // CausalConv3d(dim→dim, kernel (3,1,1), strideT 2, causal padT 1): halves T per stage.
            _downTimeConv = new CausalConv3d(weights[$"{prefix}.time_conv.weight"], tb, strideT: 2, padT: 1, padH: 0, padW: 0);
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_convW is not null) yield return _convW;
        if (_convB is not null) yield return _convB;
        if (_timeConv is not null) foreach (Tensor t in _timeConv.EnumerateWeights()) yield return t;
        if (_downTimeConv is not null) foreach (Tensor t in _downTimeConv.EnumerateWeights()) yield return t;
    }

    /// <summary>Resample. Upsample modes return <c>[B, C, Tout, 2H, 2W]</c> (time_conv doubles T from chunk 2 on);
    /// downsample modes return <c>[B, C, T, H/2, W/2]</c> (first-chunk semantics — temporal stride deferred to the
    /// encoder streaming follow-up).</summary>
    public Tensor Forward(IBackend backend, Tensor x, Wan22StreamCache? cache = null, bool applyTemporal = false)
    {
        if (_mode is Wan22ResampleMode.Downsample2d or Wan22ResampleMode.Downsample3d)
        {
            Tensor spatial = DownsampleSpatial(backend, x);
            // Whole-clip encode: apply the causal stride-2 temporal conv to halve T (skipped on the single-frame
            // / first-chunk path, where it leaves T unchanged).
            if (_mode == Wan22ResampleMode.Downsample3d && applyTemporal && _downTimeConv is not null && (int)spatial.Shape[2] > 1)
            {
                Tensor td = _downTimeConv.Forward(backend, spatial);
                spatial.Dispose();
                return td;
            }
            return spatial;
        }

        Tensor spatialIn = x;
        bool ownsSpatialIn = false;

        if (_mode == Wan22ResampleMode.Upsample3d && cache is not null && _timeConv is not null)
        {
            (bool skip, Tensor? convCache) = cache.StepTimeConv(x);
            if (!skip)
            {
                Tensor tc = _timeConv.Forward(backend, x, convCache);   // [B, 2C, T, H, W]
                convCache?.Dispose();
                spatialIn = InterleaveTemporal(tc, _dim);                // [B, C, 2T, H, W]
                tc.Dispose();
                ownsSpatialIn = true;
            }
        }

        int b = (int)spatialIn.Shape[0], c = (int)spatialIn.Shape[1], t = (int)spatialIn.Shape[2], h = (int)spatialIn.Shape[3], w = (int)spatialIn.Shape[4];
        Tensor frames = Vae3dLayout.ToFrames(spatialIn, b, c, t, h, w);   // [BT,C,H,W]
        if (ownsSpatialIn) spatialIn.Dispose();
        Tensor up = new Tensor(new TensorShape(b * t, c, h * 2, w * 2), DType.F32);
        backend.UpsampleNearest2D(up, frames, 2, 2);
        frames.Dispose();
        // Output channels come from the conv weight: Wan2.2 keeps dim→dim, Wan2.1 halves (dim→dim/2).
        int outC = (int)_convW!.Shape[0];
        Tensor conv = new Tensor(new TensorShape(b * t, outC, h * 2, w * 2), DType.F32);
        backend.Conv2D(conv, up, _convW!, _convB, 1, 1, 1, 1);
        up.Dispose();
        Tensor outT = Vae3dLayout.FromFrames(conv, b, outC, t, h * 2, w * 2);
        conv.Dispose();
        return outT;
    }

    /// <summary>Per-frame <c>ZeroPad2d((0,1,0,1))</c> + 3×3 Conv2d stride 2 — the encoder's spatial halving. The
    /// asymmetric pad (right/bottom only) is materialized manually since <see cref="IBackend.Conv2D"/> pads symmetrically.</summary>
    private Tensor DownsampleSpatial(IBackend backend, Tensor x)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        Tensor frames = Vae3dLayout.ToFrames(x, b, c, t, h, w);            // [BT,C,H,W]
        Tensor padded = new Tensor(new TensorShape(b * t, c, h + 1, w + 1), DType.F32);
        float* sp = (float*)frames.DataPointer;
        float* pp = (float*)padded.DataPointer;
        new Span<float>(pp, (int)Math.Min(int.MaxValue, padded.Shape.ElementCount)).Clear();
        for (long bc = 0; bc < (long)b * t * c; bc++)
            for (int row = 0; row < h; row++)
                Buffer.MemoryCopy(
                    sp + (bc * h + row) * w,
                    pp + (bc * (h + 1) + row) * (w + 1),
                    (long)w * 4, (long)w * 4);
        frames.Dispose();

        int oh = h / 2, ow = w / 2;   // floor((H+1−3)/2)+1 for even H
        Tensor conv = new Tensor(new TensorShape(b * t, c, oh, ow), DType.F32);
        backend.Conv2D(conv, padded, _convW!, _convB, 2, 2, 0, 0);
        padded.Dispose();
        Tensor outT = Vae3dLayout.FromFrames(conv, b, c, t, oh, ow);
        conv.Dispose();
        return outT;
    }

    /// <summary>Reshapes the time_conv output <c>[B, 2C, T, H, W]</c> into <c>[B, C, 2T, H, W]</c> by interleaving the two channel halves into consecutive frames (upstream <c>reshape(b,2,c,t,h,w) → stack(...,3) → reshape(b,c,2t,h,w)</c>): even out-frame from channel-half 0, odd from half 1.</summary>
    private static Tensor InterleaveTemporal(Tensor tc, int c)
    {
        int b = (int)tc.Shape[0], t = (int)tc.Shape[2], h = (int)tc.Shape[3], w = (int)tc.Shape[4];
        Tensor outT = new Tensor(new TensorShape([(long)b, c, 2 * t, h, w]), DType.F32);
        float* s = (float*)tc.DataPointer;   // channel layout: [half0 (c), half1 (c)]
        float* d = (float*)outT.DataPointer;
        long frame = (long)h * w;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
                for (int ti = 0; ti < t; ti++)
                    for (int p = 0; p < 2; p++)
                    {
                        int srcChan = p * c + ci;
                        long src = (((long)bi * (2 * c) + srcChan) * t + ti) * frame;
                        long dst = (((long)bi * c + ci) * (2 * t) + (2 * ti + p)) * frame;
                        Buffer.MemoryCopy(s + src, d + dst, frame * 4, frame * 4);
                    }
        return outT;
    }
}
