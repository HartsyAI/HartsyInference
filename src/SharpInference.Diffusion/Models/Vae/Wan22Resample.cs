using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Vae;

/// <summary>Wan2.2 VAE spatial resample — the up-sampling path used by the decoder (<c>Resample</c> mode <c>upsample2d</c>/<c>upsample3d</c> in <c>vae2_2.py</c>): per-frame nearest 2× upsample + a 3×3 Conv2d. Reuses the backend <see cref="IBackend.UpsampleNearest2D"/>/<see cref="IBackend.Conv2D"/> + <see cref="Vae3dLayout"/>.
///
/// <para><b>Image / first-chunk path only:</b> the temporal <c>time_conv</c> branch of <c>upsample3d</c> is skipped on the first chunk (it sets the cache "Rep" marker without convolving), so a single-frame decode needs only the spatial path here. The temporal time_conv (for multi-frame video chunks 2+) is the streaming follow-up.</para></summary>
public sealed unsafe class Wan22Resample
{
    private readonly int _dim;
    private readonly bool _temporal;
    private Tensor? _convW, _convB;
    private CausalConv3d? _timeConv;   // upsample3d temporal branch (dim → 2·dim, kernel (3,1,1)), present iff temporal

    public Wan22Resample(int dim, bool temporal = false)
    {
        _dim = dim;
        _temporal = temporal;
    }

    /// <summary>Loads the spatial conv (Sequential index 1: <c>resample.1</c>) and, for the temporal (<c>upsample3d</c>) variant, the <c>time_conv</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _convW = weights[$"{prefix}.resample.1.weight"];
        weights.TryGetValue($"{prefix}.resample.1.bias", out _convB);
        if (_temporal)
        {
            weights.TryGetValue($"{prefix}.time_conv.bias", out Tensor? tb);
            _timeConv = new CausalConv3d(weights[$"{prefix}.time_conv.weight"], tb, padT: 1, padH: 0, padW: 0);
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_convW is not null) yield return _convW;
        if (_convB is not null) yield return _convB;
        if (_timeConv is not null) foreach (Tensor t in _timeConv.EnumerateWeights()) yield return t;
    }

    /// <summary>Upsample. With a <paramref name="cache"/> on the temporal variant, the <c>time_conv</c> doubles T (skipped on the first chunk, which sets the "Rep" marker); then a per-frame spatial 2× nearest+Conv2d runs. Returns <c>[B, C, Tout, 2H, 2W]</c> (Tout = T on the first chunk, 2T after).</summary>
    public Tensor Forward(IBackend backend, Tensor x, Wan22StreamCache? cache = null)
    {
        Tensor spatialIn = x;
        bool ownsSpatialIn = false;

        if (_temporal && cache is not null && _timeConv is not null)
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
        Tensor conv = new Tensor(new TensorShape(b * t, c, h * 2, w * 2), DType.F32);
        backend.Conv2D(conv, up, _convW!, _convB, 1, 1, 1, 1);
        up.Dispose();
        Tensor outT = Vae3dLayout.FromFrames(conv, b, c, t, h * 2, w * 2);
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
