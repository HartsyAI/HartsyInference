using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Models.Hunyuan3D;

namespace HartsyInference.ThreeD.Models.TripoSr;

/// <summary>TripoSR <c>NeRFMLP</c> + <c>TriplaneNeRFRenderer.query_triplane</c>: a 3D point in [−R,R]³ is
/// scaled to [−1,1], projected onto the three planes (XY/XZ/YZ), grid-sampled
/// (<see cref="GridSampler.GridSamplePlane"/>, align_corners=False, zeros pad), concatenated to 3·C features,
/// and run through a SiLU MLP (<c>layers.0,2,…</c>) → density (1) + features (3). Density activation is
/// <c>exp(density + DensityBias)</c>; color is <c>sigmoid(features)</c>.</summary>
public sealed unsafe class TriplaneNerfDecoder
{
    private readonly TripoSrConfig _cfg;
    private Tensor? _inW, _inB;            // layers.0 : 3C -> hidden
    private Tensor[]? _midW, _midB;        // layers.2,4,… : hidden -> hidden
    private Tensor? _outW, _outB;          // layers.{2(mid+1)} : hidden -> 4

    public TriplaneNerfDecoder(TripoSrConfig cfg) => _cfg = cfg;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _inW = Hunyuan3DDit.F32(w[$"{p}layers.0.weight"]); _inB = Hunyuan3DDit.F32(w[$"{p}layers.0.bias"]);
        _midW = new Tensor[_cfg.NerfMidLayers]; _midB = new Tensor[_cfg.NerfMidLayers];
        for (int i = 0; i < _cfg.NerfMidLayers; i++)
        {
            int idx = 2 * (i + 1);
            _midW[i] = Hunyuan3DDit.F32(w[$"{p}layers.{idx}.weight"]);
            _midB[i] = Hunyuan3DDit.F32(w[$"{p}layers.{idx}.bias"]);
        }
        int outIdx = 2 * (_cfg.NerfMidLayers + 1);
        _outW = Hunyuan3DDit.F32(w[$"{p}layers.{outIdx}.weight"]); _outB = Hunyuan3DDit.F32(w[$"{p}layers.{outIdx}.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] head = [_inW, _inB, _outW, _outB];
        foreach (Tensor? t in head) if (t is not null) yield return t;
        if (_midW is not null) foreach (Tensor t in _midW) yield return t;
        if (_midB is not null) foreach (Tensor t in _midB) yield return t;
    }

    /// <summary>Decodes the activated density field over a <paramref name="resolution"/>³ grid spanning
    /// [−R,R]³ in <b>ij order</b> (x outermost, z innermost — matches TSR's grid_vertices). Feeds marching
    /// cubes at <see cref="TripoSrConfig.DensityThreshold"/>. The triplane sampling + NeRF MLP run device-resident
    /// (grid coords generated in-kernel — no host per-point loop); only the final density is read back.</summary>
    public ScalarField3D DecodeDensityField(IBackend backend, Triplane tri, int resolution, int chunkSize = 131072)
    {
        int res = resolution;
        float r = _cfg.Radius;
        long total = (long)res * res * res;
        float[] values = new float[total];
        Tensor planes = BuildPlanes(tri);
        backend.PreloadWeights([planes]);
        long produced = 0;
        while (produced < total)
        {
            int count = (int)Math.Min(chunkSize, total - produced);
            Tensor outp = EvaluateChunk(backend, planes, tri, null, produced, count, res); // [1, count, 4]
            float* op = (float*)outp.DataPointer;
            for (int i = 0; i < count; i++) values[produced + i] = MathF.Exp(op[i * 4] + _cfg.DensityBias);
            outp.Dispose();
            produced += count;
        }
        backend.FreeWeights([planes]);
        planes.Dispose();
        return new ScalarField3D { Values = values, ResX = res, ResY = res, ResZ = res, Min = (-r, -r, -r), Max = (r, r, r) };
    }

    /// <summary>Samples per-point RGB (<c>sigmoid(features)</c>) at <paramref name="points"/> (3·count xyz in
    /// [−R,R]³). Used to color mesh vertices.</summary>
    public float[] DecodeColors(IBackend backend, Triplane tri, ReadOnlySpan<float> points, int count)
    {
        Tensor planes = BuildPlanes(tri);
        backend.PreloadWeights([planes]);
        Tensor outp = EvaluateWithCoords(backend, planes, tri, points, count);
        backend.FreeWeights([planes]);
        planes.Dispose();
        float* op = (float*)outp.DataPointer;
        float[] rgb = new float[count * 3];
        for (int i = 0; i < count; i++)
            for (int c = 0; c < 3; c++) rgb[i * 3 + c] = Sigmoid(op[i * 4 + 1 + c]);
        outp.Dispose();
        return rgb;
    }

    /// <summary>Runs the MLP over <paramref name="count"/> points (xyz in [−R,R]³), returning <c>[1, count, 4]</c>
    /// (<c>[density_raw, r, g, b]</c>; density pre-exp, rgb pre-sigmoid). Exposed for parity testing.</summary>
    public Tensor Evaluate(IBackend backend, Triplane tri, ReadOnlySpan<float> coords, int count)
    {
        Tensor planes = BuildPlanes(tri);
        backend.PreloadWeights([planes]);
        Tensor outp = EvaluateWithCoords(backend, planes, tri, coords, count);
        backend.FreeWeights([planes]);
        planes.Dispose();
        return outp;
    }

    /// <summary>Uploads the triplane features once as a <c>[3,C,H,W]</c> tensor (weight-cache resident) so the
    /// grid-sample kernel + MLP never touch the host per point.</summary>
    private static Tensor BuildPlanes(Triplane tri)
    {
        int n = 3 * tri.Channels * tri.Height * tri.Width;
        Tensor planes = new(new TensorShape(3, tri.Channels, tri.Height, tri.Width), DType.F32);
        tri.Features.AsSpan(0, n).CopyTo(new Span<float>((void*)planes.DataPointer, n));
        return planes;
    }

    private Tensor EvaluateWithCoords(IBackend backend, Tensor planes, Triplane tri, ReadOnlySpan<float> coords, int count)
    {
        Tensor coordT = new(new TensorShape(count, 3), DType.F32);
        coords.Slice(0, count * 3).CopyTo(new Span<float>((void*)coordT.DataPointer, count * 3));
        Tensor outp = EvaluateChunk(backend, planes, tri, coordT, 0, count, 0);
        coordT.Dispose();
        return outp;
    }

    private Tensor EvaluateChunk(IBackend backend, Tensor planes, Triplane tri, Tensor? coords, long chunkStart, int count, int gridRes)
    {
        int c = tri.Channels, feat = 3 * c, hidden = _cfg.NerfHidden;
        Tensor f = new(new TensorShape(1, count, feat), DType.F32);
        backend.TriplaneGridSample(f, planes, coords, chunkStart, count, c, tri.Height, tri.Width, _cfg.Radius, gridRes);
        Tensor h = new(new TensorShape(1, count, hidden), DType.F32);
        backend.Linear(h, f, _inW!, _inB!); f.Dispose();
        Silu(backend, ref h);
        for (int i = 0; i < _cfg.NerfMidLayers; i++)
        {
            Tensor nh = new(new TensorShape(1, count, hidden), DType.F32);
            backend.Linear(nh, h, _midW![i], _midB![i]); h.Dispose();
            h = nh; Silu(backend, ref h);
        }
        Tensor outp = new(new TensorShape(1, count, 4), DType.F32);
        backend.Linear(outp, h, _outW!, _outB!); h.Dispose();
        return outp;
    }

    private static void Silu(IBackend backend, ref Tensor t)
    {
        Tensor o = new(t.Shape, DType.F32); backend.Silu(o, t); t.Dispose(); t = o;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
}
