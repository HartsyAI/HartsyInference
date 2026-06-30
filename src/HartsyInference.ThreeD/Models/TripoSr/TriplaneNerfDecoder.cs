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
    /// cubes at <see cref="TripoSrConfig.DensityThreshold"/>.</summary>
    public ScalarField3D DecodeDensityField(IBackend backend, Triplane tri, int resolution, int chunkSize = 32768)
    {
        int res = resolution;
        float r = _cfg.Radius;
        long total = (long)res * res * res;
        float[] values = new float[total];
        float[] coords = new float[chunkSize * 3];
        long produced = 0;
        while (produced < total)
        {
            int count = (int)Math.Min(chunkSize, total - produced);
            for (int i = 0; i < count; i++)
            {
                long lin = produced + i;
                // ij order: index = (ix·res + iy)·res + iz
                int iz = (int)(lin % res), iy = (int)((lin / res) % res), ix = (int)(lin / ((long)res * res));
                coords[i * 3] = Map(ix, res, r); coords[i * 3 + 1] = Map(iy, res, r); coords[i * 3 + 2] = Map(iz, res, r);
            }
            Tensor outp = Evaluate(backend, tri, coords.AsSpan(0, count * 3), count); // [1, count, 4]
            float* op = (float*)outp.DataPointer;
            for (int i = 0; i < count; i++) values[produced + i] = MathF.Exp(op[i * 4] + _cfg.DensityBias);
            outp.Dispose();
            produced += count;
        }
        return new ScalarField3D { Values = values, ResX = res, ResY = res, ResZ = res, Min = (-r, -r, -r), Max = (r, r, r) };
    }

    /// <summary>Samples per-point RGB (<c>sigmoid(features)</c>) at <paramref name="points"/> (3·count xyz in
    /// [−R,R]³). Used to color mesh vertices.</summary>
    public float[] DecodeColors(IBackend backend, Triplane tri, ReadOnlySpan<float> points, int count)
    {
        Tensor outp = Evaluate(backend, tri, points, count);
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
        int c = tri.Channels, feat = 3 * c, hidden = _cfg.NerfHidden;
        Tensor f = new(new TensorShape(1, count, feat), DType.F32);
        SampleTriplane(tri, coords, count, f);

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

    private void SampleTriplane(Triplane tri, ReadOnlySpan<float> coords, int count, Tensor dst)
    {
        int c = tri.Channels, h = tri.Height, wd = tri.Width;
        float r = _cfg.Radius;
        float* dp = (float*)dst.DataPointer;
        ReadOnlySpan<float> features = tri.Features;
        Span<float> tmp = stackalloc float[c];
        for (int i = 0; i < count; i++)
        {
            // scale_tensor([-R,R] -> [-1,1]) == p / R
            float gx = coords[i * 3] / r, gy = coords[i * 3 + 1] / r, gz = coords[i * 3 + 2] / r;
            long fbase = (long)i * 3 * c;
            // indices2D stack: plane0 (x,y), plane1 (x,z), plane2 (y,z); first coord -> width, second -> height.
            GridSampler.GridSamplePlane(features.Slice(tri.PlaneOffset(0), c * h * wd), c, h, wd, gx, gy, tmp); CopyTo(tmp, dp, fbase + 0 * c);
            GridSampler.GridSamplePlane(features.Slice(tri.PlaneOffset(1), c * h * wd), c, h, wd, gx, gz, tmp); CopyTo(tmp, dp, fbase + 1 * c);
            GridSampler.GridSamplePlane(features.Slice(tri.PlaneOffset(2), c * h * wd), c, h, wd, gy, gz, tmp); CopyTo(tmp, dp, fbase + 2 * c);
        }
    }

    private static void CopyTo(ReadOnlySpan<float> src, float* dst, long offset)
    {
        for (int i = 0; i < src.Length; i++) dst[offset + i] = src[i];
    }

    private static void Silu(IBackend backend, ref Tensor t)
    {
        Tensor o = new(t.Shape, DType.F32); backend.Silu(o, t); t.Dispose(); t = o;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
    private static float Map(int i, int res, float r) => -r + (res > 1 ? i / (float)(res - 1) : 0f) * (2f * r);
}
