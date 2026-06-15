using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Models.Hunyuan3D;

namespace HartsyInference.ThreeD.Models.TripoSr;

/// <summary>NeRF MLP that decodes a <see cref="Triplane"/> into density + color at any 3D point: the point is
/// projected onto the three planes, bilinearly sampled (<see cref="GridSampler.BilinearPlane"/>), the three
/// feature vectors are concatenated, and an MLP maps them to <c>[density, r, g, b]</c> (rgb via sigmoid).
/// Reused by any triplane/LRM model. Density over a grid → <see cref="ScalarField3D"/> for marching cubes.
/// <para><b>Numerics validation-pending</b> — plane→axis mapping, density activation, and MLP depth are
/// validation-gated. Structurally green on synthetic weights.</para></summary>
public sealed unsafe class TriplaneNerfDecoder
{
    private readonly TripoSrConfig _cfg;
    private Tensor? _inW, _inB;        // 3*C → hidden
    private Tensor[]? _hiddenW, _hiddenB; // hidden → hidden ×NerfLayers
    private Tensor? _outW, _outB;      // hidden → 4 (density + rgb)

    public TriplaneNerfDecoder(TripoSrConfig cfg) => _cfg = cfg;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _inW = Hunyuan3DDit.F32(w[$"{p}fc_in.weight"]); _inB = Hunyuan3DDit.F32(w[$"{p}fc_in.bias"]);
        _hiddenW = new Tensor[_cfg.NerfLayers]; _hiddenB = new Tensor[_cfg.NerfLayers];
        for (int i = 0; i < _cfg.NerfLayers; i++)
        {
            _hiddenW[i] = Hunyuan3DDit.F32(w[$"{p}fc_hidden.{i}.weight"]);
            _hiddenB[i] = Hunyuan3DDit.F32(w[$"{p}fc_hidden.{i}.bias"]);
        }
        _outW = Hunyuan3DDit.F32(w[$"{p}fc_out.weight"]); _outB = Hunyuan3DDit.F32(w[$"{p}fc_out.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] head = [_inW, _inB, _outW, _outB];
        foreach (Tensor? t in head) if (t is not null) yield return t;
        if (_hiddenW is not null) foreach (Tensor t in _hiddenW) yield return t;
        if (_hiddenB is not null) foreach (Tensor t in _hiddenB) yield return t;
    }

    /// <summary>Decodes the density field over a <paramref name="resolution"/>³ grid in <c>[-bound, bound]³</c>,
    /// evaluating points in chunks. Output feeds marching cubes at <see cref="TripoSrConfig.DensityThreshold"/>.</summary>
    public ScalarField3D DecodeDensityField(IBackend backend, Triplane tri, int resolution, float bound, int chunkSize = 32768)
    {
        int res = resolution;
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
                int x = (int)(lin % res), y = (int)((lin / res) % res), z = (int)(lin / ((long)res * res));
                coords[i * 3] = Map(x, res, bound); coords[i * 3 + 1] = Map(y, res, bound); coords[i * 3 + 2] = Map(z, res, bound);
            }
            Tensor outp = Evaluate(backend, tri, coords.AsSpan(0, count * 3), count); // [1, count, 4]
            float* op = (float*)outp.DataPointer;
            for (int i = 0; i < count; i++) values[produced + i] = op[i * 4]; // density channel
            outp.Dispose();
            produced += count;
        }
        return new ScalarField3D { Values = values, ResX = res, ResY = res, ResZ = res, Min = (-bound, -bound, -bound), Max = (bound, bound, bound) };
    }

    /// <summary>Samples per-point RGB (sigmoid of the color channels) at <paramref name="points"/>
    /// (3*count xyz). Used to color mesh vertices.</summary>
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

    /// <summary>Runs the MLP over <paramref name="count"/> points, returning <c>[1, count, 4]</c>
    /// (<c>[density, r, g, b]</c>, rgb pre-sigmoid).</summary>
    private Tensor Evaluate(IBackend backend, Triplane tri, ReadOnlySpan<float> coords, int count)
    {
        int c = tri.Channels, feat = 3 * c, hidden = _cfg.NerfHidden;
        Tensor f = new(new TensorShape(1, count, feat), DType.F32);
        SampleTriplane(tri, coords, count, f);

        Tensor h = new(new TensorShape(1, count, hidden), DType.F32);
        backend.Linear(h, f, _inW!, _inB!); f.Dispose();
        Silu(backend, ref h);
        for (int i = 0; i < _cfg.NerfLayers; i++)
        {
            Tensor nh = new(new TensorShape(1, count, hidden), DType.F32);
            backend.Linear(nh, h, _hiddenW![i], _hiddenB![i]); h.Dispose();
            h = nh; Silu(backend, ref h);
        }
        Tensor outp = new(new TensorShape(1, count, 4), DType.F32);
        backend.Linear(outp, h, _outW!, _outB!); h.Dispose();
        return outp;
    }

    private void SampleTriplane(Triplane tri, ReadOnlySpan<float> coords, int count, Tensor dst)
    {
        int c = tri.Channels, h = tri.Height, wd = tri.Width;
        float bound = _cfg.BoundingBox;
        float* dp = (float*)dst.DataPointer;
        ReadOnlySpan<float> features = tri.Features;
        Span<float> tmp = stackalloc float[c];
        for (int i = 0; i < count; i++)
        {
            float nx = (coords[i * 3] + bound) / (2 * bound);
            float ny = (coords[i * 3 + 1] + bound) / (2 * bound);
            float nz = (coords[i * 3 + 2] + bound) / (2 * bound);
            long fbase = (long)i * 3 * c;
            // Plane 0=XY (nx,ny), 1=XZ (nx,nz), 2=YZ (ny,nz).
            SamplePlane(features, tri.PlaneOffset(0), c, h, wd, nx, ny, tmp); CopyTo(tmp, dp, fbase + 0 * c);
            SamplePlane(features, tri.PlaneOffset(1), c, h, wd, nx, nz, tmp); CopyTo(tmp, dp, fbase + 1 * c);
            SamplePlane(features, tri.PlaneOffset(2), c, h, wd, ny, nz, tmp); CopyTo(tmp, dp, fbase + 2 * c);
        }
    }

    private static void SamplePlane(ReadOnlySpan<float> features, int offset, int c, int h, int w, float u, float v, Span<float> dst)
        => GridSampler.BilinearPlane(features.Slice(offset, c * h * w), c, h, w, u, v, dst);

    private static void CopyTo(ReadOnlySpan<float> src, float* dst, long offset)
    {
        for (int i = 0; i < src.Length; i++) dst[offset + i] = src[i];
    }

    private static void Silu(IBackend backend, ref Tensor t)
    {
        Tensor o = new(t.Shape, DType.F32); backend.Silu(o, t); t.Dispose(); t = o;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
    private static float Map(int i, int res, float bound) => -bound + (res > 1 ? i / (float)(res - 1) : 0f) * (2f * bound);
}
