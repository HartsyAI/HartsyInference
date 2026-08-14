using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>Pins the two LTX-2.5 NA diffusion-decoder kernels against the managed reference in <see cref="IBackend"/>,
/// which is the declared numerical authority for both. Shapes are deliberately non-square (T != H != W != heads !=
/// head_dim) so a transposed axis or a wrong stride cannot survive, and the attention cases are chosen to exercise
/// <see cref="IBackend.Na3dWindowStart"/>'s slide-inward rule on every face of the volume — an interior-only test
/// would pass a kernel that is wrong on every border, which is exactly where a window kernel diverges from the
/// naive loop.</summary>
[Collection("CudaSerial")]
public sealed unsafe class Ltx25NaDecoderKernelTests
{
    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    /// <summary>Both CUDA entry points fall back to the managed reference when the PTX is absent — and the reference
    /// is exactly what these tests compare against, so without this the whole file would pass vacuously on a box
    /// that never compiled the kernel. Assert the device path was really taken.</summary>
    private static void AssertDevicePathTaken(CudaBackend cuda)
    {
        Assert.True(cuda.Kernels is not null && cuda.Kernels.HasLtx25NaKernels,
            "ltx25_na_decoder.ptx is not loaded — this test would be comparing the managed reference against itself.");
    }

    private static Tensor Random(TensorShape shape, int seed)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        for (long i = 0; i < shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static void AssertClose(Tensor expected, Tensor actual, float tol, string what)
    {
        float* e = (float*)expected.DataPointer;
        float* a = (float*)actual.DataPointer;
        double maxErr = 0;
        long worst = -1;
        for (long i = 0; i < expected.ElementCount; i++)
        {
            double err = Math.Abs((double)e[i] - a[i]);
            if (err > maxErr) { maxErr = err; worst = i; }
        }
        Assert.True(maxErr <= tol, $"{what}: max |Δ| {maxErr:E3} > {tol:E3} at element {worst}");
    }

    /// <summary>Every case has at least one axis where the kernel exceeds or equals the axis length (so the window
    /// collapses to the whole axis) and at least one where it does not (so the window really slides).</summary>
    [Theory]
    [InlineData(5, 4, 7, 2, 8, 3, 3, 3)]      // interior + all six faces slide
    [InlineData(2, 6, 3, 3, 16, 3, 5, 5)]     // T shorter than its kernel, W equal to its kernel
    [InlineData(4, 3, 5, 1, 32, 11, 11, 11)]  // every axis shorter than the kernel — degenerates to dense
    [InlineData(7, 2, 2, 4, 8, 5, 1, 3)]      // kernel 1 on H: the no-neighbour axis
    [InlineData(3, 5, 4, 2, 64, 3, 3, 3)]     // head_dim 64, the shipped decoder's width
    public void Na3dMatchesTheManagedReference(int t, int h, int w, int heads, int headDim, int kt, int kh, int kw)
    {
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        TensorShape shape = new TensorShape([1, t, h, w, heads, headDim]);
        using Tensor q = Random(shape, seed: 11);
        using Tensor k = Random(shape, seed: 12);
        using Tensor v = Random(shape, seed: 13);

        using Tensor expected = new Tensor(shape, DType.F32);
        IBackend.Na3dReference(expected, q, k, v, kt, kh, kw, scale: 0.35f);

        using Tensor actual = new Tensor(shape, DType.F32);
        cuda.Na3d(actual, q, k, v, kt, kh, kw, scale: 0.35f);
        cuda.Sync();
        AssertDevicePathTaken(cuda);

        AssertClose(expected, actual, 2e-5f, $"na3d {t}x{h}x{w} heads={heads} hd={headDim} kernel={kt}x{kh}x{kw}");
    }

    /// <summary>The border is where a window kernel most easily diverges, so this asserts on the faces directly:
    /// every query whose window had to slide inward must still match the reference.</summary>
    [Fact]
    public void Na3dMatchesTheReferenceOnEveryFaceOfTheVolume()
    {
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        const int t = 6, h = 5, w = 7, heads = 3, headDim = 16, kt = 3, kh = 5, kw = 3;
        TensorShape shape = new TensorShape([1, t, h, w, heads, headDim]);
        using Tensor q = Random(shape, seed: 21);
        using Tensor k = Random(shape, seed: 22);
        using Tensor v = Random(shape, seed: 23);

        using Tensor expected = new Tensor(shape, DType.F32);
        IBackend.Na3dReference(expected, q, k, v, kt, kh, kw, scale: 1.0f);
        using Tensor actual = new Tensor(shape, DType.F32);
        cuda.Na3d(actual, q, k, v, kt, kh, kw, scale: 1.0f);
        cuda.Sync();
        AssertDevicePathTaken(cuda);

        float* e = (float*)expected.DataPointer;
        float* a = (float*)actual.DataPointer;
        long wStride = (long)heads * headDim, hStride = w * wStride, tStride = (long)h * hStride;
        int faces = 0;
        for (int it = 0; it < t; it++)
        for (int ih = 0; ih < h; ih++)
        for (int iw = 0; iw < w; iw++)
        {
            bool onFace = it == 0 || it == t - 1 || ih == 0 || ih == h - 1 || iw == 0 || iw == w - 1;
            if (!onFace) continue;
            faces++;
            long baseOff = it * tStride + ih * hStride + iw * wStride;
            for (long d = 0; d < wStride; d++)
            {
                double err = Math.Abs((double)e[baseOff + d] - a[baseOff + d]);
                Assert.True(err <= 2e-5f, $"face token (t={it},h={ih},w={iw}) lane {d}: |Δ| {err:E3}");
            }
        }
        Assert.Equal(t * h * w - (t - 2) * (h - 2) * (w - 2), faces);
    }

    /// <summary>A kernel covering every axis degenerates to dense attention over the whole grid, computed here
    /// independently of both implementations so a shared mistake cannot pass.</summary>
    [Fact]
    public void Na3dSpanningTheGridEqualsDenseAttention()
    {
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        const int t = 2, h = 3, w = 2, heads = 1, hd = 4;
        TensorShape shape = new TensorShape([1, t, h, w, heads, hd]);
        using Tensor q = Random(shape, seed: 31);
        using Tensor k = Random(shape, seed: 32);
        using Tensor v = Random(shape, seed: 33);
        using Tensor actual = new Tensor(shape, DType.F32);
        cuda.Na3d(actual, q, k, v, t, h, w, scale: 1.0f);
        cuda.Sync();
        AssertDevicePathTaken(cuda);

        int tokens = t * h * w;
        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer;
        float* vp = (float*)v.DataPointer, ap = (float*)actual.DataPointer;
        for (int i = 0; i < tokens; i++)
        {
            float[] scores = new float[tokens];
            float max = float.NegativeInfinity;
            for (int j = 0; j < tokens; j++)
            {
                float dot = 0f;
                for (int d = 0; d < hd; d++) dot += qp[i * hd + d] * kp[j * hd + d];
                scores[j] = dot;
                max = MathF.Max(max, dot);
            }
            float sum = 0f;
            for (int j = 0; j < tokens; j++) { scores[j] = MathF.Exp(scores[j] - max); sum += scores[j]; }
            for (int d = 0; d < hd; d++)
            {
                float acc = 0f;
                for (int j = 0; j < tokens; j++) acc += scores[j] / sum * vp[j * hd + d];
                Assert.True(Math.Abs(acc - ap[i * hd + d]) <= 2e-5f, $"token {i} lane {d}: {acc} vs {ap[i * hd + d]}");
            }
        }
    }

    /// <summary>The rope rotates three per-axis chunks by three different coordinates. The splits here are uneven
    /// (16/24/24 is the shipped head_dim-64 split) so a kernel that assumed equal thirds, or that indexed the H
    /// table with the W coordinate, cannot pass.</summary>
    [Theory]
    [InlineData(5, 4, 7, 3, 64, 16, 24)]
    [InlineData(2, 6, 3, 2, 32, 8, 12)]
    [InlineData(4, 3, 5, 5, 16, 4, 6)]
    public void Rope3dMatchesTheManagedReference(int t, int h, int w, int heads, int headDim, int splitT, int splitH)
    {
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        TensorShape shape = new TensorShape([1, t, h, w, heads, headDim]);
        int splitW = headDim - splitT - splitH;
        using Tensor cosT = Random(new TensorShape(t, splitT / 2), seed: 41);
        using Tensor sinT = Random(new TensorShape(t, splitT / 2), seed: 42);
        using Tensor cosH = Random(new TensorShape(h, splitH / 2), seed: 43);
        using Tensor sinH = Random(new TensorShape(h, splitH / 2), seed: 44);
        using Tensor cosW = Random(new TensorShape(w, splitW / 2), seed: 45);
        using Tensor sinW = Random(new TensorShape(w, splitW / 2), seed: 46);

        using Tensor expected = Random(shape, seed: 47);
        IBackend.Ltx25NaRope3dReference(expected, cosT, sinT, cosH, sinH, cosW, sinW, splitT, splitH);

        using Tensor actual = Random(shape, seed: 47);   // same seed → same starting contents
        cuda.Ltx25NaRope3d(actual, cosT, sinT, cosH, sinH, cosW, sinW, splitT, splitH);
        cuda.Sync();
        AssertDevicePathTaken(cuda);

        AssertClose(expected, actual, 1e-6f, $"rope3d {t}x{h}x{w} heads={heads} split={splitT}/{splitH}/{splitW}");
    }

    /// <summary>A zero rotation (cos=1, sin=0) must leave the tensor untouched, which catches a kernel that writes
    /// to the wrong chunk offset — the sort of error a random-table comparison can mask if both paths share it.</summary>
    [Fact]
    public void Rope3dWithIdentityTablesIsAnExactNoOp()
    {
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        const int t = 3, h = 4, w = 5, heads = 2, headDim = 64, splitT = 16, splitH = 24;
        int splitW = headDim - splitT - splitH;
        using Tensor cosT = Ones(new TensorShape(t, splitT / 2), 1f);
        using Tensor sinT = Ones(new TensorShape(t, splitT / 2), 0f);
        using Tensor cosH = Ones(new TensorShape(h, splitH / 2), 1f);
        using Tensor sinH = Ones(new TensorShape(h, splitH / 2), 0f);
        using Tensor cosW = Ones(new TensorShape(w, splitW / 2), 1f);
        using Tensor sinW = Ones(new TensorShape(w, splitW / 2), 0f);

        TensorShape shape = new TensorShape([1, t, h, w, heads, headDim]);
        using Tensor original = Random(shape, seed: 51);
        using Tensor actual = Random(shape, seed: 51);
        cuda.Ltx25NaRope3d(actual, cosT, sinT, cosH, sinH, cosW, sinW, splitT, splitH);
        cuda.Sync();
        AssertDevicePathTaken(cuda);

        float* o = (float*)original.DataPointer;
        float* a = (float*)actual.DataPointer;
        for (long i = 0; i < shape.ElementCount; i++)
            Assert.True(o[i] == a[i], $"identity rope changed element {i}: {o[i]} → {a[i]}");
    }

    private static Tensor Ones(TensorShape shape, float value)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < shape.ElementCount; i++) p[i] = value;
        return t;
    }
}
