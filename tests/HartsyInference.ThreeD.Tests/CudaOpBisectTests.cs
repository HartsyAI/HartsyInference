using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests;

/// <summary>Bisect: CPU vs CUDA parity for the individual ops the TripoSR Transformer1D backbone uses with its
/// unusual shapes (GroupNorm over [1,C,N,1], ConvTranspose2d k2/s2 upsample, Linear/SDPA over N=1024 tokens).
/// Isolates a CUDA-specific op bug. Gated on PTX presence.</summary>
public sealed unsafe class CudaOpBisectTests
{
    private readonly ITestOutputHelper _out;
    public CudaOpBisectTests(ITestOutputHelper o) => _out = o;

    private CudaBackend? Cuda()
    {
        string ptx = Path.Combine(Path.GetDirectoryName(typeof(CudaOpBisectTests).Assembly.Location)!, "Ptx");
        return Directory.Exists(ptx) ? new CudaBackend(deviceOrdinal: 0, ptxDir: ptx) : null;
    }

    private static Tensor Rand(int seed, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        Random r = new(seed); float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(r.NextDouble() * 2 - 1);
        return t;
    }

    private (double max, double corr) Cmp(Tensor a, Tensor b)
    {
        float* pa = (float*)a.DataPointer; float* pb = (float*)b.DataPointer;
        double mx = 0, dot = 0, na = 0, nb = 0;
        for (long i = 0; i < a.ElementCount; i++) { double x = pa[i], y = pb[i]; mx = Math.Max(mx, Math.Abs(x - y)); dot += x * y; na += x * x; nb += y * y; }
        return (mx, dot / (Math.Sqrt(na * nb) + 1e-12));
    }

    [Fact]
    public void GroupNorm_BackboneShape()
    {
        using CudaBackend? cuda = Cuda(); if (cuda is null) { _out.WriteLine("SKIP no PTX"); return; }
        using CpuBackend cpu = new();
        int c = 1024, n = 1024, groups = 32;
        using Tensor x = Rand(1, 1, c, n, 1);
        using Tensor w = Rand(2, c); using Tensor bta = Rand(3, c);
        using Tensor oc = new(x.Shape, DType.F32); using Tensor og = new(x.Shape, DType.F32);
        cpu.GroupNorm(oc, x, w, bta, groups, 1e-6f);
        cuda.GroupNorm(og, x, w, bta, groups, 1e-6f);
        (double mx, double corr) = Cmp(oc, og);
        _out.WriteLine($"GroupNorm [1,{c},{n},1] g{groups}: maxAbs={mx:E3} corr={corr:F8}");
        Assert.True(corr > 0.9999 && mx < 1e-3, $"GroupNorm CUDA≠CPU maxAbs={mx} corr={corr}");
    }

    [Fact]
    public void ConvTranspose2d_UpsampleShape()
    {
        using CudaBackend? cuda = Cuda(); if (cuda is null) { _out.WriteLine("SKIP no PTX"); return; }
        using CpuBackend cpu = new();
        int planes = 3, cin = 1024, cout = 40, p = 32, res = 64;
        using Tensor x = Rand(1, planes, cin, p, p);
        using Tensor w = Rand(2, cin, cout, 2, 2);   // ConvTranspose2d weight [Cin,Cout,kh,kw]
        using Tensor b = Rand(3, cout);
        using Tensor oc = new(new TensorShape(planes, cout, res, res), DType.F32);
        using Tensor og = new(new TensorShape(planes, cout, res, res), DType.F32);
        ((IBackend)cpu).ConvTranspose2d(oc, x, w, b, 2, 2, 0, 0);
        ((IBackend)cuda).ConvTranspose2d(og, x, w, b, 2, 2, 0, 0);
        (double mx, double corr) = Cmp(oc, og);
        _out.WriteLine($"ConvTranspose2d [{planes},{cin},{p},{p}]->[{planes},{cout},{res},{res}]: maxAbs={mx:E3} corr={corr:F8}");
        Assert.True(corr > 0.9999 && mx < 1e-3, $"ConvTranspose2d CUDA≠CPU maxAbs={mx} corr={corr}");
    }

    [Fact]
    public void LayerNorm_BlockShape()
    {
        using CudaBackend? cuda = Cuda(); if (cuda is null) { _out.WriteLine("SKIP no PTX"); return; }
        using CpuBackend cpu = new();
        int n = 1024, w = 1024;
        using Tensor x = Rand(1, 1, n, w); using Tensor g = Rand(2, w); using Tensor b = Rand(3, w);
        using Tensor oc = new(x.Shape, DType.F32); using Tensor og = new(x.Shape, DType.F32);
        cpu.LayerNorm(oc, x, g, b, 1e-5f);
        cuda.LayerNorm(og, x, g, b, 1e-5f);
        (double mx, double corr) = Cmp(oc, og);
        _out.WriteLine($"LayerNorm [1,{n},{w}]: maxAbs={mx:E3} corr={corr:F8}");
        Assert.True(corr > 0.9999 && mx < 1e-3, $"LayerNorm CUDA≠CPU maxAbs={mx} corr={corr}");
    }

    // Replicates the block's SelfAttn path: 3× Linear (GPU-cached output) → host ToHeads (reads .DataPointer,
    // exercising the lazy GPU→CPU sync) → SDPA. This is the Linear→DataPointer→SDPA chain the real block uses.
    [Fact]
    public void SelfAttn_LinearThenHostReshapeThenSdpa()
    {
        using CudaBackend? cuda = Cuda(); if (cuda is null) { _out.WriteLine("SKIP no PTX"); return; }
        using CpuBackend cpu = new();
        int n = 1024, w = 1024, heads = 16, hd = 64;
        using Tensor src = Rand(1, 1, n, w);
        using Tensor qW = Rand(2, w, w); using Tensor kW = Rand(3, w, w); using Tensor vW = Rand(4, w, w);
        Tensor Run(IBackend be)
        {
            Tensor q = new(new TensorShape(1, n, w), DType.F32); be.Linear(q, src, qW, null);
            Tensor k = new(new TensorShape(1, n, w), DType.F32); be.Linear(k, src, kW, null);
            Tensor v = new(new TensorShape(1, n, w), DType.F32); be.Linear(v, src, vW, null);
            Tensor qh = Heads(q, n, heads, hd), kh = Heads(k, n, heads, hd), vh = Heads(v, n, heads, hd);
            q.Dispose(); k.Dispose(); v.Dispose();
            Tensor o = new(new TensorShape(1, heads, n, hd), DType.F32);
            be.ScaledDotProductAttention(o, qh, kh, vh, null, 1f / MathF.Sqrt(hd));
            qh.Dispose(); kh.Dispose(); vh.Dispose();
            return o;
        }
        Tensor oc = Run(cpu), og = Run(cuda);
        (double mx, double corr) = Cmp(oc, og);
        _out.WriteLine($"SelfAttn(Linear→host reshape→SDPA) [n={n},w={w}]: maxAbs={mx:E3} corr={corr:F8}");
        oc.Dispose(); og.Dispose();
        Assert.True(corr > 0.9999, $"SelfAttn chain CUDA≠CPU maxAbs={mx} corr={corr}");
    }

    // Host-side head reshape [1,n,w] -> [1,heads,n,hd], reading .DataPointer (mirrors Hunyuan3DAttention.ToHeads).
    private static Tensor Heads(Tensor input, int n, int heads, int hd)
    {
        int w = heads * hd;
        Tensor o = new(new TensorShape(1, heads, n, hd), DType.F32);
        float* sp = (float*)input.DataPointer; float* dp = (float*)o.DataPointer;
        for (int s = 0; s < n; s++) for (int h = 0; h < heads; h++)
            new ReadOnlySpan<float>(sp + s * w + h * hd, hd).CopyTo(new Span<float>(dp + (h * n + s) * hd, hd));
        return o;
    }

    [Fact]
    public void Sdpa_CrossAttnShape()
    {
        using CudaBackend? cuda = Cuda(); if (cuda is null) { _out.WriteLine("SKIP no PTX"); return; }
        using CpuBackend cpu = new();
        int b = 1, h = 16, sq = 1024, sk = 1025, d = 64;
        using Tensor q = Rand(1, b, h, sq, d);
        using Tensor k = Rand(2, b, h, sk, d);
        using Tensor v = Rand(3, b, h, sk, d);
        using Tensor oc = new(new TensorShape(b, h, sq, d), DType.F32);
        using Tensor og = new(new TensorShape(b, h, sq, d), DType.F32);
        float scale = 1f / MathF.Sqrt(d);
        cpu.ScaledDotProductAttention(oc, q, k, v, null, scale);
        cuda.ScaledDotProductAttention(og, q, k, v, null, scale);
        (double mx, double corr) = Cmp(oc, og);
        _out.WriteLine($"SDPA cross [Sq={sq},Sk={sk},D={d},H={h}]: maxAbs={mx:E3} corr={corr:F8}");
        Assert.True(corr > 0.9999, $"SDPA CUDA≠CPU maxAbs={mx} corr={corr}");
    }
}
