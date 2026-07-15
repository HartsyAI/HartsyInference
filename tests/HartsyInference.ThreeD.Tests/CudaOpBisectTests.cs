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
// GpuIntegration: needs a real CUDA device. The PTX-dir existence check is not enough — the hosted
// CI runner ships the PTX but has no GPU, so CudaBackend construction/ops fail there. Runs on the
// self-hosted CUDA lane only.
[Trait("Category", "GpuIntegration")]
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

    // The dominant DiT cost was CudaBackend.Concat's per-outer-slice memcpy loop; it now routes 2-input concats
    // through the dit_concat2 kernel. These pin CUDA-kernel == CPU-loop for the two hot Hunyuan3D concat shapes.

    [Fact]
    public void Concat_LastDim_SingleBlockShape()   // single-block cat(attn[1,S,W], gelu_mlp[1,S,mlp]) along dim=2
    {
        using CudaBackend? cuda = Cuda(); if (cuda is null) { _out.WriteLine("SKIP no PTX"); return; }
        using CpuBackend cpu = new();
        int s = 4442, w = 1024, mlp = 4096;
        using Tensor a = Rand(1, 1, s, w); using Tensor b = Rand(2, 1, s, mlp);
        using Tensor oc = new(new TensorShape(1, s, w + mlp), DType.F32);
        using Tensor og = new(new TensorShape(1, s, w + mlp), DType.F32);
        cpu.Concat(oc, [a, b], 2);
        cuda.Concat(og, [a, b], 2);
        (double mx, double corr) = Cmp(oc, og);
        _out.WriteLine($"Concat dim2 [1,{s},{w}]+[1,{s},{mlp}]: maxAbs={mx:E3} corr={corr:F8}");
        Assert.True(corr > 0.99999 && mx < 1e-4, $"Concat dim2 CUDA≠CPU maxAbs={mx} corr={corr}");
    }

    [Fact]
    public void Concat_Dim1_JointShape()   // double-block joint cat(txt[1,Scond,W], img[1,N,W]) along dim=1
    {
        using CudaBackend? cuda = Cuda(); if (cuda is null) { _out.WriteLine("SKIP no PTX"); return; }
        using CpuBackend cpu = new();
        int scond = 1370, n = 3072, w = 1024;
        using Tensor a = Rand(1, 1, scond, w); using Tensor b = Rand(2, 1, n, w);
        using Tensor oc = new(new TensorShape(1, scond + n, w), DType.F32);
        using Tensor og = new(new TensorShape(1, scond + n, w), DType.F32);
        cpu.Concat(oc, [a, b], 1);
        cuda.Concat(og, [a, b], 1);
        (double mx, double corr) = Cmp(oc, og);
        _out.WriteLine($"Concat dim1 [1,{scond},{w}]+[1,{n},{w}]: maxAbs={mx:E3} corr={corr:F8}");
        Assert.True(corr > 0.99999 && mx < 1e-4, $"Concat dim1 CUDA≠CPU maxAbs={mx} corr={corr}");
    }

    [Fact]
    public void Concat_F16_LastDim()   // F16 kernel path (round-trip cast) vs CPU F32 concat
    {
        using CudaBackend? cuda = Cuda(); if (cuda is null) { _out.WriteLine("SKIP no PTX"); return; }
        using CpuBackend cpu = new();
        int s = 2048, w = 1024, mlp = 4096;
        using Tensor a = Rand(1, 1, s, w); using Tensor b = Rand(2, 1, s, mlp);
        using Tensor oc = new(new TensorShape(1, s, w + mlp), DType.F32);
        cpu.Concat(oc, [a, b], 2);
        using Tensor a16 = new(a.Shape, DType.F16); cuda.CastToF16(a16, a);
        using Tensor b16 = new(b.Shape, DType.F16); cuda.CastToF16(b16, b);
        using Tensor og16 = new(new TensorShape(1, s, w + mlp), DType.F16); cuda.Concat(og16, [a16, b16], 2);
        using Tensor og = new(og16.Shape, DType.F32); cuda.CastToF32(og, og16);
        (double mx, double corr) = Cmp(oc, og);
        _out.WriteLine($"Concat F16 dim2 [1,{s},{w}]+[1,{s},{mlp}]: maxAbs={mx:E3} corr={corr:F8}");
        Assert.True(corr > 0.9999 && mx < 1e-2, $"Concat F16 CUDA≠CPU maxAbs={mx} corr={corr}");
    }

    // Fused DiT glue: adaLN modulation + QKV-split-QK-norm (the per-forward glue-fusion round). CPU default
    // (composed reference) vs CUDA kernel.

    [Fact]
    public void LayerNormModulate_DiTShape()   // (1+scale)·LN(x)+shift, scale/shift [B,W] over seq
    {
        using CudaBackend? cuda = Cuda(); if (cuda is null) { _out.WriteLine("SKIP no PTX"); return; }
        using CpuBackend cpu = new();
        int n = 3072, w = 1024;
        using Tensor x = Rand(1, 1, n, w); using Tensor scale = Rand(2, 1, w); using Tensor shift = Rand(3, 1, w);
        using Tensor oc = new(x.Shape, DType.F32); using Tensor og = new(x.Shape, DType.F32);
        ((IBackend)cpu).LayerNormModulate(oc, x, scale, shift, 1e-6f);
        ((IBackend)cuda).LayerNormModulate(og, x, scale, shift, 1e-6f);
        (double mx, double corr) = Cmp(oc, og);
        _out.WriteLine($"LayerNormModulate [1,{n},{w}]: maxAbs={mx:E3} corr={corr:F8}");
        Assert.True(corr > 0.99999 && mx < 1e-4, $"LayerNormModulate CUDA≠CPU maxAbs={mx} corr={corr}");
    }

    [Fact]
    public void QkvSplitNorm_DiTShape()   // fused qkv[.,3W] → q,k,v[.,W] with per-head qk-rmsnorm
    {
        using CudaBackend? cuda = Cuda(); if (cuda is null) { _out.WriteLine("SKIP no PTX"); return; }
        using CpuBackend cpu = new();
        int n = 3072, heads = 16, headDim = 64, w = heads * headDim;
        using Tensor qkv = Rand(1, 1, n, 3 * w);
        using Tensor qW = Rand(2, headDim); using Tensor kW = Rand(3, headDim);
        using Tensor qc = new(new TensorShape(1, n, heads, headDim), DType.F32);
        using Tensor kc = new(new TensorShape(1, n, heads, headDim), DType.F32);
        using Tensor vc = new(new TensorShape(1, n, heads, headDim), DType.F32);
        using Tensor qg = new(qc.Shape, DType.F32); using Tensor kg = new(kc.Shape, DType.F32); using Tensor vg = new(vc.Shape, DType.F32);
        ((IBackend)cpu).QkvSplitNorm(qc, kc, vc, qkv, qW, kW, 1e-6f);
        ((IBackend)cuda).QkvSplitNorm(qg, kg, vg, qkv, qW, kW, 1e-6f);
        (double mxq, double cq) = Cmp(qc, qg); (double mxk, double ck) = Cmp(kc, kg); (double mxv, double cv) = Cmp(vc, vg);
        _out.WriteLine($"QkvSplitNorm [1,{n},{3 * w}]: q(maxAbs={mxq:E3} corr={cq:F8}) k({mxk:E3} {ck:F8}) v({mxv:E3} {cv:F8})");
        Assert.True(cq > 0.99999 && ck > 0.99999 && cv > 0.99999 && mxq < 1e-4 && mxk < 1e-4 && mxv < 1e-5,
            $"QkvSplitNorm CUDA≠CPU q({mxq},{cq}) k({mxk},{ck}) v({mxv},{cv})");
    }

    [Fact]
    public void FourierEmbed_VaeShape()   // 8-band fourier over [count,3] coords → [count,51]
    {
        using CudaBackend? cuda = Cuda(); if (cuda is null) { _out.WriteLine("SKIP no PTX"); return; }
        using CpuBackend cpu = new();
        int count = 4096, bands = 8, dim = 3 * (2 * bands + 1);
        using Tensor coords = Rand(1, count, 3);
        using Tensor oc = new(new TensorShape(1, count, dim), DType.F32);
        using Tensor og = new(new TensorShape(1, count, dim), DType.F32);
        ((IBackend)cpu).FourierEmbed(oc, coords, count, bands);
        ((IBackend)cuda).FourierEmbed(og, coords, count, bands);
        (double mx, double corr) = Cmp(oc, og);
        _out.WriteLine($"FourierEmbed [{count},3]->[{count},{dim}] bands={bands}: maxAbs={mx:E3} corr={corr:F8}");
        Assert.True(corr > 0.99999 && mx < 1e-5, $"FourierEmbed CUDA≠CPU maxAbs={mx} corr={corr}");
    }
}
