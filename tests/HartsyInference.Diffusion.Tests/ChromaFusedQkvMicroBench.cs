using System.Diagnostics;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Same-process warm A/B of the HARTSY_CHROMA_FUSED_QKV single-block path vs the split path at the
/// REAL classic-Chroma shape (hidden 3072, 24×128 heads, S=4608 ≈ 1024² joint sequence), F16 weights +
/// activations (the shipping classic-Chroma config class). Decides whether the fused path is worth a Swarm
/// deploy A/B at all (INFERENCE_ACCEL_GRIND §H3.1, measure-first). Both arms run in ONE process,
/// interleaved trials — the in-run-control lesson from the sage occupancy round. Run explicitly:
///   dotnet test --filter "FullyQualifiedName~ChromaFusedQkvMicroBench" (Category=ChromaBench, excluded from sweeps)</summary>
[Collection("CudaSerial")]
[Trait("Category", "ChromaBench")]
public sealed unsafe class ChromaFusedQkvMicroBench
{
    private const int Hidden = 3072;
    private const int Heads = 24;
    private const int HeadDim = 128;
    private const int Mlp = Hidden * 4;
    private const int TxtSeq = 512;
    private const int ImgH = 64, ImgW = 64;          // 64×64 packed = 4096 img tokens (1024² at patch 16)
    private const int Seq = TxtSeq + ImgH * ImgW;    // 4608 joint tokens

    private readonly ITestOutputHelper _output;
    public ChromaFusedQkvMicroBench(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor RandF16(int seed, params long[] shape)
    {
        Tensor f32 = new Tensor(new TensorShape(shape), DType.F32);
        Random rng = new Random(seed);
        float* p = (float*)f32.DataPointer;
        for (long i = 0; i < f32.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 0.05 - 0.025);
        Tensor f16 = f32.CastTo(DType.F16);
        f32.Dispose();
        return f16;
    }

    private static Tensor RandF32(int seed, params long[] shape)
    {
        Tensor t = new Tensor(new TensorShape(shape), DType.F32);
        Random rng = new Random(seed);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 0.05 - 0.025);
        return t;
    }

    private static Tensor ConcatRows(params Tensor[] parts)
    {
        long cols = parts[0].Shape.Rank == 2 ? parts[0].Shape[1] : 1;
        long rows = 0;
        foreach (Tensor p in parts) rows += p.Shape[0];
        long elemBytes = parts[0].DType.SizeInBytes;
        Tensor fusedT = parts[0].Shape.Rank == 2
            ? new Tensor(new TensorShape(rows, cols), parts[0].DType)
            : new Tensor(new TensorShape(rows), parts[0].DType);
        byte* dst = (byte*)fusedT.DataPointer;
        foreach (Tensor p in parts)
        {
            long bytes = p.ElementCount * elemBytes;
            Buffer.MemoryCopy((void*)p.DataPointer, dst, bytes, bytes);
            dst += bytes;
        }
        return fusedT;
    }

    /// <summary>F16 weight → fp8 e4m3 with per-tensor scale = absmax/448 (mirrors the converter's
    /// QuantizeToFp8Scaled; the fused arm quantizes the CONCAT so Q/K/V share one scale, exactly what the
    /// converter emits for BFL fp8 sources).</summary>
    private static Tensor QuantFp8(Tensor w)
    {
        Tensor f32 = w.CastTo(DType.F32);
        long n = f32.ElementCount;
        float* p = (float*)f32.DataPointer;
        float absmax = 0f;
        for (long i = 0; i < n; i++) { float a = MathF.Abs(p[i]); if (a > absmax) absmax = a; }
        float scale = absmax > 0f ? absmax / 448f : 1f;
        float inv = 1f / scale;
        for (long i = 0; i < n; i++) p[i] *= inv;
        Tensor fp8 = f32.CastTo(DType.F8E4M3);
        f32.Dispose();
        fp8.Fp8ScaleFactor = scale;
        return fp8;
    }

    [Fact]
    public void SingleBlock_Fused_Vs_Split_RealShape()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        const string p = "single_transformer_blocks.0";
        // F16 weights (the classic-Chroma shipping class); norm scales + biases stay F32 like production.
        Tensor qW = RandF16(1, Hidden, Hidden); Tensor kW = RandF16(2, Hidden, Hidden);
        Tensor vW = RandF16(3, Hidden, Hidden); Tensor mlpW = RandF16(4, Mlp, Hidden);
        Tensor qB = RandF32(5, Hidden); Tensor kB = RandF32(6, Hidden);
        Tensor vB = RandF32(7, Hidden); Tensor mlpB = RandF32(8, Mlp);
        Dictionary<string, Tensor> split = new()
        {
            [$"{p}.attn.to_q.weight"] = qW,
            [$"{p}.attn.to_k.weight"] = kW,
            [$"{p}.attn.to_v.weight"] = vW,
            [$"{p}.attn.to_q.bias"] = qB,
            [$"{p}.attn.to_k.bias"] = kB,
            [$"{p}.attn.to_v.bias"] = vB,
            [$"{p}.proj_mlp.weight"] = mlpW,
            [$"{p}.proj_mlp.bias"] = mlpB,
            [$"{p}.proj_out.weight"] = RandF16(9, Hidden, Hidden + Mlp),
            [$"{p}.proj_out.bias"] = RandF32(10, Hidden),
            [$"{p}.attn.norm_q.weight"] = RandF32(11, HeadDim),
            [$"{p}.attn.norm_k.weight"] = RandF32(12, HeadDim),
        };
        Dictionary<string, Tensor> fused = new(split);
        fused[$"{p}.linear1.weight"] = ConcatRows(qW, kW, vW, mlpW);
        fused[$"{p}.linear1.bias"] = ConcatRows(qB, kB, vB, mlpB);
        foreach (string k in new[] { "attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp" })
        {
            fused.Remove($"{p}.{k}.weight");
            fused.Remove($"{p}.{k}.bias");
        }

        ChromaSingleStreamBlock splitBlock = new ChromaSingleStreamBlock(Hidden, Heads, HeadDim);
        splitBlock.LoadWeights(split, p);
        ChromaSingleStreamBlock fusedBlock = new ChromaSingleStreamBlock(Hidden, Heads, HeadDim);
        fusedBlock.LoadWeights(fused, p);

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        FluxRope rope = new FluxRope([16, 56, 56], 10000);
        using Tensor posIds = FluxRope.BuildPositionIds(TxtSeq, ImgH, ImgW);
        rope.Precompute(posIds);

        using Tensor x = RandF16(100, 1, Seq, Hidden);
        using Tensor temb = RandF32(101, 1, 3, Hidden);

        const int Trials = 8;
        double[] splitMs = new double[Trials];
        double[] fusedMs = new double[Trials];
        // Warmup both arms (JIT + plan caches + weight upload).
        splitBlock.Forward(cuda, x, temb, rope, null).Dispose();
        fusedBlock.Forward(cuda, x, temb, rope, null).Dispose();
        cuda.Sync();
        Stopwatch sw = new Stopwatch();
        for (int t = 0; t < Trials; t++)
        {
            // Interleaved trials: any clock drift hits both arms equally (in-run control).
            sw.Restart();
            splitBlock.Forward(cuda, x, temb, rope, null).Dispose();
            cuda.Sync();
            sw.Stop();
            splitMs[t] = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            fusedBlock.Forward(cuda, x, temb, rope, null).Dispose();
            cuda.Sync();
            sw.Stop();
            fusedMs[t] = sw.Elapsed.TotalMilliseconds;
        }

        Report("F16", splitMs, fusedMs);
    }

    /// <summary>Same A/B with fp8-scaled weights + native fp8 GEMM — the actual §H3.1 target config.
    /// SKIPs on hardware without native fp8 (SM &lt; 8.9, e.g. the 3060): run with CUDA_VISIBLE_DEVICES=0
    /// on the dual-GPU box to land on the 4090.</summary>
    [Fact]
    public void SingleBlock_Fused_Vs_Split_Fp8RealShape()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        if (!cuda.Fp8Executor.IsSupported)
        {
            _output.WriteLine("SKIPPED: native fp8 GEMM needs SM 8.9+.");
            return;
        }
        cuda.EnableNativeFp8Gemm = true;

        const string p = "single_transformer_blocks.0";
        Tensor qW16 = RandF16(1, Hidden, Hidden); Tensor kW16 = RandF16(2, Hidden, Hidden);
        Tensor vW16 = RandF16(3, Hidden, Hidden); Tensor mlpW16 = RandF16(4, Mlp, Hidden);
        Tensor lin1W16 = ConcatRows(qW16, kW16, vW16, mlpW16);
        Tensor qB = RandF32(5, Hidden); Tensor kB = RandF32(6, Hidden);
        Tensor vB = RandF32(7, Hidden); Tensor mlpB = RandF32(8, Mlp);
        Tensor projOut16 = RandF16(9, Hidden, Hidden + Mlp);
        Dictionary<string, Tensor> split = new()
        {
            [$"{p}.attn.to_q.weight"] = QuantFp8(qW16),
            [$"{p}.attn.to_k.weight"] = QuantFp8(kW16),
            [$"{p}.attn.to_v.weight"] = QuantFp8(vW16),
            [$"{p}.attn.to_q.bias"] = qB,
            [$"{p}.attn.to_k.bias"] = kB,
            [$"{p}.attn.to_v.bias"] = vB,
            [$"{p}.proj_mlp.weight"] = QuantFp8(mlpW16),
            [$"{p}.proj_mlp.bias"] = mlpB,
            [$"{p}.proj_out.weight"] = QuantFp8(projOut16),
            [$"{p}.proj_out.bias"] = RandF32(10, Hidden),
            [$"{p}.attn.norm_q.weight"] = RandF32(11, HeadDim),
            [$"{p}.attn.norm_k.weight"] = RandF32(12, HeadDim),
        };
        Dictionary<string, Tensor> fused = new(split);
        fused[$"{p}.linear1.weight"] = QuantFp8(lin1W16);   // ONE common scale across q/k/v/mlp (BFL layout)
        fused[$"{p}.linear1.bias"] = ConcatRows(qB, kB, vB, mlpB);
        foreach (string k in new[] { "attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp" })
        {
            fused.Remove($"{p}.{k}.weight");
            fused.Remove($"{p}.{k}.bias");
        }
        qW16.Dispose(); kW16.Dispose(); vW16.Dispose(); mlpW16.Dispose(); lin1W16.Dispose(); projOut16.Dispose();

        ChromaSingleStreamBlock splitBlock = new ChromaSingleStreamBlock(Hidden, Heads, HeadDim);
        splitBlock.LoadWeights(split, p);
        ChromaSingleStreamBlock fusedBlock = new ChromaSingleStreamBlock(Hidden, Heads, HeadDim);
        fusedBlock.LoadWeights(fused, p);

        FluxRope rope = new FluxRope([16, 56, 56], 10000);
        using Tensor posIds = FluxRope.BuildPositionIds(TxtSeq, ImgH, ImgW);
        rope.Precompute(posIds);

        using Tensor x = RandF16(100, 1, Seq, Hidden);
        using Tensor temb = RandF32(101, 1, 3, Hidden);

        const int Trials = 8;
        double[] splitMs = new double[Trials];
        double[] fusedMs = new double[Trials];
        splitBlock.Forward(cuda, x, temb, rope, null).Dispose();
        fusedBlock.Forward(cuda, x, temb, rope, null).Dispose();
        cuda.Sync();
        Stopwatch sw = new Stopwatch();
        for (int t = 0; t < Trials; t++)
        {
            sw.Restart();
            splitBlock.Forward(cuda, x, temb, rope, null).Dispose();
            cuda.Sync();
            sw.Stop();
            splitMs[t] = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            fusedBlock.Forward(cuda, x, temb, rope, null).Dispose();
            cuda.Sync();
            sw.Stop();
            fusedMs[t] = sw.Elapsed.TotalMilliseconds;
        }
        Report("fp8", splitMs, fusedMs);
    }

    private void Report(string tag, double[] splitMs, double[] fusedMs)
    {
        int n = splitMs.Length;
        double MeanOf(double[] xs) { double s = 0; foreach (double v in xs) s += v; return s / xs.Length; }
        double StdOf(double[] xs, double mean) { double s = 0; foreach (double v in xs) s += (v - mean) * (v - mean); return Math.Sqrt(s / (xs.Length - 1)); }
        double mS = MeanOf(splitMs), mF = MeanOf(fusedMs);
        double sS = StdOf(splitMs, mS), sF = StdOf(fusedMs, mF);
        double welch = (mS - mF) / Math.Sqrt(sS * sS / n + sF * sF / n);
        _output.WriteLine($"[{tag}] split: {mS:F2} ± {sS:F2} ms | fused: {mF:F2} ± {sF:F2} ms | " +
            $"speedup {mS / mF:F3}x | Welch t={welch:F1}");
        _output.WriteLine($"[{tag}] split trials: [{string.Join(", ", splitMs.Select(v => v.ToString("F2")))}]");
        _output.WriteLine($"[{tag}] fused trials: [{string.Join(", ", fusedMs.Select(v => v.ToString("F2")))}]");
    }
}
