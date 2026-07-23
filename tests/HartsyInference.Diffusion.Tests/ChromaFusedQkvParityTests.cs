using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Parity gate for the HARTSY_CHROMA_FUSED_QKV path (INFERENCE_ACCEL_GRIND §H3.1): the fused
/// consumption (one qkv GEMM + <see cref="IBackend.QkvSplitNorm"/>, the Hunyuan3DFluxBlocks recipe) must
/// match the proven split path (3 Linears + 2 RmsNorm passes) on the SAME underlying weight values. Both
/// blocks are loaded from the same random tensors — the fused dict row-concats Q/K/V exactly as the
/// converter keeps the BFL fused layout — so any output difference is a wiring bug, not checkpoint noise.
/// CPU backend: the QkvSplitNorm default host impl is the composed reference, so the tolerance covers
/// only float summation-order noise.</summary>
public sealed unsafe class ChromaFusedQkvParityTests
{
    private const int Hidden = 256;      // 2 heads × 128 (headDim fixed at 128 by the FluxRope axes [16,56,56])
    private const int Heads = 2;
    private const int HeadDim = 128;
    private const int Mlp = Hidden * 4;
    private const int TxtSeq = 2;
    private const int ImgSeq = 4;        // 2×2 packed grid

    private readonly ITestOutputHelper _output;
    public ChromaFusedQkvParityTests(ITestOutputHelper output) => _output = output;

    private static Tensor Rand(int seed, params long[] shape)
    {
        Tensor t = new Tensor(new TensorShape(shape), DType.F32);
        Random rng = new Random(seed);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
        return t;
    }

    /// <summary>Row-concats rank-2 [N,K] (or rank-1 [N]) tensors into one tensor along dim 0.</summary>
    private static Tensor ConcatRows(params Tensor[] parts)
    {
        long cols = parts[0].Shape.Rank == 2 ? parts[0].Shape[1] : 1;
        long rows = 0;
        foreach (Tensor p in parts) rows += p.Shape[0];
        Tensor fused = parts[0].Shape.Rank == 2
            ? new Tensor(new TensorShape(rows, cols), DType.F32)
            : new Tensor(new TensorShape(rows), DType.F32);
        byte* dst = (byte*)fused.DataPointer;
        foreach (Tensor p in parts)
        {
            long bytes = p.ElementCount * sizeof(float);
            Buffer.MemoryCopy((void*)p.DataPointer, dst, bytes, bytes);
            dst += bytes;
        }
        return fused;
    }

    private static float MaxRelDiff(Tensor a, Tensor b)
    {
        float* pa = (float*)a.DataPointer;
        float* pb = (float*)b.DataPointer;
        float maxRel = 0;
        for (long i = 0; i < a.ElementCount; i++)
        {
            float diff = MathF.Abs(pa[i] - pb[i]);
            float mag = MathF.Max(MathF.Abs(pa[i]), 1e-3f);
            maxRel = MathF.Max(maxRel, diff / mag);
        }
        return maxRel;
    }

    [Fact]
    public void DoubleBlock_FusedQkv_MatchesSplitPath()
    {
        const string p = "transformer_blocks.0";
        Dictionary<string, Tensor> split = new()
        {
            [$"{p}.attn.to_q.weight"] = Rand(1, Hidden, Hidden),
            [$"{p}.attn.to_k.weight"] = Rand(2, Hidden, Hidden),
            [$"{p}.attn.to_v.weight"] = Rand(3, Hidden, Hidden),
            [$"{p}.attn.to_q.bias"] = Rand(4, Hidden),
            [$"{p}.attn.to_k.bias"] = Rand(5, Hidden),
            [$"{p}.attn.to_v.bias"] = Rand(6, Hidden),
            [$"{p}.attn.add_q_proj.weight"] = Rand(7, Hidden, Hidden),
            [$"{p}.attn.add_k_proj.weight"] = Rand(8, Hidden, Hidden),
            [$"{p}.attn.add_v_proj.weight"] = Rand(9, Hidden, Hidden),
            [$"{p}.attn.add_q_proj.bias"] = Rand(10, Hidden),
            [$"{p}.attn.add_k_proj.bias"] = Rand(11, Hidden),
            [$"{p}.attn.add_v_proj.bias"] = Rand(12, Hidden),
            [$"{p}.attn.to_out.0.weight"] = Rand(13, Hidden, Hidden),
            [$"{p}.attn.to_out.0.bias"] = Rand(14, Hidden),
            [$"{p}.attn.to_add_out.weight"] = Rand(15, Hidden, Hidden),
            [$"{p}.attn.to_add_out.bias"] = Rand(16, Hidden),
            [$"{p}.attn.norm_q.weight"] = Rand(17, HeadDim),
            [$"{p}.attn.norm_k.weight"] = Rand(18, HeadDim),
            [$"{p}.attn.norm_added_q.weight"] = Rand(19, HeadDim),
            [$"{p}.attn.norm_added_k.weight"] = Rand(20, HeadDim),
            [$"{p}.ff.net.0.proj.weight"] = Rand(21, Mlp, Hidden),
            [$"{p}.ff.net.0.proj.bias"] = Rand(22, Mlp),
            [$"{p}.ff.net.2.weight"] = Rand(23, Hidden, Mlp),
            [$"{p}.ff.net.2.bias"] = Rand(24, Hidden),
            [$"{p}.ff_context.net.0.proj.weight"] = Rand(25, Mlp, Hidden),
            [$"{p}.ff_context.net.0.proj.bias"] = Rand(26, Mlp),
            [$"{p}.ff_context.net.2.weight"] = Rand(27, Hidden, Mlp),
            [$"{p}.ff_context.net.2.bias"] = Rand(28, Hidden),
        };
        Dictionary<string, Tensor> fused = new(split);
        fused[$"{p}.attn.qkv.weight"] = ConcatRows(split[$"{p}.attn.to_q.weight"], split[$"{p}.attn.to_k.weight"], split[$"{p}.attn.to_v.weight"]);
        fused[$"{p}.attn.qkv.bias"] = ConcatRows(split[$"{p}.attn.to_q.bias"], split[$"{p}.attn.to_k.bias"], split[$"{p}.attn.to_v.bias"]);
        fused[$"{p}.attn.add_qkv.weight"] = ConcatRows(split[$"{p}.attn.add_q_proj.weight"], split[$"{p}.attn.add_k_proj.weight"], split[$"{p}.attn.add_v_proj.weight"]);
        fused[$"{p}.attn.add_qkv.bias"] = ConcatRows(split[$"{p}.attn.add_q_proj.bias"], split[$"{p}.attn.add_k_proj.bias"], split[$"{p}.attn.add_v_proj.bias"]);
        foreach (string k in new[] { "to_q", "to_k", "to_v" })
        {
            fused.Remove($"{p}.attn.{k}.weight");
            fused.Remove($"{p}.attn.{k}.bias");
        }
        foreach (string k in new[] { "add_q_proj", "add_k_proj", "add_v_proj" })
        {
            fused.Remove($"{p}.attn.{k}.weight");
            fused.Remove($"{p}.attn.{k}.bias");
        }

        ChromaDoubleStreamBlock splitBlock = new ChromaDoubleStreamBlock(Hidden, Heads, HeadDim);
        splitBlock.LoadWeights(split, p);
        ChromaDoubleStreamBlock fusedBlock = new ChromaDoubleStreamBlock(Hidden, Heads, HeadDim);
        fusedBlock.LoadWeights(fused, p);

        CpuBackend backend = new CpuBackend();
        FluxRope rope = new FluxRope([16, 56, 56], 10000);
        using Tensor posIds = FluxRope.BuildPositionIds(TxtSeq, 2, 2);
        rope.Precompute(posIds);

        using Tensor image = Rand(101, 1, ImgSeq, Hidden);
        using Tensor text = Rand(102, 1, TxtSeq, Hidden);
        using Tensor temb = Rand(103, 1, 12, Hidden);

        (Tensor imgA, Tensor txtA) = splitBlock.Forward(backend, image, text, temb, rope, null);
        (Tensor imgB, Tensor txtB) = fusedBlock.Forward(backend, image, text, temb, rope, null);

        float imgDiff = MaxRelDiff(imgA, imgB);
        float txtDiff = MaxRelDiff(txtA, txtB);
        _output.WriteLine($"double block maxRelDiff: img={imgDiff:e2} txt={txtDiff:e2}");
        Assert.True(imgDiff < 1e-4f, $"image stream diverged: {imgDiff}");
        Assert.True(txtDiff < 1e-4f, $"text stream diverged: {txtDiff}");
        imgA.Dispose(); txtA.Dispose(); imgB.Dispose(); txtB.Dispose();
    }

    [Fact]
    public void SingleBlock_FusedLinear1_MatchesSplitPath()
    {
        const string p = "single_transformer_blocks.0";
        Dictionary<string, Tensor> split = new()
        {
            [$"{p}.attn.to_q.weight"] = Rand(31, Hidden, Hidden),
            [$"{p}.attn.to_k.weight"] = Rand(32, Hidden, Hidden),
            [$"{p}.attn.to_v.weight"] = Rand(33, Hidden, Hidden),
            [$"{p}.attn.to_q.bias"] = Rand(34, Hidden),
            [$"{p}.attn.to_k.bias"] = Rand(35, Hidden),
            [$"{p}.attn.to_v.bias"] = Rand(36, Hidden),
            [$"{p}.proj_mlp.weight"] = Rand(37, Mlp, Hidden),
            [$"{p}.proj_mlp.bias"] = Rand(38, Mlp),
            [$"{p}.proj_out.weight"] = Rand(39, Hidden, Hidden + Mlp),
            [$"{p}.proj_out.bias"] = Rand(40, Hidden),
            [$"{p}.attn.norm_q.weight"] = Rand(41, HeadDim),
            [$"{p}.attn.norm_k.weight"] = Rand(42, HeadDim),
        };
        Dictionary<string, Tensor> fused = new(split);
        fused[$"{p}.linear1.weight"] = ConcatRows(split[$"{p}.attn.to_q.weight"], split[$"{p}.attn.to_k.weight"],
            split[$"{p}.attn.to_v.weight"], split[$"{p}.proj_mlp.weight"]);
        fused[$"{p}.linear1.bias"] = ConcatRows(split[$"{p}.attn.to_q.bias"], split[$"{p}.attn.to_k.bias"],
            split[$"{p}.attn.to_v.bias"], split[$"{p}.proj_mlp.bias"]);
        foreach (string k in new[] { "attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp" })
        {
            fused.Remove($"{p}.{k}.weight");
            fused.Remove($"{p}.{k}.bias");
        }

        ChromaSingleStreamBlock splitBlock = new ChromaSingleStreamBlock(Hidden, Heads, HeadDim);
        splitBlock.LoadWeights(split, p);
        ChromaSingleStreamBlock fusedBlock = new ChromaSingleStreamBlock(Hidden, Heads, HeadDim);
        fusedBlock.LoadWeights(fused, p);

        CpuBackend backend = new CpuBackend();
        FluxRope rope = new FluxRope([16, 56, 56], 10000);
        using Tensor posIds = FluxRope.BuildPositionIds(TxtSeq, 2, 2);
        rope.Precompute(posIds);

        using Tensor x = Rand(111, 1, TxtSeq + ImgSeq, Hidden);
        using Tensor temb = Rand(112, 1, 3, Hidden);

        Tensor outA = splitBlock.Forward(backend, x, temb, rope, null);
        Tensor outB = fusedBlock.Forward(backend, x, temb, rope, null);

        float diff = MaxRelDiff(outA, outB);
        _output.WriteLine($"single block maxRelDiff: {diff:e2}");
        Assert.True(diff < 1e-4f, $"single block diverged: {diff}");
        outA.Dispose(); outB.Dispose();
    }
}
