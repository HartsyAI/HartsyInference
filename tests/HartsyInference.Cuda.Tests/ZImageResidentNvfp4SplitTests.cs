using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.ModelAssets.Nvfp4;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Z-Image splits its fused <c>attention.qkv.weight</c> into three host copies at load, which is where an
/// nvfp4 checkpoint opened with <c>ResidentNvfp4</c> (Blackwell + <c>numerics.fp4Native</c>) meets the packed bytes.
/// A split that drops <see cref="Tensor.QuantInfo"/> hands <c>Linear</c> an F4E2M1 weight nothing can consume, and it
/// falls through to the generic GGUF dequant table. Runs on any CUDA card: the unpack path, not the native GEMM, is
/// what a split weight has to survive.</summary>
[Collection("CudaSerial")]
public sealed class ZImageResidentNvfp4SplitTests
{
    private readonly ITestOutputHelper _output;
    public ZImageResidentNvfp4SplitTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public unsafe void SplitQkv_KeepsAResidentNvfp4WeightConsumable()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        using CudaBackend backend = new CudaBackend(0, ptxDir);

        // h is a multiple of the 128-row block-scale swizzle tile, as Z-Image's own 3840 is.
        const int h = 256, m = 64;
        int paddedCols = (h / Nvfp4Codec.GroupSize + 3) / 4 * 4;
        using Tensor packed = new Tensor(new TensorShape(3 * h, h / 2), DType.U8);
        using Tensor scales = new Tensor(new TensorShape(3 * h, paddedCols), DType.F8E4M3);
        using Tensor global = new Tensor(new TensorShape(1), DType.F32);
        using Tensor input = new Tensor(new TensorShape(m, h), DType.F32);
        Random rng = new Random(91);
        byte* wp = (byte*)packed.DataPointer;
        for (long i = 0; i < (long)3 * h * (h / 2); i++) wp[i] = (byte)rng.Next(256);
        byte* sp = (byte*)scales.DataPointer;
        for (long i = 0; i < (long)3 * h * paddedCols; i++) sp[i] = (byte)(0x20 + rng.Next(0x21));   // 2^-3 .. 2^1
        ((float*)global.DataPointer)[0] = 0.01f;
        for (long i = 0; i < (long)m * h; i++) ((float*)input.DataPointer)[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        Assert.True(Nvfp4Codec.TryAttachResident(packed, scales, global, hasPreQuantScale: false, out Tensor fused));

        // The whole weight decoded on the host is the oracle; each third of it is what the matching split must be.
        using Tensor fusedWide = Nvfp4ResidentCodec.DequantToBf16(fused, scales, global);
        (Tensor q, Tensor k, Tensor v) = ZImageBlock.SplitQkv(fused, h);
        Tensor[] splits = [q, k, v];
        try
        {
            // fp4Native as the failing run had it: off Blackwell the dispatch gate refuses and the weight unpacks
            // per GEMM, which is the path a split weight with no companions cannot take either.
            backend.EnableNativeFp4Gemm = true;
            for (int i = 0; i < 3; i++)
            {
                Tensor split = splits[i];
                Assert.Equal(DType.F4E2M1, split.DType);
                Assert.NotNull(split.QuantInfo?.BlockScale);
                Assert.Equal(h, split.QuantInfo!.BlockScale!.Shape[0]);
                long chunkBytes = split.DType.ComputeByteCount((long)h * h);
                Assert.True(new ReadOnlySpan<byte>((byte*)split.DataPointer, (int)chunkBytes)
                    .SequenceEqual(new ReadOnlySpan<byte>(wp + i * chunkBytes, (int)chunkBytes)),
                    $"split {i} does not hold the fused weight's rows [{i * h}..{(i + 1) * h}).");

                using Tensor wideSlice = fusedWide.SliceRows((long)i * h, h);
                using Tensor got = new Tensor(new TensorShape(m, h), DType.F16);
                using Tensor want = new Tensor(new TensorShape(m, h), DType.F16);
                backend.Linear(got, input, split, null);   // threw the GGUF dequant table before the fix
                backend.Linear(want, input, wideSlice, null);
                backend.Sync();

                double sumAbs = 0, sumRefAbs = 0;
                Half* gp = (Half*)got.DataPointer;
                Half* rp = (Half*)want.DataPointer;
                for (long e = 0; e < (long)m * h; e++)
                {
                    float a = (float)gp[e], r = (float)rp[e];
                    Assert.False(float.IsNaN(a) || float.IsInfinity(a), $"split {i} output non-finite at {e}: {a}");
                    sumAbs += MathF.Abs(a - r);
                    sumRefAbs += MathF.Abs(r);
                }
                float relErr = (float)(sumAbs / Math.Max(sumRefAbs, 1e-9));
                _output.WriteLine($"split {i}: rel_err={relErr:E3} vs the host-decoded rows");
                Assert.True(relErr < 2e-2f, $"split {i} rel_err {relErr:E3} — it multiplies the wrong rows");
            }
        }
        finally
        {
            backend.FreeWeights(splits);
            foreach (Tensor split in splits) split.Dispose();
        }

        // A packed base cannot merge a LoRA into its bytes, so it carries one as a runtime adjunct — which the split
        // has to narrow with the rows, or Q/K/V silently run unpatched while out/ffn stay patched.
        using Tensor down = new Tensor(new TensorShape(4, h), DType.F32);
        using Tensor up = new Tensor(new TensorShape(3 * h, 4), DType.F32);
        LowRankAdjunct adjunct = new() { Terms = [new LowRankAdjunctTerm { Down = down, Up = up, Scale = 0.5f }] };
        using Tensor patched = fused.WithLowRankAdjunct(adjunct);
        (Tensor pq, Tensor pk, Tensor pv) = ZImageBlock.SplitQkv(patched, h);
        try
        {
            foreach (Tensor split in (Tensor[])[pq, pk, pv])
            {
                Assert.NotNull(split.LowRankAdjunct);
                Assert.Equal(h, split.LowRankAdjunct!.OutFeatures);
                Assert.Equal(h, split.LowRankAdjunct.InFeatures);
            }
        }
        finally
        {
            pq.Dispose();
            pk.Dispose();
            pv.Dispose();
            fused.Dispose();
        }
    }
}
