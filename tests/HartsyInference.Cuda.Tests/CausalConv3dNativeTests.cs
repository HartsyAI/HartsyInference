using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>CausalConv3d on CUDA through the native cuDNN 3-D convolution (<see cref="IBackend.TryConv3DFrameMajor"/>)
/// against the same layer's per-tap 2-D decomposition on the same backend. Covers a streaming cache prepend, temporal
/// stride 2, odd spatial sizes and both compute dtypes; both routes accumulate in F32, so they agree to rounding.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CausalConv3dNativeTests(ITestOutputHelper output)
{
    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(false, 1, false)]
    [InlineData(false, 2, false)]
    [InlineData(true, 1, true)]
    public void NativeConv3d_MatchesPerTapDecomposition(bool bf16, int strideT, bool withCache)
    {
        if (!CudaContext.IsAvailable()) { output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptx = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptx)) ptx = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        DType dtype = bf16 ? DType.BF16 : DType.F32;
        // Padded frames × channels × H × W must reach the channels-last threshold, or the native route declines.
        const int cIn = 48, cOut = 32, tIn = 5, h = 250, w = 400;   // 7 padded frames: 33.6M
        using Tensor weight = Random(new TensorShape([cOut, cIn, 3, 3, 3]), 1, 0.2f);
        using Tensor bias = Random(new TensorShape(cOut), 2, 0.5f);
        using Tensor input32 = Random(new TensorShape([1L, cIn, tIn, h, w]), 3, 1f);
        using Tensor cache32 = Random(new TensorShape([1L, cIn, 2, h, w]), 4, 1f);
        using Tensor input = input32.CastTo(dtype);
        using Tensor cache = cache32.CastTo(dtype);
        CausalConv3d conv = new(weight, bias, strideT: strideT, padT: 1, padH: 1, padW: 1, computeDtype: dtype);

        using CudaBackend backend = new(0, ptx);
        float[] native = Run(backend, conv, input, withCache ? cache : null, disableNative: false);
        Assert.True(backend.ChannelsLastConvCount > 0, "the native 3-D route did not run");
        float[] perTap = Run(backend, conv, input, withCache ? cache : null, disableNative: true);

        Assert.Equal(perTap.Length, native.Length);
        double maxDiff = 0, maxAbs = 0;
        for (int i = 0; i < native.Length; i++)
        {
            maxDiff = Math.Max(maxDiff, Math.Abs(native[i] - perTap[i]));
            maxAbs = Math.Max(maxAbs, Math.Abs(perTap[i]));
        }
        output.WriteLine($"bf16={bf16} strideT={strideT} cache={withCache}: max|native - perTap| = {maxDiff:E3} (max |out| {maxAbs:F2})");
        // F32 compute: TF32-class tensor-core rounding in either route. BF16: one BF16 rounding of the output.
        Assert.True(maxDiff <= (bf16 ? 3e-2 : 2e-3) * Math.Max(1, maxAbs), $"native 3-D conv diverges by {maxDiff:E3}");
    }

    private static float[] Run(CudaBackend backend, CausalConv3d conv, Tensor input, Tensor? cache, bool disableNative)
    {
        CausalConv3d.DisableNativeConv3d = disableNative;
        try
        {
            using Tensor outT = conv.Forward(backend, input, cache);
            backend.Sync();
            using Tensor f32 = outT.CastTo(DType.F32);
            return new ReadOnlySpan<float>(f32.DataPointer, (int)f32.ElementCount).ToArray();
        }
        finally
        {
            CausalConv3d.DisableNativeConv3d = false;
        }
    }

    private static Tensor Random(TensorShape shape, int seed, float scale)
    {
        Tensor t = new(shape, DType.F32);
        Random rng = new(seed);
        for (long i = 0; i < t.ElementCount; i++) ((float*)t.DataPointer)[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return t;
    }
}
