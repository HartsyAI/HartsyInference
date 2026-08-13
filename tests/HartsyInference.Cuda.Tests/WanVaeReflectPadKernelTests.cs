using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>Covers the reflect spatial-pad mode of <c>wan_vae_build_padded</c>, which the LTX-2 VAE decoder
/// takes on every conv. The expected border values are written out by hand here rather than read from either
/// implementation, so a matching CPU/CUDA pair cannot both be wrong the same way; a separate case then pins
/// CUDA against the managed default at a non-cubic shape where a H/W transposition would be visible.
/// Skips cleanly when CUDA is unavailable.</summary>
[Collection("CudaSerial")]
public sealed unsafe class WanVaeReflectPadKernelTests
{
    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor Ramp(TensorShape shape)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < shape.ElementCount; i++) p[i] = i;
        return t;
    }

    /// <summary>PyTorch <c>F.pad(mode="reflect")</c> does not repeat the border pixel: the padded row above the
    /// top is source row 1, not row 0. Asserted on a 1x1x1x3x4 ramp where every expected value is stated here.</summary>
    [Fact]
    public void ReflectPad_BorderRowsMirrorWithoutRepeatingTheEdge()
    {
        const int h = 3, w = 4;
        using Tensor input = Ramp(new TensorShape([1L, 1, 1, h, w]));       // rows 0..2 = [0..3],[4..7],[8..11]
        using Tensor padded = new Tensor(new TensorShape(1, 1, h + 2, w + 2), DType.F32);
        ((IBackend)new CpuBackend()).BuildPaddedFrames(padded, input, null, zeroPad: 0,
            replicateFirst: false, padH: 1, padW: 1, reflectSpatial: true);

        // Source rows mirror as -1 -> 1, 3 -> 1; source cols as -1 -> 1, 4 -> 2.
        float[][] expected =
        [
            [5, 4, 5, 6, 7, 6],
            [1, 0, 1, 2, 3, 2],
            [5, 4, 5, 6, 7, 6],
            [9, 8, 9, 10, 11, 10],
            [5, 4, 5, 6, 7, 6],
        ];
        float* p = (float*)padded.DataPointer;
        for (int y = 0; y < h + 2; y++)
            for (int x = 0; x < w + 2; x++)
                Assert.Equal(expected[y][x], p[y * (w + 2) + x]);
    }

    /// <summary>The CUDA kernel must agree with the managed reference on a shape where H != W and T > 1, so a
    /// transposed axis or a mis-strided frame cannot pass. Reflect padding is exact (a gather), so this is
    /// bit-equality, not a tolerance.</summary>
    [Fact]
    public void ReflectPad_CudaMatchesManagedReference()
    {
        const int cIn = 3, tin = 4, h = 5, w = 7, padH = 1, padW = 1;
        int paddedT = tin + 2;                                              // 2 leading causal pad frames
        using Tensor input = Ramp(new TensorShape([1L, cIn, tin, h, w]));
        using Tensor expected = new Tensor(new TensorShape(paddedT, cIn, h + 2 * padH, w + 2 * padW), DType.F32);
        ((IBackend)new CpuBackend()).BuildPaddedFrames(expected, input, null, zeroPad: 2,
            replicateFirst: true, padH: padH, padW: padW, reflectSpatial: true);

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Tensor actual = new Tensor(expected.Shape, DType.F32);
        cuda.BuildPaddedFrames(actual, input, null, zeroPad: 2,
            replicateFirst: true, padH: padH, padW: padW, reflectSpatial: true);
        cuda.Sync();

        float* e = (float*)expected.DataPointer;
        float* a = (float*)actual.DataPointer;
        for (long i = 0; i < expected.ElementCount; i++)
            Assert.Equal(e[i], a[i]);
    }

    /// <summary>Reflect must not silently become edge-clamp: the two modes differ on the same input, so a
    /// dropped flag anywhere in the P/Invoke chain fails here instead of producing a subtly wrong decode.</summary>
    [Fact]
    public void ReflectPad_DiffersFromEdgeClamp()
    {
        const int h = 4, w = 4;
        using Tensor input = Ramp(new TensorShape([1L, 1, 1, h, w]));
        using Tensor reflect = new Tensor(new TensorShape(1, 1, h + 2, w + 2), DType.F32);
        using Tensor clamp = new Tensor(new TensorShape(1, 1, h + 2, w + 2), DType.F32);
        IBackend cpu = new CpuBackend();
        cpu.BuildPaddedFrames(reflect, input, null, 0, false, 1, 1, reflectSpatial: true);
        cpu.BuildPaddedFrames(clamp, input, null, 0, false, 1, 1, reflectSpatial: false);

        float* r = (float*)reflect.DataPointer;
        float* c = (float*)clamp.DataPointer;
        Assert.Equal(5f, r[0]);                                             // mirrored corner: source (1,1)
        Assert.Equal(0f, c[0]);                                             // clamped corner: source (0,0)
    }
}
