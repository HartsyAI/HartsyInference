using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>CPU tests for the Chroma Radiance NeRF head fold back to pixels. The head's per-pixel stream is
/// [B·N, P², nh]; UnpatchifyTokens validates [B, N, nh·P²]. Handing it the per-patch shape made the contract read
/// B·N as the batch, which is the "[4096, 4096, 16384]; got [4096, 256, 64]" failure seen on prod at 1024².</summary>
public sealed unsafe class ChromaRadianceNerfHeadTests
{
    private const int TokenDim = 16;
    private const int NerfHidden = 8;
    private const int MaxFreqs = 2;
    private const int Depth = 2;
    private const int MlpRatio = 4;
    private const int Patch = 4;
    private const int Height = 8;
    private const int Width = 12;
    private const int Patches = Height / Patch * (Width / Patch);

    // F32 only: the CPU backend has no F16 BatchedMatMul, so the numerics.ditF16 path is a GPU-only check.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Forward_FoldsToImageShape(bool convFinalLayer)
    {
        using IDisposable _ = PinF32Activations();
        CpuBackend backend = new();
        using ChromaRadianceNerfHead head = BuildHead(convFinalLayer);

        using Tensor noisy = Random(new TensorShape(1, 3, Height, Width), seed: 21);
        using Tensor tokens = Random(new TensorShape(1, Patches, TokenDim), seed: 22);
        using Tensor x0 = head.Forward(backend, noisy, tokens);

        Assert.Equal(new TensorShape(1, 3, Height, Width), x0.Shape);
        float* p = (float*)x0.DataPointer;
        for (long i = 0; i < x0.ElementCount; i++)
            Assert.True(float.IsFinite(p[i]));
    }

    [Fact]
    public void Forward_BatchOfTwo_EqualsTwoSingleForwards()
    {
        using IDisposable _ = PinF32Activations();
        CpuBackend backend = new();
        using ChromaRadianceNerfHead head = BuildHead(convFinalLayer: true);

        using Tensor noisy = Random(new TensorShape(2, 3, Height, Width), seed: 31);
        using Tensor tokens = Random(new TensorShape(2, Patches, TokenDim), seed: 32);
        using Tensor both = head.Forward(backend, noisy, tokens);

        long imageElems = 3L * Height * Width;
        long tokenElems = (long)Patches * TokenDim;
        float* bothData = (float*)both.DataPointer;
        for (int b = 0; b < 2; b++)
        {
            using Tensor noisyOne = Slice(noisy, b, imageElems, new TensorShape(1, 3, Height, Width));
            using Tensor tokensOne = Slice(tokens, b, tokenElems, new TensorShape(1, Patches, TokenDim));
            using Tensor one = head.Forward(backend, noisyOne, tokensOne);
            float* oneData = (float*)one.DataPointer;
            for (long i = 0; i < imageElems; i++)
                Assert.Equal(oneData[i], bothData[b * imageElems + i], 5);
        }
    }

    /// <summary>The linear final layer and the GLU blocks act per pixel, so one input pixel can only move the
    /// output at that same pixel. A fold with the patch interior or the channel order wrong scatters it.</summary>
    [Fact]
    public void Forward_LinearVariant_PixelChangeStaysAtThatPixel()
    {
        using IDisposable _ = PinF32Activations();
        CpuBackend backend = new();
        using ChromaRadianceNerfHead head = BuildHead(convFinalLayer: false);

        using Tensor noisy = Random(new TensorShape(1, 3, Height, Width), seed: 41);
        using Tensor tokens = Random(new TensorShape(1, Patches, TokenDim), seed: 42);
        using Tensor before = head.Forward(backend, noisy, tokens);

        const int y = 5, x = 6;
        float* n = (float*)noisy.DataPointer;
        for (int c = 0; c < 3; c++)
            n[((long)c * Height + y) * Width + x] += 0.75f;
        using Tensor after = head.Forward(backend, noisy, tokens);

        float* a = (float*)before.DataPointer;
        float* b = (float*)after.DataPointer;
        bool changedAtPixel = false;
        for (int c = 0; c < 3; c++)
            for (int py = 0; py < Height; py++)
                for (int px = 0; px < Width; px++)
                {
                    long i = ((long)c * Height + py) * Width + px;
                    if (py == y && px == x)
                        changedAtPixel |= a[i] != b[i];
                    else
                        Assert.Equal(a[i], b[i]);
                }
        Assert.True(changedAtPixel, "perturbing an input pixel did not change that output pixel");
    }

    private static ChromaRadianceNerfHead BuildHead(bool convFinalLayer)
    {
        ChromaRadianceNerfHead head = new(Patch, NerfHidden, MaxFreqs, Depth, MlpRatio);
        head.LoadWeights(ChromaRadianceSyntheticWeights.BuildNerfHead(
            TokenDim, NerfHidden, MaxFreqs, Depth, MlpRatio, convFinalLayer));
        Assert.Equal(convFinalLayer, head.UsesConvFinalLayer);
        return head;
    }

    private static IDisposable PinF32Activations() =>
        KnobProfile.Create("radiance-nerf-act").With(EngineKnobs.DitF16, false).Push();

    private static Tensor Random(TensorShape shape, int seed)
    {
        Tensor t = new(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.ElementCount; i++)
            p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static Tensor Slice(Tensor source, int index, long elems, TensorShape shape)
    {
        Tensor t = new(shape, DType.F32);
        Buffer.MemoryCopy((float*)source.DataPointer + index * elems, (void*)t.DataPointer,
            elems * sizeof(float), elems * sizeof(float));
        return t;
    }
}
