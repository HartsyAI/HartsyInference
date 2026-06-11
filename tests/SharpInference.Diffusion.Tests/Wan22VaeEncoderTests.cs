using Xunit;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Tests.Common;

namespace SharpInference.Diffusion.Tests;

/// <summary>Structural tests for the Wan2.2 VAE encoder (single-frame / first-chunk path): 16× spatial compression to
/// the 48-channel normalized latent, encode→decode round trip through the (reused) decoder, the RGB24 byte entry, and
/// the AvgDown3D shortcut pooling invariants. Numerics vs the real checkpoint are validation-pending.</summary>
public unsafe class Wan22VaeEncoderTests
{
    private const int Dim = 8;
    private static readonly int[] DimMult = [1, 2, 4, 4];
    private static readonly bool[] TDown = [true, true, false];

    private static Wan22VaeEncoder BuildEncoder()
    {
        Wan22VaeEncoder enc = new(dim: Dim, zDim: 48, dimMult: DimMult, numResBlocks: 2, temperalDownsample: TDown);
        enc.LoadWeights(LanceSyntheticWeights.BuildVaeEncoder(Dim, 48, DimMult, 2, TDown));
        return enc;
    }

    [Fact]
    public void EncodeFrame_Compresses16xToNormalizedLatent()
    {
        CpuBackend backend = new();
        Wan22VaeEncoder enc = BuildEncoder();
        Tensor rgb = Rand5d(1, 3, 1, 32, 32, seed: 11);

        Tensor latent = enc.EncodeFrame(backend, rgb);

        Assert.Equal(5, latent.Shape.Rank);
        Assert.Equal(48, (int)latent.Shape[1]);
        Assert.Equal(1, (int)latent.Shape[2]);
        Assert.Equal(2, (int)latent.Shape[3]);     // 32 / 16
        Assert.Equal(2, (int)latent.Shape[4]);
        float* p = (float*)latent.DataPointer;
        for (long i = 0; i < latent.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }

    [Fact]
    public void EncodeThenDecode_RoundTripsShapes()
    {
        CpuBackend backend = new();
        Wan22VaeEncoder enc = BuildEncoder();
        Wan22VaeDecoder dec = new(dim: Dim, zDim: 48, dimMult: DimMult, numResBlocks: 2, temperalUpsample: [false, true, true]);
        dec.LoadWeights(LanceSyntheticWeights.BuildVae(Dim, 48, DimMult, 2, [false, true, true]));

        Tensor rgb = Rand5d(1, 3, 1, 32, 32, seed: 13);
        Tensor latent = enc.EncodeFrame(backend, rgb);
        Tensor decoded = dec.Decode(backend, latent);

        Assert.Equal(3, (int)decoded.Shape[1]);
        Assert.Equal(1, (int)decoded.Shape[2]);
        Assert.Equal(32, (int)decoded.Shape[3]);
        Assert.Equal(32, (int)decoded.Shape[4]);
    }

    [Fact]
    public void EncodeRgbFrame_AcceptsInterleavedBytes()
    {
        CpuBackend backend = new();
        Wan22VaeEncoder enc = BuildEncoder();
        byte[] rgb24 = new byte[32 * 32 * 3];
        Random rng = new(17);
        rng.NextBytes(rgb24);

        Tensor latent = enc.EncodeRgbFrame(backend, rgb24, width: 32, height: 32);
        Assert.Equal(48, (int)latent.Shape[1]);
        Assert.Equal(2, (int)latent.Shape[3]);
        Assert.Equal(2, (int)latent.Shape[4]);

        Assert.Throws<ArgumentException>(() => enc.EncodeRgbFrame(backend, new byte[10], 32, 32));
    }

    [Fact]
    public void AvgDown3D_PoolsCellsAndGroupsChannels()
    {
        // 1 channel, T=2, 2×2 spatial, factors (2,2): out 1 channel, all 8 cells in one group → plain mean.
        Tensor x = new Tensor(new TensorShape([1L, 1, 2, 2, 2]), DType.F32);
        float* p = (float*)x.DataPointer;
        for (int i = 0; i < 8; i++) p[i] = i + 1;     // mean = 4.5

        Tensor pooled = AvgDown3D.Forward(x, outChannels: 1, factorT: 2, factorS: 2);
        Assert.Equal(1, (int)pooled.Shape[2]);
        Assert.Equal(1, (int)pooled.Shape[3]);
        Assert.Equal(4.5f, *(float*)pooled.DataPointer, 4);

        // T=1 with factor_t=2: the front zero-pad halves nothing — channel-stacked output keeps the real frame.
        Tensor single = new Tensor(new TensorShape([1L, 1, 1, 2, 2]), DType.F32);
        float* sp = (float*)single.DataPointer;
        for (int i = 0; i < 4; i++) sp[i] = 2f;
        Tensor pooledSingle = AvgDown3D.Forward(single, outChannels: 2, factorT: 2, factorS: 2);   // group = 1·8/2 = 4
        Assert.Equal(1, (int)pooledSingle.Shape[2]);
        float* psp = (float*)pooledSingle.DataPointer;
        // Channel-stack order (c, t, h, w): out ch 0 = padded tap (zeros), out ch 1 = real frame (mean 2).
        Assert.Equal(0f, psp[0], 4);
        Assert.Equal(2f, psp[1], 4);
    }

    private static Tensor Rand5d(int b, int c, int t, int h, int w, int seed)
    {
        Tensor x = new Tensor(new TensorShape([(long)b, c, t, h, w]), DType.F32);
        float* p = (float*)x.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < x.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return x;
    }
}
