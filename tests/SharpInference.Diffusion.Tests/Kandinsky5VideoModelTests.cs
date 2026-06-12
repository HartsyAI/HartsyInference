using Xunit;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Denoisers.DiTBlocks;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Tests.Common;

namespace SharpInference.Diffusion.Tests;

/// <summary>Structural tests for the Kandinsky 5 video stack on CPU with tiny synthetic weights:
/// video transformer forward (T &gt; 1, 33-channel-style visual-cond packing), the T2I forward after the
/// image/video unification, RoPE scale factors, and the HunyuanVideo VAE encoder/decoder shapes.
/// Numerics vs the real checkpoints are validation-gated.</summary>
public unsafe class Kandinsky5VideoModelTests
{
    [Fact]
    public void ForwardVideo_TinyConfig_ProducesBcthwVelocity()
    {
        CpuBackend backend = new();
        Kandinsky5Config config = Kandinsky5SyntheticWeights.TinyVideoConfig;
        using Kandinsky5Transformer transformer = new(config);
        transformer.LoadWeights(Kandinsky5SyntheticWeights.BuildTransformer(config));

        // 3 latent frames, 8x8 latent, packed channels = 2*4+1 = 9.
        Tensor latent = Rand([1, config.VisualEmbedDim, 3, 8, 8], seed: 11);
        Tensor qwen = Rand([1, 5, config.InTextDim], seed: 12);
        Tensor clip = Rand([1, config.InTextDim2], seed: 13);

        Tensor velocity = transformer.ForwardVideo(backend, latent, 500f, qwen, clip, 1f, 2f, 2f);

        Assert.Equal(5, velocity.Shape.Rank);
        Assert.Equal(1, velocity.Shape[0]);
        Assert.Equal(config.OutVisualDim, velocity.Shape[1]);
        Assert.Equal(3, velocity.Shape[2]);
        Assert.Equal(8, velocity.Shape[3]);
        Assert.Equal(8, velocity.Shape[4]);
        AssertFinite(velocity);

        velocity.Dispose();
        latent.Dispose(); qwen.Dispose(); clip.Dispose();
    }

    [Fact]
    public void ForwardVideo_WrongChannelPacking_Throws()
    {
        CpuBackend backend = new();
        Kandinsky5Config config = Kandinsky5SyntheticWeights.TinyVideoConfig;
        using Kandinsky5Transformer transformer = new(config);
        transformer.LoadWeights(Kandinsky5SyntheticWeights.BuildTransformer(config));

        // Bare 4-channel latent without the cond+mask packing must fail fast.
        Tensor bare = Rand([1, config.InVisualDim, 3, 8, 8], seed: 14);
        Tensor qwen = Rand([1, 5, config.InTextDim], seed: 15);
        Tensor clip = Rand([1, config.InTextDim2], seed: 16);

        Assert.Throws<ArgumentException>(() => transformer.ForwardVideo(backend, bare, 500f, qwen, clip));

        bare.Dispose(); qwen.Dispose(); clip.Dispose();
    }

    [Fact]
    public void Forward_ImageVariant_StillProducesBchwVelocity()
    {
        CpuBackend backend = new();
        Kandinsky5Config config = Kandinsky5SyntheticWeights.TinyImageConfig;
        using Kandinsky5Transformer transformer = new(config);
        transformer.LoadWeights(Kandinsky5SyntheticWeights.BuildTransformer(config));

        Tensor latent = Rand([1, config.InVisualDim, 8, 8], seed: 21);
        Tensor qwen = Rand([1, 5, config.InTextDim], seed: 22);
        Tensor clip = Rand([1, config.InTextDim2], seed: 23);

        Tensor velocity = transformer.Forward(backend, latent, 500f, qwen, clip);

        Assert.Equal(4, velocity.Shape.Rank);
        Assert.Equal(config.OutVisualDim, velocity.Shape[1]);
        Assert.Equal(8, velocity.Shape[2]);
        Assert.Equal(8, velocity.Shape[3]);
        AssertFinite(velocity);

        velocity.Dispose();
        latent.Dispose(); qwen.Dispose(); clip.Dispose();
    }

    [Fact]
    public void Rope_ScaleFactors_DivideAngles_AndDefaultMatchesLegacy()
    {
        int[] axes = [2, 2, 2];
        Kandinsky5Rope reference = new(headDim: 6);
        reference.Precompute3D(axes, duration: 2, height: 2, width: 2);

        Kandinsky5Rope explicitOnes = new(headDim: 6);
        explicitOnes.Precompute3D(axes, 2, 2, 2, 0, 1f, 1f, 1f);

        Kandinsky5Rope scaled = new(headDim: 6);
        scaled.Precompute3D(axes, 2, 2, 2, 0, 1f, 2f, 2f);

        // Unit-pair probes: after Apply, pair (1, 0) becomes (cos θ, sin θ).
        Tensor qRef = OnesZeroPairs(8, 6);
        Tensor kRef = OnesZeroPairs(8, 6);
        reference.Apply(qRef, kRef, 1, 1, 8);

        Tensor qOnes = OnesZeroPairs(8, 6);
        Tensor kOnes = OnesZeroPairs(8, 6);
        explicitOnes.Apply(qOnes, kOnes, 1, 1, 8);

        Tensor qScaled = OnesZeroPairs(8, 6);
        Tensor kScaled = OnesZeroPairs(8, 6);
        scaled.Apply(qScaled, kScaled, 1, 1, 8);

        float* pr = (float*)qRef.DataPointer;
        float* po = (float*)qOnes.DataPointer;
        float* ps = (float*)qScaled.DataPointer;

        // Defaults must be bit-identical to the legacy (T2I) precompute.
        for (int i = 0; i < 8 * 6; i++)
            Assert.Equal(pr[i], po[i]);

        // Token (t=0, h=1, w=0) is s=2; pairs are [t-pair, h-pair, w-pair]; h-pair index 1.
        // Unscaled: θ_h = 1.0; scaled by 2 → θ_h = 0.5. The t/w angles are 0 at this token.
        int tokenOff = 2 * 6;
        Assert.Equal(MathF.Cos(1f), pr[tokenOff + 2], 5);
        Assert.Equal(MathF.Sin(1f), pr[tokenOff + 3], 5);
        Assert.Equal(MathF.Cos(0.5f), ps[tokenOff + 2], 5);
        Assert.Equal(MathF.Sin(0.5f), ps[tokenOff + 3], 5);

        qRef.Dispose(); kRef.Dispose(); qOnes.Dispose(); kOnes.Dispose(); qScaled.Dispose(); kScaled.Dispose();
    }

    [Fact]
    public void HunyuanVideoVaeDecoder_TinyConfig_DecodesExpandedClip()
    {
        CpuBackend backend = new();
        HunyuanVideoVaeConfig config = Kandinsky5SyntheticWeights.TinyVaeConfig;
        HunyuanVideoVaeDecoder decoder = new(config);
        decoder.LoadWeights(Kandinsky5SyntheticWeights.BuildVae(config));

        // T_lat=2 → F = (2-1)*4+1 = 5; spatial 4x4 → 32x32.
        Tensor latent = Rand([1, config.LatentChannels, 2, 4, 4], seed: 31);
        Tensor rgb = decoder.Decode(backend, latent);

        Assert.Equal(3, rgb.Shape[1]);
        Assert.Equal(5, rgb.Shape[2]);
        Assert.Equal(32, rgb.Shape[3]);
        Assert.Equal(32, rgb.Shape[4]);
        AssertFinite(rgb);

        rgb.Dispose();
        latent.Dispose();
    }

    [Fact]
    public void HunyuanVideoVaeEncoder_TinyConfig_EncodesToLatentGrid()
    {
        CpuBackend backend = new();
        HunyuanVideoVaeConfig config = Kandinsky5SyntheticWeights.TinyVaeConfig;
        HunyuanVideoVaeEncoder encoder = new(config);
        encoder.LoadWeights(Kandinsky5SyntheticWeights.BuildVae(config));

        // F=5 → T_lat = (5-1)/4+1 = 2; 32x32 → 4x4.
        Tensor rgb = Rand([1, 3, 5, 32, 32], seed: 32);
        Tensor latent = encoder.Encode(backend, rgb);

        Assert.Equal(config.LatentChannels, latent.Shape[1]);
        Assert.Equal(2, latent.Shape[2]);
        Assert.Equal(4, latent.Shape[3]);
        Assert.Equal(4, latent.Shape[4]);
        AssertFinite(latent);

        // Frame counts that break (F-1) % 4 == 0 fail fast.
        Tensor bad = Rand([1, 3, 4, 32, 32], seed: 33);
        Assert.Throws<ArgumentException>(() => encoder.Encode(backend, bad));

        latent.Dispose();
        rgb.Dispose();
        bad.Dispose();
    }

    /// <summary>[1, 1, seqLen, headDim] tensor whose rotation pairs are (1, 0) so RoPE writes (cos θ, sin θ).</summary>
    private static Tensor OnesZeroPairs(int seqLen, int headDim)
    {
        Tensor t = new Tensor(new TensorShape(1, 1, seqLen, headDim), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < seqLen * headDim; i++) p[i] = i % 2 == 0 ? 1f : 0f;
        return t;
    }

    private static Tensor Rand(int[] dims, int seed)
    {
        long[] d = Array.ConvertAll(dims, x => (long)x);
        Tensor t = new Tensor(new TensorShape(d), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static void AssertFinite(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.Shape.ElementCount; i++)
            Assert.True(float.IsFinite(p[i]), $"non-finite value at {i}");
    }
}
