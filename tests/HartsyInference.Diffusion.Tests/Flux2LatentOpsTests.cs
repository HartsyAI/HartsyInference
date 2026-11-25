using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;
using System.Linq;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The device-op latent layout (Pack, Patchify, the BatchNorm affine and their composites) against the scalar loops
/// it replaced. Each check is an exact element move or a per-channel affine, so the comparisons are exact or within 1e-5.</summary>
public sealed class Flux2LatentOpsTests
{
    private const int Batch = 2, Channels = 3, Patch = 2, GridH = 3, GridW = 5;
    private const int TokenChannels = Channels * Patch * Patch;
    private const float Eps = 1e-4f;

    [Fact]
    public void Pack_and_Unpack_match_the_channel_last_layout_and_invert()
    {
        float[] nchw = RandomValues(Batch * Channels * GridH * GridW, 1);
        using CpuBackend backend = new();
        using Tensor latent = Make(new TensorShape(Batch, Channels, GridH, GridW), nchw);

        using Tensor packed = Flux2LatentOps.Pack(backend, latent);
        Assert.Equal(NaivePack(nchw, Batch, Channels, GridH, GridW), packed.AsSpan<float>().ToArray());

        using Tensor unpacked = Flux2LatentOps.Unpack(backend, packed, GridH, GridW);
        Assert.Equal(nchw, unpacked.AsSpan<float>().ToArray());
    }

    [Fact]
    public void Patchify_and_Unpatchify_match_the_channel_order_and_invert()
    {
        float[] vae = RandomValues(Batch * Channels * GridH * Patch * GridW * Patch, 2);
        using CpuBackend backend = new();
        using Tensor latent = Make(new TensorShape(Batch, Channels, GridH * Patch, GridW * Patch), vae);

        using Tensor patched = Flux2LatentOps.Patchify(backend, latent, TokenChannels, Patch);
        Assert.Equal(NaivePatchify(vae, Batch, Channels, GridH, GridW, Patch), patched.AsSpan<float>().ToArray());

        using Tensor restored = Flux2LatentOps.Unpatchify(backend, patched, Channels, Patch);
        Assert.Equal(vae, restored.AsSpan<float>().ToArray());
    }

    [Fact]
    public void ToPackedTokens_matches_patchify_then_normalize_then_pack()
    {
        float[] vae = RandomValues(Batch * Channels * GridH * Patch * GridW * Patch, 3);
        float[] mean = RandomValues(TokenChannels, 4);
        float[] var = RandomValues(TokenChannels, 5).Select(v => MathF.Abs(v) + 0.5f).ToArray();
        using CpuBackend backend = new();
        using Tensor latent = Make(new TensorShape(Batch, Channels, GridH * Patch, GridW * Patch), vae);
        using Tensor meanT = Make(new TensorShape(TokenChannels), mean);
        using Tensor varT = Make(new TensorShape(TokenChannels), var);

        using Tensor tokens = Flux2LatentOps.ToPackedTokens(backend, latent, meanT, varT, Eps, Patch);

        float[] expected = NaivePack(
            NaiveNormalize(NaivePatchify(vae, Batch, Channels, GridH, GridW, Patch), Batch, TokenChannels, GridH * GridW, mean, var),
            Batch, TokenChannels, GridH, GridW);
        AssertClose(expected, tokens.AsSpan<float>().ToArray());

        using Tensor back = Flux2LatentOps.FromPackedTokens(backend, tokens, GridH, GridW, meanT, varT, Eps, Channels, Patch);
        AssertClose(vae, back.AsSpan<float>().ToArray());
    }

    [Fact]
    public void BnNormalize_and_BnUnNormalize_on_NCHW_match_the_per_channel_formula_and_invert()
    {
        float[] nchw = RandomValues(Batch * TokenChannels * GridH * GridW, 6);
        float[] mean = RandomValues(TokenChannels, 7);
        float[] var = RandomValues(TokenChannels, 8).Select(v => MathF.Abs(v) + 0.5f).ToArray();
        using CpuBackend backend = new();
        using Tensor latent = Make(new TensorShape(Batch, TokenChannels, GridH, GridW), nchw);
        using Tensor meanT = Make(new TensorShape(TokenChannels), mean);
        using Tensor varT = Make(new TensorShape(TokenChannels), var);

        using Tensor normalized = Flux2LatentOps.BnNormalize(backend, latent, meanT, varT, Eps);
        AssertClose(NaiveNormalizeNchw(nchw, Batch, TokenChannels, GridH * GridW, mean, var), normalized.AsSpan<float>().ToArray());

        using Tensor restored = Flux2LatentOps.BnUnNormalize(backend, normalized, meanT, varT, Eps);
        AssertClose(nchw, restored.AsSpan<float>().ToArray());
    }

    private static float[] NaivePack(float[] nchw, int batch, int channels, int h, int w)
    {
        float[] packed = new float[batch * h * w * channels];
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < h * w; s++)
                for (int c = 0; c < channels; c++)
                    packed[(b * h * w + s) * channels + c] = nchw[((b * channels + c) * h * w) + s];
        return packed;
    }

    private static float[] NaivePatchify(float[] vae, int batch, int channels, int h, int w, int patch)
    {
        int inW = w * patch, inH = h * patch;
        float[] output = new float[batch * channels * patch * patch * h * w];
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
                for (int py = 0; py < patch; py++)
                    for (int px = 0; px < patch; px++)
                        for (int y = 0; y < h; y++)
                            for (int x = 0; x < w; x++)
                            {
                                int outChannel = c * patch * patch + py * patch + px;
                                output[((b * channels * patch * patch + outChannel) * h + y) * w + x] =
                                    vae[((b * channels + c) * inH + y * patch + py) * inW + x * patch + px];
                            }
        return output;
    }

    private static float[] NaiveNormalize(float[] tokensNchw, int batch, int channels, int spatial, float[] mean, float[] var)
        => NaiveNormalizeNchw(tokensNchw, batch, channels, spatial, mean, var);

    private static float[] NaiveNormalizeNchw(float[] values, int batch, int channels, int spatial, float[] mean, float[] var)
    {
        float[] output = new float[values.Length];
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
                for (int s = 0; s < spatial; s++)
                {
                    int index = (b * channels + c) * spatial + s;
                    output[index] = (values[index] - mean[c]) / MathF.Sqrt(var[c] + Eps);
                }
        return output;
    }

    private static float[] RandomValues(int count, int seed)
    {
        Random generator = new(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = generator.NextSingle() * 2f - 1f;
        return values;
    }

    private static Tensor Make(TensorShape shape, float[] values)
    {
        Tensor tensor = new(shape, DType.F32);
        values.AsSpan().CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= 1e-5f, $"element {i}: expected {expected[i]}, got {actual[i]}");
    }
}
