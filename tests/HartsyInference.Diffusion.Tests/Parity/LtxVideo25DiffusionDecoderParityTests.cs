using System.Text;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Vae;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>Checks <see cref="LtxVideo25DiffusionDecoder"/> against the reference LTX-2.5 <c>NADiffusionDecoder</c> on
/// a tiny random-weight config, sharing weights, latent and noise through a fixture (C# and torch RNGs differ, so a
/// seed would prove nothing). The fixture names its weights with the real checkpoint keys, so this also covers the
/// key mapping. Regenerate with <c>tests/python-reference/ltx25_diffusion_decoder_reference.py</c>.</summary>
public sealed unsafe class LtxVideo25DiffusionDecoderParityTests
{
    private const double Tolerance = 1e-5;

    [Fact]
    public void RopeDimSplitMatchesTheReferenceDefault()
    {
        Assert.Equal((16, 24, 24), LtxVideo25DiffusionDecoderConfig.RopeDimSplit(64));
        Assert.Equal((4, 6, 6), LtxVideo25DiffusionDecoderConfig.RopeDimSplit(16));
        // 24 is the case where the first h/w split comes out odd and two dims move back to the t chunk.
        Assert.Equal((4, 10, 10), LtxVideo25DiffusionDecoderConfig.RopeDimSplit(24));
        // head_dim 8 would leave the temporal chunk empty, which must fail loudly rather than silently drop t-RoPE.
        Assert.Throws<ArgumentException>(() => LtxVideo25DiffusionDecoderConfig.RopeDimSplit(8));
    }

    [Fact]
    public void ShippedGeometryIsEightTimesTemporalAndThirtyTwoTimesSpatial()
    {
        LtxVideo25DiffusionDecoderConfig config = new LtxVideo25DiffusionDecoderConfig();
        Assert.Equal(8, config.TemporalUpscale);
        Assert.Equal(32, config.SpatialUpscale);
        Assert.Equal(2, config.TrailingPadLatentFrames);
        Assert.Equal(1, config.OutputFrames(1));
        Assert.Equal(9, config.OutputFrames(2));
        Assert.Equal(8 * 13 - 7, config.OutputFrames(13));
        Assert.Equal(1024, LtxVideo25DiffusionDecoderConfig.SwiGluHidden(256));
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void MatchesTheReferenceTinyDecoder()
    {
        string? path = FixturePath();
        if (path is null) return;   // tier-lint: guarded

        using BinaryReader reader = new BinaryReader(File.OpenRead(path));
        Assert.Equal(1, reader.ReadInt32());
        LtxVideo25DiffusionDecoderConfig config = ReadConfig(reader);
        Dictionary<string, Tensor> tensors = ReadTensors(reader);

        Dictionary<string, Tensor> weights = tensors
            .Where(kvp => kvp.Key.StartsWith("decoder.", StringComparison.Ordinal))
            .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);

        IBackend backend = new CpuBackend();
        using LtxVideo25DiffusionDecoder decoder = new LtxVideo25DiffusionDecoder(config);
        decoder.LoadWeights(weights);

        string[] unconsumed = [.. weights.Keys.Except(decoder.ConsumedKeys).Order()];
        Assert.Equal(["decoder.type_emb"], unconsumed);

        Tensor latent = tensors["__latent__"];
        Tensor noise = tensors["__noise__"];
        using (Tensor context = decoder.EncodeContext(backend, latent, out int frames, out int height, out int width))
        {
            Tensor expectedContext = tensors["__context__"];
            Assert.Equal(frames, (int)expectedContext.Shape[1]);
            Assert.Equal(height, (int)expectedContext.Shape[2]);
            Assert.Equal(width, (int)expectedContext.Shape[3]);
            double contextError = RelativeL2(context, expectedContext);
            Assert.True(contextError < Tolerance, $"stage 1-4 context relL2 {contextError:E3}");
        }

        Assert.Equal(noise.Shape, decoder.NoiseShape((int)latent.Shape[2], (int)latent.Shape[3], (int)latent.Shape[4]));
        using Tensor pixels = decoder.Decode(backend, latent, noise);
        Tensor expected = tensors["__pixels__"];
        Assert.Equal(expected.Shape, pixels.Shape);
        double error = RelativeL2(pixels, expected);
        Assert.True(error < Tolerance, $"decoded pixels relL2 {error:E3}");

        foreach (Tensor t in tensors.Values) t.Dispose();
    }

    private static LtxVideo25DiffusionDecoderConfig ReadConfig(BinaryReader reader)
    {
        int inChannels = reader.ReadInt32(), outChannels = reader.ReadInt32();
        int patch = reader.ReadInt32(), headDim = reader.ReadInt32();
        int stageCount = reader.ReadInt32();
        int[] stageChannels = ReadInts(reader, stageCount);
        int[] stageDepths = ReadInts(reader, stageCount);
        (int T, int H, int W)[] stageKernels = new (int, int, int)[stageCount];
        for (int i = 0; i < stageCount; i++)
            stageKernels[i] = (reader.ReadInt32(), reader.ReadInt32(), reader.ReadInt32());
        int upsampleCount = reader.ReadInt32();
        ((int T, int H, int W) Stride, int Reduction)[] upsamples = new ((int, int, int), int)[upsampleCount];
        for (int i = 0; i < upsampleCount; i++)
            upsamples[i] = ((reader.ReadInt32(), reader.ReadInt32(), reader.ReadInt32()), reader.ReadInt32());
        (int T, int H, int W) stage5Kernel = (reader.ReadInt32(), reader.ReadInt32(), reader.ReadInt32());
        int tEmbDim = reader.ReadInt32(), freqDim = reader.ReadInt32();
        float timestepScale = reader.ReadSingle();
        return new LtxVideo25DiffusionDecoderConfig
        {
            InChannels = inChannels,
            OutChannels = outChannels,
            PatchSize = patch,
            HeadDim = headDim,
            StageChannels = stageChannels,
            StageDepths = stageDepths,
            StageKernels = stageKernels,
            Upsamples = upsamples,
            Stage5Kernel = stage5Kernel,
            TimestepEmbedDim = tEmbDim,
            TimestepFreqDim = freqDim,
            TimestepScaleMultiplier = timestepScale,
        };
    }

    private static int[] ReadInts(BinaryReader reader, int count)
    {
        int[] values = new int[count];
        for (int i = 0; i < count; i++) values[i] = reader.ReadInt32();
        return values;
    }

    private static Dictionary<string, Tensor> ReadTensors(BinaryReader reader)
    {
        int count = reader.ReadInt32();
        Dictionary<string, Tensor> tensors = new(count);
        for (int i = 0; i < count; i++)
        {
            int nameLength = reader.ReadInt32();
            string name = Encoding.UTF8.GetString(reader.ReadBytes(nameLength));
            int rank = reader.ReadInt32();
            long[] dims = new long[rank];
            for (int d = 0; d < rank; d++) dims[d] = reader.ReadInt32();
            Tensor tensor = new Tensor(new TensorShape(dims), DType.F32);
            float* values = (float*)tensor.DataPointer;
            for (long e = 0; e < tensor.ElementCount; e++) values[e] = reader.ReadSingle();
            tensors[name] = tensor;
        }
        return tensors;
    }

    private static double RelativeL2(Tensor actual, Tensor expected)
    {
        Assert.Equal(expected.ElementCount, actual.ElementCount);
        float* a = (float*)actual.DataPointer, e = (float*)expected.DataPointer;
        double numerator = 0, denominator = 0;
        for (long i = 0; i < actual.ElementCount; i++)
        {
            double diff = a[i] - e[i];
            numerator += diff * diff;
            denominator += (double)e[i] * e[i];
        }
        return denominator > 0 ? Math.Sqrt(numerator / denominator) : Math.Sqrt(numerator);
    }

    private static string? FixturePath()
    {
        string? env = Environment.GetEnvironmentVariable("LTX25_DIFFUSION_DECODER_REFERENCE_BIN");
        if (!string.IsNullOrWhiteSpace(env)) return File.Exists(env) ? env : null;

        string? dir = AppContext.BaseDirectory;
        for (int up = 0; up < 8 && dir is not null; up++, dir = Path.GetDirectoryName(dir))
        {
            string candidate = Path.Combine(dir, "tests", "python-reference", "ltx25_diffusion_decoder_reference.bin");
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }
}
