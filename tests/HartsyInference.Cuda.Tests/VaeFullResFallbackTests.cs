using HartsyInference.Core.Configuration;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>When <see cref="VaeDecoder.DecodeTiled"/>'s full-resolution attempt fails, the tiled fallback must decode the
/// latent the caller passed. Z-Image hands the decoder a latent that exists only on the device, and the fallback's
/// <c>FreeActivations</c> used to drop that copy without a sync-back, so the tiles sliced a zeroed host buffer and every
/// seed came out the same flat gray. Both decodes here are tiled; the only difference is the failed attempt in front of
/// one of them, so any loss of the latent shows up as a mismatch. Weights are synthetic: the property is equality.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("CudaSerial")]
public sealed unsafe class VaeFullResFallbackTests
{
    private const int Tile = 24;
    private const int Overlap = 8;

    private static readonly VaeConfig SmallVae = new()
    {
        LatentChannels = 4,
        BlockOutChannels = [32, 32, 64, 64],
        LayersPerBlock = 1,
        NormNumGroups = 8,
        ScalingFactor = 0.5f,
        ShiftFactor = 0.1f,
        UsePostQuantConv = false,
        UseQuantConv = false,
    };

    private readonly ITestOutputHelper _output;

    public VaeFullResFallbackTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData("F32")]
    [InlineData("BF16")]
    public void FailedFullResAttempt_TilesTheDeviceResidentLatent(string weightDtype)
    {
        if (!TryOpen(out CudaBackend? backend)) return;
        using CudaBackend cuda = backend!;
        DType dtype = weightDtype == "BF16" ? DType.BF16 : DType.F32;
        using DecoderWeights weights = DecoderWeights.Create(SmallVae, dtype, seed: 3);
        VaeDecoder failing = weights.NewDecoder();
        VaeDecoder reference = weights.NewDecoder();
        int attempts = 0;
        failing.FullResFailureForTests = () =>
        {
            attempts++;
            return new OutOfVramException(1L << 34, 0);
        };

        TensorShape shape = new TensorShape(1, SmallVae.LatentChannels, 40, 40);
        using Tensor host = Gaussian(shape, seed: 5);
        using Tensor deviceA = DeviceResident(cuda, host);
        using Tensor afterFailure = failing.DecodeTiled(cuda, deviceA, Tile, Overlap);
        Assert.Equal(1, attempts);

        // Made after the failure: its FreeActivations would have dropped this copy too.
        using Tensor deviceB = DeviceResident(cuda, host);
        Tensor tiledOnly;
        using (KnobProfile.Create("always-tiled").With(EngineKnobs.VaeFullres, false).Push())
        {
            tiledOnly = reference.DecodeTiled(cuda, deviceB, Tile, Overlap);
        }
        using (tiledOnly)
        {
            double spread = StdDev(tiledOnly);
            _output.WriteLine($"{weightDtype}: tiled-only std {spread:F4}, after-failure std {StdDev(afterFailure):F4}");
            Assert.True(spread > 1e-3, $"the tiled-only decode is flat (std {spread}), so equality would prove nothing");
            AssertBitIdentical(tiledOnly, afterFailure);
        }
    }

    /// <summary>One failure used to make every later decode in the process tiled. It now only covers latents at least
    /// as large as the one that failed.</summary>
    [Fact]
    public void FailedFullResAttempt_SkipsOnlyLatentsAtLeastThatLarge()
    {
        if (!TryOpen(out CudaBackend? backend)) return;
        using CudaBackend cuda = backend!;
        using DecoderWeights weights = DecoderWeights.Create(SmallVae, DType.F32, seed: 7);
        VaeDecoder decoder = weights.NewDecoder();
        int attempts = 0;
        decoder.FullResFailureForTests = () => ++attempts == 1 ? new OutOfVramException(1L << 34, 0) : null;

        DecodeAndDispose(cuda, decoder, 40, 40);
        Assert.Equal(1, attempts);
        DecodeAndDispose(cuda, decoder, 40, 40);
        DecodeAndDispose(cuda, decoder, 48, 40);
        Assert.Equal(1, attempts);
        DecodeAndDispose(cuda, decoder, 32, 32);
        Assert.Equal(2, attempts);
    }

    private bool TryOpen(out CudaBackend? backend)
    {
        backend = null;
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return false;
        }
        backend = new CudaBackend(0, PtxDir());
        // The full-res attempt only runs with this much free (its estimate plus 1.5 GB headroom); below it the decode
        // tiles without trying and there is no failure to recover from.
        if (backend.FreeMemoryBytes() < 2L << 30)
        {
            _output.WriteLine("SKIPPED: under 2 GB of free VRAM, the full-res attempt would not run");
            backend.Dispose();
            backend = null;
            return false;
        }
        return true;
    }

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        return Directory.Exists(dir)
            ? dir
            : Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
    }

    private static void DecodeAndDispose(CudaBackend cuda, VaeDecoder decoder, int height, int width)
    {
        using Tensor latent = Gaussian(new TensorShape(1, SmallVae.LatentChannels, height, width), seed: height * 31 + width);
        using Tensor image = decoder.DecodeTiled(cuda, latent, Tile, Overlap);
        Assert.Equal((long)height * VaeDecoder.SpatialScale, image.Shape[2]);
    }

    /// <summary>A copy of <paramref name="host"/> whose only valid data is on the device, as an on-device unpatchify leaves it.</summary>
    private static Tensor DeviceResident(CudaBackend cuda, Tensor host)
    {
        Tensor device = new Tensor(host.Shape, DType.F32);
        cuda.Scale(device, host, 1f);
        Assert.True(GpuTransferHelper.TryGetCachedDevice(device, out _),
            "Scale must leave its output on the device for this test to mean anything.");
        return device;
    }

    private static Tensor Gaussian(TensorShape shape, int seed)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        for (long i = 0; i < shape.ElementCount; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            p[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        return t;
    }

    private static double StdDev(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        double sum = 0, sumSq = 0;
        for (long i = 0; i < t.ElementCount; i++)
        {
            sum += p[i];
            sumSq += (double)p[i] * p[i];
        }
        double mean = sum / t.ElementCount;
        return Math.Sqrt(Math.Max(0, sumSq / t.ElementCount - mean * mean));
    }

    private static void AssertBitIdentical(Tensor expected, Tensor actual)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        int* e = (int*)expected.DataPointer, a = (int*)actual.DataPointer;
        long mismatches = 0, first = -1;
        for (long i = 0; i < expected.ElementCount; i++)
        {
            if (e[i] == a[i]) continue;
            if (first < 0) first = i;
            mismatches++;
        }
        Assert.True(mismatches == 0, $"{mismatches} of {expected.ElementCount} pixels differ, first at {first}: "
            + $"{BitConverter.Int32BitsToSingle(e[Math.Max(0, first)])} vs {BitConverter.Int32BitsToSingle(a[Math.Max(0, first)])}");
    }

    /// <summary>Seeded AutoencoderKL decoder weights for <see cref="SmallVae"/>, cast to the decoder's dtype.</summary>
    private sealed class DecoderWeights : IDisposable
    {
        private readonly Dictionary<string, Tensor> _weights = new();

        public static DecoderWeights Create(VaeConfig config, DType dtype, int seed)
        {
            DecoderWeights result = new DecoderWeights();
            Random rng = new Random(seed);
            int[] channels = config.BlockOutChannels;
            int mid = channels[^1];
            result.Conv(rng, dtype, "decoder.conv_in", mid, config.LatentChannels, 3);
            result.ResNet(rng, dtype, "decoder.mid_block.resnets.0", mid, mid);
            string attention = "decoder.mid_block.attentions.0";
            result.Norm(rng, dtype, $"{attention}.group_norm", mid);
            foreach (string projection in new[] { "to_q", "to_k", "to_v", "to_out.0" })
                result.Linear(rng, dtype, $"{attention}.{projection}", mid, mid);
            result.ResNet(rng, dtype, "decoder.mid_block.resnets.1", mid, mid);
            int previous = mid;
            for (int block = 0; block < channels.Length; block++)
            {
                int outCh = channels[channels.Length - 1 - block];
                for (int res = 0; res <= config.LayersPerBlock; res++)
                    result.ResNet(rng, dtype, $"decoder.up_blocks.{block}.resnets.{res}", res == 0 ? previous : outCh, outCh);
                if (block < channels.Length - 1)
                    result.Conv(rng, dtype, $"decoder.up_blocks.{block}.upsamplers.0.conv", outCh, outCh, 3);
                previous = outCh;
            }
            result.Norm(rng, dtype, "decoder.conv_norm_out", channels[0]);
            result.Conv(rng, dtype, "decoder.conv_out", 3, channels[0], 3);
            return result;
        }

        public VaeDecoder NewDecoder()
        {
            VaeDecoder decoder = new VaeDecoder(SmallVae);
            decoder.LoadWeights(_weights);
            return decoder;
        }

        public void Dispose()
        {
            foreach (Tensor tensor in _weights.Values) tensor.Dispose();
            _weights.Clear();
        }

        private void ResNet(Random rng, DType dtype, string prefix, int inCh, int outCh)
        {
            Norm(rng, dtype, $"{prefix}.norm1", inCh);
            Conv(rng, dtype, $"{prefix}.conv1", outCh, inCh, 3);
            Norm(rng, dtype, $"{prefix}.norm2", outCh);
            Conv(rng, dtype, $"{prefix}.conv2", outCh, outCh, 3);
            if (inCh != outCh) Conv(rng, dtype, $"{prefix}.conv_shortcut", outCh, inCh, 1);
        }

        private void Norm(Random rng, DType dtype, string prefix, int ch)
        {
            Add($"{prefix}.weight", Uniform(rng, new TensorShape(ch), 0.9f, 1.1f), dtype);
            Add($"{prefix}.bias", Uniform(rng, new TensorShape(ch), -0.1f, 0.1f), dtype);
        }

        private void Conv(Random rng, DType dtype, string prefix, int outCh, int inCh, int kernel)
        {
            float bound = MathF.Sqrt(3f / (inCh * kernel * kernel));
            Add($"{prefix}.weight", Uniform(rng, new TensorShape(outCh, inCh, kernel, kernel), -bound, bound), dtype);
            Add($"{prefix}.bias", Uniform(rng, new TensorShape(outCh), -0.05f, 0.05f), dtype);
        }

        private void Linear(Random rng, DType dtype, string prefix, int outCh, int inCh)
        {
            float bound = MathF.Sqrt(3f / inCh);
            Add($"{prefix}.weight", Uniform(rng, new TensorShape(outCh, inCh), -bound, bound), dtype);
            Add($"{prefix}.bias", Uniform(rng, new TensorShape(outCh), -0.05f, 0.05f), dtype);
        }

        private void Add(string key, Tensor f32, DType dtype)
        {
            if (dtype == DType.F32)
            {
                _weights[key] = f32;
                return;
            }
            _weights[key] = f32.CastTo(dtype);
            f32.Dispose();
        }

        private static Tensor Uniform(Random rng, TensorShape shape, float low, float high)
        {
            Tensor t = new Tensor(shape, DType.F32);
            float* p = (float*)t.DataPointer;
            for (long i = 0; i < shape.ElementCount; i++) p[i] = low + (float)rng.NextDouble() * (high - low);
            return t;
        }
    }
}
