using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.FastFrames;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.Video;
using HartsyInference.Video.Pipelines;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Engine.Tests;

/// <summary>A fact that is reported as skipped, not passed, unless <c>FF_REAL_PARITY_DIR</c> names a folder.</summary>
internal sealed class RealWeightsParityFactAttribute : FactAttribute
{
    public RealWeightsParityFactAttribute()
    {
        if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("FF_REAL_PARITY_DIR")))
        {
            Skip = "SKIPPED: FF_REAL_PARITY_DIR is not set; the real-weights parity run needs the released weights and the reference.";
        }
    }
}

/// <summary>The released FastFrames generator, KV refiner and pixel decoder (S or L) against Apple's own run on the same saved inputs. The
/// reference comes from <c>tools/parity/dump_fastframes_reference.py checkpoint</c>, which writes <c>io.safetensors</c> (saved noise,
/// transitions, prompt embeddings, latents and pixels). This test loads the same weights through the recipe's own readers and runs
/// the engine rollout on the backend chosen by the environment.
///
/// Reference precision: S is compared with upstream's FP32 run; L with upstream's BF16 run, which is L's native precision (an FP32 L
/// reference does not fit an 80 GB card with the DiT). A BF16 reference agrees with the engine only to about the BF16 floor: on S,
/// upstream's own BF16 run differs from its FP32 run by 0.10 to 0.12 in latent relative RMS.
///
/// Environment (the test is skipped unless <c>FF_REAL_PARITY_DIR</c> is set):
///   FF_REAL_PARITY_DIR      folder with framewise-2step.pt, kv_refiner.safetensors, pixel_decoder.safetensors and io.safetensors
///   FF_REAL_PARITY_GENERATOR  generator file name in that folder (default framewise-2step.pt for S; dit.safetensors for L)
///   FF_REAL_PARITY_BACKEND  cuda (default), vulkan or cpu
///   FF_REAL_PARITY_BOUND    relative RMS bound per frame (default 1e-2; provisional until the BF16-vs-FP32 floor is measured)
///
/// The metric is relative RMS, ||engine - upstream|| / ||upstream||, for each latent frame and each pixel frame. The pixel comparison
/// drops the first three decoded frames, as upstream's align_frames does. The engine's lane is F16 on CUDA and F32 on CPU. The
/// reference is upstream's FP32 run for S and its BF16 run for L (the dump's <c>--precision</c>), so compare a result only with a
/// reference of the same model and precision.</summary>
public sealed class FastFramesRealWeightsParityTests(ITestOutputHelper output)
{
    private const int AlignedDroppedPixelFrames = 3;

    [RealWeightsParityFact]
    [Trait("Category", "GpuIntegration")]
    public void RealWeightsMatchUpstreamReference()
    {
        string dir = Environment.GetEnvironmentVariable("FF_REAL_PARITY_DIR")!;
        string generatorFile = Environment.GetEnvironmentVariable("FF_REAL_PARITY_GENERATOR") ?? "framewise-2step.pt";

        string backendName = Environment.GetEnvironmentVariable("FF_REAL_PARITY_BACKEND") ?? "cuda";
        double bound = double.TryParse(Environment.GetEnvironmentVariable("FF_REAL_PARITY_BOUND"), NumberStyles.Float,
            CultureInfo.InvariantCulture, out double parsed) ? parsed : 1e-2;

        List<IDisposable> loaders = [];
        try
        {
            using IBackend backend = backendName switch
            {
                "cpu" => new CpuBackend(),
                "vulkan" => BackendFactory.CreateVulkan(),
                _ => BackendFactory.CreateCuda(0),
            };
            bool castForCpu = backend is CpuBackend;
            DType pickleDType = castForCpu ? DType.F32 : DType.F16;

            Dictionary<string, Tensor> generator = FastFramesRecipe.ReadGenerator(
                Path.Combine(dir, generatorFile), pickleDType, castForCpu, loaders);
            WanVideoCheckpointConverter.ConvertedWeights converted = WanVideoCheckpointConverter.Convert(generator);
            WanVideoConfig wan = WanConfigDetector.Detect(converted.Transformer);
            FastFramesConfig config = FastFramesRecipe.ConfigFor(wan.InnerDim);

            WanVideoTransformer transformer = new(wan);
            loaders.Add(transformer);
            transformer.LoadWeights(converted.Transformer);

            CheckpointSource refinerSource = CheckpointSource.Open(Path.Combine(dir, "kv_refiner.safetensors"));
            loaders.Add(refinerSource);
            FastFramesKvRefiner refiner = new(FastFramesKvRefiner.DetectConfig(refinerSource.Weights, wan.NumHeads));
            loaders.Add(refiner);
            refiner.LoadWeights(refinerSource.Weights);

            CheckpointSource decoderSource = CheckpointSource.Open(Path.Combine(dir, "pixel_decoder.safetensors"));
            loaders.Add(decoderSource);
            FastFramesPixelDecoder decoder = new(FastFramesPixelDecoder.DetectConfig(decoderSource.Weights));
            loaders.Add(decoder);
            decoder.LoadWeights(decoderSource.Weights);

            using CheckpointSource io = CheckpointSource.Open(Path.Combine(dir, "io.safetensors"));
            int latentValues = Channels * config.LatentHeight * config.LatentWidth;
            int frames = (int)(io.Weights["noise"].ElementCount / latentValues);
            Assert.Equal(frames * latentValues, (int)io.Weights["noise"].ElementCount);
            // The reference's prompt embeddings are already padded to the generator's 512 rows (see the dumper).
            using Tensor prompt = new(new TensorShape(512, 4096), DType.F32);
            io.Weights["prompt_embeds"].AsReadOnlySpan<float>().CopyTo(prompt.AsSpan<float>());
            SavedNoise noise = new(io.Weights["noise"], io.Weights["transition"], frames, latentValues);

            FastFramesPipeline pipeline = new(backend, transformer, refiner, decoder, config);
            List<FastFramesFrame> got = pipeline.Rollout(prompt, noise, frames).ToList();
            Assert.Equal(frames, got.Count);

            // Latents: [frames, 16, H, W] per the reference.
            float[] refLatents = io.Weights["latents"].AsReadOnlySpan<float>().ToArray();
            double worstLatent = 0;
            for (int f = 0; f < frames; f++)
            {
                double rel = RelativeRms(got[f].Latent, refLatents.AsSpan(f * latentValues, latentValues));
                output.WriteLine($"latent frame {f}: relative RMS {rel:G4}");
                worstLatent = Math.Max(worstLatent, rel);
            }

            // Pixels: the engine's decoded frames minus the first three, against the reference's [pixelFrames, 3, H, W].
            float[] refPixels = io.Weights["pixels"].AsReadOnlySpan<float>().ToArray();
            int perPixelFrame = got[0].Pixels.Length / got[0].PixelFrames;
            float[] decoded = got.SelectMany(fr => fr.Pixels).ToArray();
            Assert.Equal(refPixels.Length, decoded.Length - AlignedDroppedPixelFrames * perPixelFrame);
            double worstPixel = 0;
            for (int p = 0; p < refPixels.Length / perPixelFrame; p++)
            {
                double rel = RelativeRms(
                    decoded.AsSpan((p + AlignedDroppedPixelFrames) * perPixelFrame, perPixelFrame),
                    refPixels.AsSpan(p * perPixelFrame, perPixelFrame));
                worstPixel = Math.Max(worstPixel, rel);
            }
            output.WriteLine($"pixel frames: worst relative RMS {worstPixel:G4} over {refPixels.Length / perPixelFrame} frames");
            output.WriteLine($"backend {backendName}, {frames} latent frames, bound {bound:G3}, worst latent {worstLatent:G4}");

            Assert.True(worstLatent <= bound, $"worst latent relative RMS {worstLatent:G4} exceeds {bound:G3}.");
            Assert.True(worstPixel <= bound, $"worst pixel relative RMS {worstPixel:G4} exceeds {bound:G3}.");
        }
        finally
        {
            for (int i = loaders.Count - 1; i >= 0; i--)
            {
                loaders[i].Dispose();
            }
        }
    }

    private const int Channels = 16;

    private static double RelativeRms(ReadOnlySpan<float> actual, ReadOnlySpan<float> reference)
    {
        double num = 0, den = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double d = actual[i] - reference[i];
            num += d * d;
            den += (double)reference[i] * reference[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    /// <summary>The saved noise from the reference: the initial noise per frame and the transitions per step and frame.</summary>
    private sealed class SavedNoise(Tensor initial, Tensor transition, int frames, int frameValues) : IFastFramesNoise
    {
        private readonly float[] _initial = initial.AsReadOnlySpan<float>().ToArray();
        private readonly float[] _transition = transition.AsReadOnlySpan<float>().ToArray();

        public float[] Initial(int frame) => _initial.AsSpan(frame * frameValues, frameValues).ToArray();

        public float[] Transition(int step, int frame) => _transition.AsSpan((step * frames + frame) * frameValues, frameValues).ToArray();
    }
}
