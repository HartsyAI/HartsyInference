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

/// <summary>The released S generator, KV refiner and pixel decoder against Apple's own FP32 run on the same saved inputs. The
/// reference comes from <c>tools/parity/dump_fastframes_reference.py checkpoint</c>, which writes <c>io.safetensors</c> (saved noise,
/// transitions, prompt embeddings, latents and pixels). This test loads the same weights through the recipe's own readers and runs
/// the engine rollout on the backend chosen by the environment.
///
/// Environment (the test is skipped unless <c>FF_REAL_PARITY_DIR</c> is set):
///   FF_REAL_PARITY_DIR      folder with framewise-2step.pt, kv_refiner.safetensors, pixel_decoder.safetensors and io.safetensors
///   FF_REAL_PARITY_GENERATOR  generator file name in that folder (default framewise-2step.pt for S; dit.safetensors for L)
///   FF_REAL_PARITY_BACKEND  cuda (default) or cpu
///   FF_REAL_PARITY_BOUND    relative RMS bound per frame (default 1e-2; provisional until the BF16-vs-FP32 floor is measured)
///
/// The metric is relative RMS, ||engine - upstream|| / ||upstream||, for each latent frame and each pixel frame. The pixel comparison
/// drops the first three decoded frames, as upstream's align_frames does. The engine's lane is F16 on CUDA and F32 on CPU, so the
/// result is compared against the FP32 reference, not against an identical lane.</summary>
public sealed class FastFramesRealWeightsParityTests(ITestOutputHelper output)
{
    private const int AlignedDroppedPixelFrames = 3;

    [Fact]
    public void RealSWeightsMatchUpstreamFp32Run()
    {
        string? dir = Environment.GetEnvironmentVariable("FF_REAL_PARITY_DIR");
        string generatorFile = Environment.GetEnvironmentVariable("FF_REAL_PARITY_GENERATOR") ?? "framewise-2step.pt";
        if (string.IsNullOrWhiteSpace(dir))
        {
            return;
        }

        string backendName = Environment.GetEnvironmentVariable("FF_REAL_PARITY_BACKEND") ?? "cuda";
        double bound = double.TryParse(Environment.GetEnvironmentVariable("FF_REAL_PARITY_BOUND"), NumberStyles.Float,
            CultureInfo.InvariantCulture, out double parsed) ? parsed : 1e-2;

        List<IDisposable> loaders = [];
        try
        {
            using IBackend backend = backendName == "cpu" ? new CpuBackend() : BackendFactory.CreateCuda(0);
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
