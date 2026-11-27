using System.Buffers.Binary;
using System.Text;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The avatar loop's per-step dump: the reference's file names, and values that obey the scheduler step they describe.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatStepDumpTests
{
    private const int Frames = 3, Height = 4, Width = 4, TextLength = 6, Steps = 3;
    private static readonly int Hw = Height * Width;

    [Theory]
    [InlineData(LongCatAvatarMode.At2v)]
    [InlineData(LongCatAvatarMode.Ai2v)]
    public void EachStepWritesItsTimestepNoiseAndLatentAndTheyFollowTheScheduler(LongCatAvatarMode mode)
    {
        LongCatConfig config = LongCatTestData.TinyAvatar;
        int cond = mode == LongCatAvatarMode.At2v ? 0 : 1, noisy = Frames - cond, channels = config.InChannels;
        Dictionary<string, Tensor> weights = LongCatTestData.AvatarWeights(config, 71);
        string root = Path.Combine(Path.GetTempPath(), "longcat-step-dump-" + Guid.NewGuid().ToString("N"));
        try
        {
            Random rng = new(5);
            float[] noise = LongCatTestData.RandomValues(rng, channels * Frames * Hw);
            float[]? condLatents = cond == 0 ? null : LongCatTestData.RandomValues(rng, channels * cond * Hw);
            float[] audio = LongCatTestData.RandomValues(rng, Frames * config.AudioContextTokens * config.AudioOutputDim);
            LongCatPrompt prompt = new(LongCatTestData.RandomValues(rng, TextLength * config.CaptionChannels), TextLength, TextLength);
            using CpuBackend backend = new();
            using LongCatTransformer transformer = new(config);
            transformer.LoadWeights(weights);
            LongCatAvatarPipeline pipeline = new(backend, transformer);
            LongCatAvatarRun run = new(channels, Frames, Height, Width, noise, prompt, prompt, audio, audio, Steps,
                LongCatAvatarSchedule.Kind.Linear, 7f, 2f, 1.5f, mode, cond, condLatents);

            float[] undumped = pipeline.Denoise(run).Latents;
            Assert.False(Directory.Exists(root));
            LongCatAvatarResult result;
            using (KnobProfile.Create("longcat-dump").With(EngineKnobs.LongCatStepDump, root).Push())
            {
                result = pipeline.Denoise(run);
                pipeline.Denoise(run);
            }

            // Reading the tensors back changes nothing.
            Assert.Equal(undumped, result.Latents);
            // Two calls of three steps, three files each, numbered by call.
            List<string> expected = [];
            for (int call = 0; call < 2; call++)
                for (int step = 0; step < Steps; step++)
                    foreach (string name in new[] { "latent", "noise", "t" })
                        expected.Add($"call{call:D2}_step{step:D2}_{name}.npy");
            Assert.Equal(expected, Directory.GetFiles(root).Select(file => Path.GetFileName(file)).Order(StringComparer.Ordinal));

            float[] previous = LongCatFrameOps.JoinFrames(condLatents, LongCatFrameOps.SteppedFrames(noise, channels, Frames, Hw, cond),
                channels, Frames, Hw, cond);
            for (int step = 0; step < Steps; step++)
            {
                Assert.Equal(result.Timesteps[step], Scalar(Path.Combine(root, $"call00_step{step:D2}_t.npy")));
                float[] output = Read(root, step, "noise", [1, channels, noisy, Height, Width]);
                float[] latent = Read(root, step, "latent", [1, channels, Frames, Height, Width]);
                float dt = result.Sigmas[step + 1] - result.Sigmas[step];
                for (int c = 0; c < channels; c++)
                    for (int f = 0; f < Frames; f++)
                        for (int e = 0; e < Hw; e++)
                        {
                            int at = (c * Frames + f) * Hw + e;
                            float wanted = f < cond ? condLatents![(c * cond + f) * Hw + e]
                                : previous[at] + dt * output[(c * noisy + f - cond) * Hw + e];
                            Assert.Equal(wanted, latent[at], 5e-6f);
                        }
                previous = latent;
            }
            Assert.Equal(result.Latents, previous);
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
            if (Directory.Exists(root))
                Directory.Delete(root, recursive: true);
        }
    }

    private static float[] Read(string root, int step, string name, int[] shape)
    {
        VisionArray array = VisionArrayCodec.FromNpy(File.ReadAllBytes(Path.Combine(root, $"call00_step{step:D2}_{name}.npy")));
        Assert.Equal(shape, array.Shape);
        return array.Data;
    }

    /// <summary>A zero-dimensional <c>.npy</c>, which the vision codec does not read: the header names an empty shape and one value follows.</summary>
    private static float Scalar(string path)
    {
        byte[] file = File.ReadAllBytes(path);
        int header = BinaryPrimitives.ReadUInt16LittleEndian(file.AsSpan(8));
        Assert.Contains("'shape': ()", Encoding.ASCII.GetString(file, 10, header));
        Assert.Equal(10 + header + sizeof(float), file.Length);
        return BinaryPrimitives.ReadSingleLittleEndian(file.AsSpan(10 + header));
    }
}
