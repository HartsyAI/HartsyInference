using HartsyInference.Audio.Models.Music;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>End-to-end parity for MiniMax Music 3's flow stage: the reference's frame hiddens go in, the reference's
/// noise draws are forced, and every window's latents plus the stitched waveform are compared. This is what covers
/// the hand-written parts the component tests cannot — the overlap blend, the condition splice, the carry slicing,
/// the Euler timing, and the crop/stitch geometry working together over more than one window.
///
/// <para>Runs on CUDA when a device is present (the CPU path is roughly an hour for the same work). Skips without
/// <c>HARTSY_MINIMAX_MUSIC3_PATH</c> and the reference dump.</para></summary>
[Trait("Category", "Integration")]
[Trait("Category", "Slow")]
public sealed unsafe class MiniMaxMusic3FlowParityTests(ITestOutputHelper output)
{
    private const double LatentTolerance = 5e-3;
    private const double AudioTolerance = 5e-3;

    [Fact]
    public void Denoise_MatchesDiffusersReference()
    {
        string? checkpoint = Environment.GetEnvironmentVariable("HARTSY_MINIMAX_MUSIC3_PATH");
        MiniMaxMusic3Reference? reference = MiniMaxMusic3Reference.TryLoad();
        if (checkpoint is null || reference is null || !reference.Has("flow_audio"))
        {
            return;
        }

        List<SafeTensorsLoader> loaders = [];
        IBackend backend = CreateBackend();
        try
        {
            using MiniMaxMusic3ConditionEncoder conditionEncoder = new MiniMaxMusic3ConditionEncoder();
            conditionEncoder.LoadWeights(Open(checkpoint, "condition_encoder", loaders));
            using MiniMaxMusic3Dit dit = new MiniMaxMusic3Dit();
            dit.LoadWeights(Open(checkpoint, "transformer", loaders));
            using MiniMaxMusic3Vocoder vocoder = new MiniMaxMusic3Vocoder();
            vocoder.LoadWeights(Open(checkpoint, "vocoder", loaders));

            int frames = reference.Meta("flow_frames").GetInt32();
            int steps = reference.Meta("flow_steps").GetInt32();
            float[] frameHiddens = reference.Read("flow_frame_hiddens");
            int windows = reference.Meta("flow_chunk_starts").GetArrayLength();
            List<float[]> forcedNoise = [];
            for (int i = 0; i < windows; i++)
            {
                forcedNoise.Add(reference.Read($"flow_noise_{i}"));
            }
            Assert.Equal(windows, MiniMaxMusic3FlowPipeline.ChunkStarts(frames).Length);

            using MiniMaxMusic3FlowPipeline pipeline = new MiniMaxMusic3FlowPipeline(backend, conditionEncoder, dit);
            Tensor[] chunks = pipeline.Denoise(frameHiddens, frames, steps,
                MiniMaxMusic3FlowPipeline.DefaultCfgScale, seed: 0, forcedNoise: forcedNoise);
            Tensor[] waveforms = new Tensor[chunks.Length];
            try
            {
                Assert.Equal(windows, chunks.Length);
                for (int i = 0; i < chunks.Length; i++)
                {
                    float[] expected = reference.Read($"flow_chunk_{i}");
                    (double meanAbs, double maxAbs, double correlation) =
                        MiniMaxMusic3Reference.Compare(chunks[i].AsReadOnlySpan<float>(), expected);
                    output.WriteLine($"[MiniMaxMusic3Flow] window {i} meanAbs={meanAbs:E3} maxAbs={maxAbs:E3} corr={correlation:F8}");
                    Assert.Equal(expected.Length, (int)chunks[i].Shape.ElementCount);
                    Assert.True(meanAbs < LatentTolerance,
                        $"window {i} latents diverge: meanAbs={meanAbs:E3}, maxAbs={maxAbs:E3}, corr={correlation:F8}");
                    waveforms[i] = vocoder.Decode(backend, chunks[i]);
                }

                (float[] left, float[] right) = MiniMaxMusic3FlowPipeline.Stitch(waveforms, MiniMaxMusic3Vocoder.LatentHopLength);
                int[] audioShape = reference.Shape("flow_audio");
                Assert.Equal(audioShape[2], left.Length);

                float[] expectedAudio = reference.Read("flow_audio");
                (double leftMean, double leftMax, double leftCorrelation) =
                    MiniMaxMusic3Reference.Compare(left, expectedAudio.AsSpan(0, left.Length));
                (double rightMean, _, double rightCorrelation) =
                    MiniMaxMusic3Reference.Compare(right, expectedAudio.AsSpan(left.Length, right.Length));
                output.WriteLine($"[MiniMaxMusic3Flow] audio L meanAbs={leftMean:E3} maxAbs={leftMax:E3} corr={leftCorrelation:F8}, "
                    + $"R meanAbs={rightMean:E3} corr={rightCorrelation:F8}");
                Assert.True(leftMean < AudioTolerance && rightMean < AudioTolerance,
                    $"stitched audio diverges: L meanAbs={leftMean:E3}, R meanAbs={rightMean:E3}");
            }
            finally
            {
                foreach (Tensor waveform in waveforms)
                {
                    waveform?.Dispose();
                }
                foreach (Tensor chunk in chunks)
                {
                    chunk.Dispose();
                }
            }
        }
        finally
        {
            (backend as IDisposable)?.Dispose();
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
        }
    }

    private static IBackend CreateBackend()
    {
        try
        {
            return new CudaBackend();
        }
        catch (Exception)
        {
            return new CpuBackend();   // tier-lint: guarded
        }
    }

    private static Dictionary<string, Tensor> Open(string checkpoint, string subfolder, List<SafeTensorsLoader> loaders)
    {
        string[] shards = Directory.GetFiles(Path.Combine(checkpoint, subfolder), "*.safetensors");
        Array.Sort(shards, StringComparer.Ordinal);
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (string shard in shards)
        {
            SafeTensorsLoader loader = new SafeTensorsLoader();
            loader.Load(shard);
            loaders.Add(loader);
            foreach (KeyValuePair<string, Tensor> entry in loader.GetAllTensors())
            {
                weights[entry.Key] = entry.Value;
            }
        }
        return weights;
    }
}
