using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight parity for MiniMax Music 3's autoregressive stage. The reference's sampled codes are
/// teacher-forced so the comparison is of the model, not of two incompatible samplers — which is also what makes
/// the frame-0 rule (its codes are generated and fed back but its hidden state is NOT emitted) and the layer-major
/// ordering of the frame hiddens observable. Skips without <c>HARTSY_MINIMAX_MUSIC3_PATH</c> and the dump.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "Slow")]
public sealed unsafe class MiniMaxMusic3ArParityTests(ITestOutputHelper output)
{
    private const double HiddenTolerance = 2e-2;
    private const double FrameHiddenTolerance = 2e-2;

    /// <summary>CUDA when a device is present. This test ran CPU-only until 2026-08-14, which is why it stayed green
    /// through a depth-decoder bug that only manifests on the GPU.</summary>
    private static IBackend CreateBackend()
    {
        if (Environment.GetEnvironmentVariable("HARTSY_MM3_FORCE_CPU") == "1")
        {
            return new CpuBackend();   // tier-lint: guarded
        }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            return new CpuBackend();   // tier-lint: guarded
        }
        try
        {
            return new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
        }
        catch (Exception)
        {
            return new CpuBackend();   // tier-lint: guarded
        }
    }

    [Fact]
    public void Generate_MatchesDiffusersReferenceUnderTeacherForcing()
    {
        string? checkpoint = Environment.GetEnvironmentVariable("HARTSY_MINIMAX_MUSIC3_PATH");
        MiniMaxMusic3Reference? reference = MiniMaxMusic3Reference.TryLoad();
        if (checkpoint is null || reference is null || !reference.Has("ar_frame_hiddens"))
        {
            return;
        }
        string languageModelDirectory = Path.Combine(checkpoint, "language_model");
        string depthPath = Path.Combine(checkpoint, "rvq_depth_decoder", "diffusion_pytorch_model.safetensors");
        string tokenizerPath = Path.Combine(checkpoint, "tokenizer", "tokenizer.json");
        if (!Directory.Exists(languageModelDirectory) || !File.Exists(depthPath) || !File.Exists(tokenizerPath))
        {
            return;
        }

        List<SafeTensorsLoader> loaders = [];
        try
        {
            Dictionary<string, Tensor> languageWeights = new Dictionary<string, Tensor>(StringComparer.Ordinal);
            string[] shards = Directory.GetFiles(languageModelDirectory, "*.safetensors");
            Array.Sort(shards, StringComparer.Ordinal);
            foreach (string shard in shards)
            {
                SafeTensorsLoader loader = new SafeTensorsLoader();
                loader.Load(shard);
                loaders.Add(loader);
                foreach (KeyValuePair<string, Tensor> entry in loader.GetAllTensors())
                {
                    languageWeights[entry.Key] = entry.Value;
                }
            }
            SafeTensorsLoader depthLoader = new SafeTensorsLoader();
            depthLoader.Load(depthPath);
            loaders.Add(depthLoader);

            using MiniMaxMusic3GlobalLm languageModel = new MiniMaxMusic3GlobalLm();
            languageModel.LoadWeights(languageWeights);
            using MiniMaxMusic3DepthDecoder depthDecoder = new MiniMaxMusic3DepthDecoder();
            depthDecoder.LoadWeights(depthLoader.GetAllTensors());

            using FileStream tokenizerJson = File.OpenRead(tokenizerPath);
            GgufTokenizer tokenizer = HfTokenizerJson.LoadByteLevelBpe(tokenizerJson);
            (int[] conditional, int[] unconditional) = MiniMaxMusic3Prompt.Tokenize(
                tokenizer,
                reference.Meta("ar_caption").GetString()!,
                reference.Meta("ar_lyrics").GetString()!);

            int[] referenceIds = [.. reference.Meta("ar_text_ids").EnumerateArray().Select(id => id.GetInt32())];
            Assert.Equal(referenceIds, conditional);

            List<int[]> forced = [];
            foreach (System.Text.Json.JsonElement row in reference.Meta("ar_codes").EnumerateArray())
            {
                int[] codes = [.. row.EnumerateArray().Select(code => code.GetInt32())];
                // The reference records the semantic slot as the sampled TOKEN id, not the codebook index.
                codes[0] -= MiniMaxMusic3GlobalLm.AudioCodeOffset;
                forced.Add(codes);
            }

            int frames = reference.Meta("ar_frames").GetInt32();
            using MiniMaxMusic3ArPipeline pipeline = new MiniMaxMusic3ArPipeline(languageModel, depthDecoder);
            IBackend backend = CreateBackend();
            output.WriteLine($"[MiniMaxMusic3Ar] backend={backend.GetType().Name}");
            (float[] frameHiddens, int produced) = pipeline.Generate(
                backend, conditional, unconditional, frames, seed: 0, forcedCodes: forced);

            Assert.Equal(frames, produced);
            float[] expected = reference.Read("ar_frame_hiddens");
            Assert.Equal(expected.Length, frameHiddens.Length);
            (double meanAbs, double maxAbs, double correlation) =
                MiniMaxMusic3Reference.Compare(frameHiddens, expected);
            output.WriteLine($"[MiniMaxMusic3Ar] frames={produced} meanAbs={meanAbs:E3} maxAbs={maxAbs:E3} corr={correlation:F8}");

            // The reference runs the language model in F32 while this loads the checkpoint's BF16 weights, so the
            // gate is correlation plus a loose absolute bound rather than an F32-noise tolerance.
            Assert.True(correlation > 0.999,
                $"frame hiddens diverge: corr={correlation:F8}, meanAbs={meanAbs:E3}, maxAbs={maxAbs:E3}");
            Assert.True(meanAbs < FrameHiddenTolerance,
                $"frame hiddens diverge: meanAbs={meanAbs:E3}, maxAbs={maxAbs:E3}, corr={correlation:F8}");

            // Frame 0's hidden state must NOT appear: the emitted window starts at the reference's capture index 1.
            float[] firstCapture = reference.Read("ar_last_hidden_0");
            (_, _, double frameZeroCorrelation) = MiniMaxMusic3Reference.Compare(
                frameHiddens.AsSpan(0, MiniMaxMusic3GlobalLm.HiddenSize), firstCapture.AsSpan(0, MiniMaxMusic3GlobalLm.HiddenSize));
            float[] secondCapture = reference.Read("ar_last_hidden_1");
            (double secondMeanAbs, _, double secondCorrelation) = MiniMaxMusic3Reference.Compare(
                frameHiddens.AsSpan(0, MiniMaxMusic3GlobalLm.HiddenSize), secondCapture.AsSpan(0, MiniMaxMusic3GlobalLm.HiddenSize));
            output.WriteLine($"[MiniMaxMusic3Ar] emitted-frame-0 vs capture0 corr={frameZeroCorrelation:F6}, vs capture1 corr={secondCorrelation:F6}");
            Assert.True(secondCorrelation > frameZeroCorrelation,
                "the first emitted frame matches the prefill hidden state — the frame-0 skip is missing.");
            Assert.True(secondMeanAbs < HiddenTolerance,
                $"the first emitted frame does not match the reference's second capture: meanAbs={secondMeanAbs:E3}");
        }
        finally
        {
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
        }
    }
}
