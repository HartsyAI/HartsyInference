using HartsyInference.Audio.Models.LongCat;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Shared;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Engine.Tests;

/// <summary>The avatar's driving-speech features on a CUDA device with the real Whisper-large-v3 encoder, released with
/// <c>FreeWeights</c> after the first read and read again: the second read is bit-identical to the first, the release frees device memory,
/// and the re-upload is timed. Reports only the numbers; the pass is the identity and the release.</summary>
/// <remarks>Env-gated: runs when <c>HARTSY_WHISPER_LARGE_V3_SAFETENSORS</c> names an existing <c>model.safetensors</c> and a CUDA device opens.
/// Otherwise it is reported as skipped, never as passed.</remarks>
[Trait("Category", "GpuIntegration")]
public sealed class LongCatWhisperResidencyGpuTests(ITestOutputHelper output)
{
    public const string EnvName = "HARTSY_WHISPER_LARGE_V3_SAFETENSORS";

    [WhisperResidencyFact]
    public void FeaturesAreIdenticalAfterTheEncoderWeightsAreFreedAndUploadedAgain()
    {
        string path = Environment.GetEnvironmentVariable(EnvName)!;
        using CudaBackend backend = GpuDevices.OpenCuda();

        SafeTensorsLoader loader = new();
        loader.Load(path);
        try
        {
            Dictionary<string, Tensor> encoderWeights = loader.GetAllTensors()
                .Where(pair => pair.Key.StartsWith("model.encoder.", StringComparison.Ordinal))
                .ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal);
            WhisperEncoder encoder = new(WhisperConfig.LargeV3);
            encoder.LoadWeights(encoderWeights);
            MelSpectrogramExtractor mel = new(MelSpectrogramExtractor.WhisperConfig(WhisperConfig.LargeV3.NumMelBins));

            // Two seconds of a 440 Hz tone at 16 kHz: the content is not the point, the identity and the release are.
            float[] speech = new float[32_000];
            for (int i = 0; i < speech.Length; i++) speech[i] = 0.2f * MathF.Sin(2f * MathF.PI * 440f * i / 16_000f);

            long beforeFirst = backend.FreeMemoryBytes();
            float[] first = LongCatWhisperFeatures.Extract(backend, encoder, mel, speech, fps: 25);
            long afterFirst = backend.FreeMemoryBytes();

            backend.FreeWeights(encoder.EnumerateWeights());
            long afterFree = backend.FreeMemoryBytes();

            System.Diagnostics.Stopwatch timer = System.Diagnostics.Stopwatch.StartNew();
            float[] second = LongCatWhisperFeatures.Extract(backend, encoder, mel, speech, fps: 25);
            timer.Stop();

            long releasedMiB = (afterFree - afterFirst) >> 20;
            output.WriteLine($"weights={path}");
            output.WriteLine($"features={first.Length} floats; free before={beforeFirst >> 20} MiB, after first read={afterFirst >> 20} MiB, "
                + $"after FreeWeights={afterFree >> 20} MiB (released {releasedMiB} MiB); second read (re-upload) {timer.Elapsed.TotalMilliseconds:F0} ms.");

            Assert.Equal(first.Length, second.Length);
            Assert.True(first.AsSpan().SequenceEqual(second), "The features after FreeWeights and a re-upload differ from the first read.");
            Assert.True(afterFree > afterFirst, $"FreeWeights released no device memory (free {afterFirst >> 20} MiB before, {afterFree >> 20} MiB after).");
        }
        finally
        {
            loader.Dispose();
        }
    }

    /// <summary>Runs only when the env var names an existing file and a CUDA device opens; otherwise the test is reported as skipped.</summary>
    private sealed class WhisperResidencyFactAttribute : FactAttribute
    {
        public WhisperResidencyFactAttribute()
        {
            string? path = Environment.GetEnvironmentVariable(EnvName);
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
                Skip = $"SKIPPED: {EnvName} is unset or not an existing file.";
            else if (GpuDevices.CudaFailure is { } failure)
                Skip = $"SKIPPED: no usable CUDA device ({failure}).";
        }
    }
}
