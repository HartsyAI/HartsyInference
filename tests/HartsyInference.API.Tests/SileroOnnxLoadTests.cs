using HartsyInference.Audio.Models.Wake;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Engine.Audio.Wake;
using HartsyInference.ModelAssets.Onnx;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>Reading Silero's weights straight out of the upstream ONNX.
///
/// <para>The engine has no ONNX executor, so an ONNX file is only ever a weight container — but this one keeps
/// its fifteen tensors as anonymous <c>Constant</c> nodes inside the <c>then_branch</c> of an <c>If</c> on
/// sample rate, where an initializer-only reader cannot see them and where there are no names to bind by. That
/// is worth loading directly rather than through a conversion step, because it means the model installs from
/// its canonical MIT source with nobody hosting a repacked copy.</para>
///
/// <para>Everything here fails silently if it is wrong. Picking the 8 kHz branch gives a model that loads, runs,
/// and scores speech incorrectly. Swapping the two same-shaped LSTM matrices does the same. So the test that
/// matters is the last one: identical probabilities from the ONNX and from a safetensors file already verified
/// against onnxruntime to 1e-6.</para>
///
/// <para>Point <c>HARTSYINFERENCE_WAKE_MODELS</c> at a wake model root holding <c>vad/silero_vad.onnx</c> to
/// run; the comparison also needs <c>vad/silero_vad_16k.safetensors</c> beside it and
/// <c>HARTSYINFERENCE_WAKE_REF_DIR</c> for the audio.</para></summary>
public sealed class SileroOnnxLoadTests
{
    private static string? ModelsDir => Environment.GetEnvironmentVariable("HARTSYINFERENCE_WAKE_MODELS");
    private static string? RefDir => Environment.GetEnvironmentVariable("HARTSYINFERENCE_WAKE_REF_DIR");

    private static string? OnnxPath =>
        ModelsDir is null ? null : Path.Combine(ModelsDir, "vad", "silero_vad.onnx");

    private static string? SafeTensorsPath =>
        ModelsDir is null ? null : Path.Combine(ModelsDir, "vad", "silero_vad_16k.safetensors");

    /// <summary>The subgraph walk finds the branch's constants at all — an initializer-only reader finds zero.</summary>
    [Fact]
    public void SubgraphConstants_FindsTheSixteenKilohertzBranch()
    {
        if (OnnxPath is null || !File.Exists(OnnxPath)) return;

        using OnnxWeightLoader loader = new();
        loader.Load(OnnxPath);

        Assert.Empty(loader.GetAllTensors());          // nothing at the top level: the point of the exercise
        Assert.NotEmpty(loader.Model!.Subgraphs);

        IReadOnlyList<Tensor> constants = loader.SubgraphConstants("then_branch");
        Assert.NotEmpty(constants);

        // The fifteen shapes the architecture is made of must all be present.
        (int[] Shape, int Count)[] expected =
        [
            ([258, 1, 256], 1), ([128, 129, 3], 1), ([64, 128, 3], 1), ([64, 64, 3], 1), ([128, 64, 3], 1),
            ([512, 128], 2), ([512], 2), ([1, 128, 1], 1),
        ];
        foreach ((int[] shape, int count) in expected)
        {
            int found = constants.Count(t => HasShape(t, shape));
            Assert.True(found >= count,
                $"expected at least {count} tensor(s) of shape [{string.Join(",", shape)}], found {found}");
        }
    }

    /// <summary>Both branches are present and distinct, which is what makes picking the right one a real choice.</summary>
    [Fact]
    public void BothSampleRateBranches_AreParsed()
    {
        if (OnnxPath is null || !File.Exists(OnnxPath)) return;

        using OnnxWeightLoader loader = new();
        loader.Load(OnnxPath);
        Assert.Contains(loader.Model!.Subgraphs, s => s.AttributeName == "then_branch");
        Assert.Contains(loader.Model!.Subgraphs, s => s.AttributeName == "else_branch");
    }

    /// <summary>The load-bearing one: the ONNX path and the onnxruntime-verified safetensors path must score
    /// identically. A wrong branch or a swapped LSTM pair fails here and nowhere else.</summary>
    [Fact]
    public void OnnxWeights_ScoreIdenticallyToVerifiedSafeTensors()
    {
        if (OnnxPath is null || SafeTensorsPath is null || RefDir is null) return;
        if (!File.Exists(OnnxPath) || !File.Exists(SafeTensorsPath)) return;
        string audioPath = Path.Combine(RefDir, "silero_input.bin");
        if (!File.Exists(audioPath)) return;

        float[] audio = ReadF32(audioPath);
        int chunks = Math.Min(120, audio.Length / SileroVad.WindowSamples);
        Assert.True(chunks > 0);

        float[] fromOnnx = Score(OnnxPath, audio, chunks);
        float[] fromSafeTensors = Score(SafeTensorsPath, audio, chunks);

        float worst = 0f;
        for (int i = 0; i < chunks; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(fromOnnx[i] - fromSafeTensors[i]));
        }
        Assert.True(worst < 1e-6f, $"ONNX and safetensors probabilities differ by {worst} over {chunks} chunks");

        // And the audio is continuous speech, so a model that loaded but scores nothing would still agree with
        // itself. It must actually hear the speech.
        Assert.True(fromOnnx.Count(p => p >= 0.5f) > chunks / 2,
            $"only {fromOnnx.Count(p => p >= 0.5f)} of {chunks} chunks scored as speech on continuous speech");
    }

    private static float[] Score(string weightsPath, float[] audio, int chunks)
    {
        // A model root holding only the one file under test, so LoadVad's own preference order cannot quietly
        // pick the other format. Goes through the service's loader rather than a copy of it, so the test covers
        // the path production actually takes.
        string root = Path.Combine(Path.GetTempPath(), "silero-onnx-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(root, "vad"));
        try
        {
            File.Copy(weightsPath, Path.Combine(root, "vad", Path.GetFileName(weightsPath)));
            using WakeModelSet models = new(root);
            Assert.True(models.LoadVad(), $"LoadVad found nothing usable in '{root}/vad'");
            SileroVadStream stream = models.CreateVad(minSilenceMs: 100)
                ?? throw new InvalidOperationException($"CreateVad returned null for '{weightsPath}'");
            try
            {
                return ScoreWith(stream, audio, chunks);
            }
            finally
            {
                stream.Model.Dispose();
            }
        }
        finally
        {
            try { Directory.Delete(root, recursive: true); } catch (IOException) { }
        }
    }

    private static float[] ScoreWith(SileroVadStream stream, float[] audio, int chunks)
    {
        using CpuBackend backend = new();

        float[] probabilities = new float[chunks];
        for (int i = 0; i < chunks; i++)
        {
            stream.Push(backend, audio.AsSpan(i * SileroVad.WindowSamples, SileroVad.WindowSamples), out _);
            probabilities[i] = stream.LastProbability;
        }
        return probabilities;
    }

    private static bool HasShape(Tensor tensor, int[] shape)
    {
        if (tensor.Shape.Rank != shape.Length) return false;
        for (int i = 0; i < shape.Length; i++)
        {
            if (tensor.Shape[i] != shape[i]) return false;
        }
        return true;
    }

    private static float[] ReadF32(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        float[] values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, values.Length * sizeof(float));
        return values;
    }
}
