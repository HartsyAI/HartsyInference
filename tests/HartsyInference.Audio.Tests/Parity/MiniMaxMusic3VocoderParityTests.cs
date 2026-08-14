using HartsyInference.Audio.Models.Music;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight parity for the MiniMax Music 3 Flow-VAE vocoder against the diffusers reference. Decodes the
/// dump's latents and compares the stereo waveform sample-for-sample. Skips without
/// <c>HARTSY_MINIMAX_MUSIC3_PATH</c> and the reference dump.</summary>
[Trait("Category", "Integration")]
public sealed unsafe class MiniMaxMusic3VocoderParityTests(ITestOutputHelper output)
{
    private const double MaxAbsErrorTolerance = 1e-4;

    [Fact]
    public void Decode_MatchesDiffusersReference() => Run("vocoder_short_in", "vocoder_short_out");

    /// <summary>The full 137-latent window. Slow on a CPU backend (tens of minutes), so it is tiered out of the
    /// default Integration run — the short case above covers the same code path.</summary>
    [Fact]
    [Trait("Category", "Slow")]
    public void Decode_MatchesDiffusersReference_FullWindow() => Run("dit_latents", "vocoder_out");

    private void Run(string latentsName, string expectedName)
    {
        string? checkpoint = Environment.GetEnvironmentVariable("HARTSY_MINIMAX_MUSIC3_PATH");
        MiniMaxMusic3Reference? reference = MiniMaxMusic3Reference.TryLoad();
        if (checkpoint is null || reference is null || !reference.Has(expectedName))
        {
            return;
        }
        string weightsPath = Path.Combine(checkpoint, "vocoder", "diffusion_pytorch_model.safetensors");
        if (!File.Exists(weightsPath))
        {
            return;
        }

        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(weightsPath);
        using MiniMaxMusic3Vocoder vocoder = new MiniMaxMusic3Vocoder();
        vocoder.LoadWeights(loader.GetAllTensors());

        int[] shape = reference.Shape(latentsName);
        float[] latentValues = reference.Read(latentsName);
        using Tensor latents = new Tensor(new TensorShape(shape[0], shape[1], shape[2]), DType.F32);
        latentValues.CopyTo(new Span<float>((float*)latents.DataPointer, latentValues.Length));

        CpuBackend backend = new CpuBackend();
        using Tensor waveform = vocoder.Decode(backend, latents);

        int[] expectedShape = reference.Shape(expectedName);
        Assert.Equal(expectedShape[2], (int)waveform.Shape[2]);
        Assert.Equal(shape[2] * MiniMaxMusic3Vocoder.LatentHopLength, (int)waveform.Shape[2]);

        float[] expected = reference.Read(expectedName);
        ReadOnlySpan<float> actual = new ReadOnlySpan<float>((float*)waveform.DataPointer, (int)waveform.Shape.ElementCount);
        (double meanAbs, double maxAbs, double correlation) = MiniMaxMusic3Reference.Compare(actual, expected);
        output.WriteLine($"[MiniMaxMusic3Vocoder:{expectedName}] samples={actual.Length} meanAbs={meanAbs:E3} maxAbs={maxAbs:E3} corr={correlation:F8}");

        // The two channels come from a channel fold, not a batch axis — identical halves would mean the fold is wrong.
        int perChannel = (int)waveform.Shape[2];
        double channelDelta = 0d;
        for (int i = 0; i < perChannel; i++)
        {
            channelDelta = Math.Max(channelDelta, Math.Abs(actual[i] - actual[perChannel + i]));
        }
        Assert.True(channelDelta > 1e-4, $"left and right are identical (max delta {channelDelta:E3}) — the stereo channel fold is wrong.");
        Assert.True(maxAbs < MaxAbsErrorTolerance, $"vocoder diverges: maxAbs={maxAbs:E3}, meanAbs={meanAbs:E3}, corr={correlation:F8}");
    }
}
