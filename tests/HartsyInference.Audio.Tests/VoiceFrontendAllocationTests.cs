using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Audio.Models.Wake;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>The voice front end runs on a real-time audio thread, a frame every 20 ms for the life of a call, so it
/// must not allocate: every managed allocation there eventually costs a GC pause on the thread with the deadline.
/// Synthetic weights of the real shapes, because what is measured is the code path, not the model — the networks
/// still run in full on every frame. Inside <see cref="CpuParallel.EnterInline"/>, as the audio thread runs them.</summary>
public sealed class VoiceFrontendAllocationTests
{
    private const int Warmup = 50;
    private const int Frames = 1_000;

    internal static readonly (string Name, long[] Shape)[] RnnoiseLayout =
    [
        ("conv1.weight", [128, 65, 3]), ("conv1.bias", [128]), ("conv2.weight", [384, 128, 3]), ("conv2.bias", [384]),
        ("gru1.weight_ih_l0", [1152, 384]), ("gru1.weight_hh_l0", [1152, 384]), ("gru1.bias_ih_l0", [1152]),
        ("gru1.bias_hh_l0", [1152]), ("gru2.weight_ih_l0", [1152, 384]), ("gru2.weight_hh_l0", [1152, 384]),
        ("gru2.bias_ih_l0", [1152]), ("gru2.bias_hh_l0", [1152]), ("gru3.weight_ih_l0", [1152, 384]),
        ("gru3.weight_hh_l0", [1152, 384]), ("gru3.bias_ih_l0", [1152]), ("gru3.bias_hh_l0", [1152]),
        ("dense_out.weight", [32, 1536]), ("dense_out.bias", [32]), ("vad_dense.weight", [1, 1536]),
        ("vad_dense.bias", [1]),
    ];

    private static readonly (string Name, long[] Shape)[] SileroLayout =
    [
        ("stft_conv.weight", [258, 1, 256]), ("conv1.weight", [128, 129, 3]), ("conv1.bias", [128]),
        ("conv2.weight", [64, 128, 3]), ("conv2.bias", [64]), ("conv3.weight", [64, 64, 3]), ("conv3.bias", [64]),
        ("conv4.weight", [128, 64, 3]), ("conv4.bias", [128]), ("lstm_cell.weight_ih", [512, 128]),
        ("lstm_cell.weight_hh", [512, 128]), ("lstm_cell.bias_ih", [512]), ("lstm_cell.bias_hh", [512]),
        ("final_conv.weight", [1, 128, 1]), ("final_conv.bias", [1]),
    ];

    [Theory]
    [InlineData(RnnoisePrecision.Float)]
    [InlineData(RnnoisePrecision.Int8)]
    public void RnnoiseStream_At16k_AllocatesNothingPerFrame(RnnoisePrecision precision)
    {
        using RnnoiseWeights weights = new();
        Load(RnnoiseLayout, seed: 1, weights.Load);
        if (precision == RnnoisePrecision.Int8) LoadInt8Tables(weights, seed: 11);
        Assert.Equal(precision, weights.Precision);
        using CpuBackend backend = new();
        using RnnoiseStream stream = new(weights, 16_000);
        float[] input = new float[320];
        float[] output = new float[320 + stream.FrameSize];
        Random rng = new(2);

        using CpuParallel.InlineScope inline = CpuParallel.EnterInline();
        long allocated = 0;
        for (int f = 0; f < Warmup + Frames; f++)
        {
            // int16-scale noise: loud enough that the denoiser never takes its silence shortcut.
            for (int i = 0; i < input.Length; i++) input[i] = (float)((rng.NextDouble() * 2 - 1) * 3000);
            long before = GC.GetAllocatedBytesForCurrentThread();
            stream.Process(backend, input, output);
            if (f >= Warmup) allocated += GC.GetAllocatedBytesForCurrentThread() - before;
        }
        Assert.Equal(0, allocated);
    }

    [Fact]
    public void SileroVad_AllocatesNothingPerChunk()
    {
        using SileroVad vad = new();
        Load(SileroLayout, seed: 3, vad.LoadWeights);
        using CpuBackend backend = new();
        float[] chunk = new float[SileroVad.WindowSamples];
        Random rng = new(4);

        using CpuParallel.InlineScope inline = CpuParallel.EnterInline();
        long allocated = 0;
        for (int f = 0; f < Warmup + Frames; f++)
        {
            for (int i = 0; i < chunk.Length; i++) chunk[i] = (float)((rng.NextDouble() * 2 - 1) * 0.1);
            long before = GC.GetAllocatedBytesForCurrentThread();
            vad.Process(backend, chunk);
            if (f >= Warmup) allocated += GC.GetAllocatedBytesForCurrentThread() - before;
        }
        Assert.Equal(0, allocated);
    }

    /// <summary>Random int8 tables in upstream's arrangement (<see cref="RnnoiseInt8Tables.Arrays"/>), scaled so the
    /// products land where the F32 layers' do, added to already-loaded <paramref name="weights"/>.</summary>
    internal static void LoadInt8Tables(RnnoiseWeights weights, int seed)
    {
        Random rng = new(seed);
        Dictionary<string, Tensor> tables = new(StringComparer.Ordinal);
        try
        {
            foreach ((string name, int count) in RnnoiseInt8Tables.Arrays)
            {
                bool int8 = name.EndsWith("_weights_int8", StringComparison.Ordinal);
                Tensor tensor = new(new TensorShape(count), int8 ? DType.I8 : DType.F32);
                tables[name] = tensor;
                if (int8)
                {
                    Span<sbyte> codes = tensor.AsSpan<sbyte>();
                    for (int i = 0; i < codes.Length; i++) codes[i] = (sbyte)rng.Next(-127, 128);
                    continue;
                }
                bool scale = name.EndsWith("_scale", StringComparison.Ordinal);
                Span<float> values = tensor.AsSpan<float>();
                for (int i = 0; i < values.Length; i++)
                    values[i] = scale ? (float)(3e-6 * (0.5 + rng.NextDouble())) : (float)((rng.NextDouble() * 2 - 1) * 0.05);
            }
            weights.LoadInt8Tables(tables);
        }
        finally
        {
            foreach (Tensor tensor in tables.Values) tensor.Dispose();
        }
    }

    /// <summary>Hands the model small random weights of the given shapes. Both models copy what they are given, so
    /// the originals are disposed straight after. <see cref="RnnoisePairTests"/> uses it too.</summary>
    internal static void Load((string Name, long[] Shape)[] layout, int seed,
        Action<IReadOnlyDictionary<string, Tensor>> load)
    {
        Random rng = new(seed);
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal);
        foreach ((string name, long[] shape) in layout)
        {
            Tensor tensor = new(new TensorShape(shape), DType.F32);
            Span<float> values = tensor.AsSpan<float>();
            for (int i = 0; i < values.Length; i++) values[i] = (float)((rng.NextDouble() * 2 - 1) * 0.05);
            weights[name] = tensor;
        }
        load(weights);
        foreach (Tensor tensor in weights.Values) tensor.Dispose();
    }
}
