using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Ernie5;
using HartsyInference.ModelAssets.BlockScale;
using Xunit;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>The FP8 checkpoint path on the CPU against its own dequantized F32 twin: the same weights, read as E4M3 with their block scales
/// and as the F32 values those decode to, must give the same logits. Needs no GPU, so it runs in the CPU lane.</summary>
public sealed class Ernie5Fp8ReferenceTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-ernie5-fp8-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    [Fact]
    public void Cpu_SyntheticErnieFp8_LogitsMatchDequantizedF32()
    {
        string fp8 = Path.Combine(_root, "fp8"), twin = Path.Combine(_root, "twin");
        Ernie5SyntheticCheckpoint.Write(fp8, storage: Ernie5SyntheticCheckpoint.Storage.Fp8);
        Ernie5SyntheticCheckpoint.Write(twin, storage: Ernie5SyntheticCheckpoint.Storage.Fp8Dequantized);
        using CpuBackend cpu = new();
        float[] want = Ernie5ModelGpuParityTests.Logits(twin, cpu, resident: false);
        float[] got = Ernie5ModelGpuParityTests.Logits(fp8, cpu, resident: false);
        Ernie5ModelGpuParityTests.AssertClose(want, got, "CPU FP8");
    }

    [Fact]
    public void Cpu_SyntheticErnieFp8_ResidentWeights_LogitsMatchDequantizedF32()
    {
        string fp8 = Path.Combine(_root, "fp8"), twin = Path.Combine(_root, "twin");
        Ernie5SyntheticCheckpoint.Write(fp8, storage: Ernie5SyntheticCheckpoint.Storage.Fp8);
        Ernie5SyntheticCheckpoint.Write(twin, storage: Ernie5SyntheticCheckpoint.Storage.Fp8Dequantized);
        using CpuBackend cpu = new();
        float[] want = Ernie5ModelGpuParityTests.Logits(twin, cpu, resident: true);
        float[] got = Ernie5ModelGpuParityTests.Logits(fp8, cpu, resident: true);
        Ernie5ModelGpuParityTests.AssertClose(want, got, "CPU FP8 resident");
    }

    // Each packed weight, decoded by the block codec and run through the FP8 dispatch, must equal its F32 twin: this localizes a
    // mismatch of the model to the read or to the linear, weight by weight.
    [Fact]
    public void Fp8LinearsMatchTheirTwinWeights()
    {
        string fp8 = Path.Combine(_root, "fp8"), twin = Path.Combine(_root, "twin");
        Ernie5SyntheticCheckpoint.Write(fp8, storage: Ernie5SyntheticCheckpoint.Storage.Fp8);
        Ernie5SyntheticCheckpoint.Write(twin, storage: Ernie5SyntheticCheckpoint.Storage.Fp8Dequantized);
        using Ernie5CheckpointWeights packed = Ernie5CheckpointWeights.Open(fp8, block: 128);
        using Ernie5CheckpointWeights plain = Ernie5CheckpointWeights.Open(twin, block: 128);
        using CpuBackend cpu = new();
        Random rng = new(3);
        int checkedWeights = 0;
        foreach (string name in packed.Names.ToList())
        {
            using Tensor stored = packed.Read(name);
            if (stored.DType != DType.F8E4M3) continue;
            checkedWeights++;
            using Tensor want = plain.Read(name);
            int n = checked((int)stored.Shape[0]), k = checked((int)stored.Shape[1]);

            using Tensor decoded = Fp8BlockF32Codec.Dequantize(stored, stored.QuantInfo!.Fp8BlockScaleInv!, 128, 128);
            Assert.True(want.AsReadOnlySpan<float>().SequenceEqual(decoded.AsReadOnlySpan<float>()), $"{name}: the block decode differs from the twin.");

            using Tensor input = new(new TensorShape(1, k), DType.F32);
            Span<float> x = input.AsSpan<float>();
            for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
            using Tensor packedOut = new(new TensorShape(1, n), DType.F32);
            using Tensor twinOut = new(new TensorShape(1, n), DType.F32);
            cpu.Linear(packedOut, input, stored, null);
            cpu.Linear(twinOut, input, want, null);
            ReadOnlySpan<float> a = packedOut.AsReadOnlySpan<float>(), b = twinOut.AsReadOnlySpan<float>();
            for (int i = 0; i < n; i++)
                Assert.True(MathF.Abs(a[i] - b[i]) <= 1e-4f * MathF.Max(1f, MathF.Abs(b[i])), $"{name} output {i}: packed {a[i]}, twin {b[i]}.");
        }
        Assert.True(checkedWeights > 0, "The FP8 checkpoint holds no block-scaled weight.");
    }
}
