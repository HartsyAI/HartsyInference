using HartsyInference.Audio.Models.Separation;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine;
using Xunit;

namespace HartsyInference.Audio.Tests.Separation;

public sealed unsafe class KimVocal2NetworkTests
{
    private static float[] Run(KimVocal2Reference reference, float[] spec, string backendName = "cpu")
    {
        using IBackend backend = BackendFactory.Create(backendName);
        using KimVocal2Network network = new(reference.Weights);
        using Tensor input = new(new TensorShape(1, KimVocal2Reference.Channels, KimVocal2Reference.DimF, KimVocal2Reference.Frames), DType.F32);
        spec.AsSpan().CopyTo(new Span<float>((void*)input.DataPointer, spec.Length));
        using Tensor output = network.Forward(backend, input);
        Assert.Equal(input.Shape, output.Shape);
        return new ReadOnlySpan<float>((void*)output.DataPointer, spec.Length).ToArray();
    }

    [Fact]
    public void ForwardMatchesThePlainLoopReference() => AssertMatchesReference("cpu");

    [GpuBackendTheory("cuda")]
    [Trait("Category", "GpuIntegration")]
    [InlineData("cuda")]
    public void ForwardOnCudaMatchesThePlainLoopReference(string backendName) => AssertMatchesReference(backendName);

    [GpuBackendTheory("vulkan")]
    [Trait("Category", "GpuIntegration")]
    [InlineData("vulkan")]
    public void ForwardOnVulkanMatchesThePlainLoopReference(string backendName) => AssertMatchesReference(backendName);

    private static void AssertMatchesReference(string backendName)
    {
        using KimVocal2Reference reference = new(11);
        Random rng = new(12);
        float[] spec = new float[KimVocal2Reference.Channels * KimVocal2Reference.DimF * KimVocal2Reference.Frames];
        for (int i = 0; i < spec.Length; i++) spec[i] = (float)(rng.NextDouble() - 0.5);

        float[] expected = reference.Forward(spec, KimVocal2Reference.Frames);
        float[] actual = Run(reference, spec, backendName);

        double num = 0, den = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = expected[i] - actual[i];
            num += d * d;
            den += (double)expected[i] * expected[i];
        }
        Assert.True(den > 0, "the reference output is all zero, so the comparison proves nothing");
        Assert.True(Math.Sqrt(num / den) < 1e-5, $"relRMS {Math.Sqrt(num / den):E2}");
    }

    [Fact]
    public void GeometryIsReadFromTheWeights()
    {
        using KimVocal2Reference reference = new(1);
        using KimVocal2Network network = new(reference.Weights);
        Assert.Equal(KimVocal2Reference.Channels, network.InChannels);
        Assert.Equal(KimVocal2Reference.DimF, network.DimF);
        Assert.Equal(2, network.Levels);
    }

    [Fact]
    public void MissingLayerFailsFast()
    {
        using KimVocal2Reference reference = new(2);
        reference.Weights.Remove("decoding_blocks.1.tdf.2.weight");
        InvalidDataException ex = Assert.Throws<InvalidDataException>(() => new KimVocal2Network(reference.Weights));
        Assert.Contains("decoding_blocks.1.tdf.2.weight", ex.Message);
    }

    [Fact]
    public void TimeAxisNotDivisibleByTheDownsamplingIsRefused()
    {
        using KimVocal2Reference reference = new(3);
        using IBackend backend = new CpuBackend();
        using KimVocal2Network network = new(reference.Weights);
        using Tensor input = new(new TensorShape(1, KimVocal2Reference.Channels, KimVocal2Reference.DimF, 6), DType.F32);
        Assert.Throws<ArgumentException>(() => network.Forward(backend, input));
    }
}
