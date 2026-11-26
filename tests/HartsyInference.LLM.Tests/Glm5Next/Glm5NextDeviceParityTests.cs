using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.LLM.Glm5Next;
using HartsyInference.LLM.Transformer;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Glm5Next;

/// <summary>The tiny-config glm5_next decoder on a GPU backend against transformers' logits (the same fixture and tolerance as
/// <see cref="Glm5NextReferenceParityTests"/>). A test whose device or kernels are absent is reported as skipped, never passed.</summary>
public sealed class Glm5NextDeviceParityTests(ITestOutputHelper output)
{
    private const float Tolerance = 5e-4f;
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Glm5Next", "tiny");

    [Glm5NextCudaFact]
    public void Cuda_TinyDecoder_LogitsMatchTheReference() => AssertMatchesReference(() => new CudaBackend(0, CudaPtxDirectory.Resolve()), "CUDA");

    [Glm5NextVulkanFact]
    public void Vulkan_TinyDecoder_LogitsMatchTheReference() => AssertMatchesReference(() => new VulkanBackend(), "Vulkan");

    private void AssertMatchesReference(Func<IBackend> makeBackend, string name)
    {
        Glm5NextConfig c = Glm5NextHfConfig.Parse(File.ReadAllText(Path.Combine(Dir, "config.json")));
        int[] ids = [.. NpyReader.ReadInt64(Path.Combine(Dir, "input_ids.npy")).Data.Select(v => (int)v)];
        float[] expected = NpyReader.ReadFloat32(Path.Combine(Dir, "logits.npy")).Data;
        using IBackend backend = makeBackend();
        using BlockFp8CheckpointWeights source = BlockFp8CheckpointWeights.Open(Dir, Glm5NextTextModel.ModelType);
        using Glm5NextModel model = new(backend, c, source);
        using Glm5NextState state = new(c, ids.Length);
        float[] logits = model.Forward(ids, state);

        Assert.Equal(expected.Length, logits.Length);
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++) worst = MathF.Max(worst, MathF.Abs(expected[i] - logits[i]));
        output.WriteLine($"{name} tiny glm5_next logits: max abs difference {worst:E3} against the reference ({expected.Length} values)");
        Assert.True(worst < Tolerance, $"{name}: max abs difference {worst} exceeds {Tolerance}");
    }

    private static class Probe
    {
        public static readonly Lazy<string?> Cuda = new(() =>
        {
            try { using CudaBackend _ = new(0, CudaPtxDirectory.Resolve()); return null; }
            catch (Exception e) { return e.Message; }
        });

        public static readonly Lazy<string?> Vulkan = new(() =>
        {
            try { using VulkanBackend _ = new(); return null; }
            catch (Exception e) { return e.Message; }
        });
    }

    /// <summary>A fact reported as skipped when no CUDA device opens or its PTX cannot be found.</summary>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
    private sealed class Glm5NextCudaFactAttribute : FactAttribute
    {
        public Glm5NextCudaFactAttribute()
        {
            string? failure = Probe.Cuda.Value;
            if (failure is not null) Skip = $"SKIPPED: no usable CUDA device or PTX ({failure}); GPU check did not run.";
        }
    }

    /// <summary>A fact reported as skipped when no Vulkan device opens.</summary>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
    private sealed class Glm5NextVulkanFactAttribute : FactAttribute
    {
        public Glm5NextVulkanFactAttribute()
        {
            string? failure = Probe.Vulkan.Value;
            if (failure is not null) Skip = $"SKIPPED: no usable Vulkan device ({failure}); GPU check did not run.";
        }
    }
}
