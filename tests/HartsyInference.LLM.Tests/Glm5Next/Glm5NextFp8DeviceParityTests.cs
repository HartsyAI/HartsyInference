using HartsyInference.Core.Backends;
using HartsyInference.Cuda;
using HartsyInference.LLM.Glm5Next;
using HartsyInference.LLM.Tests.Parity;
using HartsyInference.LLM.Transformer;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Glm5Next;

/// <summary>The tiny block-scaled FP8 glm5_next checkpoint on a GPU backend against transformers on the dequantized weights (the same fixture
/// and bound as <see cref="Glm5NextFp8ParityTests"/>). The packed FP8 projections run through <c>Fp8BlockLinear</c> on the device. A lane with no
/// device or kernels is reported as skipped (see <see cref="DeviceProbe"/>).</summary>
public sealed class Glm5NextFp8DeviceParityTests(ITestOutputHelper output)
{
    private const int Block = 16;
    private const float Tolerance = 5e-4f;
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Glm5Next", "tiny_fp8");

    [CudaFact]
    public void Cuda_Fp8Decoder_PrefillLogitsMatchTheReference() => AssertPrefill(() => new CudaBackend(0, CudaPtxDirectory.Resolve()), "CUDA");

    [VulkanFact]
    public void Vulkan_Fp8Decoder_PrefillLogitsMatchTheReference() => AssertPrefill(() => new VulkanBackend(), "Vulkan");

    private void AssertPrefill(Func<IBackend> makeBackend, string name)
    {
        Glm5NextConfig c = Glm5NextHfConfig.Parse(File.ReadAllText(Path.Combine(Dir, "config.json")));
        int[] ids = [.. NpyReader.ReadInt64(Path.Combine(Dir, "input_ids.npy")).Data.Select(v => (int)v)];
        float[] expected = NpyReader.ReadFloat32(Path.Combine(Dir, "fp8_logits.npy")).Data;
        using IBackend backend = makeBackend();
        using BlockFp8CheckpointWeights source = BlockFp8CheckpointWeights.Open(Dir, Glm5NextTextModel.ModelType, block: Block);
        using Glm5NextModel model = new(backend, c, source);
        using Glm5NextState state = new(c, ids.Length);
        float[] logits = model.Forward(ids, state);

        Assert.Equal(expected.Length, logits.Length);
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++) worst = MathF.Max(worst, MathF.Abs(expected[i] - logits[i]));
        output.WriteLine($"{name} FP8 tiny glm5_next prefill: max abs difference {worst:E3} against transformers on the dequantized weights ({expected.Length} values)");
        Assert.True(worst < Tolerance, $"{name} FP8 prefill: max abs difference {worst} exceeds {Tolerance}");
    }
}
