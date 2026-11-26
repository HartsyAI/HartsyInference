using HartsyInference.Core.Backends;
using HartsyInference.Cuda;
using HartsyInference.LLM.Glm5Next;
using HartsyInference.LLM.Tests.Parity;
using HartsyInference.LLM.Transformer;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Glm5Next;

/// <summary>The tiny-config glm5_next decoder on a GPU backend against transformers' logits: the same fixture and tolerance as
/// <see cref="Glm5NextReferenceParityTests"/>, the prefill and the token-by-token decode. A lane with no device or kernels is reported as skipped (see <see cref="DeviceProbe"/>).</summary>
public sealed class Glm5NextDeviceParityTests(ITestOutputHelper output)
{
    private const float Tolerance = 5e-4f;
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Glm5Next", "tiny");

    [CudaFact]
    public void Cuda_TinyDecoder_PrefillLogitsMatchTheReference() => AssertPrefill(() => new CudaBackend(0, CudaPtxDirectory.Resolve()), "CUDA");

    [VulkanFact]
    public void Vulkan_TinyDecoder_PrefillLogitsMatchTheReference() => AssertPrefill(() => new VulkanBackend(), "Vulkan");

    [CudaFact]
    public void Cuda_TinyDecoder_TokenByTokenDecodeMatchesTheReference() => AssertDecode(() => new CudaBackend(0, CudaPtxDirectory.Resolve()), "CUDA");

    [VulkanFact]
    public void Vulkan_TinyDecoder_TokenByTokenDecodeMatchesTheReference() => AssertDecode(() => new VulkanBackend(), "Vulkan");

    private void AssertPrefill(Func<IBackend> makeBackend, string name)
    {
        int[] ids = Ids();
        float[] logits = Run(makeBackend, ids, decodeOneByOne: false);
        Compare($"{name} prefill", logits, NpyReader.ReadFloat32(Path.Combine(Dir, "logits.npy")).Data);
    }

    private void AssertDecode(Func<IBackend> makeBackend, string name)
    {
        int[] ids = Ids();
        float[] logits = Run(makeBackend, ids, decodeOneByOne: true);
        Compare($"{name} decode", logits, NpyReader.ReadFloat32(Path.Combine(Dir, "decode_logits.npy")).Data);
    }

    private static int[] Ids() => [.. NpyReader.ReadInt64(Path.Combine(Dir, "input_ids.npy")).Data.Select(v => (int)v)];

    private static float[] Run(Func<IBackend> makeBackend, int[] ids, bool decodeOneByOne)
    {
        Glm5NextConfig c = Glm5NextHfConfig.Parse(File.ReadAllText(Path.Combine(Dir, "config.json")));
        using IBackend backend = makeBackend();
        using BlockFp8CheckpointWeights source = BlockFp8CheckpointWeights.Open(Dir, Glm5NextTextModel.ModelType);
        using Glm5NextModel model = new(backend, c, source);
        using Glm5NextState state = new(c, ids.Length);
        if (!decodeOneByOne) return model.Forward(ids, state);
        List<float> logits = [];
        foreach (int id in ids) logits.AddRange(model.Forward([id], state));
        return [.. logits];
    }

    private void Compare(string what, float[] actual, float[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++) worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]));
        output.WriteLine($"{what} tiny glm5_next logits: max abs difference {worst:E3} against the reference ({expected.Length} values)");
        Assert.True(worst < Tolerance, $"{what}: max abs difference {worst} exceeds {Tolerance}");
    }
}
