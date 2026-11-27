using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Glm5Next;
using HartsyInference.LLM.Transformer;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Glm5Next;

/// <summary>The tiny glm5_next decoder with its FFN and DSA projections stored as block-scaled FP8 (E4M3, 16x16 blocks, F32 <c>weight_scale_inv</c>)
/// against transformers on the same weights dequantized to F32 (<c>tools/parity/glm5_next_fp8_reference.py</c>, <c>fp8_logits.npy</c>). Both sides see
/// exactly <c>fp8 * scale_inv</c>, so the bound is the float32 arithmetic, not the quantization.</summary>
public sealed class Glm5NextFp8ParityTests(ITestOutputHelper output)
{
    private const int Block = 16;
    private const float Tolerance = 5e-4f;
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Glm5Next", "tiny_fp8");

    [Fact]
    public void Fp8Decoder_PrefillLogitsMatchTransformersOnTheDequantizedWeights()
    {
        Glm5NextConfig c = Glm5NextHfConfig.Parse(File.ReadAllText(Path.Combine(Dir, "config.json")));
        int[] ids = [.. NpyReader.ReadInt64(Path.Combine(Dir, "input_ids.npy")).Data.Select(v => (int)v)];
        float[] expected = NpyReader.ReadFloat32(Path.Combine(Dir, "fp8_logits.npy")).Data;
        using BlockFp8CheckpointWeights source = BlockFp8CheckpointWeights.Open(Dir, Glm5NextTextModel.ModelType, block: Block);
        using Glm5NextModel model = new(new CpuBackend(), c, source);
        using Glm5NextState state = new(c, ids.Length);
        float[] logits = model.Forward(ids, state);

        Assert.Equal(expected.Length, logits.Length);
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++) worst = MathF.Max(worst, MathF.Abs(expected[i] - logits[i]));
        output.WriteLine($"FP8 tiny glm5_next prefill: max abs difference {worst:E3} against transformers on the dequantized weights ({expected.Length} values)");
        Assert.True(worst < Tolerance, $"FP8 prefill: max abs difference {worst} exceeds {Tolerance}");
    }
}
