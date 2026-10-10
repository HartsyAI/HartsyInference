using HartsyInference.Core.Configuration;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu.Moe;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;
using static HartsyInference.Cpu.Tests.Moe.MoeTestData;

namespace HartsyInference.Cpu.Tests.Moe;

/// <summary>
/// The packed CPU kernels on real Qwen3-30B-A3B Q4_K_M experts, read in place from the GGUF's stacked expert tensors as the
/// offload path will read them. Q4_K_M keeps gate and up in Q4_K and stores half the layers' down projections in Q6_K, so one
/// layer of each kind runs; each expert is compared with the F32 reference on the same dequantized weights.
/// </summary>
[Trait("Category", "Integration")]
public sealed unsafe class GgufMixedQuantExpertTests(ITestOutputHelper output)
{
    private const string RelativePath = "llm/moe-parity/qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf";

    [Fact]
    public void Qwen3Q4KMExperts_MatchTheF32Reference_ForQ4KAndQ6KDownLayers()
    {
        string root = EngineKnobs.ModelsRoot.Value ?? "/mnt/model-storage/Models";
        string path = Path.Combine(root, RelativePath);
        if (!RealWeightGate.Require(output.WriteLine, path)) return;
        using GgufLoader gguf = new();
        gguf.Load(path);

        int layers = (int)gguf.Metadata.GetUInt32("qwen3moe.block_count");
        int? q4Layer = null, q6Layer = null;
        for (int l = 0; l < layers && (q4Layer is null || q6Layer is null); l++)
        {
            DType down = gguf.Descriptors[$"blk.{l}.ffn_down_exps.weight"].DType;
            if (down == DType.Q4_K) q4Layer ??= l;
            if (down == DType.Q6_K) q6Layer ??= l;
        }
        Assert.NotNull(q4Layer);
        Assert.NotNull(q6Layer);

        foreach (int layer in new[] { q4Layer!.Value, q6Layer!.Value })
        {
            foreach (int expert in new[] { 0, 77, 127 })
            {
                using Tensor gate = ExpertSlice(gguf, layer, "gate", expert);
                using Tensor up = ExpertSlice(gguf, layer, "up", expert);
                using Tensor down = ExpertSlice(gguf, layer, "down", expert);
                int hidden = (int)gate.Shape[1];
                int inter = (int)gate.Shape[0];
                ExpertDTypes dtypes = new(gate.DType, up.DType, down.DType);
                F32ExpertWeights reference = new(hidden, inter, Dequantized(gate), Dequantized(up), Dequantized(down));

                const int rows = 4;
                float[] x = Random(rows * hidden, seed: layer * 1000 + expert, scale: 1f);
                float[] expected = new float[rows * hidden];
                ExpertProgramReference.Apply(ExpertProgram.Swiglu, reference, x, rows, expected);
                float[] actual = new float[rows * hidden];
                CpuExpertKernels.Apply(ExpertProgram.Swiglu, dtypes, hidden, inter, Bytes(gate), Bytes(up), Bytes(down), x, rows, actual);

                float maxAbs = 0f, maxRef = 0f;
                for (int i = 0; i < expected.Length; i++)
                {
                    maxAbs = MathF.Max(maxAbs, MathF.Abs(actual[i] - expected[i]));
                    maxRef = MathF.Max(maxRef, MathF.Abs(expected[i]));
                }
                float relative = maxAbs / maxRef;
                output.WriteLine($"layer {layer} expert {expert} {dtypes.Gate.Name}/{dtypes.Up.Name}/{dtypes.Down.Name}: relative {relative:E3}");
                Assert.True(relative <= 2.5e-2f, $"layer {layer} expert {expert}: relative error {relative:E3}.");
            }
        }
    }

    /// <summary>One expert's rows of a stacked <c>ffn_{proj}_exps</c> tensor: a view, no copy.</summary>
    private static Tensor ExpertSlice(GgufLoader gguf, int layer, string projection, int expert)
    {
        Tensor stacked = gguf.GetTensor($"blk.{layer}.ffn_{projection}_exps.weight");
        // GGUF dims are fastest first: [cols, rows, experts]. Viewed as [experts·rows, cols], expert e is a contiguous row range.
        long cols = stacked.Shape[0], rows = stacked.Shape[1], experts = stacked.Shape[2];
        return stacked.Reshape(new TensorShape(experts * rows, cols)).SliceRows(expert * rows, rows);
    }

    private static float[] Dequantized(Tensor quant)
    {
        using Tensor f32 = GgufDequantizer.Dequantize(quant, DType.F32);
        return f32.AsReadOnlySpan<float>().ToArray();
    }

    private static ReadOnlySpan<byte> Bytes(Tensor t) => new(t.DataPointer, checked((int)t.DType.ComputeByteCount(t.ElementCount)));
}
