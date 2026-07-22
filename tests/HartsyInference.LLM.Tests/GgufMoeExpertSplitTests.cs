using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Regression test for the gemma4-moe <c>KeyNotFoundException</c> (real checkpoints — e.g.
/// unsloth/gemma-4-26B-A4B-it-GGUF — fuse the gate+up expert projections into one <c>ffn_gate_up_exps</c> tensor,
/// llama.cpp's <c>LLM_TENSOR_FFN_GATE_UP_EXPS</c>, instead of separate <c>ffn_gate_exps</c>/<c>ffn_up_exps</c>).
/// Verifies <see cref="GgufLanguageModel.SplitStackedExperts"/> splits the fused tensor into the exact per-expert
/// <c>gate_proj</c>/<c>up_proj</c> byte ranges <see cref="MoeFeedForward.LoadWeights"/> expects (gate = first
/// <c>inter</c> rows of each expert's <c>2*inter</c>-row block, up = the next <c>inter</c> rows — matching ggml
/// <c>build_moe_ffn</c>'s <c>gate_up_exps</c> view split), alongside the pre-existing separate <c>down_exps</c> split.</summary>
public sealed unsafe class GgufMoeExpertSplitTests
{
    private static Tensor F2(int rows, int cols)
    {
        Tensor t = new(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                p[r * cols + c] = r * 100f + c;
        return t;
    }

    private static float At(Tensor t, int row, int col) => ((float*)t.DataPointer)[row * (int)t.Shape[1] + col];

    [Fact]
    public void SplitStackedExperts_FusedGateUpExps_SplitsGateThenUpPerExpert()
    {
        const int e = 3, inter = 2, hidden = 4;
        TransformerConfig cfg = new()
        {
            HiddenSize = hidden, NumLayers = 1, NumHeads = 1, NumKvHeads = 1, HeadDim = hidden,
            IntermediateSize = inter, VocabSize = 8, MaxPositionEmbeddings = 8,
            Moe = new MoeConfig { NumExperts = e, NumExpertsPerTok = 1, MoeIntermediateSize = inter, Scoring = MoeScoring.Softmax },
        };

        Tensor gateUp = F2(e * 2 * inter, hidden);   // rows [expert0 gate|expert0 up|expert1 gate|expert1 up|...]
        Tensor down = F2(e * hidden, inter);         // pre-existing separate-tensor split path, unaffected by this fix
        Dictionary<string, Tensor> w = new()
        {
            ["model.layers.0.mlp.gate_up_exps.weight"] = gateUp,
            ["model.layers.0.mlp.down_exps.weight"] = down,
        };

        GgufLanguageModel.SplitStackedExperts(w, cfg);

        Assert.False(w.ContainsKey("model.layers.0.mlp.gate_up_exps.weight"));
        Assert.False(w.ContainsKey("model.layers.0.mlp.down_exps.weight"));

        for (int x = 0; x < e; x++)
        {
            Tensor gate = w[$"model.layers.0.mlp.experts.{x}.gate_proj.weight"];
            Tensor up = w[$"model.layers.0.mlp.experts.{x}.up_proj.weight"];
            Assert.Equal(new TensorShape(inter, hidden), gate.Shape);
            Assert.Equal(new TensorShape(inter, hidden), up.Shape);

            int gateRowBase = x * 2 * inter;
            int upRowBase = gateRowBase + inter;
            for (int r = 0; r < inter; r++)
                for (int c = 0; c < hidden; c++)
                {
                    Assert.Equal((gateRowBase + r) * 100f + c, At(gate, r, c));
                    Assert.Equal((upRowBase + r) * 100f + c, At(up, r, c));
                }

            Tensor downProj = w[$"model.layers.0.mlp.experts.{x}.down_proj.weight"];
            Assert.Equal(new TensorShape(hidden, inter), downProj.Shape);
            for (int r = 0; r < hidden; r++)
                for (int c = 0; c < inter; c++)
                    Assert.Equal((x * hidden + r) * 100f + c, At(downProj, r, c));
        }
    }
}
