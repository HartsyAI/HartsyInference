using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Ssm;
using HartsyInference.ModelAssets.Gguf;
using Xunit;
using static HartsyInference.LLM.Tests.Ssm.SenseNovaLoopReference;

namespace HartsyInference.LLM.Tests.Ssm;

/// <summary>The official <c>sensenova_loop</c> GGUF layout (OpenSenseNova/llama.cpp @ 24ea7bb, <c>conversion/sensenova_loop.py</c> and
/// <c>conversion/qwen.py</c>): each physical layer stored once at its first execution slot, fused projections, value heads in tiled order, and
/// <c>hc_*_fn</c> routers. A tiny file in that layout, written here by an independent re-statement of the converter, must load into the same
/// model as the HuggingFace route.</summary>
public sealed unsafe class SenseNovaLoopGgufTests
{
    private const int Streams = 2;

    [Theory]
    [InlineData(2, 1, 16)]
    [InlineData(4, 2, 8)]
    public void Untile_UndoesTheConvertersTiling(int keyHeads, int repeat, int rowsPerHead)
    {
        int heads = keyHeads * repeat, cols = 32;
        float[] grouped = SenseNovaLoopTests.Gaussian(new Random(3), (5 + heads * rowsPerHead) * cols, 1f);
        float[] tiled = TileRows(grouped, cols, 5, rowsPerHead, keyHeads, repeat);
        using Tensor src = Matrix(tiled, 5 + heads * rowsPerHead, cols);
        List<Tensor> owned = [];
        Tensor back = Qwen35Model.UntileRows(src, 5, keyHeads, repeat, rowsPerHead, owned);
        Assert.Equal(grouped, back.AsReadOnlySpan<float>().ToArray());
        foreach (Tensor t in owned) t.Dispose();
    }

    [Fact]
    public void Untile_RowsOfAQuantizedTensorMoveAsWholeRows()
    {
        const int keyHeads = 2, repeat = 2, rowsPerHead = 4, cols = 64, rows = keyHeads * repeat * rowsPerHead;
        float[] grouped = SenseNovaLoopTests.Gaussian(new Random(9), rows * cols, 1f);
        using Tensor groupedQ = GgufQuantizer.Quantize(Matrix(grouped, rows, cols), DType.Q8_0);
        using Tensor expected = GgufDequantizer.Dequantize(groupedQ, DType.F32);
        // The tiled file holds the same quantized rows in tiled order: quantization is per row, so permuting the f32 rows first is equivalent.
        using Tensor tiledQ = GgufQuantizer.Quantize(Matrix(TileRows(expected.AsReadOnlySpan<float>().ToArray(), cols, 0, rowsPerHead, keyHeads, repeat), rows, cols), DType.Q8_0);
        List<Tensor> owned = [];
        Tensor back = Qwen35Model.UntileRows(tiledQ, 0, keyHeads, repeat, rowsPerHead, owned);
        using Tensor widened = GgufDequantizer.Dequantize(back, DType.F32);
        Assert.Equal(expected.AsReadOnlySpan<float>().ToArray(), widened.AsReadOnlySpan<float>().ToArray());
        foreach (Tensor t in owned) t.Dispose();
    }

    [Fact]
    public void Untile_ColumnsOfAQuantizedMatrixComeBackAsF16()
    {
        const int keyHeads = 2, repeat = 2, headCols = 8, rows = 6, cols = keyHeads * repeat * headCols;
        float[] grouped = SenseNovaLoopTests.Gaussian(new Random(4), rows * cols, 1f);
        float[] tiled = TileColumns(grouped, rows, headCols, keyHeads, repeat);
        using Tensor tiledQ = GgufQuantizer.Quantize(Matrix(tiled, rows, cols), DType.Q8_0);
        using Tensor widened = GgufDequantizer.Dequantize(tiledQ, DType.F32);
        float[] expected = TileColumns(widened.AsReadOnlySpan<float>().ToArray(), rows, headCols, keyHeads, repeat, inverse: true);
        List<Tensor> owned = [];
        Tensor back = Qwen35Model.UntileColumns(tiledQ, keyHeads, repeat, headCols, owned);
        Assert.Equal(DType.F16, back.DType);
        using Tensor backF32 = back.CastTo(DType.F32);
        float[] got = backF32.AsReadOnlySpan<float>().ToArray();
        for (int i = 0; i < got.Length; i++) Assert.Equal((float)(Half)expected[i], got[i]);
        foreach (Tensor t in owned) t.Dispose();
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TinyGguf_LoadsIntoTheSameModelAsTheHuggingFaceRoute(bool quantized)
    {
        (Dictionary<string, float[]> raw, Dictionary<string, Tensor> hfWeights) = SenseNovaLoopTests.TinyWeights(11, Streams);
        int[] ids = [3, 17, 5, 29, 8, 1, 22];
        string path = Path.Combine(Path.GetTempPath(), $"sensenova_loop_tiny_{Guid.NewGuid():N}.gguf");
        try
        {
            WriteTinyGguf(path, raw, quantized);
            using CpuBackend backend = new();
            string json = SenseNovaLoopTests.TinyConfig(Streams);
            using Qwen35Model fromHf = Qwen35Model.FromHuggingFace(hfWeights, Qwen35HfConfig.FromJson(json), 32, loop: SenseNovaLoopConfig.FromJson(json));
            float[] want = quantized ? [] : fromHf.ForwardHiddenStates(backend, ids);

            using Qwen35Model fromGguf = Qwen35Model.Load(path);
            Assert.Equal(6, fromGguf.NumLayers);
            Assert.NotNull(fromGguf.LoopPlan);
            Assert.Equal(YarnType, fromGguf.RopeScaling.Type);
            Assert.Equal(1.0, fromGguf.RopeScaling.Factor / 2.0);
            // Each stored tensor appears once for the device whatever number of slots read it.
            List<Tensor> weights = fromGguf.EnumerateWeights().ToList();
            Assert.Equal(weights.Count, new HashSet<Tensor>(weights, ReferenceEqualityComparer.Instance).Count);
            if (quantized)
            {
                // The CPU backend cannot multiply by quantized weights (CUDA and Vulkan can), so the quantized file is checked by what it
                // loads as: row-permuted tensors stay Q8_0, the column-permuted ssm_out of the two linear layers is widened to F16.
                Assert.Contains(weights, t => t.DType == DType.Q8_0);
                Assert.Equal(2, weights.Count(t => t.DType == DType.F16));
                return;
            }
            float[] got = fromGguf.ForwardHiddenStates(backend, ids);

            double num = 0, den = 0;
            for (int i = 0; i < got.Length; i++)
            {
                num += (want[i] - (double)got[i]) * (want[i] - (double)got[i]);
                den += (double)want[i] * want[i];
            }
            double rel = Math.Sqrt(num / den);
            Assert.True(rel < 1e-5, $"{(quantized ? "Q8_0" : "F32")} GGUF vs HF route: hidden states relRMS {rel}");
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static Core.Rope.RopeScalingType YarnType => Core.Rope.RopeScalingType.Yarn;

    // ---- the converter, restated ---------------------------------------------------------------------------------------------

    /// <summary>Writes the tiny checkpoint the way the fork's converter lays out a GGUF, with Q8_0 where a row length allows when
    /// <paramref name="quantize"/> is set.</summary>
    private static void WriteTinyGguf(string path, Dictionary<string, float[]> hf, bool quantize)
    {
        const string root = "model.language_model";
        int[] plan = [0, 1, 2, 1, 2, 3];
        List<Tensor> keep = [];
        using GgufWriter writer = new(path);
        const string arch = "sensenova_loop";
        writer.SetMetadata("general.architecture", arch);
        writer.SetMetadata($"{arch}.block_count", (uint)(plan.Length + 1));
        writer.SetMetadata($"{arch}.nextn_predict_layers", 1u);
        writer.SetMetadata($"{arch}.context_length", 64u);
        writer.SetMetadata($"{arch}.embedding_length", (uint)Hidden);
        writer.SetMetadata($"{arch}.feed_forward_length", (uint)Ffn);
        writer.SetMetadata($"{arch}.attention.head_count", (uint)Heads);
        writer.SetMetadata($"{arch}.attention.head_count_kv", (uint)KvHeads);
        writer.SetMetadata($"{arch}.attention.key_length", (uint)HeadDim);
        writer.SetMetadata($"{arch}.attention.value_length", (uint)HeadDim);
        writer.SetMetadata($"{arch}.attention.layer_norm_rms_epsilon", 1e-6f);
        writer.SetMetadata($"{arch}.attention.recurrent_layers", plan.Select(p => Linear[p]).Append(false).ToArray());
        writer.SetMetadata($"{arch}.full_attention_interval", 4u);
        writer.SetMetadata($"{arch}.rope.dimension_count", (uint)RotaryDim);
        writer.SetMetadata($"{arch}.rope.dimension_sections", new[] { 1, 1, 0, 0 });
        writer.SetMetadata($"{arch}.rope.freq_base", (float)RopeTheta);
        writer.SetMetadata($"{arch}.rope.scaling.type", "yarn");
        writer.SetMetadata($"{arch}.rope.scaling.factor", (float)YarnFactor);
        writer.SetMetadata($"{arch}.rope.scaling.original_context_length", (uint)YarnOriginal);
        writer.SetMetadata($"{arch}.ssm.conv_kernel", (uint)ConvKernel);
        writer.SetMetadata($"{arch}.ssm.state_size", (uint)KeyDim);
        writer.SetMetadata($"{arch}.ssm.group_count", (uint)KeyHeads);
        writer.SetMetadata($"{arch}.ssm.time_step_rank", (uint)ValueHeads);
        writer.SetMetadata($"{arch}.ssm.inner_size", (uint)(ValueHeads * ValueDim));
        writer.SetMetadata($"{arch}.loop.start", 1u);
        writer.SetMetadata($"{arch}.loop.end", 3u);
        writer.SetMetadata($"{arch}.loop.count", 2u);
        writer.SetMetadata($"{arch}.mhc.start", 1u);
        writer.SetMetadata($"{arch}.mhc.end", 3u);
        writer.SetMetadata($"{arch}.mhc.streams", (uint)Streams);
        writer.SetMetadata($"{arch}.mhc.iterations", 20u);
        writer.SetMetadata($"{arch}.mhc.fixed_identity", false);
        writer.SetMetadata($"{arch}.mhc.pre_scale", 1.0f);

        Tensor Add(string name, float[] data, int rows, int cols, bool tryQuantize)
        {
            Tensor t = cols > 0 ? Matrix(data, rows, cols) : Vector(data);
            keep.Add(t);
            if (tryQuantize && quantize && cols % 32 == 0)
            {
                Tensor q = GgufQuantizer.Quantize(t, DType.Q8_0);
                keep.Add(q);
                writer.AddTensor(name, q);
                return q;
            }
            writer.AddTensor(name, t);
            return t;
        }
        float[] OnePlus(float[] w) => w.Select(v => v + 1f).ToArray();

        Add("token_embd.weight", hf[$"{root}.embed_tokens.weight"], Vocab, Hidden, false);
        Add("output_norm.weight", OnePlus(hf[$"{root}.norm.weight"]), 0, 0, false);
        for (int slot = 0; slot < plan.Length; slot++)
        {
            int p = plan[slot];
            if (Array.IndexOf(plan, p) != slot) continue;   // later visits are not stored
            string src = $"{root}.layers.{p}", dst = $"blk.{slot}";
            Add($"{dst}.attn_norm.weight", OnePlus(hf[$"{src}.input_layernorm.weight"]), 0, 0, false);
            Add($"{dst}.post_attention_norm.weight", OnePlus(hf[$"{src}.post_attention_layernorm.weight"]), 0, 0, false);
            Add($"{dst}.ffn_gate.weight", hf[$"{src}.mlp.gate_proj.weight"], Ffn, Hidden, true);
            Add($"{dst}.ffn_up.weight", hf[$"{src}.mlp.up_proj.weight"], Ffn, Hidden, true);
            Add($"{dst}.ffn_down.weight", hf[$"{src}.mlp.down_proj.weight"], Hidden, Ffn, true);
            if (Linear[p])
            {
                int convDim = 2 * KeyHeads * KeyDim + ValueHeads * ValueDim, keyWidth = KeyHeads * KeyDim, repeat = ValueHeads / KeyHeads;
                float[] qkv = TileRows(hf[$"{src}.linear_attn.in_proj_qkv.weight"], Hidden, 2 * keyWidth, ValueDim, KeyHeads, repeat);
                float[] z = TileRows(hf[$"{src}.linear_attn.in_proj_z.weight"], Hidden, 0, ValueDim, KeyHeads, repeat);
                Add($"{dst}.attn_qkv_gate.weight", qkv.Concat(z).ToArray(), convDim + ValueHeads * ValueDim, Hidden, true);
                float[] b = TileRows(hf[$"{src}.linear_attn.in_proj_b.weight"], Hidden, 0, 1, KeyHeads, repeat);
                float[] a = TileRows(hf[$"{src}.linear_attn.in_proj_a.weight"], Hidden, 0, 1, KeyHeads, repeat);
                Add($"{dst}.ssm_beta_alpha.weight", b.Concat(a).ToArray(), 2 * ValueHeads, Hidden, true);
                Add($"{dst}.ssm_conv1d.weight", TileRows(hf[$"{src}.linear_attn.conv1d.weight"], ConvKernel, 2 * keyWidth, ValueDim, KeyHeads, repeat), convDim, ConvKernel, false);
                float[] aLog = hf[$"{src}.linear_attn.A_log"].Select(v => -(float)Math.Exp(v)).ToArray();
                Add($"{dst}.ssm_a", TileRows(aLog, 1, 0, 1, KeyHeads, repeat), 0, 0, false);
                Add($"{dst}.ssm_dt.bias", TileRows(hf[$"{src}.linear_attn.dt_bias"], 1, 0, 1, KeyHeads, repeat), 0, 0, false);
                Add($"{dst}.ssm_norm.weight", hf[$"{src}.linear_attn.norm.weight"], 0, 0, false);
                float[] outTiled = TileColumns(hf[$"{src}.linear_attn.out_proj.weight"], Hidden, ValueDim, KeyHeads, repeat);
                Add($"{dst}.ssm_out.weight", outTiled, Hidden, ValueHeads * ValueDim, true);
            }
            else
            {
                Add($"{dst}.attn_q.weight", hf[$"{src}.self_attn.q_proj.weight"], Heads * HeadDim * 2, Hidden, true);
                Add($"{dst}.attn_k.weight", hf[$"{src}.self_attn.k_proj.weight"], KvHeads * HeadDim, Hidden, true);
                Add($"{dst}.attn_v.weight", hf[$"{src}.self_attn.v_proj.weight"], KvHeads * HeadDim, Hidden, true);
                Add($"{dst}.attn_output.weight", hf[$"{src}.self_attn.o_proj.weight"], Hidden, Heads * HeadDim, true);
                Add($"{dst}.attn_q_norm.weight", OnePlus(hf[$"{src}.self_attn.q_norm.weight"]), 0, 0, false);
                Add($"{dst}.attn_k_norm.weight", OnePlus(hf[$"{src}.self_attn.k_norm.weight"]), 0, 0, false);
            }
            if (p is >= 1 and < 3)
            {
                int mix = 2 * Streams + Streams * Streams;
                Add($"{dst}.hc_attn_fn.weight", hf[$"{src}.mhc_attn.weight"], mix, Streams * Hidden, false);
                Add($"{dst}.hc_attn_fn.bias", hf[$"{src}.mhc_attn.bias"], 0, 0, false);
                Add($"{dst}.hc_ffn_fn.weight", hf[$"{src}.mhc_ffn.weight"], mix, Streams * Hidden, false);
                Add($"{dst}.hc_ffn_fn.bias", hf[$"{src}.mhc_ffn.bias"], 0, 0, false);
            }
        }
        writer.Flush();
        foreach (Tensor t in keep) t.Dispose();
    }

    /// <summary>Moves value heads from transformers' grouped order to llama.cpp's tiled order (<c>_reorder_v_heads</c>): tiled head
    /// <c>t = repeatIndex * keyHeads + keyIndex</c> holds grouped head <c>keyIndex * repeat + repeatIndex</c>. With <paramref name="inverse"/>
    /// it goes the other way. Rows before <paramref name="firstRow"/> stay.</summary>
    private static float[] TileRows(float[] data, int rowLen, int firstRow, int rowsPerHead, int keyHeads, int repeat, bool inverse = false)
    {
        float[] result = (float[])data.Clone();
        for (int t = 0; t < keyHeads * repeat; t++)
        {
            int keyIndex = t % keyHeads, repeatIndex = t / keyHeads, grouped = keyIndex * repeat + repeatIndex;
            int from = inverse ? t : grouped, to = inverse ? grouped : t;
            Array.Copy(data, (long)(firstRow + from * rowsPerHead) * rowLen, result, (long)(firstRow + to * rowsPerHead) * rowLen, (long)rowsPerHead * rowLen);
        }
        return result;
    }

    private static float[] TileColumns(float[] data, int rows, int headCols, int keyHeads, int repeat, bool inverse = false)
    {
        float[] result = (float[])data.Clone();
        int cols = keyHeads * repeat * headCols;
        for (int r = 0; r < rows; r++)
            for (int t = 0; t < keyHeads * repeat; t++)
            {
                int keyIndex = t % keyHeads, repeatIndex = t / keyHeads, grouped = keyIndex * repeat + repeatIndex;
                int from = inverse ? t : grouped, to = inverse ? grouped : t;
                Array.Copy(data, r * cols + from * headCols, result, r * cols + to * headCols, headCols);
            }
        return result;
    }

    private static Tensor Matrix(float[] data, int rows, int cols)
    {
        Tensor t = new(new TensorShape(rows, cols), DType.F32);
        data.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static Tensor Vector(float[] data)
    {
        Tensor t = new(new TensorShape(data.Length), DType.F32);
        data.CopyTo(t.AsSpan<float>());
        return t;
    }
}
