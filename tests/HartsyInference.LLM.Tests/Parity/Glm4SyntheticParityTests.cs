using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Transformer;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>Synthetic GLM-4 forward-pass parity check against real HF <c>transformers</c> source (no download
/// needed — see <c>tests/python-reference/dump_glm4_synthetic_ref.py</c>). Loads a tiny random-weight
/// Glm4ForCausalLM's exact weights into <see cref="GenericTransformer"/> with a hand-built config mirroring
/// <see cref="GgufConfigFactory"/>'s real glm4 branch (Interleaved RoPE, partial rotary, sandwich norm, QKV
/// bias, untied head, 8:1 GQA — the same order as production's 16:1), runs the same 16-token sequence on the
/// F32 <see cref="CpuBackend"/>, and diffs final logits. Passes at float32 rounding noise (~1e-6), which
/// proves the F32 forward math (RoPE pairing/width, sandwich-norm slots, GQA broadcast, gate/up split, QKV
/// bias) is correct — the still-open 2026-07-22 numeric-retrieval bug against the real checkpoint (see
/// MODEL_STATUS_LLM.md) is therefore isolated to the quantized (Q4_K / <c>--low-vram-quant</c>) compute path,
/// the only way to run the real 9B checkpoint on a 12 GB card.</summary>
public sealed class Glm4SyntheticParityTests
{
    private readonly ITestOutputHelper _output;
    public Glm4SyntheticParityTests(ITestOutputHelper output) => _output = output;

    private static readonly string RefDir = Path.Combine(
        AppContext.BaseDirectory, "..", "..", "..", "..", "python-reference", "glm4_ref");

    private static Dictionary<string, (long[] Shape, string File)> LoadManifest(string dir)
    {
        Dictionary<string, (long[], string)> m = [];
        foreach (string line in File.ReadAllLines(Path.Combine(dir, "manifest.tsv")))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            string[] parts = line.Split('\t');
            long[] shape = [.. parts[1].Split(',').Select(long.Parse)];
            m[parts[0]] = (shape, parts[2]);
        }
        return m;
    }

    private static Dictionary<string, int> LoadMeta(string dir)
    {
        Dictionary<string, int> m = [];
        foreach (string line in File.ReadAllLines(Path.Combine(dir, "meta.tsv")))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            string[] parts = line.Split('\t');
            m[parts[0]] = (int)double.Parse(parts[1]);
        }
        return m;
    }

    private static unsafe Tensor LoadTensor(string dir, long[] shape, string file)
    {
        byte[] bytes = File.ReadAllBytes(Path.Combine(dir, file));
        Tensor t = new(new TensorShape(shape), DType.F32);
        fixed (byte* src = bytes)
            Buffer.MemoryCopy(src, t.DataPointer, bytes.Length, bytes.Length);
        return t;
    }

    private static float[] ReadFloats(string dir, string file)
    {
        byte[] bytes = File.ReadAllBytes(Path.Combine(dir, file));
        float[] result = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, result, 0, bytes.Length);
        return result;
    }

    [Fact]
    public void SyntheticGlm4_MatchesHfTransformers_FinalLogits()
    {
        if (!Directory.Exists(RefDir))
        {
            _output.WriteLine($"SKIPPED: reference dir not found: {RefDir}. Run dump_glm4_synthetic_ref.py first.");
            return;
        }

        Dictionary<string, (long[] Shape, string File)> manifest = LoadManifest(RefDir);
        Dictionary<string, int> meta = LoadMeta(RefDir);

        TransformerConfig cfg = new()
        {
            HiddenSize = meta["hidden"],
            NumLayers = meta["layers"],
            NumHeads = meta["heads"],
            NumKvHeads = meta["kv_heads"],
            HeadDim = meta["head_dim"],
            RotaryDim = meta["rotary_dim"],
            IntermediateSize = meta["intermediate"],
            VocabSize = meta["vocab"],
            RopeTheta = 10000f,
            RmsNormEps = 1e-6f,
            AttentionBias = true,
            TieWordEmbeddings = false,
            Rope = RopeStyle.Interleaved,
            SandwichNorm = true,
            Activation = ActivationKind.Silu,
        };

        Dictionary<string, Tensor> weights = [];
        foreach ((string name, (long[] shape, string file)) in manifest)
        {
            if (!name.StartsWith("w::", StringComparison.Ordinal)) continue;
            weights[name[3..]] = LoadTensor(RefDir, shape, file);
        }

        // Mirror GgufLanguageModel.SplitFusedPhi's glm4 branch: fused [gate|up] rows split at `intermediate`.
        int ffn = meta["intermediate"];
        for (int i = 0; i < meta["layers"]; i++)
        {
            string p = $"model.layers.{i}";
            Tensor gu = weights[$"{p}.mlp.gate_up_proj.weight"];
            long hiddenDim = gu.Shape[1];
            Tensor gate = new(new TensorShape(ffn, hiddenDim), DType.F32);
            Tensor up = new(new TensorShape(ffn, hiddenDim), DType.F32);
            unsafe
            {
                long rowBytes = hiddenDim * sizeof(float);
                Buffer.MemoryCopy(gu.DataPointer, gate.DataPointer, ffn * rowBytes, ffn * rowBytes);
                Buffer.MemoryCopy((byte*)gu.DataPointer + ffn * rowBytes, up.DataPointer, ffn * rowBytes, ffn * rowBytes);
            }
            weights[$"{p}.mlp.gate_proj.weight"] = gate;
            weights[$"{p}.mlp.up_proj.weight"] = up;
            weights.Remove($"{p}.mlp.gate_up_proj.weight");
            gu.Dispose();
        }

        using CpuBackend backend = new();
        using GenericTransformer transformer = new(cfg);
        transformer.LoadWeights(weights, "model", "lm_head.weight");

        (long[] idsShape, string idsFile) = manifest["input_ids"];
        float[] idsF = ReadFloats(RefDir, idsFile);
        int[] tokenIds = [.. idsF.Select(f => (int)f)];
        _ = idsShape;

        using KvCache cache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim);
        using Tensor hidden = transformer.Forward(backend, tokenIds, 0, cache);
        using Tensor logits = transformer.ProjectLogits(backend, hidden, tokenIds.Length);

        (long[] refShape, string refFile) = manifest["logits"];
        float[] refLogits = ReadFloats(RefDir, refFile);

        Assert.Equal(refShape[0] * refShape[1], (long)refLogits.Length);
        Span<float> ours = logits.AsSpan<float>();
        Assert.Equal(refLogits.Length, ours.Length);

        double maxAbsDiff = 0;
        int maxAbsDiffIdx = -1;
        double sumSqDiff = 0, sumSqRef = 0;
        for (int i = 0; i < refLogits.Length; i++)
        {
            double diff = Math.Abs(ours[i] - refLogits[i]);
            if (diff > maxAbsDiff) { maxAbsDiff = diff; maxAbsDiffIdx = i; }
            sumSqDiff += diff * diff;
            sumSqRef += (double)refLogits[i] * refLogits[i];
        }
        double relError = Math.Sqrt(sumSqDiff / Math.Max(sumSqRef, 1e-12));
        int vocab = (int)refShape[1];
        int badPos = maxAbsDiffIdx / vocab, badVocab = maxAbsDiffIdx % vocab;
        _output.WriteLine($"maxAbsDiff={maxAbsDiff:E4} at pos={badPos} vocab={badVocab}; relError={relError:E4}");
        _output.WriteLine($"ours[last8]  = [{string.Join(",", ours[^8..].ToArray().Select(x => x.ToString("F4")))}]");
        _output.WriteLine($"ref[last8]   = [{string.Join(",", refLogits[^8..].Select(x => x.ToString("F4")))}]");

        if (maxAbsDiff > 1e-2)
        {
            // Diverged -- dump per-layer hidden states to localize which layer first went wrong.
            for (int hi = 0; hi < meta["layers"] + 1; hi++)
            {
                (long[] hShape, string hFile) = manifest[$"hidden_{hi}"];
                float[] refHidden = ReadFloats(RefDir, hFile);
                _ = hShape;
                _output.WriteLine($"hidden_{hi} ref[0][:8] = [{string.Join(",", refHidden[..8].Select(x => x.ToString("F4")))}]");
            }
        }

        Assert.True(maxAbsDiff < 1e-2, $"Synthetic glm4 logits diverge from HF transformers: maxAbsDiff={maxAbsDiff:E4} relError={relError:E4} at seq pos {badPos}");
    }
}
