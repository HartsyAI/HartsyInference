using System.Text.Json;
using HartsyInference.Audio.Models.Haru;
using HartsyInference.Audio.Models.VibeVoice;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.Haru;

/// <summary>Per-tensor re-quantization error of the released Haru-4B language model at Q8_0 and Q6_K, to find a tensor that breaks a K-quant.
/// Opt-in: <c>HARU_REAL_DIR</c> names the folder holding <c>model.safetensors</c> and <c>HARU_DIAG=1</c> runs it.</summary>
[Trait("Category", "Integration")]
public sealed unsafe class HaruQuantErrorDiagnostic(ITestOutputHelper output)
{
    [Fact]
    public void PerTensorError_AtQ8AndK_Quants()
    {
        string? dir = Environment.GetEnvironmentVariable("HARU_REAL_DIR");
        if (string.IsNullOrEmpty(dir) || Environment.GetEnvironmentVariable("HARU_DIAG") != "1" || !File.Exists(Path.Combine(dir, "model.safetensors")))
        {
            return;
        }
        string[] suffixes =
        [
            ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
            ".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight", ".linear_attn.out_proj.weight",
            ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight",
        ];
        int[] layers = [0, 1, 2, 3, 15, 30, 31];
        DType[] targets = [DType.Q8_0, DType.Q6_K, DType.Q4_K];
        using HfQuantCheckpoint checkpoint = HfQuantCheckpoint.OpenFiles([Path.Combine(dir, "model.safetensors")]);
        output.WriteLine("tensor | shape | " + string.Join(" | ", targets.Select(t => $"{t.Name} relRMS / maxAbs")));
        foreach (int layer in layers)
        {
            foreach (string suffix in suffixes)
            {
                string name = $"lm.layers.{layer}{suffix}";
                if (!checkpoint.Weights.TryGetValue(name, out HfQuantWeight? w))
                {
                    continue;
                }
                List<string> cells = [];
                foreach (DType target in targets)
                {
                    GgufQuantError error = new();
                    using Tensor t = checkpoint.Produce(name, target, error);
                    cells.Add($"{error.RelativeRms:F4} / {error.MaxAbs:F3}");
                }
                output.WriteLine($"{name} | {w.Shape} | {string.Join(" | ", cells)}");
            }
        }
    }

    /// <summary>Whether the CUDA linear on a K-quant weight equals a plain matmul over the same weight dequantized to F32: separates a kernel
    /// error from the quantization error itself. <c>HARU_DIAG=2</c>.</summary>
    [Fact]
    public unsafe void CudaKQuantLinear_EqualsTheDequantizedMatmul()
    {
        string? dir = Environment.GetEnvironmentVariable("HARU_REAL_DIR");
        if (string.IsNullOrEmpty(dir) || Environment.GetEnvironmentVariable("HARU_DIAG") != "2" || !File.Exists(Path.Combine(dir, "model.safetensors")))
        {
            return;
        }
        using IBackend backend = BackendFactory.Create("cuda");
        using HfQuantCheckpoint checkpoint = HfQuantCheckpoint.OpenFiles([Path.Combine(dir, "model.safetensors")]);
        string[] names = ["lm.layers.0.mlp.gate_proj.weight", "lm.layers.0.mlp.down_proj.weight", "lm.layers.3.self_attn.q_proj.weight", "lm.layers.31.linear_attn.out_proj.weight"];
        Random random = new(1);
        foreach (string name in names)
        {
            foreach (DType dtype in new[] { DType.Q8_0, DType.Q6_K, DType.Q5_K, DType.Q4_K })
            {
                if (name.Contains("linear_attn", StringComparison.Ordinal) && !checkpoint.Weights.ContainsKey(name))
                {
                    continue;
                }
                using Tensor weight = checkpoint.Produce(name, dtype);
                using Tensor plain = GgufDequantizer.Dequantize(weight, DType.F32);
                int outDim = (int)weight.Shape[0], inDim = (int)weight.Shape[1];
                foreach (int rows in new[] { 1, 16 })
                {
                    using Tensor input = new(new TensorShape(1, rows, inDim), DType.F32);
                    float* x = (float*)input.DataPointer;
                    for (int i = 0; i < rows * inDim; i++)
                    {
                        x[i] = (float)(random.NextDouble() * 2 - 1);
                    }
                    using Tensor got = new(new TensorShape(1, rows, outDim), DType.F32);
                    backend.Linear(got, input, weight, null);
                    float* g = (float*)got.DataPointer;
                    float* w = (float*)plain.DataPointer;
                    double err = 0, norm = 0;
                    for (int r = 0; r < rows; r++)
                    {
                        for (int o = 0; o < outDim; o += 7)
                        {
                            double acc = 0;
                            for (int k = 0; k < inDim; k++)
                            {
                                acc += (double)w[(long)o * inDim + k] * x[r * inDim + k];
                            }
                            double d = g[r * outDim + o] - acc;
                            err += d * d;
                            norm += acc * acc;
                        }
                    }
                    output.WriteLine($"{name} {dtype.Name} rows={rows}: relative error of the CUDA linear against the dequantized matmul = {Math.Sqrt(err / norm):E3}");
                }
            }
        }
    }

    /// <summary>Which tensors of the language model carry the K-quant sensitivity: the prompt prefill at Q8_0 is the reference and every
    /// variant swaps some tensors for Q6_K; the distance of the last hidden row (and of the F32 PAD/EOP logits) from the reference ranks them.
    /// <c>HARU_DIAG=3</c>, <c>HARU_CACHE_DIR</c> holds the Q8_0 cache.</summary>
    [Fact]
    public unsafe void PrefillSensitivity_PerTensorClassAndLayerRange()
    {
        string? dir = Environment.GetEnvironmentVariable("HARU_REAL_DIR");
        if (string.IsNullOrEmpty(dir) || Environment.GetEnvironmentVariable("HARU_DIAG") != "3" || !File.Exists(Path.Combine(dir, "model.safetensors")))
        {
            return;
        }
        string cacheDir = Environment.GetEnvironmentVariable("HARU_CACHE_DIR") ?? Path.GetTempPath();
        using IBackend backend = BackendFactory.Create("cuda");
        HaruConfig config = HaruConfig.FromJson(File.ReadAllText(Path.Combine(dir, "config.json")));
        HaruTextCodec text;
        using (FileStream json = File.OpenRead(Path.Combine(dir, "tokenizer.json")))
        {
            text = HaruTextCodec.From(HaruTokenizer.Load(json, config));
        }
        using SafeTensorsLoader expected = new();
        expected.Load(Path.Combine(dir, "real_session_expected.safetensors"));
        List<float[]> draws = [];
        for (int i = 0; expected.Descriptors.ContainsKey($"run.draw.{i}"); i++)
        {
            draws.Add(HaruReferenceModel.Read(expected.GetTensor($"run.draw.{i}")));
        }
        using JsonDocument meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "real_session_expected.json")));
        string card = meta.RootElement.GetProperty("card").GetString()!;
        float[] voice = HaruReferenceModel.Read(expected.GetTensor("in.voice"));

        HfQuantCheckpoint checkpoint = HfQuantCheckpoint.OpenFiles([Path.Combine(dir, "model.safetensors")]);
        (IReadOnlyDictionary<string, Tensor> q8, IDisposable owner) = HaruWeights.Load(checkpoint, HaruPrecision.Q8_0, Path.Combine(cacheDir, "haru-test-Q8_0.gguf"));
        using (owner)
        {
            VibeVoiceAcousticTokenizerModel codecModel = HaruCodec.Create(config.Codec, q8, keepStoredWeight: true);
            using HaruCodecSession codec = new(codecModel, backend, config.LatentDim, config.Codec.FixStd);
            HaruPrompt prompt = HaruPrompt.Build(config, text, codec, new HaruScriptedNoise(draws), card, voice, speakFirst: false);
            output.WriteLine($"prompt: {prompt.Length} positions");
            int warned = 0;

            (float[] Hidden, float[] Exact) Run(IReadOnlyDictionary<string, Tensor> weights)
            {
                using HaruLm lm = new(config, weights, prompt.Length + 8);
                WhisperOps.PreloadOrStream(backend, [.. lm.Trunk.EnumerateWeights(), .. lm.EnumerateWeights()], "diag", ref warned);
                lm.Reset();
                int d = config.LatentDim;
                Tensor? last = null;
                for (int start = 0; start < prompt.Length; start += 512)
                {
                    int rows = Math.Min(512, prompt.Length - start);
                    using Tensor embeds = lm.Embed(backend, prompt.Tokens.AsSpan(start, rows), prompt.ModelLatents.AsSpan(start * d, rows * d),
                        prompt.ModelMask.AsSpan(start, rows), prompt.UserLatents.AsSpan(start * d, rows * d), prompt.UserMask.AsSpan(start, rows));
                    using Tensor hidden = lm.Forward(backend, embeds, rows);
                    last?.Dispose();
                    last = new Tensor(new TensorShape(1, 1, lm.HiddenSize), DType.F32);
                    new ReadOnlySpan<float>((float*)hidden.DataPointer + (long)(rows - 1) * lm.HiddenSize, lm.HiddenSize)
                        .CopyTo(new Span<float>((void*)last.DataPointer, lm.HiddenSize));
                }
                float[] h = new ReadOnlySpan<float>((void*)last!.DataPointer, lm.HiddenSize).ToArray();
                Span<float> exact = stackalloc float[2];
                lm.StreamLogits(last, 0, [config.PadTokenId, config.EopTokenId], exact);
                last.Dispose();
                return (h, exact.ToArray());
            }

            (float[] refHidden, float[] refExact) = Run(q8);
            string[] classes = ["attn", "gate", "up", "down"];
            bool InClass(string name, string cls) => cls switch
            {
                "attn" => new[] { ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
                    ".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight", ".linear_attn.out_proj.weight" }.Any(s => name.EndsWith(s, StringComparison.Ordinal)),
                "gate" => name.EndsWith(".mlp.gate_proj.weight", StringComparison.Ordinal),
                "up" => name.EndsWith(".mlp.up_proj.weight", StringComparison.Ordinal),
                _ => name.EndsWith(".mlp.down_proj.weight", StringComparison.Ordinal),
            };
            bool IsMatrix(string n) => n.StartsWith("lm.layers.", StringComparison.Ordinal) && classes.Any(c => InClass(n, c));
            int LayerOf(string n) => int.Parse(n.Split('.')[2]);
            List<(string Label, Func<string, bool> Pick)> variants = [];
            foreach (string c in classes)
            {
                string cls = c;
                variants.Add(($"class {cls}, all layers", n => IsMatrix(n) && InClass(n, cls)));
            }
            foreach ((int lo, int hi) in new[] { (0, 7), (8, 15), (16, 23), (24, 31) })
            {
                int a = lo, b = hi;
                variants.Add(($"layers {a}-{b}, all classes", n => IsMatrix(n) && LayerOf(n) >= a && LayerOf(n) <= b));
            }
            variants.Add(("every matrix", IsMatrix));
            foreach ((string label, Func<string, bool> pick) in variants)
            {
                Dictionary<string, Tensor> mixed = new(q8, StringComparer.Ordinal);
                List<Tensor> made = [];
                foreach (string name in q8.Keys.Where(pick).ToList())
                {
                    Tensor t6 = checkpoint.Produce(name, DType.Q6_K);
                    made.Add(t6);
                    mixed[name] = t6;
                }
                (float[] h, float[] e) = Run(mixed);
                double err = 0, norm = 0;
                for (int i = 0; i < h.Length; i++)
                {
                    err += (double)(h[i] - refHidden[i]) * (h[i] - refHidden[i]);
                    norm += (double)refHidden[i] * refHidden[i];
                }
                output.WriteLine($"{label} ({made.Count} tensors at Q6_K): hidden relative error {Math.Sqrt(err / norm):E3}, PAD logit {refExact[0]:F3} -> {e[0]:F3}, EOP logit {refExact[1]:F3} -> {e[1]:F3}");
                foreach (Tensor t6 in made)
                {
                    t6.Dispose();
                }
            }
        }
    }
}
