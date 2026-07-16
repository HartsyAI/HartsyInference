using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight parity for the Lance MoT backbone vs the upstream reference. Replays the fixed forward dumped by <c>tests/python-reference/dump_lance_reference.py</c> (same token ids, noise, timestep) on the CPU backend in F32 and diffs the M-RoPE positions, every dumped layer, and the velocity output. Skips cleanly unless <c>LANCE_3B_DIR</c> points at the checkpoint and <c>LANCE_PARITY_DIR</c> at the reference dump. ~25 GB host RAM (weights cast to F32) + several minutes of CPU GEMM.</summary>
[Trait("Category", "Integration")]
public unsafe class LanceRealWeightParityTests
{
    private readonly ITestOutputHelper _output;
    public LanceRealWeightParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Backbone_FixedForward_MatchesReferenceDump()
    {
        string ckptDir = TestPaths.Lance.Dir;
        string parityDir = Environment.GetEnvironmentVariable("LANCE_PARITY_DIR") ?? "";
        if (!Directory.Exists(ckptDir)) { _output.WriteLine($"SKIPPED: Lance checkpoint not found: {ckptDir}"); return; }
        if (parityDir.Length == 0 || !File.Exists(Path.Combine(parityDir, "manifest.json")))
        {
            _output.WriteLine("SKIPPED: LANCE_PARITY_DIR unset or missing manifest.json (run dump_lance_reference.py first).");
            return;
        }

        System.Text.Json.JsonDocument manifest = System.Text.Json.JsonDocument.Parse(
            File.ReadAllText(Path.Combine(parityDir, "manifest.json")));
        System.Text.Json.JsonElement root = manifest.RootElement;
        int gridH = root.GetProperty("grid")[1].GetInt32();
        int gridW = root.GetProperty("grid")[2].GetInt32();
        int nVae = root.GetProperty("n_vae").GetInt32();
        int seq = root.GetProperty("seq").GetInt32();
        float timestep = root.GetProperty("timestep").GetSingle();
        int[] prefix = ReadIntArray(root, "prefix_tokens");
        int[] caption = ReadIntArray(root, "caption_tokens");
        int[] mid = ReadIntArray(root, "mid_tokens");
        int numLayers = root.GetProperty("num_layers").GetInt32();

        LanceConfig cfg = LanceConfig.Image;
        LancePromptTemplate template = new() { PrefixTokens = prefix, MidTokens = mid, TrailingEos = true };
        using LanceSequence sequence = LancePipelineCommon.BuildGenSequence(template, caption, cfg, 1, gridH, gridW);
        Assert.Equal(seq, sequence.UndIdx.Length + sequence.GenIdx.Length);

        // ── position ids must match the upstream get_rope_index exactly ──
        float[] refPos = ReadF32(Path.Combine(parityDir, "pos_ids.bin"), seq * 3);
        float* posPtr = (float*)sequence.PositionIds.DataPointer;
        for (int i = 0; i < seq * 3; i++)
            Assert.True(Math.Abs(posPtr[i] - refPos[i]) < 0.5f,
                $"pos_ids mismatch at flat {i} (token {i / 3}, axis {i % 3}): engine {posPtr[i]} vs ref {refPos[i]}");
        _output.WriteLine("pos_ids: exact match vs upstream get_rope_index");

        // ── load real weights (cast to F32 for the CPU kernels) ──
        string dumpDir = Path.Combine(Path.GetTempPath(), "lance_parity_engine_" + Environment.ProcessId);
        Environment.SetEnvironmentVariable("LANCE_DEBUG_DIR", dumpDir);
        _output.WriteLine($"Loading checkpoint (engine dumps → {dumpDir}) ...");
        (LanceCheckpointConverter.ConvertedWeights conv, IReadOnlyList<SafeTensorsLoader> loaders) =
            LanceCheckpointConverter.LoadAndConvert(ckptDir);
        try
        {
            Dictionary<string, Tensor> f32 = new(conv.Transformer.Count);
            foreach (KeyValuePair<string, Tensor> kv in conv.Transformer)
                f32[kv.Key] = kv.Value.DType == DType.F32 ? kv.Value : kv.Value.CastTo(DType.F32);

            using LanceTransformer transformer = new(cfg);
            transformer.LoadWeights(f32);
            using CpuBackend backend = new();

            Tensor latents = new Tensor(new TensorShape(nVae, cfg.PatchFeatureDim), DType.F32);
            float[] noise = ReadF32(Path.Combine(parityDir, "noise.bin"), nVae * cfg.PatchFeatureDim);
            noise.CopyTo(new Span<float>((float*)latents.DataPointer, noise.Length));
            int[] latentPosIds = LancePipelineCommon.BuildLatentPositionIds(1, gridH, gridW, cfg.MaxLatentSize);

            _output.WriteLine($"Forward: seq={seq}, nVae={nVae}, t={timestep} ...");
            Tensor velocity = transformer.Forward(backend, sequence.TextTokenIds, latents, latentPosIds, timestep,
                sequence.PositionIds, sequence.UndIdx, sequence.GenIdx, sequence.AttentionMask);
            latents.Dispose();

            // ── per-layer diff (engine LANCE_DEBUG_DIR dumps vs reference dumps) ──
            (double avg0, double max0) = Diff(Path.Combine(dumpDir, "layers", "packed_in.bin"),
                Path.Combine(parityDir, "layers", "packed_in.bin"));
            _output.WriteLine($"packed_in: avg={avg0:E3} max={max0:E3}");
            Assert.True(avg0 < 1e-3, $"packed_in diverged: avg={avg0:E3}");
            for (int i = 0; i < numLayers; i++)
            {
                string refLayer = Path.Combine(parityDir, "layers", $"layers_{i}.bin");
                string engLayer = Path.Combine(dumpDir, "layers", $"layers_{i}.bin");
                if (!File.Exists(refLayer) || !File.Exists(engLayer)) continue;
                (double avg, double max) = Diff(engLayer, refLayer);
                _output.WriteLine($"layers.{i}: avg={avg:E3} max={max:E3}");
            }

            float[] refVel = ReadF32(Path.Combine(parityDir, "output_velocity.bin"), nVae * cfg.PatchFeatureDim);
            float* v = (float*)velocity.DataPointer;
            double sumErr = 0, maxErr = 0;
            for (int i = 0; i < refVel.Length; i++)
            {
                double e = Math.Abs(v[i] - refVel[i]);
                sumErr += e;
                if (e > maxErr) maxErr = e;
            }
            double avgErr = sumErr / refVel.Length;
            double corr = Correlation(v, refVel);
            velocity.Dispose();
            _output.WriteLine($"velocity: avg={avgErr:E3} max={maxErr:E3} corr={corr:F6}");
            Assert.True(avgErr < 5e-3, $"velocity avg err {avgErr:E3} exceeds 5e-3");
            Assert.True(corr > 0.999, $"velocity correlation {corr:F6} below 0.999");
        }
        finally
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
        }
    }

    [Fact]
    public void Tokenizer_TemplateSegments_MatchReferenceDump()
    {
        string ckptDir = TestPaths.Lance.Dir;
        string parityDir = Environment.GetEnvironmentVariable("LANCE_PARITY_DIR") ?? "";
        string tokenizerJson = Path.Combine(ckptDir, "tokenizer.json");
        if (!File.Exists(tokenizerJson)) { _output.WriteLine("SKIPPED: checkpoint tokenizer.json not found."); return; }
        if (parityDir.Length == 0 || !File.Exists(Path.Combine(parityDir, "manifest.json")))
        {
            _output.WriteLine("SKIPPED: LANCE_PARITY_DIR unset (run dump_lance_reference.py first).");
            return;
        }

        System.Text.Json.JsonDocument manifest = System.Text.Json.JsonDocument.Parse(
            File.ReadAllText(Path.Combine(parityDir, "manifest.json")));
        System.Text.Json.JsonElement root = manifest.RootElement;
        string prompt = root.GetProperty("prompt").GetString()!;

        // tokenizer.json → GgufTokenizer: byte-level BPE with the exact pre-tokenizer Split regex.
        // (The two-file Qwen2Tokenizer path mis-splits space+punct like ' "' — 330 vs [220, 1].)
        using FileStream fs = File.OpenRead(tokenizerJson);
        HartsyInference.Tokenizers.GgufTokenizer tokenizer = HartsyInference.Tokenizers.HfTokenizerJson.LoadByteLevelBpe(fs);
        LancePromptTemplate template = LancePromptTemplate.Create(tokenizer.EncodeOrdinary, LanceConfig.Image, video: false);

        Assert.Equal(ReadIntArray(root, "prefix_tokens"), template.PrefixTokens);
        Assert.Equal(ReadIntArray(root, "mid_tokens"), template.MidTokens);
        Assert.Equal(ReadIntArray(root, "caption_tokens"), tokenizer.EncodeOrdinary(prompt));
        _output.WriteLine("Engine byte-level BPE template segments match the HF reference tokenizer exactly.");
    }

    private static int[] ReadIntArray(System.Text.Json.JsonElement root, string name)
    {
        System.Text.Json.JsonElement arr = root.GetProperty(name);
        int[] result = new int[arr.GetArrayLength()];
        for (int i = 0; i < result.Length; i++) result[i] = arr[i].GetInt32();
        return result;
    }

    private static float[] ReadF32(string path, int expected)
    {
        byte[] bytes = File.ReadAllBytes(path);
        Assert.Equal(expected * 4, bytes.Length);
        float[] result = new float[expected];
        Buffer.BlockCopy(bytes, 0, result, 0, bytes.Length);
        return result;
    }

    private static (double Avg, double Max) Diff(string enginePath, string refPath)
    {
        byte[] a = File.ReadAllBytes(enginePath);
        byte[] b = File.ReadAllBytes(refPath);
        Assert.Equal(b.Length, a.Length);
        int n = a.Length / 4;
        float[] fa = new float[n], fb = new float[n];
        Buffer.BlockCopy(a, 0, fa, 0, a.Length);
        Buffer.BlockCopy(b, 0, fb, 0, b.Length);
        double sum = 0, max = 0;
        for (int i = 0; i < n; i++)
        {
            double e = Math.Abs(fa[i] - fb[i]);
            sum += e;
            if (e > max) max = e;
        }
        return (sum / n, max);
    }

    private static double Correlation(float* a, float[] b)
    {
        int n = b.Length;
        double ma = 0, mb = 0;
        for (int i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
        ma /= n; mb /= n;
        double num = 0, da = 0, db = 0;
        for (int i = 0; i < n; i++)
        {
            double xa = a[i] - ma, xb = b[i] - mb;
            num += xa * xb; da += xa * xa; db += xb * xb;
        }
        return num / Math.Sqrt(da * db + 1e-12);
    }
}
