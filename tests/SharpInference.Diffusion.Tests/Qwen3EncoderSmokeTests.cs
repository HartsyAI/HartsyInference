using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Cuda;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tests.Common;
using SharpInference.Tokenizers;

namespace SharpInference.Diffusion.Tests;

/// <summary>
/// Smoke tests for the Llama-style encoder running Qwen3-4B as a text feature extractor (Flux.2 Klein
/// dependency). Validates: weights load, tokenizer encodes, forward pass returns finite hidden states
/// of the right shape. Doesn't validate against a Python reference — that's a separate validation task.
/// </summary>
public sealed class Qwen3EncoderSmokeTests
{
    private static string Qwen3WeightsPath => TestPaths.TextEncoders.Qwen3_4B;
    private static string Qwen3VocabPath => TestPaths.Tokenizers.Qwen3Vocab;
    private static string Qwen3MergesPath => TestPaths.Tokenizers.Qwen3Merges;

    private readonly ITestOutputHelper _output;

    public Qwen3EncoderSmokeTests(ITestOutputHelper output) => _output = output;

    /// <summary>Tokenize a prompt with the Qwen3 BPE — confirms vocab+merges load and produce reasonable output.</summary>
    [Fact]
    public void Qwen3Tokenizer_EncodesPrompt()
    {
        if (!File.Exists(Qwen3VocabPath) || !File.Exists(Qwen3MergesPath))
        {
            _output.WriteLine($"SKIPPED: Qwen3 tokenizer files not found ({Qwen3VocabPath}, {Qwen3MergesPath})");
            return;
        }

        using Qwen3Tokenizer tok = new Qwen3Tokenizer(Qwen3VocabPath, Qwen3MergesPath, maxLength: 64);
        string prompt = "A photograph of an astronaut riding a horse";
        IReadOnlyList<int> raw = tok.EncodeRaw(prompt);
        int[] padded = tok.Encode(prompt, appendEos: true);
        int[] mask = Qwen3Tokenizer.CreateAttentionMask(padded);

        _output.WriteLine($"Raw token count: {raw.Count}");
        _output.WriteLine($"Raw ids: [{string.Join(", ", raw)}]");
        _output.WriteLine($"Padded length: {padded.Length}, mask sum: {mask.Sum()}");

        Assert.True(raw.Count > 0, "Tokenizer produced an empty token list");
        Assert.Equal(64, padded.Length);
        // EOS should be at position raw.Count (right after the real tokens).
        Assert.Equal(Qwen3Tokenizer.EosTokenId, padded[raw.Count]);
        Assert.True(mask.Sum() == raw.Count + 1, $"Mask sum {mask.Sum()} should equal {raw.Count + 1} (raw tokens + EOS)");
        // All padded slots after EOS should be the BOS pad token.
        for (int i = raw.Count + 1; i < padded.Length; i++)
            Assert.Equal(Qwen3Tokenizer.BosTokenId, padded[i]);
    }

    /// <summary>End-to-end CPU smoke test: load Qwen3-4B safetensors, run a single forward pass on a short prompt,
    /// confirm output shape is [B, S, 2560] and values are finite. This is slow on CPU (4B params) — use small seqLen.</summary>
    [Fact]
    public void Qwen3Encoder_CpuForward_ProducesFiniteHiddenStates()
    {
        if (!File.Exists(Qwen3WeightsPath))
        {
            _output.WriteLine($"SKIPPED: Qwen3 weights not found: {Qwen3WeightsPath}");
            return;
        }
        if (!File.Exists(Qwen3VocabPath) || !File.Exists(Qwen3MergesPath))
        {
            _output.WriteLine("SKIPPED: Qwen3 tokenizer files not found");
            return;
        }

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[1/4] Loading {Path.GetFileName(Qwen3WeightsPath)}...");
        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(Qwen3WeightsPath);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();
        _output.WriteLine($"  {weights.Count} tensors loaded in {sw.ElapsedMilliseconds}ms");

        _output.WriteLine("[2/4] Casting BF16 → F32 (LlamaStyleEncoder uses F32 throughout)...");
        sw.Restart();
        Dictionary<string, Tensor> f32Weights = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            f32Weights[kvp.Key] = kvp.Value.DType == DType.F32 ? kvp.Value : kvp.Value.CastTo(DType.F32);
        }
        sw.Stop();
        _output.WriteLine($"  Cast in {sw.ElapsedMilliseconds}ms");

        _output.WriteLine("[3/4] Building encoder + loading weights...");
        sw.Restart();
        LlamaStyleEncoder encoder = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_4B);
        encoder.LoadWeights(f32Weights);
        sw.Stop();
        _output.WriteLine($"  Loaded in {sw.ElapsedMilliseconds}ms");

        using Qwen3Tokenizer tok = new Qwen3Tokenizer(Qwen3VocabPath, Qwen3MergesPath, maxLength: 16);
        int[] tokens = tok.Encode("A horse on the moon", appendEos: true);
        int[][] batch = [tokens];

        _output.WriteLine("[4/4] Running CPU forward pass (4B params, expect ~minutes)...");
        sw.Restart();
        using CpuBackend backend = new CpuBackend();
        Tensor hidden = encoder.Encode(backend, batch);
        sw.Stop();
        _output.WriteLine($"  Forward done in {sw.ElapsedMilliseconds}ms, output shape={hidden.Shape}, dtype={hidden.DType}");

        Assert.Equal(1, hidden.Shape[0]);
        Assert.Equal(16, hidden.Shape[1]);
        Assert.Equal(2560, hidden.Shape[2]);
        Assert.Equal(DType.F32, hidden.DType);

        ReadOnlySpan<float> data = hidden.AsReadOnlySpan<float>();
        int nan = 0, inf = 0;
        float min = float.MaxValue, max = float.MinValue;
        double sumAbs = 0;
        for (int i = 0; i < data.Length; i++)
        {
            float v = data[i];
            if (float.IsNaN(v)) nan++;
            else if (float.IsInfinity(v)) inf++;
            else
            {
                if (v < min) min = v;
                if (v > max) max = v;
                sumAbs += Math.Abs(v);
            }
        }
        _output.WriteLine($"  Hidden state stats: nan={nan}, inf={inf}, min={min:F4}, max={max:F4}, abs_mean={sumAbs / data.Length:F4}");

        Assert.Equal(0, nan);
        Assert.Equal(0, inf);
        Assert.True(sumAbs > 0, "Hidden states are all zero — forward pass produced no signal");

        hidden.Dispose();
        encoder.Dispose();
    }

    /// <summary>
    /// Klein-shaped multi-layer encoder forward: concatenates Qwen3 hidden states from layers
    /// (9, 18, 27) into a [B, S, 3*2560 = 7680] tensor, matching the input shape Klein's
    /// <c>context_embedder = Linear(7680, 3072)</c> expects. Uses the chat-formatted prompt so the
    /// hidden states match what diffusers' Klein pipeline produces.
    /// </summary>
    [Fact]
    public void Qwen3Encoder_KleinShape_MultiLayerHiddenStates()
    {
        if (!File.Exists(Qwen3WeightsPath))
        {
            _output.WriteLine($"SKIPPED: Qwen3 weights not found: {Qwen3WeightsPath}");
            return;
        }
        if (!File.Exists(Qwen3VocabPath) || !File.Exists(Qwen3MergesPath))
        {
            _output.WriteLine("SKIPPED: Qwen3 tokenizer files not found");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(Qwen3EncoderSmokeTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[1/4] Loading {Path.GetFileName(Qwen3WeightsPath)}...");
        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(Qwen3WeightsPath);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();
        _output.WriteLine($"  {weights.Count} tensors loaded in {sw.ElapsedMilliseconds}ms");

        _output.WriteLine("[2/4] Casting BF16 → F32...");
        sw.Restart();
        Dictionary<string, Tensor> f32Weights = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
            f32Weights[kvp.Key] = kvp.Value.DType == DType.F32 ? kvp.Value : kvp.Value.CastTo(DType.F32);
        _output.WriteLine($"  Cast in {sw.ElapsedMilliseconds}ms");

        _output.WriteLine("[3/4] Building encoder + loading weights...");
        LlamaStyleEncoder encoder = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_4B);
        encoder.LoadWeights(f32Weights);

        // Klein uses chat-formatted input padded to max_length=512.
        const int MaxLen = 64; // shorter for the smoke test — full Klein uses 512
        using Qwen3Tokenizer tok = new Qwen3Tokenizer(Qwen3VocabPath, Qwen3MergesPath, maxLength: MaxLen);
        int[] tokens = tok.EncodeChat("A photograph of an astronaut riding a horse");
        int[][] batch = [tokens];
        _output.WriteLine($"  Chat-formatted token count: {tokens.Length}, first 16 ids: [{string.Join(", ", tokens[..16])}]");

        _output.WriteLine("[4/4] Running GPU multi-layer forward (layers 9, 18, 27)...");
        sw.Restart();
        using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
        // Klein's `text_encoder_out_layers = (9, 18, 27)` in HF convention (k=0 embeddings, k=N
        // post-layer-(N-1)).
        Tensor klein = encoder.EncodeMultiLayer(backend, batch, [9, 18, 27]);
        backend.Sync();
        sw.Stop();
        _output.WriteLine($"  Multi-layer forward done in {sw.ElapsedMilliseconds}ms, shape={klein.Shape}, dtype={klein.DType}");

        Assert.Equal(1, klein.Shape[0]);
        Assert.Equal(MaxLen, klein.Shape[1]);
        Assert.Equal(3 * 2560, klein.Shape[2]); // 7680 — matches Klein's joint_attention_dim

        ReadOnlySpan<float> data = klein.AsReadOnlySpan<float>();
        int nan = 0, inf = 0;
        float min = float.MaxValue, max = float.MinValue;
        double sumAbs = 0;
        for (int i = 0; i < data.Length; i++)
        {
            float v = data[i];
            if (float.IsNaN(v)) nan++;
            else if (float.IsInfinity(v)) inf++;
            else { if (v < min) min = v; if (v > max) max = v; sumAbs += Math.Abs(v); }
        }
        _output.WriteLine($"  Stats: nan={nan}, inf={inf}, min={min:F4}, max={max:F4}, abs_mean={sumAbs / data.Length:F4}");

        Assert.Equal(0, nan);
        Assert.Equal(0, inf);
        Assert.True(sumAbs > 0);

        klein.Dispose();
        encoder.Dispose();
    }

    /// <summary>Same as the CPU test but on the GPU. Faster, lets us run with longer prompts.</summary>
    [Fact]
    public void Qwen3Encoder_GpuForward_ProducesFiniteHiddenStates()
    {
        if (!File.Exists(Qwen3WeightsPath))
        {
            _output.WriteLine($"SKIPPED: Qwen3 weights not found: {Qwen3WeightsPath}");
            return;
        }
        if (!File.Exists(Qwen3VocabPath) || !File.Exists(Qwen3MergesPath))
        {
            _output.WriteLine("SKIPPED: Qwen3 tokenizer files not found");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(Qwen3EncoderSmokeTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[1/4] Loading {Path.GetFileName(Qwen3WeightsPath)}...");
        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(Qwen3WeightsPath);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();
        _output.WriteLine($"  {weights.Count} tensors loaded in {sw.ElapsedMilliseconds}ms");

        _output.WriteLine("[2/4] Casting BF16 → F32...");
        sw.Restart();
        Dictionary<string, Tensor> f32Weights = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
            f32Weights[kvp.Key] = kvp.Value.DType == DType.F32 ? kvp.Value : kvp.Value.CastTo(DType.F32);
        _output.WriteLine($"  Cast in {sw.ElapsedMilliseconds}ms");

        _output.WriteLine("[3/4] Building encoder + loading weights...");
        sw.Restart();
        LlamaStyleEncoder encoder = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_4B);
        encoder.LoadWeights(f32Weights);
        _output.WriteLine($"  Loaded in {sw.ElapsedMilliseconds}ms");

        using Qwen3Tokenizer tok = new Qwen3Tokenizer(Qwen3VocabPath, Qwen3MergesPath, maxLength: 64);
        int[] tokens = tok.Encode("A photograph of an astronaut riding a horse", appendEos: true);
        int[][] batch = [tokens];

        _output.WriteLine("[4/4] Running GPU forward pass...");
        sw.Restart();
        using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
        Tensor hidden = encoder.Encode(backend, batch);
        backend.Sync();
        sw.Stop();
        _output.WriteLine($"  Forward done in {sw.ElapsedMilliseconds}ms, output shape={hidden.Shape}, dtype={hidden.DType}");

        Assert.Equal(1, hidden.Shape[0]);
        Assert.Equal(64, hidden.Shape[1]);
        Assert.Equal(2560, hidden.Shape[2]);

        ReadOnlySpan<float> data = hidden.AsReadOnlySpan<float>();
        int nan = 0, inf = 0;
        float min = float.MaxValue, max = float.MinValue;
        double sumAbs = 0;
        for (int i = 0; i < data.Length; i++)
        {
            float v = data[i];
            if (float.IsNaN(v)) nan++;
            else if (float.IsInfinity(v)) inf++;
            else { if (v < min) min = v; if (v > max) max = v; sumAbs += Math.Abs(v); }
        }
        _output.WriteLine($"  Hidden state stats: nan={nan}, inf={inf}, min={min:F4}, max={max:F4}, abs_mean={sumAbs / data.Length:F4}");

        Assert.Equal(0, nan);
        Assert.Equal(0, inf);
        Assert.True(sumAbs > 0);

        hidden.Dispose();
        encoder.Dispose();
    }
}
