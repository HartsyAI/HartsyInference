using System.Text.Json;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Layer-by-layer parity harness for the Ideogram 4 transformer. Loads the tiny
/// synthetic checkpoint + fixed inputs written by <c>dump_ideogram4_full_forward.py</c>,
/// runs <see cref="Ideogram4Transformer"/> on the CPU backend with <c>IDEOGRAM4_DEBUG_DIR</c>
/// set, then leaves the dump for <c>diff_ideogram4_layers.py</c> to compare against the
/// Python reference taps. Self-contained: no real 9.3B checkpoint and no GPU needed — this
/// validates the transformer MATH (MRoPE interleave, masked add, scale-only/tanh-gated
/// sandwich blocks, the final layer). Skips cleanly when the Python dump is absent.
///
/// Run order: <c>python3 tests/python-reference/dump_ideogram4_full_forward.py</c> first,
/// then this test, then <c>python3 tests/python-reference/diff_ideogram4_layers.py</c>.</summary>
[Trait("Category", "Integration")]
public unsafe class Ideogram4DiffTests
{
    private readonly ITestOutputHelper _output;
    public Ideogram4DiffTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Transformer_Matches_PythonReference_LayerByLayer_Cpu()
    {
        string repoRoot = RepoRoot.Path;
        string refRoot = Path.Combine(repoRoot, "tests/python-reference/ideogram4_reference_tensors");
        string fwdDir = Path.Combine(refRoot, "full_forward");
        string inputsDir = Path.Combine(fwdDir, "inputs");
        string ckpt = Path.Combine(refRoot, "transformer.safetensors");

        if (!Directory.Exists(inputsDir) || !File.Exists(ckpt))
        {
            _output.WriteLine($"SKIPPED: Python reference not found at {refRoot}");
            _output.WriteLine("  Run: python3 tests/python-reference/dump_ideogram4_full_forward.py");
            return;
        }

        // ── Read meta and build the matching tiny config ──
        using JsonDocument meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(inputsDir, "meta.json")));
        JsonElement m = meta.RootElement;
        int emb = m.GetProperty("emb").GetInt32();
        int layers = m.GetProperty("layers").GetInt32();
        int heads = m.GetProperty("heads").GetInt32();
        int ffn = m.GetProperty("ffn").GetInt32();
        int adaln = m.GetProperty("adaln").GetInt32();
        int inCh = m.GetProperty("in_ch").GetInt32();
        int llmDim = m.GetProperty("llm_dim").GetInt32();
        int theta = m.GetProperty("theta").GetInt32();
        int[] section = [.. m.GetProperty("section").EnumerateArray().Select(e => e.GetInt32())];
        float blockEps = m.GetProperty("block_eps").GetSingle();
        int L = m.GetProperty("L").GetInt32();
        float timestep = m.GetProperty("timestep").GetSingle();

        Ideogram4Config config = Ideogram4Config.V4 with
        {
            EmbDim = emb,
            NumLayers = layers,
            NumHeads = heads,
            IntermediateSize = ffn,
            AdalnDim = adaln,
            InChannels = inCh,
            LlmFeaturesDim = llmDim,
            RopeTheta = theta,
            MropeSection = (section[0], section[1], section[2]),
            NormEps = blockEps,
        };
        _output.WriteLine($"Config: emb={emb} layers={layers} heads={heads} head_dim={config.HeadDim} L={L} t={timestep}");

        // ── Dump dir + env (set BEFORE any Forward so Ideogram4DebugDump resolves it) ──
        string csDumpDir = Path.Combine(repoRoot, "Output", "ideogram4_csharp_dump");
        if (Directory.Exists(csDumpDir)) Directory.Delete(csDumpDir, recursive: true);
        Directory.CreateDirectory(Path.Combine(csDumpDir, "layers"));
        Environment.SetEnvironmentVariable("IDEOGRAM4_DEBUG_DIR", csDumpDir);

        // ── Load synthetic weights ──
        using SafeTensorsLoader loader = new();
        loader.Load(ckpt);
        Dictionary<string, Tensor> weights = new();
        foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
            weights[kvp.Key] = kvp.Value;

        using Ideogram4Transformer transformer = new(config);
        transformer.LoadWeights(weights);

        // ── Load fixed inputs ──
        Tensor x = LoadF32(Path.Combine(inputsDir, "x.bin"), 1, L, inCh);
        Tensor llm = LoadF32(Path.Combine(inputsDir, "llm_features.bin"), 1, L, llmDim);
        Tensor posIds = LoadF32(Path.Combine(inputsDir, "position_ids.bin"), 1, L, 3);
        int[] indicator = LoadI32(Path.Combine(inputsDir, "indicator.bin"), L);

        // ── Standalone MRoPE dump (so RoPE bugs localize without the whole forward) ──
        Ideogram4Mrope rope = new(config.HeadDim, config.RopeTheta, config.MropeSection);
        (Tensor cos, Tensor sin) = rope.BuildCosSin(posIds);
        WriteRawF32(Path.Combine(csDumpDir, "layers", "mrope_cos.bin"), cos);
        WriteRawF32(Path.Combine(csDumpDir, "layers", "mrope_sin.bin"), sin);
        cos.Dispose();
        sin.Dispose();

        // ── Forward (dumps input_proj, hidden_in, adaln_input, layers.i, output_velocity) ──
        // The Forward API takes text rows + image tokens separately (the reference files are full-sequence);
        // output_velocity is image rows only — diff_ideogram4_layers.py compares against the reference tail.
        int numText = 0;
        while (numText < L && indicator[numText] == 3) numText++;
        Tensor llmText = SliceRowsHost(llm, 0, numText, llmDim);
        Tensor imgTokens = SliceRowsHost(x, numText, L - numText, inCh);
        using CpuBackend backend = new();
        _output.WriteLine($"Running Ideogram4Transformer.Forward (CPU, numText={numText})...");
        Tensor velocity = transformer.Forward(backend, numText > 0 ? llmText : null, imgTokens, timestep, posIds, indicator, null);

        float* vptr = (float*)velocity.DataPointer;
        int n = (int)velocity.ElementCount;
        double sum = 0, sumAbs = 0, sumSq = 0;
        for (int i = 0; i < n; i++) { sum += vptr[i]; sumAbs += Math.Abs(vptr[i]); sumSq += vptr[i] * vptr[i]; }
        double mean = sum / n;
        _output.WriteLine($"output_velocity: shape={velocity.Shape}, mean={mean:F6}, abs_mean={sumAbs / n:F6}, std={Math.Sqrt(sumSq / n - mean * mean):F6}");

        velocity.Dispose();
        llmText.Dispose();
        imgTokens.Dispose();
        x.Dispose();
        llm.Dispose();
        posIds.Dispose();
        Environment.SetEnvironmentVariable("IDEOGRAM4_DEBUG_DIR", null);

        // Sanity: the forward dumped the expected taps.
        Assert.True(File.Exists(Path.Combine(csDumpDir, "output_velocity.bin")), "output_velocity not dumped — IDEOGRAM4_DEBUG_DIR may have been cached before this test set it.");
        Assert.True(File.Exists(Path.Combine(csDumpDir, "layers", $"layers_{layers - 1}.bin")), "last block not dumped");

        _output.WriteLine($"\nC# dump written to: {csDumpDir}");
        _output.WriteLine("Now run: python3 tests/python-reference/diff_ideogram4_layers.py");
    }

    private static Tensor SliceRowsHost(Tensor src, int startRow, int rows, int dim)
    {
        Tensor outT = new Tensor(new TensorShape(1, rows, dim), DType.F32);
        long bytes = (long)rows * dim * sizeof(float);
        Buffer.MemoryCopy((float*)src.DataPointer + (long)startRow * dim, (void*)outT.DataPointer, bytes, bytes);
        return outT;
    }

    private static Tensor LoadF32(string path, params long[] dims)
    {
        TensorShape shape = dims.Length switch
        {
            2 => new TensorShape(dims[0], dims[1]),
            3 => new TensorShape(dims[0], dims[1], dims[2]),
            _ => throw new ArgumentException($"rank {dims.Length}"),
        };
        Tensor t = new(shape, DType.F32);
        byte[] raw = File.ReadAllBytes(path);
        long expected = shape.ElementCount * sizeof(float);
        if (raw.LongLength != expected)
            throw new InvalidDataException($"{path}: expected {expected} bytes, got {raw.LongLength}");
        fixed (byte* src = raw) Buffer.MemoryCopy(src, (void*)t.DataPointer, raw.LongLength, raw.LongLength);
        return t;
    }

    private static int[] LoadI32(string path, int count)
    {
        byte[] raw = File.ReadAllBytes(path);
        if (raw.LongLength != (long)count * sizeof(int))
            throw new InvalidDataException($"{path}: expected {count * sizeof(int)} bytes, got {raw.LongLength}");
        int[] result = new int[count];
        fixed (byte* src = raw) fixed (int* dst = result)
            Buffer.MemoryCopy(src, dst, raw.LongLength, raw.LongLength);
        return result;
    }

    private static void WriteRawF32(string path, Tensor t)
    {
        long bytes = t.ElementCount * sizeof(float);
        byte[] buffer = new byte[bytes];
        fixed (byte* dst = buffer) Buffer.MemoryCopy((void*)t.DataPointer, dst, bytes, bytes);
        File.WriteAllBytes(path, buffer);
    }
}
