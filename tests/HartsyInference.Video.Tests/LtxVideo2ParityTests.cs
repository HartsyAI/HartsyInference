using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.ModelHandler.SafeTensors;

namespace HartsyInference.Video.Tests;

/// <summary>Numerical parity harness (C# side) for the LTX-2.3 dual-stream DiT. Loads the tiny random weights +
/// inputs produced by <c>Parity/ltx2_transformer_parity_dump.py</c> (diffusers reference), runs
/// <see cref="LtxVideo2Transformer.Forward"/> with the SAME tiny config on the CPU backend, and prints per-stage
/// relL2 (post-proj_in, each block, final velocity) vs the reference .bin dumps. The first stage whose relL2 jumps
/// from ~float-epsilon to O(1) localizes the divergent op.
///
/// <para>Run order: (1) <c>python tests/HartsyInference.Video.Tests/Parity/ltx2_transformer_parity_dump.py
/// $LTX2_PARITY_DIR</c> using the hfvenv, then (2) this test. Skips cleanly if the dump dir is absent.</para></summary>
public unsafe class LtxVideo2ParityTests
{
    private readonly ITestOutputHelper _output;
    public LtxVideo2ParityTests(ITestOutputHelper output) => _output = output;

    // Must match Parity/ltx2_transformer_parity_dump.py exactly.
    private const int TLat = 2, HLat = 3, WLat = 4, Sv = TLat * HLat * WLat;   // 24 video tokens
    private const int Sa = 5, Lv = 7, La = 6;
    private const int InCh = 8, Inner = 16, AudioInner = 8, OutCh = 8, NumLayers = 2;
    private const double Fps = 24.0;
    private const float Timestep = 500.0f;

    private static LtxVideo2Config TinyConfig() => new()
    {
        InChannels = InCh, OutChannels = OutCh,
        NumHeads = 2, HeadDim = 8,                 // InnerDim = 16
        CrossAttentionDim = Inner,
        AudioInChannels = InCh, AudioOutChannels = OutCh,
        AudioNumHeads = 2, AudioHeadDim = 4,       // AudioInnerDim = 8
        AudioCrossAttentionDim = AudioInner,
        NumLayers = NumLayers, FfnMultiplier = 4,
        SelfAttnModParams = 9, CrossAttnMod = true,
        NormEps = 1e-6f,
        QkNormEps = 1e-6f,                         // match diffusers norm_eps (prod uses 1e-5 — see findings)
        RopeTheta = 10000f, RopeBaseNumFrames = 20, RopeBaseHeight = 2048, RopeBaseWidth = 2048,
        CausalOffset = 1,
        AudioSamplingRate = 16000, AudioHopLength = 160, AudioScaleFactor = 4, AudioPosEmbedMaxPos = 20,
        VaeSpatialCompression = 32, VaeTemporalCompression = 8,
        TimestepScaleMultiplier = 1000, CrossAttnTimestepScaleMultiplier = 1000,
    };

    [Fact]
    public void LtxVideo2_Transformer_PerBlockParity_VsDiffusers()
    {
        string dir = Environment.GetEnvironmentVariable("LTX2_PARITY_DIR") ?? "/tmp/ltx2_parity";
        string weights = Path.Combine(dir, "weights.safetensors");
        if (!File.Exists(weights))
        {
            _output.WriteLine($"SKIPPED: parity dumps not found in {dir}. Run Parity/ltx2_transformer_parity_dump.py first.");
            return;
        }

        // Reference dumps.
        Dictionary<string, float[]> refDump = new()
        {
            ["projin"] = ReadBin(Path.Combine(dir, "projin.bin")),
            ["out_velocity"] = ReadBin(Path.Combine(dir, "out_velocity.bin")),
        };
        for (int i = 0; i < NumLayers; i++) refDump[$"block{i}"] = ReadBin(Path.Combine(dir, $"block{i}.bin"));

        // --- Isolated RoPE parity: LtxVideo2Rope.BuildVideo vs the diffusers video self-attn rope. ---
        // This localizes a spatial-periodicity (grid) bug in the rope grid / coordinate normalization independently
        // of the attention. Uses the SAME rope construction args the transformer uses (see LtxVideo2Transformer ctor).
        string ropeCosPath = Path.Combine(dir, "rope_video_cos.bin");
        if (File.Exists(ropeCosPath))
        {
            int[] videoScale = { 8, 32, 32 };
            LtxVideo2Rope rope = HartsyInference.Diffusion.Models.Denoisers.DiTBlocks.LtxVideo2Rope.ForVideoSelf(
                Inner, 10000.0, 20, 2048, 2048, videoScale, causalOffset: 1);
            (Tensor cos, Tensor sin) = rope.BuildVideo(TLat, HLat, WLat, Fps);
            double cosRel = RelL2(HostFloats(cos), ReadBin(ropeCosPath));
            double sinRel = RelL2(HostFloats(sin), ReadBin(Path.Combine(dir, "rope_video_sin.bin")));
            _output.WriteLine($"RoPE video cos relL2 = {cosRel:E4}   sin relL2 = {sinRel:E4}   " +
                $"{(cosRel < 1e-4 && sinRel < 1e-4 ? "OK" : "*** ROPE DIVERGENCE (grid suspect) ***")}");
            cos.Dispose(); sin.Dispose();
        }

        // Weights + config.
        using SafeTensorsLoader loader = new();
        loader.Load(weights);
        LtxVideo2Config cfg = TinyConfig();
        using LtxVideo2Transformer transformer = new(cfg);
        transformer.LoadWeights(loader.GetAllTensors());

        // Inputs (row-major f32, B=1).
        Tensor videoTokens = LoadTensor(Path.Combine(dir, "input_video.bin"), Sv, InCh);
        Tensor audioTokens = LoadTensor(Path.Combine(dir, "input_audio.bin"), Sa, InCh);
        Tensor encVideo = LoadTensor(Path.Combine(dir, "input_enc_video.bin"), Lv, Inner);
        Tensor encAudio = LoadTensor(Path.Combine(dir, "input_enc_audio.bin"), La, AudioInner);

        CpuBackend backend = new();

        // Per-stage relL2 via the OnBlockOutput hook (index -1 = post-proj_in, 0..N-1 = block outputs).
        List<(string Stage, double RelL2)> results = new();
        transformer.OnBlockOutput = (idx, hidden, _) =>
        {
            string stage = idx < 0 ? "projin" : $"block{idx}";
            double rel = RelL2(HostFloats(hidden), refDump[stage]);
            results.Add((stage, rel));
        };

        (Tensor video, Tensor audio) = transformer.Forward(backend, videoTokens, audioTokens, encVideo, encAudio,
            Timestep, (TLat, HLat, WLat), Sa, Fps, null, null);

        double outRel = RelL2(HostFloats(video), refDump["out_velocity"]);

        _output.WriteLine("LTX-2.3 DiT per-stage relL2 (C# vs diffusers, tiny matched config):");
        foreach ((string stage, double rel) in results)
            _output.WriteLine($"  {stage,-8} relL2 = {rel:E4}   {(rel < 1e-3 ? "OK" : "*** DIVERGENCE ***")}");
        _output.WriteLine($"  {"velocity",-8} relL2 = {outRel:E4}   {(outRel < 1e-3 ? "OK" : "*** DIVERGENCE ***")}");

        // First divergent stage (for the report). Threshold 1e-3 tolerates f32/TF32 noise while catching real bugs.
        (string Stage, double RelL2) first = default;
        foreach ((string stage, double rel) in results) if (rel >= 1e-3) { first = (stage, rel); break; }
        if (first.Stage is not null)
            _output.WriteLine($"FIRST DIVERGENCE at stage '{first.Stage}' (relL2 {first.RelL2:E4}) — the bug is in that block's ops.");
        else if (outRel >= 1e-3)
            _output.WriteLine($"Blocks match but final velocity diverges (relL2 {outRel:E4}) — bug is in norm_out/proj_out/OutputLayer.");
        else
            _output.WriteLine("FULL PARITY — transformer code is faithful to diffusers; grid bug is in config values / weight mapping / pipeline, not the DiT code.");

        video.Dispose(); audio.Dispose();
        videoTokens.Dispose(); audioTokens.Dispose(); encVideo.Dispose(); encAudio.Dispose();

        // Assert the worst per-stage divergence so the run fails loudly when the DiT diverges.
        double worst = outRel;
        foreach ((_, double rel) in results) worst = Math.Max(worst, rel);
        Assert.True(worst < 1e-2, $"LTX-2 DiT diverges from diffusers (worst relL2 {worst:E4}); see per-stage log above.");
    }

    private static Tensor LoadTensor(string path, int rows, int cols)
    {
        float[] data = ReadBin(path);
        if (data.Length != (long)rows * cols)
            throw new InvalidDataException($"{Path.GetFileName(path)}: {data.Length} floats != {rows}x{cols}.");
        Tensor t = new(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < data.Length; i++) p[i] = data[i];
        return t;
    }

    private static float[] ReadBin(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        float[] f = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, f, 0, f.Length * 4);
        return f;
    }

    private static float[] HostFloats(Tensor t)
    {
        long n = t.Shape.ElementCount;
        float[] f = new float[n];
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < n; i++) f[i] = p[i];
        return f;
    }

    private static double RelL2(float[] a, float[] b)
    {
        if (a.Length != b.Length) return double.NaN;
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = (double)a[i] - b[i];
            num += d * d;
            den += (double)b[i] * b[i];
        }
        return Math.Sqrt(num / (den + 1e-12));
    }
}
