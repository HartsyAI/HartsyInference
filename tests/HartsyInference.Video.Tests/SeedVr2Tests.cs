using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Video.Pipelines;
using HartsyInference.Vision.Codec;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Video.Tests;

/// <summary>The SeedVR2 verification chain, staged so a failure bisects to its stage (recorded gate
/// results: PARITY_VERIFICATION.md). Reference dumps for the env-gated facts come from
/// <c>tests/python-reference/seedvr2_reference/</c> and <c>Parity/seedvr2_transformer_parity_dump.py</c>.</summary>
public sealed class SeedVr2Tests
{
    private readonly ITestOutputHelper _output;

    public SeedVr2Tests(ITestOutputHelper output)
    {
        _output = output;
    }

    #region Window partition (Unit tier — committed fixture, no weights)

    /// <summary>Exact-equality vs slices emitted by ByteDance's <c>models/dit_v2/window.py</c>; regenerate
    /// the fixture with <c>fixtures/seedvr2_window_fixture_dump.py</c>.</summary>
    [Fact]
    public void WindowSlices_MatchReference_AllFixtureCases()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "fixtures", "seedvr2_windows.json");
        Assert.True(File.Exists(path), $"Fixture missing: {path}");

        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        int cases = 0, slicesChecked = 0;
        foreach (JsonElement el in doc.RootElement.EnumerateArray())
        {
            int[] size = el.GetProperty("size").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            int[] num = el.GetProperty("num").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            string method = el.GetProperty("method").GetString()!;
            List<int[]> expected = el.GetProperty("slices").EnumerateArray()
                .Select(s => s.EnumerateArray().Select(v => v.GetInt32()).ToArray()).ToList();

            SeedVr2WindowSlice[] actual = method == "720pwin_by_size_bysize"
                ? SeedVr2Windowing.MakeWindows(size[0], size[1], size[2], num[0], num[1], num[2])
                : SeedVr2Windowing.MakeShiftedWindows(size[0], size[1], size[2], num[0], num[1], num[2]);

            Assert.True(expected.Count == actual.Length,
                $"{method} ({size[0]},{size[1]},{size[2]}): expected {expected.Count} windows, got {actual.Length}");
            for (int i = 0; i < actual.Length; i++)
            {
                int[] e = expected[i];
                SeedVr2WindowSlice a = actual[i];
                Assert.True(
                    e[0] == a.T0 && e[1] == a.T1 && e[2] == a.H0 && e[3] == a.H1 && e[4] == a.W0 && e[5] == a.W1,
                    $"{method} ({size[0]},{size[1]},{size[2]}) window {i}: expected " +
                    $"[{e[0]},{e[1]},{e[2]},{e[3]},{e[4]},{e[5]}], got " +
                    $"[{a.T0},{a.T1},{a.H0},{a.H1},{a.W0},{a.W1}]");
            }
            cases++;
            slicesChecked += actual.Length;
        }
        _output.WriteLine($"Verified {cases} cases, {slicesChecked} slices, exact match.");
        Assert.True(cases >= 40, "Fixture unexpectedly small — regeneration may have truncated it.");
    }

    /// <summary>Independent structural property (not fixture-derived): the unshifted partition covers every
    /// token exactly once.</summary>
    [Fact]
    public void RegularWindows_TileGridExactly()
    {
        foreach ((int t, int h, int w) in new[] { (7, 45, 80), (13, 60, 106), (1, 1, 1), (31, 44, 79) })
        {
            SeedVr2WindowSlice[] slices = SeedVr2Windowing.MakeWindows(t, h, w, 4, 3, 3);
            bool[] covered = new bool[t * h * w];
            foreach (SeedVr2WindowSlice s in slices)
            {
                for (int it = s.T0; it < s.T1; it++)
                for (int ih = s.H0; ih < s.H1; ih++)
                for (int iw = s.W0; iw < s.W1; iw++)
                {
                    int idx = (it * h + ih) * w + iw;
                    Assert.False(covered[idx], $"Token ({it},{ih},{iw}) covered twice in ({t},{h},{w})");
                    covered[idx] = true;
                }
            }
            Assert.DoesNotContain(false, covered);
        }
    }

    #endregion

    #region Preprocessing (cut_videos Unit; resize parity Integration)

    /// <summary>Causal-VAE frame contract: 1 for stills, else next (T−1) % 4 == 0.</summary>
    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 5)]
    [InlineData(4, 5)]
    [InlineData(5, 5)]
    [InlineData(6, 9)]
    [InlineData(7, 9)]
    [InlineData(9, 9)]
    [InlineData(24, 25)]
    [InlineData(25, 25)]
    public void PaddedFrameCount_MatchesCutVideos(int frames, int expected)
        => Assert.Equal(expected, SeedVr2Preprocess.PaddedFrameCount(frames));

    /// <summary>Preprocess chain vs ByteDance's own transforms; the antialiased bicubic is
    /// float-tolerance, the rest exact.</summary>
    [Fact]
    [Trait("Category", "Integration")]
    public void Preprocess_MatchesReference_AllCases()
    {
        string? refPath = Env("SEEDVR2_PRE_REF");
        if (refPath is null || !File.Exists(refPath))
        {
            _output.WriteLine("SKIPPED: set SEEDVR2_PRE_REF to the preprocess reference dump.");
            return;
        }
        string[] caseNames =
            ["up_360p_t5", "up_240p_t1", "up_odd_t7", "down_1080p_t5", "tiny_t3", "nondiv_t9", "exact_720p_t5"];

        using SafeTensorsLoader loader = new();
        loader.Load(refPath);

        double worst = 0;
        foreach (string name in caseNames)
        {
            Tensor input = loader.GetTensor($"{name}.input");     // (T,3,H,W) u8
            Tensor expected = loader.GetTensor($"{name}.output"); // (3,T',H',W') f32
            int t = (int)input.Shape[0], h = (int)input.Shape[2], w = (int)input.Shape[3];

            List<byte[]> frames = new List<byte[]>(t);
            ReadOnlySpan<byte> u8 = input.AsSpan<byte>();
            for (int f = 0; f < t; f++)
            {
                byte[] rgb = new byte[h * w * 3];
                for (int c = 0; c < 3; c++)
                {
                    int plane = (f * 3 + c) * h * w;
                    for (int i = 0; i < h * w; i++)
                        rgb[i * 3 + c] = u8[plane + i];
                }
                frames.Add(rgb);
            }

            SeedVr2Preprocess.Result actual = SeedVr2Preprocess.Run(frames, w, h, 1280L * 720L);

            Assert.True(expected.Shape[1] == actual.Frames && expected.Shape[2] == actual.Height
                && expected.Shape[3] == actual.Width,
                $"{name}: shape (3,{actual.Frames},{actual.Height},{actual.Width}) vs reference " +
                $"(3,{expected.Shape[1]},{expected.Shape[2]},{expected.Shape[3]})");

            ReadOnlySpan<float> exp = expected.AsSpan<float>();
            double maxAbs = 0;
            for (int i = 0; i < exp.Length; i++)
            {
                double d = Math.Abs(exp[i] - actual.Data[i]);
                if (d > maxAbs)
                    maxAbs = d;
            }
            _output.WriteLine($"{name}: out (3,{actual.Frames},{actual.Height},{actual.Width}) maxAbs {maxAbs:e2}");
            worst = Math.Max(worst, maxAbs);
            Assert.True(maxAbs <= 1e-5, $"{name}: maxAbs {maxAbs:e2} exceeds 1e-5");
        }
        _output.WriteLine($"All {caseNames.Length} cases within tolerance; worst maxAbs {worst:e2}.");
    }

    #endregion

    #region Checkpoint conversion (negatives Unit; real-weight inventory Integration)

    /// <summary>Structural negatives on synthetic dictionaries — no weights needed.</summary>
    [Fact]
    public void Converter_RejectsMissingAndUnknownKeys()
    {
        Dictionary<string, Tensor> synthetic = new();
        Assert.ThrowsAny<Exception>(() => SeedVr2CheckpointConverter.Convert(synthetic));
    }

    /// <summary>Real 3B checkpoint: full inventory consumed and <see cref="SeedVr2Config.Detect"/>
    /// reproduces the published dims, including the mm_layers=10 separate/shared boundary.</summary>
    [Fact]
    [Trait("Category", "Integration")]
    public void RealCheckpoint_ConvertsAndDetects3BConfig()
    {
        string? path = Env("SEEDVR2_DIT");
        if (path is null || !File.Exists(path))
        {
            _output.WriteLine("SKIPPED: set SEEDVR2_DIT to the converted DiT safetensors.");
            return;
        }

        (Dictionary<string, Tensor> weights, SafeTensorsLoader loader) =
            SeedVr2CheckpointConverter.LoadAndConvert(path);
        using SafeTensorsLoader _ = loader;

        // 635 checkpoint tensors − 32 recomputable per-block RoPE freq buffers = 603 consumed.
        Assert.Equal(603, weights.Count);

        SeedVr2Config detected = SeedVr2Config.Detect(weights);
        Assert.Equal(SeedVr2Config.Seedvr2_3B, detected);
        _output.WriteLine($"Detected: dim={detected.VidDim} heads={detected.Heads} layers={detected.NumLayers} " +
            $"mm={detected.MmLayers} in={detected.InChannels} out={detected.OutChannels} mlp={detected.MlpDim}");

        Assert.True(weights.ContainsKey("blocks.9.attn.proj_qkv.vid.weight"));
        Assert.True(weights.ContainsKey("blocks.9.attn.proj_qkv.txt.weight"));
        Assert.False(weights.ContainsKey("blocks.9.attn.proj_qkv.all.weight"));
        Assert.True(weights.ContainsKey("blocks.10.attn.proj_qkv.all.weight"));
        Assert.False(weights.ContainsKey("blocks.10.attn.proj_qkv.vid.weight"));
    }

    #endregion

    #region VAE parity (Integration — real weights vs Python dump)

    /// <summary>Encoder mean/logvar + decoder RGB vs the real-weight Python reference (basic_forward
    /// path, no slicing).</summary>
    [Fact]
    [Trait("Category", "Integration")]
    public void Vae_EncoderAndDecoder_MatchRealWeightReference()
    {
        string? vaePath = Env("SEEDVR2_VAE");
        string? refPath = Env("SEEDVR2_VAE_REF");
        if (vaePath is null || refPath is null || !File.Exists(vaePath) || !File.Exists(refPath))
        {
            _output.WriteLine("SKIPPED: set SEEDVR2_VAE and SEEDVR2_VAE_REF.");
            return;
        }

        using SafeTensorsLoader weightsLoader = new();
        weightsLoader.Load(vaePath);
        Dictionary<string, Tensor> weights = weightsLoader.GetAllTensors();
        using SafeTensorsLoader refLoader = new();
        refLoader.Load(refPath);

        IBackend backend = new CpuBackend();
        SeedVr2VaeConfig config = SeedVr2VaeConfig.Default;

        SeedVr2VaeEncoder encoder = new(config);
        encoder.LoadWeights(weights);
        (Tensor mean, Tensor logvar) = encoder.Encode(backend, refLoader.GetTensor("enc.input"));
        backend.Sync();
        double meanRel = RelL2(mean, refLoader.GetTensor("enc.mean"));
        double logvarRel = RelL2(logvar, refLoader.GetTensor("enc.logvar"));
        _output.WriteLine($"encoder: mean relL2 {meanRel:e2}, logvar relL2 {logvarRel:e2}");

        SeedVr2VaeDecoder decoder = new(config);
        decoder.LoadWeights(weights);
        Tensor decoded = decoder.Decode(backend, refLoader.GetTensor("dec.input"));
        backend.Sync();
        double decRel = RelL2(decoded, refLoader.GetTensor("dec.output"));
        _output.WriteLine($"decoder: output relL2 {decRel:e2}");

        Assert.True(meanRel < 1e-3, $"encoder mean relL2 {meanRel:e2} exceeds 1e-3");
        Assert.True(logvarRel < 1e-3, $"encoder logvar relL2 {logvarRel:e2} exceeds 1e-3");
        Assert.True(decRel < 1e-3, $"decoder relL2 {decRel:e2} exceeds 1e-3");
    }

    #endregion

    #region VAE BF16 activations (GpuIntegration — bf16 memory mode vs f32, staged conv → encoder → decoder)

    /// <summary>BF16 VAE activation mode (the 720p+ memory path) vs the F32 reference path on CUDA, staged
    /// so a failure localizes: bare CausalConv3d first, then full encoder, then full decoder.</summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void Vae_Bf16Activations_MatchF32_OnCuda()
    {
        string? vaePath = Env("SEEDVR2_VAE");
        if (vaePath is null || !File.Exists(vaePath))
        {
            _output.WriteLine("SKIPPED: set SEEDVR2_VAE.");
            return;
        }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(SeedVr2Tests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine("SKIPPED: no Ptx dir (CUDA backend unavailable).");
            return;
        }
        IBackend backend = new HartsyInference.Cuda.CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);

        using SafeTensorsLoader weightsLoader = new();
        weightsLoader.Load(vaePath);
        Dictionary<string, Tensor> weights = weightsLoader.GetAllTensors();

        // Stage 1: bare conv (encoder conv_in, k3 causal, 3→128) on a seeded clip.
        Tensor clip = SeededClip(1, 3, 5, 32, 48, seed: 42);
        CausalConv3d convF32 = SeedVr2VaeOps.Conv(weights, "encoder.conv_in");
        CausalConv3d convBf16 = SeedVr2VaeOps.Conv(weights, "encoder.conv_in", computeDtype: DType.BF16);
        Tensor outF32 = convF32.Forward(backend, clip);
        backend.Sync();
        Tensor clipBf = clip.CastTo(DType.BF16);
        Tensor outBfRaw = convBf16.Forward(backend, clipBf);
        backend.Sync();
        Tensor outBf = outBfRaw.CastTo(DType.F32);
        double convRel = RelL2(outBf, outF32);
        _output.WriteLine($"conv_in bf16-vs-f32 relL2 {convRel:e2}");
        Assert.True(convRel < 2e-2, $"CausalConv3d bf16 relL2 {convRel:e2} exceeds 2e-2");
        outF32.Dispose(); outBfRaw.Dispose(); outBf.Dispose(); clipBf.Dispose();

        // Stage 2: full encoder.
        SeedVr2VaeEncoder encF32 = new(SeedVr2VaeConfig.Default);
        encF32.LoadWeights(weights);
        SeedVr2VaeEncoder encBf16 = new(SeedVr2VaeConfig.Default with { ActivationDType = DType.BF16 });
        encBf16.LoadWeights(weights);
        (Tensor meanF, Tensor logvarF) = encF32.Encode(backend, clip);
        backend.Sync();
        (Tensor meanB, Tensor logvarB) = encBf16.Encode(backend, clip);
        backend.Sync();
        double meanRel = RelL2(meanB, meanF), logvarRel = RelL2(logvarB, logvarF);
        _output.WriteLine($"encoder bf16-vs-f32: mean relL2 {meanRel:e2}, logvar relL2 {logvarRel:e2}");
        Assert.True(meanRel < 3e-2, $"encoder mean bf16 relL2 {meanRel:e2} exceeds 3e-2");
        meanF.Dispose(); logvarF.Dispose(); meanB.Dispose(); logvarB.Dispose(); clip.Dispose();

        // Stage 2b: encoder under the pipeline's weight-residency cycle (Preload → encode → Sync → Free →
        // TrimMemoryPool) — the phase staging is the only structural difference from the bare call above.
        backend.PreloadWeights(encBf16.EnumerateWeights());
        Tensor clip2 = SeededClip(1, 3, 5, 32, 48, seed: 42);
        (Tensor meanP, Tensor logvarP) = encBf16.Encode(backend, clip2);
        backend.Sync();
        backend.FreeWeights(encBf16.EnumerateWeights());
        backend.TrimMemoryPool();
        (Tensor meanF2, Tensor logvarF2) = encF32.Encode(backend, clip2);
        backend.Sync();
        double meanRelP = RelL2(meanP, meanF2);
        _output.WriteLine($"encoder (preload cycle) bf16-vs-f32: mean relL2 {meanRelP:e2}");
        Assert.True(meanRelP < 3e-2, $"encoder preload-cycle bf16 relL2 {meanRelP:e2} exceeds 3e-2");
        meanP.Dispose(); logvarP.Dispose(); meanF2.Dispose(); logvarF2.Dispose(); clip2.Dispose();

        // Stage 3: full decoder on a seeded latent.
        Tensor latent = SeededClip(1, 16, 2, 6, 8, seed: 7);
        SeedVr2VaeDecoder decF32 = new(SeedVr2VaeConfig.Default);
        decF32.LoadWeights(weights);
        SeedVr2VaeDecoder decBf16 = new(SeedVr2VaeConfig.Default with { ActivationDType = DType.BF16 });
        decBf16.LoadWeights(weights);
        Tensor decF = decF32.Decode(backend, latent);
        backend.Sync();
        Tensor decB = decBf16.Decode(backend, latent);
        backend.Sync();
        double decRel = RelL2(decB, decF);
        _output.WriteLine($"decoder bf16-vs-f32: relL2 {decRel:e2}");
        Assert.True(decRel < 5e-2, $"decoder bf16 relL2 {decRel:e2} exceeds 5e-2");
        decF.Dispose(); decB.Dispose(); latent.Dispose();
    }

    private static Tensor SeededClip(int b, int c, int t, int h, int w, int seed)
    {
        Tensor clip = new Tensor(new TensorShape([(long)b, c, t, h, w]), DType.F32);
        Span<float> s = clip.AsSpan<float>();
        Random rng = new(seed);
        for (int i = 0; i < s.Length; i++)
            s[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return clip;
    }

    #endregion

    #region DiT parity (Integration — tiny seeded config, per-block bisection)

    // Must match Parity/seedvr2_transformer_parity_dump.py exactly.
    private const int TinyVidDim = 128, TinyTxtInDim = 32;
    private const int TinyT = 5, TinyH = 90, TinyW = 160, TinyTxtLen = 7;
    private const float TinyTimestep = 937.0f;

    private static SeedVr2Config TinyConfig => new()
    {
        VidDim = TinyVidDim, TxtInDim = TinyTxtInDim, EmbDim = 6 * TinyVidDim, Heads = 1, HeadDim = 128,
        MlpDim = 512, NumLayers = 4, MmLayers = 2, InChannels = 33, OutChannels = 16,
    };

    /// <summary>SeedVr2Dit vs ByteDance's NaDiT at a tiny seeded-random config, with per-block relL2 and
    /// first-divergence reporting.</summary>
    [Fact]
    [Trait("Category", "Integration")]
    public void Dit_TinyConfig_ForwardMatchesReference_PerBlock()
    {
        string dir = Env("SEEDVR2_PARITY_DIR") ?? "/tmp/seedvr2_parity";
        if (!File.Exists(Path.Combine(dir, "weights.safetensors")))
        {
            _output.WriteLine($"SKIPPED: no dump at {dir} — run seedvr2_transformer_parity_dump.py first.");
            return;
        }

        using SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(dir, "weights.safetensors"));
        Dictionary<string, Tensor> weights = new(
            loader.GetAllTensors().Where(kv => !kv.Key.EndsWith(".rope.rope.freqs", StringComparison.Ordinal)));

        SeedVr2Dit dit = new(TinyConfig);
        dit.LoadWeights(weights);
        IBackend backend = new CpuBackend();

        Tensor latent = LoadBin(Path.Combine(dir, "input_latent.bin"), [TinyT, TinyH, TinyW, 33]);
        Tensor txt = LoadBin(Path.Combine(dir, "input_txt.bin"), [TinyTxtLen, TinyTxtInDim]);

        string? firstDivergence = null;
        dit.OnBlockOutput = (idx, vid, txtTok) =>
        {
            if (idx < 0)
                return;
            double vidRel = RelL2(vid, Path.Combine(dir, $"block{idx}_vid.bin"));
            double txtRel = RelL2(txtTok, Path.Combine(dir, $"block{idx}_txt.bin"));
            _output.WriteLine($"block{idx}: vid relL2 {vidRel:e2}, txt relL2 {txtRel:e2}");
            if (firstDivergence is null && (vidRel > 1e-3 || txtRel > 1e-3))
                firstDivergence = $"block{idx}";
        };

        Tensor output = dit.Forward(backend, latent, txt, TinyTimestep);
        backend.Sync();
        double outRel = RelL2(output, Path.Combine(dir, "output.bin"));
        _output.WriteLine($"output: relL2 {outRel:e2}");
        if (firstDivergence is not null)
            _output.WriteLine($"FIRST DIVERGENCE at '{firstDivergence}' — the bug is in/before that stage.");
        Assert.True(outRel < 1e-3, $"final output relL2 {outRel:e2} exceeds 1e-3" +
            (firstDivergence is null ? "" : $" (first divergence: {firstDivergence})"));
    }

    private static SeedVr2Config TinyV1Config => new()
    {
        VidDim = TinyVidDim, TxtInDim = TinyTxtInDim, EmbDim = 6 * TinyVidDim, Heads = 1, HeadDim = 128,
        MlpDim = 512, NumLayers = 4, MmLayers = 4, InChannels = 33, OutChannels = 16,
        SwiGluMlp = false, HasTailNorm = false, PixelRope = true, LastLayerVidOnly = false,
    };

    /// <summary>v1 NaDiT (7B architecture, models/dit) vs ByteDance's reference at a tiny seeded config —
    /// pixel rope, plain GELU-tanh MLP, full-split blocks, no tail norm, no last-layer text shortcut.</summary>
    [Fact]
    [Trait("Category", "Integration")]
    public void Dit_TinyConfigV1_ForwardMatchesReference_PerBlock()
    {
        string dir = Env("SEEDVR2_PARITY_V1_DIR") ?? "/tmp/seedvr2_parity_v1";
        if (!File.Exists(Path.Combine(dir, "weights.safetensors")))
        {
            _output.WriteLine($"SKIPPED: no dump at {dir} — run seedvr2_transformer_v1_parity_dump.py first.");
            return;
        }

        using SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(dir, "weights.safetensors"));
        Dictionary<string, Tensor> weights = new(
            loader.GetAllTensors().Where(kv => !kv.Key.EndsWith(".rope.rope.freqs", StringComparison.Ordinal)));

        SeedVr2Config detected = SeedVr2Config.Detect(weights);
        Assert.True(detected.PixelRope, "Detect should flag the v1 (plain-MLP + no-tail) signature as PixelRope.");
        Assert.False(detected.LastLayerVidOnly, "v1 must not apply the v2 last-layer text shortcut.");

        SeedVr2Dit dit = new(TinyV1Config);
        dit.LoadWeights(weights);
        IBackend backend = new CpuBackend();

        Tensor latent = LoadBin(Path.Combine(dir, "input_latent.bin"), [TinyT, TinyH, TinyW, 33]);
        Tensor txt = LoadBin(Path.Combine(dir, "input_txt.bin"), [TinyTxtLen, TinyTxtInDim]);

        string? firstDivergence = null;
        dit.OnBlockOutput = (idx, vid, txtTok) =>
        {
            if (idx < 0)
                return;
            double vidRel = RelL2(vid, Path.Combine(dir, $"block{idx}_vid.bin"));
            double txtRel = RelL2(txtTok, Path.Combine(dir, $"block{idx}_txt.bin"));
            _output.WriteLine($"block{idx}: vid relL2 {vidRel:e2}, txt relL2 {txtRel:e2}");
            if (firstDivergence is null && (vidRel > 1e-3 || txtRel > 1e-3))
                firstDivergence = $"block{idx}";
        };

        Tensor output = dit.Forward(backend, latent, txt, TinyTimestep);
        backend.Sync();
        double outRel = RelL2(output, Path.Combine(dir, "output.bin"));
        _output.WriteLine($"output: relL2 {outRel:e2}");
        if (firstDivergence is not null)
            _output.WriteLine($"FIRST DIVERGENCE at '{firstDivergence}' — the bug is in/before that stage.");
        Assert.True(outRel < 1e-3, $"final output relL2 {outRel:e2} exceeds 1e-3" +
            (firstDivergence is null ? "" : $" (first divergence: {firstDivergence})"));
    }

    #endregion

    #region Full-pipeline E2E (GpuIntegration — real weights, reference noises injected)

    /// <summary>Full C# restore vs the real-weight Python reference, reference noises injected via
    /// <see cref="SeedVr2RestorePipeline.NoiseHook"/> (torch RNG is unmatchable).</summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void E2e_BigBuckBunny_RestoreMatchesPythonReference()
    {
        string? dit = Env("SEEDVR2_DIT");
        string? vae = Env("SEEDVR2_VAE");
        string? emb = Env("SEEDVR2_EMB");
        string? refPath = Env("SEEDVR2_E2E_REF");
        string? framesDir = Env("SEEDVR2_FRAMES");
        if (dit is null || vae is null || emb is null || refPath is null || framesDir is null)
        {
            _output.WriteLine("SKIPPED: set SEEDVR2_DIT/VAE/EMB/E2E_REF/FRAMES.");
            return;
        }

        (Dictionary<string, Tensor> ditWeights, SafeTensorsLoader ditLoader) =
            SeedVr2CheckpointConverter.LoadAndConvert(dit);
        using SafeTensorsLoader _ = ditLoader;
        using SafeTensorsLoader vaeLoader = new();
        vaeLoader.Load(vae);
        Dictionary<string, Tensor> vaeWeights = vaeLoader.GetAllTensors();
        using SafeTensorsLoader embLoader = new();
        embLoader.Load(emb);
        Tensor posEmb = embLoader.GetTensor("pos_emb").CastTo(DType.F32);

        using SafeTensorsLoader refLoader = new();
        refLoader.Load(refPath);
        Tensor posteriorNoise = refLoader.GetTensor("posterior_noise");
        Tensor initNoise = refLoader.GetTensor("init_noise");
        Tensor expected = refLoader.GetTensor("output");   // (3,F,H,W) [-1,1]

        string backendSel = Env("SEEDVR2_E2E_BACKEND") ?? "cuda";
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(SeedVr2Tests).Assembly.Location)!, "Ptx");
        IBackend backend = backendSel == "cpu" || !Directory.Exists(ptxDir)
            ? new CpuBackend()
            : new HartsyInference.Cuda.CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);

        SeedVr2Config config = SeedVr2Config.Detect(ditWeights);
        SeedVr2Dit model = new(config);
        model.LoadWeights(ditWeights);
        SeedVr2VaeEncoder encoder = new(SeedVr2VaeConfig.Default);
        encoder.LoadWeights(vaeWeights);
        SeedVr2VaeDecoder decoder = new(SeedVr2VaeConfig.Default);
        decoder.LoadWeights(vaeWeights);

        List<byte[]> frames = new();
        int w = 0, h = 0;
        foreach (string f in Directory.GetFiles(framesDir, "*.png").OrderBy(x => x))
        {
            (byte[] rgb, int fw, int fh) = PngDecoder.DecodeFromFile(f);
            frames.Add(rgb);
            (w, h) = (fw, fh);
        }
        _output.WriteLine($"input: {frames.Count} frames {w}x{h}, backend {backendSel}");

        using SeedVr2RestorePipeline pipeline = new(backend, model, encoder, decoder, posEmb);
        pipeline.NoiseHook = (kind, chunk, shape) =>
        {
            Tensor src = kind == "posterior" ? posteriorNoise : initNoise;
            Assert.Equal(shape.ElementCount, src.Shape.ElementCount);
            Tensor copy = new Tensor(shape, DType.F32);
            src.AsSpan<float>().CopyTo(copy.AsSpan<float>());
            return copy;
        };

        // f32 whole-clip VAE at 720p-area exceeds 24 GB; the parity gate runs reduced.
        long area = long.TryParse(Env("SEEDVR2_AREA"), out long a) ? a : 1280L * 720L;
        long t0 = Environment.TickCount64;
        (List<byte[]> restored, int outW, int outH) = pipeline.Restore(
            frames, w, h, new SeedVr2RestoreOptions { ClipFrames = frames.Count, TargetArea = area });
        _output.WriteLine($"restored {restored.Count} frames {outW}x{outH} in {Environment.TickCount64 - t0} ms");

        int refF = (int)expected.Shape[1], refH = (int)expected.Shape[2], refW = (int)expected.Shape[3];
        Assert.Equal(refF, restored.Count);
        Assert.Equal((refH, refW), (outH, outW));

        ReadOnlySpan<float> exp = expected.AsSpan<float>();
        double ssimSum = 0, mseSum = 0;
        for (int f = 0; f < refF; f++)
        {
            byte[] refRgb = new byte[outW * outH * 3];
            for (int c = 0; c < 3; c++)
            {
                long plane = ((long)c * refF + f) * outH * outW;
                for (int i = 0; i < outH * outW; i++)
                {
                    float v = (Math.Clamp(exp[(int)(plane + i)], -1f, 1f) * 0.5f + 0.5f) * 255f;
                    refRgb[i * 3 + c] = (byte)Math.Clamp(MathF.Round(v), 0f, 255f);
                }
            }
            double ssim = Helpers.Ssim.Compute(restored[f], refRgb, outW, outH);
            double mse = 0;
            for (int i = 0; i < refRgb.Length; i++)
            {
                double d = restored[f][i] - refRgb[i];
                mse += d * d;
            }
            mse /= refRgb.Length;
            ssimSum += ssim;
            mseSum += mse;
            _output.WriteLine($"frame {f}: SSIM {ssim:f5}  PSNR {10 * Math.Log10(255.0 * 255.0 / Math.Max(mse, 1e-9)):f2} dB");
        }
        double meanSsim = ssimSum / refF;
        _output.WriteLine($"MEAN: SSIM {meanSsim:f5}  PSNR {10 * Math.Log10(255.0 * 255.0 / Math.Max(mseSum / refF, 1e-9)):f2} dB");
        Assert.True(meanSsim >= 0.995, $"mean SSIM {meanSsim:f5} below 0.995 gate");
    }

    #endregion

    #region Shared helpers

    private static string? Env(string name)
    {
        string? v = Environment.GetEnvironmentVariable(name);
        return string.IsNullOrWhiteSpace(v) ? null : v;
    }

    private static Tensor LoadBin(string path, long[] shape)
    {
        byte[] bytes = File.ReadAllBytes(path);
        Tensor tensor = new Tensor(new TensorShape(shape), DType.F32);
        bytes.AsSpan().CopyTo(System.Runtime.InteropServices.MemoryMarshal.AsBytes(tensor.AsSpan<float>()));
        return tensor;
    }

    private static double RelL2(Tensor actual, Tensor expected)
    {
        Assert.Equal(expected.Shape.ElementCount, actual.Shape.ElementCount);
        ReadOnlySpan<float> a = actual.AsSpan<float>();
        ReadOnlySpan<float> e = expected.AsSpan<float>();
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - e[i];
            num += d * d;
            den += (double)e[i] * e[i];
        }
        return Math.Sqrt(num / (den + 1e-12));
    }

    private static double RelL2(Tensor actual, string refPath)
    {
        float[] expected = new float[new FileInfo(refPath).Length / 4];
        using (FileStream fs = File.OpenRead(refPath))
            fs.ReadExactly(System.Runtime.InteropServices.MemoryMarshal.AsBytes(expected.AsSpan()));
        ReadOnlySpan<float> a = actual.AsSpan<float>();
        Assert.Equal(expected.Length, a.Length);
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - expected[i];
            num += d * d;
            den += (double)expected[i] * expected[i];
        }
        return Math.Sqrt(num / (den + 1e-12));
    }

    #endregion
}
