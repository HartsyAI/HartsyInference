using System.Linq;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Video.Tests;

/// <summary>Advisor-directed gate before building SmoothQuant (W8A8_HANDOFF.md item 1): SmoothQuant only
/// helps if activation-side quantization is the dominant error source — it migrates difficulty FROM
/// activations INTO weights, so if weight quantization already dominates, smoothing makes things worse,
/// not better. This isolates each operand's contribution via fake-quant/dequant (quantize one operand
/// with the exact production math, keep the other F32, run the real F32 GEMM, relL2 vs the true F32×F32
/// reference) on a REAL captured Kandinsky5 Linear operand pair — not synthetic data, since stage 4a/4b
/// already found real checkpoints are 3-5x worse than synthetic. The captured operands come from
/// CudaBackend.CaptureW8A8Operands, a test-only hook fired once from inside a real ForwardVideo pass.
/// Quantization formulas mirror native/cuda/dequant/w8a8.cu (activation: per-row absmax/127, round,
/// clamp[-127,127]) and CudaBackend.QuantizeWeightForW8A8 (weight: per-output-channel absmax/127) exactly,
/// so the ablation measures the SAME quantization the real kernels perform, just staged separately.
/// Run explicitly:
///   CUDA_VISIBLE_DEVICES=1 dotnet test --filter "FullyQualifiedName~Kandinsky5W8A8OperandAblationTests"
/// </summary>
[Trait("Category", "W8A8Bench")]
public sealed unsafe class Kandinsky5W8A8OperandAblationTests
{
    private readonly ITestOutputHelper _output;
    public Kandinsky5W8A8OperandAblationTests(ITestOutputHelper output) => _output = output;

    private static string T2VDir => Environment.GetEnvironmentVariable("KANDINSKY5_T2V_DIR")
        ?? Path.Combine(TestPaths.ModelsDir, "Stable-Diffusion", "Kandinsky5", "Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers");

    [Fact]
    public void W8A8_OperandAblation_ActivationVsWeight()
    {
        string transformerDir = Path.Combine(T2VDir, "transformer");
        if (!Directory.Exists(transformerDir)) { _output.WriteLine($"SKIPPED: T2V transformer dir not found: {transformerDir} (set KANDINSKY5_T2V_DIR)."); return; }
        if (!File.Exists(TestPaths.Kandinsky5.PromptQwenEmbeds) || !File.Exists(TestPaths.Kandinsky5.PromptClipPooled))
        { _output.WriteLine("SKIPPED: pre-computed Qwen/CLIP embeddings missing (see dump_kandinsky5_embeddings.py)."); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(Kandinsky5W8A8OperandAblationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }
        if (!File.Exists(Path.Combine(ptxDir, "w8a8.ptx"))) { _output.WriteLine("SKIPPED: w8a8.ptx missing"); return; }
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        (Kandinsky5CheckpointConverter.ConvertedWeights converted, List<SafeTensorsLoader> loaders) =
            Kandinsky5CheckpointConverter.LoadDiffusersFolder(transformerDir);
        try
        {
            Dictionary<string, Tensor> tw = new(converted.Transformer.Count);
            foreach ((string k, Tensor v) in converted.Transformer)
                tw[k] = v.DType == DType.BF16 ? v.CastTo(DType.F16) : v;
            Kandinsky5Config config = Kandinsky5Config.VideoLite2B;
            using Kandinsky5Transformer transformer = new(config);
            transformer.LoadWeights(tw);

            CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            try
            {
                _output.WriteLine($"Device: {backend.Capabilities.Name} (CUDA_VISIBLE_DEVICES=" +
                    $"{Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES") ?? "unset"})");

                using Tensor qwen = LoadF32Tensor(TestPaths.Kandinsky5.PromptQwenEmbeds, config.InTextDim);
                using Tensor clip = LoadPooled(TestPaths.Kandinsky5.PromptClipPooled, config.InTextDim2);

                const int width = 512, height = 512, numFrames = 25;
                int tLat = (numFrames - 1) / 4 + 1, hLat = height / 8, wLat = width / 8;
                int latCh = config.InVisualDim;
                TensorShape latentShape = new TensorShape([1L, latCh, tLat, hLat, wLat]);
                TensorShape maskShape = new TensorShape([1L, 1, tLat, hLat, wLat]);

                const int steps = 30;
                FlowMatchEulerDiscreteScheduler scheduler = new(5.0f);
                scheduler.SetTimesteps(steps);
                ReadOnlySpan<float> timesteps = scheduler.Timesteps;
                float midT = timesteps[steps / 2];
                (float scaleT, float scaleH, float scaleW) = Kandinsky5VideoPipeline.GetRopeScaleFactor(height, width);

                using Tensor noisy = SeedGenerator.CreateNoise(latentShape, seed: 42);
                using Tensor condLatent = Zeros(latentShape);
                using Tensor condMask = Zeros(maskShape);
                using Tensor packed = PackVisualCond(noisy, condLatent, condMask, latCh);

                backend.PreloadWeights(transformer.EnumerateWeights());

                // Capture ONE real Linear's real (input, weight) operands off a live forward pass, at a
                // mid-schedule timestep. The hook fires once (first W8A8-eligible Linear call) and clears
                // itself; W8A8 must be enabled for the eligibility gate (M>=32 etc.) to fire the capture,
                // but the capture itself happens BEFORE any quantization runs — it snapshots the same F32
                // operands the F16/F32 GEMM path would have used.
                // Skip the shallow, t-invariant projections: the text-embed proj (M=~38 text tokens, K=3584
                // Qwen dim) AND the visual patch-embed (M in the thousands but K=132 — the packed
                // noisy/cond/mask channels x patch volume, applied before any adaLN t-modulation, and in
                // T2V mode half those channels are artificially all-zero from the zero image-conditioning
                // here, which degenerates channel statistics). K>=1792=ModelDim targets a genuine
                // mid-network hidden-state Linear (attention/FFN proj on real, t-modulated activations).
                float[]? capIn = null, capW = null;
                int capM = 0, capK = 0, capN = 0;
                Action<float[], int, int, float[], int, Tensor> hook = null!;
                hook = (inArr, m, k, wArr, n, wT) =>
                {
                    if (m < 500 || k < 1792) { CudaBackend.CaptureW8A8Operands = hook; return; }
                    capIn = inArr; capM = m; capK = k;
                    capW = wArr; capN = n;
                };
                CudaBackend.CaptureW8A8Operands = hook;
                backend.EnableW8A8 = true;
                Tensor captureOut = transformer.ForwardVideo(backend, packed, midT, qwen, clip, scaleT, scaleH, scaleW);
                backend.Sync();
                captureOut.Dispose();
                backend.FreeActivations(trimPool: false);
                backend.FreeWeights(transformer.EnumerateWeights());
                CudaBackend.CaptureW8A8Operands = null; // in case nothing fired

                Assert.True(capIn is not null && capW is not null,
                    "No W8A8-eligible Linear call was captured — check EnableW8A8/eligibility gate.");
                _output.WriteLine($"Captured operand pair: input[{capM},{capK}], weight[{capN},{capK}]");

                // ── Reference: true F32 x^T·w^T (row-major [M,K] x [N,K]^T -> [M,N]) ──
                float[] refOut = MatMulTransposeB(capIn!, capM, capK, capW!, capN);

                // ── A-only: fake-quant/dequant the ACTIVATION per-row (mirrors w8a8_quant_rowwise_*),
                // weight stays exact F32 ──
                float[] actFakeQ = FakeQuantRowwise(capIn!, capM, capK);
                float[] aOnlyOut = MatMulTransposeB(actFakeQ, capM, capK, capW!, capN);
                double aOnlyRel = RelL2(aOnlyOut, refOut);

                // ── W-only: fake-quant/dequant the WEIGHT per-output-channel (mirrors
                // QuantizeWeightForW8A8), activation stays exact F32 ──
                float[] wFakeQ = FakeQuantPerChannel(capW!, capN, capK);
                float[] wOnlyOut = MatMulTransposeB(capIn!, capM, capK, wFakeQ, capN);
                double wOnlyRel = RelL2(wOnlyOut, refOut);

                // ── Both (sanity: should roughly bracket the real W8A8 GEMM's relL2, since both operands
                // are quantized exactly as production does, just via fake-quant instead of the real int8
                // GEMM+dequant epilogue) ──
                float[] bothOut = MatMulTransposeB(actFakeQ, capM, capK, wFakeQ, capN);
                double bothRel = RelL2(bothOut, refOut);

                _output.WriteLine($"A-only (activation quantized, weight F32) relL2: {aOnlyRel:e3}");
                _output.WriteLine($"W-only (weight quantized, activation F32) relL2: {wOnlyRel:e3}");
                _output.WriteLine($"Both (both quantized, fake-quant path)   relL2: {bothRel:e3}");
                _output.WriteLine("(this is ONE layer's local fake-quant error, not comparable in magnitude " +
                    "to the full-chain e2e relL2 measured elsewhere — only the A-vs-W RATIO below is the signal here)");
                double ratio = aOnlyRel / Math.Max(wOnlyRel, 1e-12);
                _output.WriteLine($"A-only / W-only ratio: {ratio:F2} " +
                    "(>>1 => activation-dominated, SmoothQuant has headroom; <<1 or ~1 => weight-side dominates/comparable, SmoothQuant may not help or could hurt)");
            }
            catch (Exception realEx)
            {
                _output.WriteLine($"REAL EXCEPTION (pre-Dispose): {realEx}");
                throw;
            }
            finally
            {
                CudaBackend.CaptureW8A8Operands = null;
                try { backend.Dispose(); }
                catch (Exception disposeEx) { _output.WriteLine($"DISPOSE-TIME EXCEPTION: {disposeEx}"); }
            }
        }
        finally { foreach (SafeTensorsLoader l in loaders) l.Dispose(); }
    }

    /// <summary>Second half of the advisor's gate: given A-only dominates W-only (confirmed above),
    /// SmoothQuant's premise is FIXED outlier channels (a few input channels with systematically larger
    /// magnitude, stable in identity, not just magnitude, across inputs) — that's what lets a per-channel
    /// factor computed once from a small sample generalize. This captures the SAME first-eligible Linear's
    /// real input activation at 3 timesteps (early/mid/late across the flow-match schedule) and checks:
    /// (1) heavy tail — is max channel absmax >> median channel absmax? (2) stability — do the top-10
    /// outlier channel INDICES stay the same across timesteps, even as magnitude drifts? If both hold,
    /// first-batch calibration (compute s_j once, reuse for the whole generation) is sufficient — the
    /// per-row dynamic quant already in production handles the magnitude drift. If indices drift, a
    /// single-sample calibration would miss channels that only become outliers later in the schedule.</summary>
    [Fact]
    public void W8A8_ActivationChannelOutlier_Stability()
    {
        string transformerDir = Path.Combine(T2VDir, "transformer");
        if (!Directory.Exists(transformerDir)) { _output.WriteLine($"SKIPPED: T2V transformer dir not found: {transformerDir} (set KANDINSKY5_T2V_DIR)."); return; }
        if (!File.Exists(TestPaths.Kandinsky5.PromptQwenEmbeds) || !File.Exists(TestPaths.Kandinsky5.PromptClipPooled))
        { _output.WriteLine("SKIPPED: pre-computed Qwen/CLIP embeddings missing (see dump_kandinsky5_embeddings.py)."); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(Kandinsky5W8A8OperandAblationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }
        if (!File.Exists(Path.Combine(ptxDir, "w8a8.ptx"))) { _output.WriteLine("SKIPPED: w8a8.ptx missing"); return; }
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        (Kandinsky5CheckpointConverter.ConvertedWeights converted, List<SafeTensorsLoader> loaders) =
            Kandinsky5CheckpointConverter.LoadDiffusersFolder(transformerDir);
        try
        {
            Dictionary<string, Tensor> tw = new(converted.Transformer.Count);
            foreach ((string k, Tensor v) in converted.Transformer)
                tw[k] = v.DType == DType.BF16 ? v.CastTo(DType.F16) : v;
            Kandinsky5Config config = Kandinsky5Config.VideoLite2B;
            using Kandinsky5Transformer transformer = new(config);
            transformer.LoadWeights(tw);

            CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            try
            {
                _output.WriteLine($"Device: {backend.Capabilities.Name} (CUDA_VISIBLE_DEVICES=" +
                    $"{Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES") ?? "unset"})");

                using Tensor qwen = LoadF32Tensor(TestPaths.Kandinsky5.PromptQwenEmbeds, config.InTextDim);
                using Tensor clip = LoadPooled(TestPaths.Kandinsky5.PromptClipPooled, config.InTextDim2);

                const int width = 512, height = 512, numFrames = 25;
                int tLat = (numFrames - 1) / 4 + 1, hLat = height / 8, wLat = width / 8;
                int latCh = config.InVisualDim;
                TensorShape latentShape = new TensorShape([1L, latCh, tLat, hLat, wLat]);
                TensorShape maskShape = new TensorShape([1L, 1, tLat, hLat, wLat]);

                const int steps = 30;
                FlowMatchEulerDiscreteScheduler scheduler = new(5.0f);
                scheduler.SetTimesteps(steps);
                ReadOnlySpan<float> timesteps = scheduler.Timesteps;
                int[] probeSteps = [0, steps / 2, steps - 1];
                (float scaleT, float scaleH, float scaleW) = Kandinsky5VideoPipeline.GetRopeScaleFactor(height, width);

                using Tensor noisy = SeedGenerator.CreateNoise(latentShape, seed: 42);
                using Tensor condLatent = Zeros(latentShape);
                using Tensor condMask = Zeros(maskShape);
                using Tensor packed = PackVisualCond(noisy, condLatent, condMask, latCh);

                backend.PreloadWeights(transformer.EnumerateWeights());

                List<float[]> perChannelAbsMax = new();
                int capK = 0;
                foreach (int stepIdx in probeSteps)
                {
                    float t = timesteps[stepIdx];
                    // Same shallow-projection skip as the ablation test: only accept a genuine mid-network
                    // hidden-state Linear (M in the thousands, K>=ModelDim), not the t-invariant text-embed
                    // or visual patch-embed projections.
                    float[]? capIn = null;
                    int m = 0, k = 0;
                    Action<float[], int, int, float[], int, Tensor> hook = null!;
                    hook = (inArr, mm, kk, wArr, nn, wT) =>
                    {
                        if (mm < 500 || kk < 1792) { CudaBackend.CaptureW8A8Operands = hook; return; }
                        capIn = inArr; m = mm; k = kk;
                    };
                    CudaBackend.CaptureW8A8Operands = hook;
                    backend.EnableW8A8 = true;
                    Tensor outp = transformer.ForwardVideo(backend, packed, t, qwen, clip, scaleT, scaleH, scaleW);
                    backend.Sync();
                    outp.Dispose();
                    backend.FreeActivations(trimPool: false);
                    CudaBackend.CaptureW8A8Operands = null;

                    Assert.True(capIn is not null, $"No capture fired at step {stepIdx}.");
                    capK = k;
                    float[] chAbsMax = new float[k];
                    for (int r = 0; r < m; r++)
                    {
                        int baseIdx = r * k;
                        for (int c = 0; c < k; c++)
                        {
                            float a = MathF.Abs(capIn![baseIdx + c]);
                            if (a > chAbsMax[c]) chAbsMax[c] = a;
                        }
                    }
                    perChannelAbsMax.Add(chAbsMax);
                    _output.WriteLine($"step {stepIdx} (t={t:F1}): captured [{m},{k}]");
                }
                backend.FreeWeights(transformer.EnumerateWeights());

                const int topK = 10;
                List<int[]> topChannelsPerStep = new();
                foreach (float[] chAbsMax in perChannelAbsMax)
                {
                    int[] idx = Enumerable.Range(0, capK).OrderByDescending(i => chAbsMax[i]).Take(topK).ToArray();
                    float median = chAbsMax.OrderBy(v => v).ElementAt(capK / 2);
                    float max = chAbsMax.Max();
                    _output.WriteLine($"  max channel absmax={max:e3}, median={median:e3}, max/median={max / Math.Max(median, 1e-12f):F1}x, top-{topK} channels={string.Join(",", idx)}");
                    topChannelsPerStep.Add(idx);
                }

                for (int i = 1; i < topChannelsPerStep.Count; i++)
                {
                    int overlap = topChannelsPerStep[0].Intersect(topChannelsPerStep[i]).Count();
                    _output.WriteLine($"top-{topK} overlap step0 vs step{i}: {overlap}/{topK}");
                }

                // Top-K overlap is fragile to reordering within a broad shoulder of similarly-large
                // channels — a channel dropping from rank 9 to rank 12 reads as "lost" even though its
                // magnitude barely moved. Pearson correlation of the FULL per-channel absmax profile is
                // the robust signal: high correlation despite top-K churn means mild reordering (sparse
                // calibration suffices); low correlation means the outlier structure itself is moving
                // (calibration must sample densely across the schedule either way — this only sets density).
                for (int i = 1; i < perChannelAbsMax.Count; i++)
                {
                    double r = PearsonCorrelation(perChannelAbsMax[0], perChannelAbsMax[i]);
                    _output.WriteLine($"per-channel absmax Pearson r, step0 vs step{i}: {r:F3}");
                }
            }
            catch (Exception realEx)
            {
                _output.WriteLine($"REAL EXCEPTION (pre-Dispose): {realEx}");
                throw;
            }
            finally
            {
                CudaBackend.CaptureW8A8Operands = null;
                try { backend.Dispose(); }
                catch (Exception disposeEx) { _output.WriteLine($"DISPOSE-TIME EXCEPTION: {disposeEx}"); }
            }
        }
        finally { foreach (SafeTensorsLoader l in loaders) l.Dispose(); }
    }

    /// <summary>Offline go/no-go for the SmoothQuant subsystem BEFORE building it (advisor-directed):
    /// max-aggregate per-channel activation absmax across the 3 schedule-spanning timesteps already
    /// validated as necessary (Pearson r drops to 0.432 at the schedule extremes — single-sample
    /// calibration would miss real drift), derive s_j = (actMax_j / wMax_j)^alpha per SmoothQuant, apply
    /// X_hat=X/s, W_hat=W*s (product-preserving, so the true F32 reference is unchanged), fake-quant BOTH
    /// with the exact production formulas, and check whether the smoothed relL2 drops from the unsmoothed
    /// "Both" baseline toward the W-only floor. The fake-vs-real GEMM offset is identical in both the
    /// smoothed and unsmoothed arms (same fake-quant path), so it cancels in the comparison — only the
    /// DELTA needs to be trustworthy here, not the absolute magnitude. Sweeps alpha in {0.3, 0.5, 0.7} to
    /// pick a production value for free. Captures the SAME K=1792 mid-network Linear already validated
    /// (attention-shaped: N=K=1792) AND a K=7168 one (FFN-shaped) so the result isn't layer-idiosyncratic.
    /// Zero production/kernel changes — pure host-side math on already-captured real operands.</summary>
    [Fact]
    public void W8A8_SmoothQuant_OfflineGate()
    {
        string transformerDir = Path.Combine(T2VDir, "transformer");
        if (!Directory.Exists(transformerDir)) { _output.WriteLine($"SKIPPED: T2V transformer dir not found: {transformerDir} (set KANDINSKY5_T2V_DIR)."); return; }
        if (!File.Exists(TestPaths.Kandinsky5.PromptQwenEmbeds) || !File.Exists(TestPaths.Kandinsky5.PromptClipPooled))
        { _output.WriteLine("SKIPPED: pre-computed Qwen/CLIP embeddings missing (see dump_kandinsky5_embeddings.py)."); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(Kandinsky5W8A8OperandAblationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }
        if (!File.Exists(Path.Combine(ptxDir, "w8a8.ptx"))) { _output.WriteLine("SKIPPED: w8a8.ptx missing"); return; }
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        (Kandinsky5CheckpointConverter.ConvertedWeights converted, List<SafeTensorsLoader> loaders) =
            Kandinsky5CheckpointConverter.LoadDiffusersFolder(transformerDir);
        try
        {
            Dictionary<string, Tensor> tw = new(converted.Transformer.Count);
            foreach ((string k, Tensor v) in converted.Transformer)
                tw[k] = v.DType == DType.BF16 ? v.CastTo(DType.F16) : v;
            Kandinsky5Config config = Kandinsky5Config.VideoLite2B;
            using Kandinsky5Transformer transformer = new(config);
            transformer.LoadWeights(tw);

            CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            try
            {
                _output.WriteLine($"Device: {backend.Capabilities.Name} (CUDA_VISIBLE_DEVICES=" +
                    $"{Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES") ?? "unset"})");

                using Tensor qwen = LoadF32Tensor(TestPaths.Kandinsky5.PromptQwenEmbeds, config.InTextDim);
                using Tensor clip = LoadPooled(TestPaths.Kandinsky5.PromptClipPooled, config.InTextDim2);

                const int width = 512, height = 512, numFrames = 25;
                int tLat = (numFrames - 1) / 4 + 1, hLat = height / 8, wLat = width / 8;
                int latCh = config.InVisualDim;
                TensorShape latentShape = new TensorShape([1L, latCh, tLat, hLat, wLat]);
                TensorShape maskShape = new TensorShape([1L, 1, tLat, hLat, wLat]);

                const int steps = 30;
                FlowMatchEulerDiscreteScheduler scheduler = new(5.0f);
                scheduler.SetTimesteps(steps);
                ReadOnlySpan<float> timesteps = scheduler.Timesteps;
                int[] probeSteps = [0, steps / 2, steps - 1];
                (float scaleT, float scaleH, float scaleW) = Kandinsky5VideoPipeline.GetRopeScaleFactor(height, width);

                using Tensor noisy = SeedGenerator.CreateNoise(latentShape, seed: 42);
                using Tensor condLatent = Zeros(latentShape);
                using Tensor condMask = Zeros(maskShape);
                using Tensor packed = PackVisualCond(noisy, condLatent, condMask, latCh);

                backend.PreloadWeights(transformer.EnumerateWeights());

                // Capture two distinct-shape deep Linears in the SAME forward pass: the first K>=1792 call
                // ("attn-like", N==K) establishes shape A; the first call with a DIFFERENT shape after that
                // establishes shape B ("ffn-like", expected K=7168=FfDim). Once both shapes are known (from
                // step 0), later steps just match by shape. Each capture records weight once (t-invariant,
                // re-captured redundantly each step but identical) and activation per probeStep (t-dependent).
                (int n, int k)? shapeA = null, shapeB = null;
                (float[] weight, int n, int k, List<float[]> actPerStep)? layerA = null;
                (float[] weight, int n, int k, List<float[]> actPerStep)? layerB = null;
                foreach (int stepIdx in probeSteps)
                {
                    float t = timesteps[stepIdx];
                    float[]? aIn = null, aW = null, bIn = null, bW = null;
                    int aM = 0, aK = 0, aN = 0, bK = 0, bN = 0;
                    Action<float[], int, int, float[], int, Tensor> hook = null!;
                    hook = (inArr, mm, kk, wArr, nn, wT) =>
                    {
                        if (mm < 500 || kk < 1792) { CudaBackend.CaptureW8A8Operands = hook; return; }
                        bool matchesA = shapeA is { } sa ? (kk == sa.k && nn == sa.n) : aIn is null;
                        bool matchesB = shapeB is { } sb && kk == sb.k && nn == sb.n;
                        if (aIn is null && matchesA)
                        {
                            aIn = inArr; aW = wArr; aM = mm; aK = kk; aN = nn;
                        }
                        else if (bIn is null && !matchesA && (matchesB || shapeB is null))
                        {
                            bIn = inArr; bW = wArr; bK = kk; bN = nn;
                        }
                        bool stillWantA = aIn is null;
                        bool stillWantB = bIn is null && (shapeB is not null || shapeA is not null);
                        if (stillWantA || stillWantB) CudaBackend.CaptureW8A8Operands = hook;
                    };
                    CudaBackend.CaptureW8A8Operands = hook;
                    backend.EnableW8A8 = true;
                    Tensor outp = transformer.ForwardVideo(backend, packed, t, qwen, clip, scaleT, scaleH, scaleW);
                    backend.Sync();
                    outp.Dispose();
                    backend.FreeActivations(trimPool: false);
                    CudaBackend.CaptureW8A8Operands = null;
                    Assert.True(aIn is not null, $"No capture fired at step {stepIdx}.");

                    shapeA ??= (aN, aK);
                    layerA ??= (aW!, aN, aK, new List<float[]>());
                    layerA.Value.actPerStep.Add(aIn!);
                    if (bIn is not null)
                    {
                        shapeB ??= (bN, bK);
                        layerB ??= (bW!, bN, bK, new List<float[]>());
                        layerB.Value.actPerStep.Add(bIn!);
                    }
                }
                backend.FreeWeights(transformer.EnumerateWeights());

                Assert.True(layerA is not null, "No layer A captured.");
                _output.WriteLine($"Layer A: weight[{layerA.Value.n},{layerA.Value.k}] ({layerA.Value.actPerStep.Count} activation samples)");
                RunOfflineSmoothQuantSweep("A", layerA.Value.actPerStep, layerA.Value.weight, layerA.Value.n, layerA.Value.k);
                if (layerB is not null)
                {
                    _output.WriteLine($"Layer B: weight[{layerB.Value.n},{layerB.Value.k}] ({layerB.Value.actPerStep.Count} activation samples)");
                    RunOfflineSmoothQuantSweep("B", layerB.Value.actPerStep, layerB.Value.weight, layerB.Value.n, layerB.Value.k);
                }
                else
                {
                    _output.WriteLine("Layer B: not found (only one distinct deep-Linear shape encountered before probe budget ran out).");
                }
            }
            catch (Exception realEx)
            {
                _output.WriteLine($"REAL EXCEPTION (pre-Dispose): {realEx}");
                throw;
            }
            finally
            {
                CudaBackend.CaptureW8A8Operands = null;
                try { backend.Dispose(); }
                catch (Exception disposeEx) { _output.WriteLine($"DISPOSE-TIME EXCEPTION: {disposeEx}"); }
            }
        }
        finally { foreach (SafeTensorsLoader l in loaders) l.Dispose(); }
    }

    /// <summary>Advisor-directed follow-up (2026-07-24): the 2-layer offline gate predicted a 40% relL2
    /// win from SmoothQuant, but applying alpha=0.7 uniformly to all 335 W8A8-eligible weights made the
    /// real e2e SSIM WORSE (0.9144 vs the pre-SmoothQuant 0.9211) — a genuine, surprising contradiction.
    /// The two possible explanations: (a) uniform alpha hurts a meaningful fraction of layers (the ones
    /// the first ablation warned about — weight-dominated or balanced, not activation-dominated), or
    /// (b) smoothing helps EVERY layer's local error but still regresses e2e because it redistributes
    /// error into channels the downstream (later blocks / VAE) is more sensitive to. This test answers
    /// (a) vs (b) directly: for every W8A8-eligible Linear encountered across 3 calibration timesteps
    /// (same capture mechanism as CalibrateSmoothQuant in Kandinsky5W8A8SsimAbTests.cs), compute the
    /// fake-quant relL2 with and without alpha=0.7 smoothing using the SAME production formulas
    /// (FakeQuantRowwise/FakeQuantPerChannel/MatMulTransposeB), using the LAST calibration timestep's
    /// activation as the test point and actMax aggregated across all 3 (mirrors the real calibration
    /// exactly). Reports how many layers are helped vs hurt and the aggregate. Zero new GPU passes beyond
    /// the 3 calibration forwards already required.</summary>
    [Fact]
    public void W8A8_SmoothQuant_AllLayers_OfflineGate()
    {
        string transformerDir = Path.Combine(T2VDir, "transformer");
        if (!Directory.Exists(transformerDir)) { _output.WriteLine($"SKIPPED: T2V transformer dir not found: {transformerDir} (set KANDINSKY5_T2V_DIR)."); return; }
        if (!File.Exists(TestPaths.Kandinsky5.PromptQwenEmbeds) || !File.Exists(TestPaths.Kandinsky5.PromptClipPooled))
        { _output.WriteLine("SKIPPED: pre-computed Qwen/CLIP embeddings missing (see dump_kandinsky5_embeddings.py)."); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(Kandinsky5W8A8OperandAblationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }
        if (!File.Exists(Path.Combine(ptxDir, "w8a8.ptx"))) { _output.WriteLine("SKIPPED: w8a8.ptx missing"); return; }
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        (Kandinsky5CheckpointConverter.ConvertedWeights converted, List<SafeTensorsLoader> loaders) =
            Kandinsky5CheckpointConverter.LoadDiffusersFolder(transformerDir);
        try
        {
            Dictionary<string, Tensor> tw = new(converted.Transformer.Count);
            foreach ((string k, Tensor v) in converted.Transformer)
                tw[k] = v.DType == DType.BF16 ? v.CastTo(DType.F16) : v;
            Kandinsky5Config config = Kandinsky5Config.VideoLite2B;
            using Kandinsky5Transformer transformer = new(config);
            transformer.LoadWeights(tw);

            CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            try
            {
                _output.WriteLine($"Device: {backend.Capabilities.Name} (CUDA_VISIBLE_DEVICES=" +
                    $"{Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES") ?? "unset"})");

                using Tensor qwen = LoadF32Tensor(TestPaths.Kandinsky5.PromptQwenEmbeds, config.InTextDim);
                using Tensor clip = LoadPooled(TestPaths.Kandinsky5.PromptClipPooled, config.InTextDim2);

                const int width = 512, height = 512, numFrames = 25;
                int tLat = (numFrames - 1) / 4 + 1, hLat = height / 8, wLat = width / 8;
                int latCh = config.InVisualDim;
                TensorShape latentShape = new TensorShape([1L, latCh, tLat, hLat, wLat]);
                TensorShape maskShape = new TensorShape([1L, 1, tLat, hLat, wLat]);

                const int steps = 30;
                FlowMatchEulerDiscreteScheduler scheduler = new(5.0f);
                scheduler.SetTimesteps(steps);
                ReadOnlySpan<float> timesteps = scheduler.Timesteps;
                int[] probeSteps = [0, steps / 2, steps - 1];
                (float scaleT, float scaleH, float scaleW) = Kandinsky5VideoPipeline.GetRopeScaleFactor(height, width);

                using Tensor noisy = SeedGenerator.CreateNoise(latentShape, seed: 42);
                using Tensor condLatent = Zeros(latentShape);
                using Tensor condMask = Zeros(maskShape);
                using Tensor packed = PackVisualCond(noisy, condLatent, condMask, latCh);

                backend.PreloadWeights(transformer.EnumerateWeights());

                Dictionary<Tensor, float[]> actMax = new();
                Dictionary<Tensor, float[]> wMax = new();
                // Only the SCALAR delta result is retained per layer — NOT the raw M×K activation/weight
                // arrays (335 layers × up to ~200MB each would blow host memory). Computed inline inside
                // the hook, on the LAST calibration step's capture (by then actMax/wMax are fully
                // aggregated across all 3 steps for that weight), then discarded.
                const double alpha = 0.7;
                List<(double unsmoothed, double smoothed, double ratio)> perLayer = new();

                backend.EnableW8A8 = true;
                for (int si = 0; si < probeSteps.Length; si++)
                {
                    int stepIdx = probeSteps[si];
                    bool isLast = si == probeSteps.Length - 1;
                    float t = timesteps[stepIdx];
                    Action<float[], int, int, float[], int, Tensor> hook = null!;
                    hook = (inArr, m, k, wArr, n, wT) =>
                    {
                        if (!actMax.TryGetValue(wT, out float[]? am))
                        {
                            am = new float[k];
                            actMax[wT] = am;
                        }
                        for (int r = 0; r < m; r++)
                        {
                            int baseIdx = r * k;
                            for (int c = 0; c < k; c++)
                            {
                                float a = MathF.Abs(inArr[baseIdx + c]);
                                if (a > am[c]) am[c] = a;
                            }
                        }
                        if (!wMax.TryGetValue(wT, out float[]? wm))
                        {
                            wm = new float[k];
                            for (int ni = 0; ni < n; ni++)
                            {
                                int baseIdx = ni * k;
                                for (int c = 0; c < k; c++)
                                {
                                    float a = MathF.Abs(wArr[baseIdx + c]);
                                    if (a > wm[c]) wm[c] = a;
                                }
                            }
                            wMax[wT] = wm;
                        }
                        if (isLast)
                        {
                            // Cap M for the reference/fake-quant GEMMs — this is a host-CPU O(M*N*K) matmul,
                            // not the real GPU GEMM, and 335 layers at full M (up to ~7168) is far too slow
                            // for a diagnostic pass. Row-subsampling the activation preserves the per-channel
                            // quantization behavior (rows are independent for the ratio signal we need) while
                            // cutting cost proportionally. A real (properly-sized) subarray, not just a
                            // smaller m against the full backing array, so FakeQuantRowwise/MatMulTransposeB's
                            // .Length-based allocations stay proportional to the capped size too.
                            int mCapped = Math.Min(m, 256);
                            float[] inArrCapped = mCapped == m ? inArr : inArr.AsSpan(0, mCapped * k).ToArray();
                            (double unsmoothedRel, double smoothedRel) = ComputeSmoothQuantDelta(inArrCapped, mCapped, k, wArr, n, am, wm, alpha);
                            double ratio = smoothedRel / Math.Max(unsmoothedRel, 1e-12);
                            lock (perLayer) perLayer.Add((unsmoothedRel, smoothedRel, ratio));
                        }
                        CudaBackend.CaptureW8A8Operands = hook;
                    };
                    CudaBackend.CaptureW8A8Operands = hook;
                    Tensor outp = transformer.ForwardVideo(backend, packed, t, qwen, clip, scaleT, scaleH, scaleW);
                    backend.Sync();
                    outp.Dispose();
                    backend.FreeActivations(trimPool: false);
                    CudaBackend.CaptureW8A8Operands = null;
                    _output.WriteLine($"calibration pass step {stepIdx}: {actMax.Count} distinct weights seen so far");
                }
                backend.FreeWeights(transformer.EnumerateWeights());

                int helped = perLayer.Count(x => x.ratio < 0.98);
                int hurt = perLayer.Count(x => x.ratio > 1.02);
                int neutral = perLayer.Count - helped - hurt;
                double sumUnsmoothed = perLayer.Sum(x => x.unsmoothed);
                double sumSmoothed = perLayer.Sum(x => x.smoothed);

                _output.WriteLine($"Layers analyzed: {perLayer.Count} | helped(>2% better)={helped} hurt(>2% worse)={hurt} neutral={neutral}");
                _output.WriteLine($"Sum unsmoothed relL2 = {sumUnsmoothed:e3} | Sum smoothed relL2 = {sumSmoothed:e3} | " +
                    $"aggregate ratio = {sumSmoothed / Math.Max(sumUnsmoothed, 1e-12):F3}");
                List<(double unsmoothed, double smoothed, double ratio)> worst =
                    perLayer.OrderByDescending(x => x.ratio).Take(10).ToList();
                _output.WriteLine("Worst 10 layers by smoothed/unsmoothed ratio (>1 = smoothing hurt this layer):");
                foreach ((double u, double sm, double r) in worst)
                    _output.WriteLine($"  unsmoothed={u:e3} smoothed={sm:e3} ratio={r:F3}");
            }
            catch (Exception realEx)
            {
                _output.WriteLine($"REAL EXCEPTION (pre-Dispose): {realEx}");
                throw;
            }
            finally
            {
                CudaBackend.CaptureW8A8Operands = null;
                try { backend.Dispose(); }
                catch (Exception disposeEx) { _output.WriteLine($"DISPOSE-TIME EXCEPTION: {disposeEx}"); }
            }
        }
        finally { foreach (SafeTensorsLoader l in loaders) l.Dispose(); }
    }

    /// <summary>Measure-first gate for the NEXT lever (per-group weight quant), before writing a single
    /// line of grouped-dequant kernel code — advisor-directed, 2026-07-24, following the SmoothQuant e2e
    /// regression finding. Checks the quadrature hypothesis (do A-only/W-only errors combine as
    /// Both ≈ sqrt(A²+W²)?) on real captured Kandinsky5 layers, which — if it holds — puts a hard CEILING
    /// on what per-group weight quant can achieve: driving W-only to zero only pulls Both down to the
    /// A-only floor, so if activation dominates (A-only >> W-only, as the prior deep-layer ablation found:
    /// A=1.098e-2 vs W=5.413e-3, ratio 2.03), grouped weight quant's local-relL2 ceiling is small — and per
    /// the SmoothQuant finding, local relL2 has already shown it can ANTI-CORRELATE with e2e SSIM on this
    /// model (-29% aggregate local relL2 produced -0.007 e2e SSIM), so even a "capped but real" local win
    /// is not a trustworthy predictor here. This test reports the numbers; it does NOT build the kernel —
    /// that decision is the user's per the same discipline that stopped the SmoothQuant alpha-tuning loop.
    /// Reuses the SAME Layer A(attn 1792×1792)/B(FFN 7168×1792) capture as W8A8_SmoothQuant_OfflineGate.</summary>
    [Fact]
    public void W8A8_GroupWeightQuant_QuadratureCheck()
    {
        string transformerDir = Path.Combine(T2VDir, "transformer");
        if (!Directory.Exists(transformerDir)) { _output.WriteLine($"SKIPPED: T2V transformer dir not found: {transformerDir} (set KANDINSKY5_T2V_DIR)."); return; }
        if (!File.Exists(TestPaths.Kandinsky5.PromptQwenEmbeds) || !File.Exists(TestPaths.Kandinsky5.PromptClipPooled))
        { _output.WriteLine("SKIPPED: pre-computed Qwen/CLIP embeddings missing (see dump_kandinsky5_embeddings.py)."); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(Kandinsky5W8A8OperandAblationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }
        if (!File.Exists(Path.Combine(ptxDir, "w8a8.ptx"))) { _output.WriteLine("SKIPPED: w8a8.ptx missing"); return; }
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        (Kandinsky5CheckpointConverter.ConvertedWeights converted, List<SafeTensorsLoader> loaders) =
            Kandinsky5CheckpointConverter.LoadDiffusersFolder(transformerDir);
        try
        {
            Dictionary<string, Tensor> tw = new(converted.Transformer.Count);
            foreach ((string k, Tensor v) in converted.Transformer)
                tw[k] = v.DType == DType.BF16 ? v.CastTo(DType.F16) : v;
            Kandinsky5Config config = Kandinsky5Config.VideoLite2B;
            using Kandinsky5Transformer transformer = new(config);
            transformer.LoadWeights(tw);

            CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            try
            {
                _output.WriteLine($"Device: {backend.Capabilities.Name} (CUDA_VISIBLE_DEVICES=" +
                    $"{Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES") ?? "unset"})");

                using Tensor qwen = LoadF32Tensor(TestPaths.Kandinsky5.PromptQwenEmbeds, config.InTextDim);
                using Tensor clip = LoadPooled(TestPaths.Kandinsky5.PromptClipPooled, config.InTextDim2);

                const int width = 512, height = 512, numFrames = 25;
                int tLat = (numFrames - 1) / 4 + 1, hLat = height / 8, wLat = width / 8;
                int latCh = config.InVisualDim;
                TensorShape latentShape = new TensorShape([1L, latCh, tLat, hLat, wLat]);
                TensorShape maskShape = new TensorShape([1L, 1, tLat, hLat, wLat]);

                FlowMatchEulerDiscreteScheduler scheduler = new(5.0f);
                scheduler.SetTimesteps(30);
                float midT = scheduler.Timesteps[15];
                (float scaleT, float scaleH, float scaleW) = Kandinsky5VideoPipeline.GetRopeScaleFactor(height, width);

                using Tensor noisy = SeedGenerator.CreateNoise(latentShape, seed: 42);
                using Tensor condLatent = Zeros(latentShape);
                using Tensor condMask = Zeros(maskShape);
                using Tensor packed = PackVisualCond(noisy, condLatent, condMask, latCh);

                backend.PreloadWeights(transformer.EnumerateWeights());

                // Single-timestep capture of shape-A and shape-B (mirrors W8A8_SmoothQuant_OfflineGate's
                // two-target hook, but only ONE pass needed here — the quadrature check doesn't need
                // multi-timestep activation stats, unlike SmoothQuant calibration).
                (int n, int k)? shapeA = null;
                float[]? aIn = null, aW = null, bIn = null, bW = null;
                int aM = 0, aK = 0, aN = 0, bK = 0, bN = 0;
                Action<float[], int, int, float[], int, Tensor> hook = null!;
                hook = (inArr, mm, kk, wArr, nn, wT) =>
                {
                    if (mm < 500 || kk < 1792) { CudaBackend.CaptureW8A8Operands = hook; return; }
                    bool matchesA = shapeA is { } sa ? (kk == sa.k && nn == sa.n) : aIn is null;
                    if (aIn is null && matchesA) { aIn = inArr; aW = wArr; aM = mm; aK = kk; aN = nn; }
                    else if (bIn is null && !matchesA) { bIn = inArr; bW = wArr; bK = kk; bN = nn; }
                    shapeA ??= aIn is not null ? (aN, aK) : null;
                    if (aIn is null || bIn is null) CudaBackend.CaptureW8A8Operands = hook;
                };
                CudaBackend.CaptureW8A8Operands = hook;
                backend.EnableW8A8 = true;
                Tensor outp = transformer.ForwardVideo(backend, packed, midT, qwen, clip, scaleT, scaleH, scaleW);
                backend.Sync();
                outp.Dispose();
                backend.FreeActivations(trimPool: false);
                backend.FreeWeights(transformer.EnumerateWeights());
                CudaBackend.CaptureW8A8Operands = null;

                Assert.True(aIn is not null, "No layer A captured.");
                _output.WriteLine($"Layer A: input[{aM},{aK}] weight[{aN},{aK}]");
                RunQuadratureCheck("A", aIn!, aM, aK, aW!, aN);
                if (bIn is not null)
                {
                    int bM = bIn.Length / bK;
                    _output.WriteLine($"Layer B: input[{bM},{bK}] weight[{bN},{bK}]");
                    RunQuadratureCheck("B", bIn, bM, bK, bW!, bN);
                }
                else
                {
                    _output.WriteLine("Layer B: not found (only one distinct deep-Linear shape encountered).");
                }
            }
            catch (Exception realEx)
            {
                _output.WriteLine($"REAL EXCEPTION (pre-Dispose): {realEx}");
                throw;
            }
            finally
            {
                CudaBackend.CaptureW8A8Operands = null;
                try { backend.Dispose(); }
                catch (Exception disposeEx) { _output.WriteLine($"DISPOSE-TIME EXCEPTION: {disposeEx}"); }
            }
        }
        finally { foreach (SafeTensorsLoader l in loaders) l.Dispose(); }
    }

    private void RunQuadratureCheck(string label, float[] act, int m, int k, float[] w, int n)
    {
        float[] refOut = MatMulTransposeB(act, m, k, w, n);

        float[] aOnlyOut = MatMulTransposeB(FakeQuantRowwise(act, m, k), m, k, w, n);
        double aOnly = RelL2(aOnlyOut, refOut);
        float[] wOnlyOut = MatMulTransposeB(act, m, k, FakeQuantPerChannel(w, n, k), n);
        double wOnlyBaseline = RelL2(wOnlyOut, refOut);
        float[] bothOut = MatMulTransposeB(FakeQuantRowwise(act, m, k), m, k, FakeQuantPerChannel(w, n, k), n);
        double bothBaseline = RelL2(bothOut, refOut);

        double quadraturePredicted = Math.Sqrt(aOnly * aOnly + wOnlyBaseline * wOnlyBaseline);
        _output.WriteLine($"[{label}] A-only={aOnly:e3}  W-only(per-row)={wOnlyBaseline:e3}  Both(per-row)={bothBaseline:e3}");
        _output.WriteLine($"[{label}] quadrature-predicted Both = sqrt(A²+W²) = {quadraturePredicted:e3} " +
            $"(actual/predicted = {bothBaseline / quadraturePredicted:F3})");

        double ceilingBoth = Math.Sqrt(aOnly * aOnly); // W-only -> 0 limit
        _output.WriteLine($"[{label}] perfect-weight-quant ceiling: Both -> {ceilingBoth:e3} " +
            $"({(1.0 - ceilingBoth / bothBaseline) * 100:F1}% max possible local relL2 reduction)");

        foreach (int groupSize in new[] { 128, 64, 32 })
        {
            if (k % groupSize != 0) continue;
            float[] wGroupOut = MatMulTransposeB(act, m, k, FakeQuantPerGroup(w, n, k, groupSize), n);
            double wGroupOnly = RelL2(wGroupOut, refOut);
            float[] bothGroupOut = MatMulTransposeB(FakeQuantRowwise(act, m, k), m, k, FakeQuantPerGroup(w, n, k, groupSize), n);
            double bothGroup = RelL2(bothGroupOut, refOut);
            _output.WriteLine($"[{label}] group={groupSize,4}: W-only={wGroupOnly:e3} " +
                $"({(wOnlyBaseline > 0 ? wGroupOnly / wOnlyBaseline : 1.0):F2}x per-row W-only)  " +
                $"Both={bothGroup:e3} ({(bothBaseline > 0 ? bothGroup / bothBaseline : 1.0):F2}x per-row Both, " +
                $"{(1.0 - bothGroup / bothBaseline) * 100:F1}% local relL2 reduction vs current production)");
        }
    }

    private void RunOfflineSmoothQuantSweep(string label, List<float[]> actPerStep, float[] w, int n, int k)
    {
        // Use the LAST probe (latest, most-drifted timestep — the one least like the others per the
        // Pearson-correlation finding) as the "test" activation; actMax is aggregated across ALL samples
        // (mirrors production: calibrate across the schedule, then serve any point on it).
        float[] xTest = actPerStep[^1];
        int m = xTest.Length / k;

        float[] actMax = new float[k];
        foreach (float[] act in actPerStep)
            for (int i = 0; i < act.Length; i++)
            {
                int c = i % k;
                float a = MathF.Abs(act[i]);
                if (a > actMax[c]) actMax[c] = a;
            }
        float[] wMax = new float[k];
        for (int ni = 0; ni < n; ni++)
        {
            int baseIdx = ni * k;
            for (int c = 0; c < k; c++)
            {
                float a = MathF.Abs(w[baseIdx + c]);
                if (a > wMax[c]) wMax[c] = a;
            }
        }

        float[] refOut = MatMulTransposeB(xTest, m, k, w, n);
        float[] unsmoothedBoth = MatMulTransposeB(FakeQuantRowwise(xTest, m, k), m, k, FakeQuantPerChannel(w, n, k), n);
        double unsmoothedRel = RelL2(unsmoothedBoth, refOut);
        float[] wOnly = MatMulTransposeB(xTest, m, k, FakeQuantPerChannel(w, n, k), n);
        double wOnlyRel = RelL2(wOnly, refOut);
        _output.WriteLine($"[{label}] unsmoothed Both relL2={unsmoothedRel:e3}, W-only floor relL2={wOnlyRel:e3}");

        foreach (double alpha in new[] { 0.3, 0.5, 0.7, 0.8, 0.9 })
        {
            float[] s = new float[k];
            for (int c = 0; c < k; c++)
            {
                double sv = actMax[c] > 0 && wMax[c] > 0
                    ? Math.Pow(actMax[c], alpha) / Math.Pow(wMax[c], 1.0 - alpha)
                    : 1.0;
                s[c] = (float)Math.Clamp(sv, 1e-3, 1e3);
            }
            float[] xHat = new float[xTest.Length];
            for (int r = 0; r < m; r++)
                for (int c = 0; c < k; c++)
                    xHat[r * k + c] = xTest[r * k + c] / s[c];
            float[] wHat = new float[w.Length];
            for (int ni = 0; ni < n; ni++)
                for (int c = 0; c < k; c++)
                    wHat[ni * k + c] = w[ni * k + c] * s[c];

            float[] smoothedBoth = MatMulTransposeB(FakeQuantRowwise(xHat, m, k), m, k, FakeQuantPerChannel(wHat, n, k), n);
            double smoothedRel = RelL2(smoothedBoth, refOut);
            _output.WriteLine($"[{label}] alpha={alpha:F1}: smoothed Both relL2={smoothedRel:e3} " +
                $"({(smoothedRel < unsmoothedRel ? "IMPROVED" : "WORSE")} vs unsmoothed, " +
                $"{(smoothedRel <= wOnlyRel * 1.1 ? "at/near W-only floor" : $"{smoothedRel / wOnlyRel:F2}x the W-only floor")})");
        }
    }

    /// <summary>Mirrors native/cuda/dequant/w8a8.cu w8a8_quant_rowwise_{f16,f32}: per-row absmax/127
    /// symmetric quant, round-to-nearest, clamp[-127,127], then immediately dequantized back to F32
    /// (fake-quant) so the error can be measured through a plain F32 GEMM.</summary>
    /// <summary>Fake-quant relL2 with vs without SmoothQuant smoothing for one Linear's real captured
    /// operands — the per-layer unit of work behind <c>W8A8_SmoothQuant_AllLayers_OfflineGate</c>.</summary>
    private static (double unsmoothed, double smoothed) ComputeSmoothQuantDelta(
        float[] act, int m, int k, float[] w, int n, float[] actMax, float[] wMax, double alpha)
    {
        float[] refOut = MatMulTransposeB(act, m, k, w, n);
        float[] unsmoothedOut = MatMulTransposeB(FakeQuantRowwise(act, m, k), m, k, FakeQuantPerChannel(w, n, k), n);
        double unsmoothedRel = RelL2(unsmoothedOut, refOut);

        float[] s = new float[k];
        for (int c = 0; c < k; c++)
        {
            double sv = actMax[c] > 0 && wMax[c] > 0 ? Math.Pow(actMax[c], alpha) / Math.Pow(wMax[c], 1.0 - alpha) : 1.0;
            s[c] = (float)Math.Clamp(sv, 1e-3, 1e3);
        }
        float[] xHat = new float[(long)m * k];
        for (int r = 0; r < m; r++)
            for (int c = 0; c < k; c++)
                xHat[r * k + c] = act[r * k + c] / s[c];
        float[] wHat = new float[w.Length];
        for (int ni = 0; ni < n; ni++)
            for (int c = 0; c < k; c++)
                wHat[ni * k + c] = w[ni * k + c] * s[c];
        float[] smoothedOut = MatMulTransposeB(FakeQuantRowwise(xHat, m, k), m, k, FakeQuantPerChannel(wHat, n, k), n);
        double smoothedRel = RelL2(smoothedOut, refOut);
        return (unsmoothedRel, smoothedRel);
    }

    private static float[] FakeQuantRowwise(float[] x, int rows, int cols)
    {
        float[] result = new float[x.Length];
        for (int r = 0; r < rows; r++)
        {
            int baseIdx = r * cols;
            float amax = 0f;
            for (int c = 0; c < cols; c++)
            {
                float a = MathF.Abs(x[baseIdx + c]);
                if (a > amax) amax = a;
            }
            float scale = amax > 0f ? amax / 127f : 1f;
            float inv = amax > 0f ? 127f / amax : 0f;
            for (int c = 0; c < cols; c++)
            {
                int iv = (int)MathF.Round(x[baseIdx + c] * inv);
                if (iv > 127) iv = 127;
                if (iv < -127) iv = -127;
                result[baseIdx + c] = iv * scale;
            }
        }
        return result;
    }

    /// <summary>Mirrors CudaBackend.QuantizeWeightForW8A8: per-output-channel (row of [N,K]) absmax/127
    /// symmetric quant, round-to-nearest, clamp[-127,127], then immediately dequantized back to F32.</summary>
    private static float[] FakeQuantPerChannel(float[] w, int n, int k)
    {
        float[] result = new float[w.Length];
        for (int ni = 0; ni < n; ni++)
        {
            int baseIdx = ni * k;
            float amax = 0f;
            for (int ki = 0; ki < k; ki++)
            {
                float a = MathF.Abs(w[baseIdx + ki]);
                if (a > amax) amax = a;
            }
            float scale = amax > 0f ? amax / 127f : 1f;
            float inv = amax > 0f ? 127f / amax : 0f;
            for (int ki = 0; ki < k; ki++)
            {
                int iv = (int)MathF.Round(w[baseIdx + ki] * inv);
                if (iv > 127) iv = 127;
                if (iv < -127) iv = -127;
                result[baseIdx + ki] = iv * scale;
            }
        }
        return result;
    }

    /// <summary>Per-group variant of <see cref="FakeQuantPerChannel"/>: instead of one absmax/127 scale
    /// per output row (over the full K), each row is split into contiguous groups of <paramref name="groupSize"/>
    /// input channels, each with its own scale — the standard group-quant scheme (GPTQ/AWQ-style), a
    /// candidate lever for the weight-quantization floor SmoothQuant can't touch. <paramref name="groupSize"/>
    /// must evenly divide k.</summary>
    private static float[] FakeQuantPerGroup(float[] w, int n, int k, int groupSize)
    {
        if (k % groupSize != 0) throw new ArgumentException($"groupSize {groupSize} does not evenly divide k={k}");
        int groupsPerRow = k / groupSize;
        float[] result = new float[w.Length];
        for (int ni = 0; ni < n; ni++)
        {
            int rowBase = ni * k;
            for (int g = 0; g < groupsPerRow; g++)
            {
                int gBase = rowBase + g * groupSize;
                float amax = 0f;
                for (int ki = 0; ki < groupSize; ki++)
                {
                    float a = MathF.Abs(w[gBase + ki]);
                    if (a > amax) amax = a;
                }
                float scale = amax > 0f ? amax / 127f : 1f;
                float inv = amax > 0f ? 127f / amax : 0f;
                for (int ki = 0; ki < groupSize; ki++)
                {
                    int iv = (int)MathF.Round(w[gBase + ki] * inv);
                    if (iv > 127) iv = 127;
                    if (iv < -127) iv = -127;
                    result[gBase + ki] = iv * scale;
                }
            }
        }
        return result;
    }

    /// <summary>out[M,N] = x[M,K] · w[N,K]^T (the exact operand layout CudaBackend.Linear uses).</summary>
    private static float[] MatMulTransposeB(float[] x, int m, int k, float[] w, int n)
    {
        float[] result = new float[(long)m * n];
        System.Threading.Tasks.Parallel.For(0, m, mi =>
        {
            int xBase = mi * k;
            int oBase = mi * n;
            for (int ni = 0; ni < n; ni++)
            {
                int wBase = ni * k;
                double sum = 0;
                for (int ki = 0; ki < k; ki++)
                    sum += (double)x[xBase + ki] * w[wBase + ki];
                result[oBase + ni] = (float)sum;
            }
        });
        return result;
    }

    private static double PearsonCorrelation(float[] a, float[] b)
    {
        double meanA = 0, meanB = 0;
        for (int i = 0; i < a.Length; i++) { meanA += a[i]; meanB += b[i]; }
        meanA /= a.Length; meanB /= b.Length;
        double cov = 0, varA = 0, varB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double da = a[i] - meanA, db = b[i] - meanB;
            cov += da * db; varA += da * da; varB += db * db;
        }
        return cov / Math.Sqrt(Math.Max(varA * varB, 1e-30));
    }

    private static double RelL2(float[] a, float[] b)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - b[i];
            num += d * d;
            den += (double)b[i] * b[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    private static Tensor Zeros(TensorShape shape)
    {
        Tensor t = new Tensor(shape, DType.F32);
        new Span<float>((float*)t.DataPointer, checked((int)shape.ElementCount)).Clear();
        return t;
    }

    /// <summary>Ports <c>Kandinsky5VideoPipeline.PackVisualCond</c> (private): concat <c>[noisy(16), cond(16), mask(1)]</c> along the channel axis.</summary>
    private static Tensor PackVisualCond(Tensor noisy, Tensor condLatent, Tensor condMask, int latCh)
    {
        long b = noisy.Shape[0], t = noisy.Shape[2], h = noisy.Shape[3], w = noisy.Shape[4];
        Tensor packed = new Tensor(new TensorShape([b, 2L * latCh + 1, t, h, w]), DType.F32);
        long per = t * h * w;
        float* dst = (float*)packed.DataPointer;
        float* pn = (float*)noisy.DataPointer;
        float* pc = (float*)condLatent.DataPointer;
        float* pm = (float*)condMask.DataPointer;
        long chOut = 2L * latCh + 1;
        for (long bi = 0; bi < b; bi++)
        {
            long dstBase = bi * chOut * per;
            Buffer.MemoryCopy(pn + bi * latCh * per, dst + dstBase, latCh * per * 4, latCh * per * 4);
            Buffer.MemoryCopy(pc + bi * latCh * per, dst + dstBase + latCh * per, latCh * per * 4, latCh * per * 4);
            Buffer.MemoryCopy(pm + bi * per, dst + dstBase + 2L * latCh * per, per * 4, per * 4);
        }
        return packed;
    }

    /// <summary>Raw headerless F32 <c>[seq, dim]</c> → <c>[1, seq, dim]</c>.</summary>
    private static Tensor LoadF32Tensor(string path, int embedDim)
    {
        byte[] data = File.ReadAllBytes(path);
        long totalFloats = data.Length / sizeof(float);
        if (totalFloats % embedDim != 0)
            throw new InvalidOperationException($"{path}: {totalFloats} floats not a multiple of {embedDim}.");
        int seqLen = (int)(totalFloats / embedDim);
        Tensor result = new Tensor(new TensorShape(1, seqLen, embedDim), DType.F32);
        fixed (byte* src = data) Buffer.MemoryCopy(src, (void*)result.DataPointer, data.Length, data.Length);
        return result;
    }

    /// <summary>Raw headerless F32 <c>[dim]</c> → <c>[1, dim]</c>.</summary>
    private static Tensor LoadPooled(string path, int embedDim)
    {
        byte[] data = File.ReadAllBytes(path);
        if (data.Length / sizeof(float) != embedDim)
            throw new InvalidOperationException($"{path}: expected {embedDim} floats.");
        Tensor result = new Tensor(new TensorShape(1, embedDim), DType.F32);
        fixed (byte* src = data) Buffer.MemoryCopy(src, (void*)result.DataPointer, data.Length, data.Length);
        return result;
    }
}
