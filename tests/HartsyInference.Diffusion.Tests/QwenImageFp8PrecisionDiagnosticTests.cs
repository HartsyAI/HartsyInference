using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>DIAGNOSTIC (not a regression gate) — investigates the cross-ordinal SSIM collapse documented in
/// <c>benchmarks/results/2026-08-05_multigpu_speeds.md</c>'s ‡ footnote (SM 8.9 4090 + SM 8.6 3060). Two facts,
/// both real Edit-2511 fp8mixed weights, 2026-08-06:
///
/// <para><b>1) <see cref="NativeFp8ActivationQuant_NotWeakCardWeightCast_ExplainsCrossOrdinalDivergence"/></b> —
/// isolates ONE real block at its REAL preceding-block input activation (streamed unsharded forward, block-by-block
/// preload/free so VRAM never exceeds ~1 block) on four independent <see cref="CudaBackend"/> instances (flags
/// fixed at construction, never mutated — rules out cast/scale-cache contamination): native-ordinal0 (default 4090,
/// native fp8 tensor cores), trueF32-ordinal0 (same GPU, <c>EnableNativeFp8Gemm=false</c> +
/// <c>EnableFp8F32Gemm=true</c> — the closest-to-exact GEMM available), bf16-ordinal1 (default 3060, the original
/// hypothesis's suspect), f32fix-ordinal1 (3060, <c>EnableFp8F32Gemm=true</c> — the original hypothesis's proposed
/// fix). At block 59 (|absmax| 8.4e7): bf16-ordinal1 vs native-ordinal0 relL2 = 3.51 (reproduces the ‡ footnote);
/// forcing f32fix-ordinal1 barely moves it (3.50) — REFUTES "the non-native stage's weight-cast precision is the
/// cause". Instead native-ordinal0 vs trueF32-ordinal0 (same GPU, only the native e4m3 ACTIVATION-quant path
/// differs) is itself 0.80 relL2 — the lossy step is the native fp8 GEMM's per-TENSOR dynamic e4m3 activation
/// quantization, and it runs on the 4090, in the unsharded baseline too. bf16-ordinal1 vs trueF32-ordinal0 is only
/// 2.0e-3; f32fix-ordinal1 vs trueF32-ordinal0 is 3.9e-5 — the "weak" card is far CLOSER to a high-precision
/// reference than the "strong" card's native path. A magnitude-scaling control (block 30, smaller |absmax| 3.56e7)
/// shows a smaller native-vs-trueF32 gap (0.19 vs 0.80) — backs the causal claim that this is an outlier-channel
/// per-TENSOR e4m3 quantization failure, not a generic fp8 weakness.</para>
///
/// <para><b>2) <see cref="WholeModelRegimeChange_NoSharding_NoBoundary_VelocityDivergence"/></b> — the follow-up
/// that shows the sharding mechanism is correct but explains why a stage-only code fix fails. Forcing the WHOLE
/// process (baseline AND sharded engine, via env var) onto one regime DID restore SSIM to 0.9922 on
/// <c>QwenImageDitShardingEngineTests</c>' real 1024² gate — proof the cross-device sharding itself is faithful
/// once both sides compute in the same precision regime. But wiring the equalization into the sharded engine
/// alone (leaving the separately-constructed unsharded-baseline engine on its default native path, since it never
/// populates <c>DitShardBackends</c>) measured SSIM 0.1017, WORSE than doing nothing (0.1797 default) — now
/// baseline and sharded are two cleanly-computed but different-regime images. This fact explains why matching
/// only works when BOTH sides match: running the SAME 60-block forward, no sharding, no boundary crossing at all,
/// once with <c>EnableNativeFp8Gemm=true</c> and once <c>false</c> (both fresh instances, same physical GPU, same
/// weights, same random input) already diverges at velocity relL2 0.93 — essentially as large as the actual
/// cross-device divergence. This is depth-compounding, not chaos: block 59 alone already measured relL2 0.80
/// against a true-F32 reference in isolation (fact 1), and that per-block error compounds across all 60 blocks of
/// one forward pass — not an iterative-sampling instability (this is a single denoise step, no trajectory
/// feedback). "Equalizing" the regime across shard stages only, while a differently-constructed baseline stays on
/// a different regime, does not make the two match. Same class of finding as <c>MiniMaxH3Recipe</c>'s
/// already-accepted cross-device limitation (its own footnote: whole-model ordinal-0-vs-ordinal-1 SSIM 0.1377, no
/// sharding at all). No code fix is wired into <c>QwenImageRecipe.cs</c> as a result — see that file's comment at
/// the DiT-sharding block for the pointer back here.</para></summary>
/// <remarks>Same static-init caveat as <c>QwenImageDitShardingEngineTests.DitSharding_NonResident_...</c>: both
/// facts drive <c>QwenImageTransformer.Forward</c>, which latches <c>QWEN_IMAGE_DEBUG_DIR</c> into
/// <c>QwenImageDebugDump</c>'s process-static <c>_dumpDir</c> field on first touch. Fact 1 sets/clears the env var
/// and deletes its own dump dir when done; if fact 2 runs in the SAME process afterward, its <c>Forward</c> call
/// still dumps into fact 1's (now-deleted) directory and throws <c>DirectoryNotFoundException</c> — a test-process
/// artifact, not a finding. Run each fact filter-isolated (`--filter FullyQualifiedName~<MethodName>`) rather than
/// by class name.</remarks>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class QwenImageFp8PrecisionDiagnosticTests
{
    private readonly ITestOutputHelper _output;
    public QwenImageFp8PrecisionDiagnosticTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public unsafe void NativeFp8ActivationQuant_NotWeakCardWeightCast_ExplainsCrossOrdinalDivergence()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.QwenImageEdit.Edit2511Fp8;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(QwenImageFp8PrecisionDiagnosticTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        string dumpDir = Path.Combine(Path.GetTempPath(), $"qwen-fp8-diag-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dumpDir);
        Stopwatch sw = Stopwatch.StartNew();
        try
        {
            _output.WriteLine($"[1/4] Loading Qwen-Image Edit fp8 checkpoint from {checkpoint}...");
            (QwenImageCheckpointConverter.ConvertedWeights converted, HartsyInference.ModelAssets.SafeTensors.SafeTensorsLoader loader) =
                QwenImageCheckpointConverter.LoadAndConvert(checkpoint);
            try
            {
                QwenImageConfig config = QwenImageConfig.V1;
                QwenImageTransformer transformer = new(config);
                transformer.LoadWeights(converted.Transformer);
                _output.WriteLine($"  loaded {converted.Transformer.Count} tensors, {sw.Elapsed.TotalSeconds:F1}s");

                using CudaBackend streamBackend = new(deviceOrdinal: 0, ptxDir: ptxDir);
                streamBackend.CacheWeightCasts = false;
                if (!streamBackend.EnableNativeFp8Gemm)
                {
                    _output.WriteLine("SKIPPED: ordinal 0 on this box is not a native-fp8 (SM 8.9+) device — this diagnostic assumes this box's known topology.");
                    return;
                }

                const int hPacked = 16, wPacked = 16;
                int patchDim = config.PatchSize * config.PatchSize * config.InChannels;
                using Tensor packedLatent = RandF32(new TensorShape(1, hPacked * wPacked, patchDim), seed: 1, scale: 0.5f);
                using Tensor encoderHidden = RandF32(new TensorShape(1, 32, config.ContextDim), seed: 2, scale: 0.2f);

                Environment.SetEnvironmentVariable("QWEN_IMAGE_DEBUG_DIR", dumpDir);
                streamBackend.PreloadWeights(transformer.EnumerateSharedWeights());
                int loadedBlock = -1;
                transformer.BeforeBlockForward = i =>
                {
                    if (loadedBlock >= 0) streamBackend.FreeWeights(transformer.EnumerateBlockRangeWeights(loadedBlock, loadedBlock + 1));
                    streamBackend.PreloadWeights(transformer.EnumerateBlockRangeWeights(i, i + 1));
                    loadedBlock = i;
                };
                _output.WriteLine($"[2/4] Streaming reference forward (all {config.Depth} blocks, ordinal 0, ~1 block resident at a time, dumping every block's I/O)...");
                using Tensor referenceVelocity = transformer.Forward(streamBackend, packedLatent, encoderHidden, timestep: 0.5f, hPacked, wPacked);
                AssertFinite(referenceVelocity);
                transformer.BeforeBlockForward = null;
                Environment.SetEnvironmentVariable("QWEN_IMAGE_DEBUG_DIR", null);
                _output.WriteLine($"  streamed forward done, {sw.Elapsed.TotalSeconds:F1}s total.");

                TensorShape imgShape = new(1, hPacked * wPacked, config.HiddenSize);
                TensorShape txtShape = new(1, 32, config.HiddenSize);
                TensorShape tembShape = new(1, config.HiddenSize);
                QwenImageRope rope = new(theta: config.RopeTheta, ropeText: config.RopeText);
                int txtPositionStart = QwenImageRope.ComputeTextPositionStart(hPacked, wPacked);

                // Four INDEPENDENT backend instances (two per physical GPU) — flags fixed at construction, never
                // mutated mid-test, so no run can contaminate another via a shared cast/input-scale cache keyed to
                // the same instance (EnsureFp8InputScaleDev, weight-cast cache).
                using CudaBackend nativeBackend = new(deviceOrdinal: 0, ptxDir: ptxDir);
                nativeBackend.CacheWeightCasts = false;
                using CudaBackend trueF32Backend = new(deviceOrdinal: 0, ptxDir: ptxDir)
                {
                    CacheWeightCasts = false,
                    EnableNativeFp8Gemm = false,
                    EnableFp8F32Gemm = true,
                };
                using CudaBackend bf16Backend = new(deviceOrdinal: 1, ptxDir: ptxDir);
                bf16Backend.CacheWeightCasts = false;
                using CudaBackend f32FixBackend = new(deviceOrdinal: 1, ptxDir: ptxDir)
                {
                    CacheWeightCasts = false,
                    EnableFp8F32Gemm = true,
                };

                _output.WriteLine("[3/4] block 59 (deepest — matches the ‡ footnote's Control B split=59) at its REAL block-58 input:");
                RunBlockComparison(59, config, converted, rope, hPacked, wPacked, txtPositionStart, imgShape, txtShape, tembShape,
                    dumpDir, nativeBackend, trueF32Backend, bf16Backend, f32FixBackend);

                _output.WriteLine("[4/4] Magnitude-scaling control — block 30 (mid-depth, smaller activation) at its REAL block-29 input:");
                RunBlockComparison(30, config, converted, rope, hPacked, wPacked, txtPositionStart, imgShape, txtShape, tembShape,
                    dumpDir, nativeBackend, trueF32Backend, bf16Backend, f32FixBackend);
            }
            finally
            {
                loader.Dispose();
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("QWEN_IMAGE_DEBUG_DIR", null);
            try { Directory.Delete(dumpDir, recursive: true); } catch { /* best-effort cleanup, not test-relevant */ }
        }
    }

    /// <summary>Discriminating follow-up: does the native-vs-BF16-cast GEMM regime alone (NO sharding, NO boundary
    /// crossing at all — one physical GPU, one backend instance per run) move the WHOLE-MODEL output enough to
    /// explain image-level SSIM collapse by itself? If yes, "make the regime uniform across shard stages" only
    /// makes two runs agree with each other while both stay far from a high-precision trajectory — informational
    /// reclassification (MiniMax-H3's existing precedent) is the honest call, not a fix. If the whole-model relL2
    /// is small, the per-block error measured in the other fact does NOT propagate on its own, which means the
    /// production 0.18 really was specifically the boundary discontinuity (mixing regimes mid-trajectory), and
    /// equalizing the regime is worth wiring all the way to the baseline path.</summary>
    [Fact]
    public unsafe void WholeModelRegimeChange_NoSharding_NoBoundary_VelocityDivergence()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string checkpoint = TestPaths.QwenImageEdit.Edit2511Fp8;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(QwenImageFp8PrecisionDiagnosticTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[1/3] Loading Qwen-Image Edit fp8 checkpoint from {checkpoint}...");
        (QwenImageCheckpointConverter.ConvertedWeights converted, HartsyInference.ModelAssets.SafeTensors.SafeTensorsLoader loader) =
            QwenImageCheckpointConverter.LoadAndConvert(checkpoint);
        try
        {
            QwenImageConfig config = QwenImageConfig.V1;
            const int hPacked = 16, wPacked = 16;
            int patchDim = config.PatchSize * config.PatchSize * config.InChannels;
            using Tensor packedLatent = RandF32(new TensorShape(1, hPacked * wPacked, patchDim), seed: 1, scale: 0.5f);
            using Tensor encoderHidden = RandF32(new TensorShape(1, 32, config.ContextDim), seed: 2, scale: 0.2f);

            using Tensor nativeVelocity = RunStreamedWholeModel(converted, config, packedLatent, encoderHidden, hPacked, wPacked,
                ptxDir, nativeFp8: true);
            _output.WriteLine($"  [2/3] native-fp8 whole-model forward done, {sw.Elapsed.TotalSeconds:F1}s total.");
            using Tensor nonNativeVelocity = RunStreamedWholeModel(converted, config, packedLatent, encoderHidden, hPacked, wPacked,
                ptxDir, nativeFp8: false);
            _output.WriteLine($"  [3/3] non-native whole-model forward done, {sw.Elapsed.TotalSeconds:F1}s total.");

            double relL2 = RelL2(nonNativeVelocity, nativeVelocity);
            _output.WriteLine($"SUMMARY: whole-model (60 blocks, no sharding, no boundary) native-vs-nonnative velocity relL2 = {relL2:E3}");
            _output.WriteLine(relL2 > 0.05
                ? "  => LARGE: the regime change alone is enough to diverge the output — depth-compounded per-block error, not a boundary-specific defect."
                : "  => SMALL: the per-block error measured in isolation does not propagate on its own — the boundary discontinuity is the real mechanism.");
        }
        finally
        {
            loader.Dispose();
        }
    }

    private static unsafe Tensor RunStreamedWholeModel(QwenImageCheckpointConverter.ConvertedWeights converted,
        QwenImageConfig config, Tensor packedLatent, Tensor encoderHidden, int hPacked, int wPacked, string ptxDir,
        bool nativeFp8)
    {
        QwenImageTransformer transformer = new(config);
        transformer.LoadWeights(converted.Transformer);
        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir) { CacheWeightCasts = false, EnableNativeFp8Gemm = nativeFp8 };

        backend.PreloadWeights(transformer.EnumerateSharedWeights());
        int loadedBlock = -1;
        transformer.BeforeBlockForward = i =>
        {
            if (loadedBlock >= 0) backend.FreeWeights(transformer.EnumerateBlockRangeWeights(loadedBlock, loadedBlock + 1));
            backend.PreloadWeights(transformer.EnumerateBlockRangeWeights(i, i + 1));
            loadedBlock = i;
        };
        Tensor velocity = transformer.Forward(backend, packedLatent, encoderHidden, timestep: 0.5f, hPacked, wPacked);
        AssertFinite(velocity);
        transformer.BeforeBlockForward = null;
        transformer.Dispose();
        Tensor host = Materialize(velocity);
        velocity.Dispose();
        return host;
    }

    private unsafe void RunBlockComparison(int blockIdx, QwenImageConfig config,
        QwenImageCheckpointConverter.ConvertedWeights converted, QwenImageRope rope, int hPacked, int wPacked,
        int txtPositionStart, TensorShape imgShape, TensorShape txtShape, TensorShape tembShape, string dumpDir,
        CudaBackend nativeBackend, CudaBackend trueF32Backend, CudaBackend bf16Backend, CudaBackend f32FixBackend)
    {
        string imgPath = Path.Combine(dumpDir, "layers", $"block_{blockIdx - 1}_image.bin");
        string txtPath = Path.Combine(dumpDir, "layers", $"block_{blockIdx - 1}_text.bin");
        string tembPath = Path.Combine(dumpDir, "layers", "time_text_embed.bin");
        Assert.True(File.Exists(imgPath), $"missing dump: {imgPath}");
        Assert.True(File.Exists(txtPath), $"missing dump: {txtPath}");
        Assert.True(File.Exists(tembPath), $"missing dump: {tembPath}");

        using Tensor probe = LoadRawF32(imgPath, imgShape);
        _output.WriteLine($"  block {blockIdx} image-stream input |absmax| = {AbsMax(probe):E3}");

        QwenImageBlock block = new(config.HiddenSize, config.NumHeads, config.HeadDim,
            (int)(config.HiddenSize * config.MlpRatio), config.QkNormEps);
        block.LoadWeights(converted.Transformer, $"transformer_blocks.{blockIdx}");

        (Tensor nativeImg, Tensor nativeTxt) = RunOne(nativeBackend, block, imgPath, txtPath, tembPath, imgShape, txtShape, tembShape, rope, hPacked, wPacked, txtPositionStart);
        (Tensor trueImg, Tensor trueTxt) = RunOne(trueF32Backend, block, imgPath, txtPath, tembPath, imgShape, txtShape, tembShape, rope, hPacked, wPacked, txtPositionStart);
        (Tensor bf16Img, Tensor bf16Txt) = RunOne(bf16Backend, block, imgPath, txtPath, tembPath, imgShape, txtShape, tembShape, rope, hPacked, wPacked, txtPositionStart);
        (Tensor f32Img, Tensor f32Txt) = RunOne(f32FixBackend, block, imgPath, txtPath, tembPath, imgShape, txtShape, tembShape, rope, hPacked, wPacked, txtPositionStart);

        double nativeVsTrue = RelL2(nativeImg, trueImg);
        double bf16VsNative = RelL2(bf16Img, nativeImg);
        double f32VsNative = RelL2(f32Img, nativeImg);
        double bf16VsTrue = RelL2(bf16Img, trueImg);
        double f32VsTrue = RelL2(f32Img, trueImg);
        _output.WriteLine($"    native-ordinal0 vs trueF32-ordinal0 (isolates native e4m3 activation quant): img relL2={nativeVsTrue:E3}");
        _output.WriteLine($"    bf16-ordinal1 vs native-ordinal0 (original hypothesis's comparison):        img relL2={bf16VsNative:E3}");
        _output.WriteLine($"    f32fix-ordinal1 vs native-ordinal0 (proposed fix's comparison):              img relL2={f32VsNative:E3}");
        _output.WriteLine($"    bf16-ordinal1 vs trueF32-ordinal0 (weak card vs high-precision reference):   img relL2={bf16VsTrue:E3}");
        _output.WriteLine($"    f32fix-ordinal1 vs trueF32-ordinal0:                                          img relL2={f32VsTrue:E3}");

        nativeImg.Dispose(); nativeTxt.Dispose();
        trueImg.Dispose(); trueTxt.Dispose();
        bf16Img.Dispose(); bf16Txt.Dispose();
        f32Img.Dispose(); f32Txt.Dispose();
    }

    private static unsafe (Tensor img, Tensor txt) RunOne(CudaBackend backend, QwenImageBlock block,
        string imgPath, string txtPath, string tembPath, TensorShape imgShape, TensorShape txtShape, TensorShape tembShape,
        QwenImageRope rope, int hPacked, int wPacked, int txtPositionStart)
    {
        backend.PreloadWeights(block.EnumerateWeights());
        using Tensor img = LoadRawF32(imgPath, imgShape);
        using Tensor txt = LoadRawF32(txtPath, txtShape);
        using Tensor temb = LoadRawF32(tembPath, tembShape);
        (Tensor outImg, Tensor outTxt) = block.Forward(backend, img, txt, temb, rope, hPacked, wPacked, txtPositionStart);
        // Materialize to a fresh host tensor immediately — DataPointer's EnsureCpuData is lazy, and this
        // instance's backend/flags must not be touched again before every comparison reads these bytes.
        Tensor hostImg = Materialize(outImg);
        Tensor hostTxt = Materialize(outTxt);
        outImg.Dispose();
        outTxt.Dispose();
        return (hostImg, hostTxt);
    }

    private static unsafe Tensor Materialize(Tensor t)
    {
        Tensor copy = new(t.Shape, DType.F32);
        using Tensor f32 = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        Buffer.MemoryCopy(f32.DataPointer, copy.DataPointer, copy.ElementCount * sizeof(float), f32.ElementCount * sizeof(float));
        return copy;
    }

    private static unsafe Tensor RandF32(TensorShape s, int seed, float scale)
    {
        Tensor t = new(s, DType.F32);
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        long n = s.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private static unsafe Tensor LoadRawF32(string path, TensorShape shape)
    {
        Tensor t = new(shape, DType.F32);
        byte[] bytes = File.ReadAllBytes(path);
        long expectedBytes = shape.ElementCount * sizeof(float);
        Assert.True(bytes.LongLength == expectedBytes, $"{path}: expected {expectedBytes} bytes, got {bytes.LongLength}.");
        fixed (byte* src = bytes)
        {
            Buffer.MemoryCopy(src, t.DataPointer, expectedBytes, expectedBytes);
        }
        return t;
    }

    private static unsafe double AbsMax(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        double m = 0;
        for (long i = 0; i < t.ElementCount; i++) m = Math.Max(m, Math.Abs((double)p[i]));
        return m;
    }

    private static unsafe void AssertFinite(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }

    private static unsafe double RelL2(Tensor a, Tensor b)
    {
        float* pa = (float*)a.DataPointer, pb = (float*)b.DataPointer;
        double num = 0, den = 0;
        for (long i = 0; i < a.ElementCount; i++)
        {
            double d = pa[i] - pb[i];
            num += d * d;
            den += (double)pb[i] * pb[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }
}
