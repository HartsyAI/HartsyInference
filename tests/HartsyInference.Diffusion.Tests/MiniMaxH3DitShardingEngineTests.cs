using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>MiniMax-H3 DiT sharding through the FULL engine path (<c>InferenceEngine</c> → <c>MiniMaxH3Recipe</c> →
/// <c>MiniMaxH3RecipePipeline</c>) with <see cref="PlacementConfig.EnableDitSharding"/> — a real short
/// text-to-video generation, not the synthetic bit-parity fixture <see cref="MiniMaxH3DitShardingTests"/> covers,
/// nor the raw-transformer VRAM-pooling proof <see cref="MiniMaxH3DitShardingVramTests"/> covers.</summary>
/// <remarks><para>A real unsharded baseline IS obtainable on this box: the ~19.5 GB fp8 DiT and the ~14.6 GB text
/// encoder are never co-resident (<c>MiniMaxH3RecipePipeline</c> frees the encoder before the DiT denoise loop
/// runs), and the DiT alone preloads resident on the 4090 (measured ~50 ms/step, not the multi-second-per-step
/// signature of block-streaming).</para>
/// <para><b>History of the cross-device gate (rewritten 2026-08-06 — the original causal story was wrong).</b>
/// The cross-device run used to score SSIM ~0.17 (a regular-grid mosaic), and both it AND its justifying control
/// (whole model unsharded on ordinal 1 → SSIM ≈ 0.14) were blamed on SM 8.6 fp8-dequant rounding amplified by
/// this checkpoint's huge residuals. The REAL cause of both numbers was a bug: <c>Modulate</c> emits e4m3
/// activations (stored value = real/input_scale) with the fp8-emit flag on regardless of SM, but only the
/// native-fp8 GEMM branch folded <c>input.Fp8ScaleFactor</c> back into alpha — the SM 8.6 fallback dequantized
/// scale-blind, so every tail-block <c>qkv</c>/<c>fc1</c> output on the 3060 was off by its input-scale factor.
/// Confirmed by experiment (2026-08-06): disabling the emit (<c>HARTSY_MODULATE_EMIT_FP8=0</c>) raised
/// cross-device SSIM 0.17 → 0.9795; after folding the scale in <c>CudaBackend.LinearImpl</c>'s fallback,
/// defaults measure <b>0.9597</b> (emission on, its perf kept). The residual gap from 1.0 IS the genuine SM 8.6
/// story: BF16/F16-fallback GEMMs at residual magnitudes ~2.7e6 (see <c>MiniMaxH3Recipe</c>), plus e4m3
/// re-quantization of modulated activations. Gates: same-device split SSIM > 0.99 (sharding math exact —
/// measured 1.0000), cross-device SSIM > 0.90 (real bar again; measured 0.9597 default / 0.9795 emit-off),
/// plus coherence + two-card VRAM-pooling bounds.</para></remarks>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class MiniMaxH3DitShardingEngineTests
{
    private readonly ITestOutputHelper _output;
    public MiniMaxH3DitShardingEngineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task DitSharding_RealEngine_SameDeviceSplitIsExact_CrossDevicePoolsVramAndStaysCoherent()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        string checkpoint = TestPaths.MiniMaxH3.DitFp8;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(MiniMaxH3DitShardingEngineTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        (double free0, long total0, double free1, long total1) = ProbeFreeGb(ptxDir);
        _output.WriteLine($"Free VRAM — ordinal 0: {free0:F2}/{total0 >> 30} GB, ordinal 1: {free1:F2}/{total1 >> 30} GB.");
        if (free0 + free1 < 26.0 || Math.Min(free0, free1) < 6.0)
        {
            _output.WriteLine("SKIPPED: not enough pooled VRAM for the ~19.5 GB DiT across both cards.");
            return;
        }

        // nvidia-smi reports its own device index, independent of and NOT necessarily matching the CUDA driver's
        // ordinal assignment — match by total capacity (24 GB vs 12 GB here) instead of assuming index order.
        long[] nvidiaSmiTotalMib = QueryTotalMib();
        int smiIndexForOrdinal0 = ClosestByTotal(nvidiaSmiTotalMib, total0);
        int smiIndexForOrdinal1 = ClosestByTotal(nvidiaSmiTotalMib, total1);
        bool canSampleVram = smiIndexForOrdinal0 >= 0 && smiIndexForOrdinal1 >= 0 && smiIndexForOrdinal0 != smiIndexForOrdinal1;
        _output.WriteLine(canSampleVram
            ? $"nvidia-smi index for ordinal 0: {smiIndexForOrdinal0}; ordinal 1: {smiIndexForOrdinal1}."
            : "nvidia-smi unavailable or ambiguous — VRAM pooling assertion will be skipped.");

        ModelSpec spec = ModelResolver.Resolve("minimax-h3", checkpoint, Modality.Video);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: minimax-h3 not resolvable with the explicit path."); return; }

        // Smallest real geometry the H3 grid allows: Frames=5 is already frame-aligned (17k+5, k=0) giving 2 video
        // latent frames and 8 audio latent frames; 256x160 is the 32-pixel-aligned canvas VramTests already probed.
        // Steps=12: measured identical (~0.17) cross-device SSIM at Steps=4 and Steps=12, ruling out "unconverged
        // flow-match trajectory" as the explanation — the divergence is the fp8 dequant path, not step count.
        VideoRequest request = new VideoRequest
        {
            Prompt = "a red ball bouncing on a wooden table",
            Width = 256,
            Height = 160,
            Frames = 5,
            Steps = 12,
            CfgScale = 1.0f,
            Seed = 42,
        };

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine("[1/3] UNSHARDED baseline (ordinal 0 alone — the ~19.5 GB DiT resident; the TE is freed " +
            "before the DiT loads, so co-residency is not the constraint)...");
        VideoGenerationResult baseline;
        using (InferenceEngine unshardedEngine = new InferenceEngine("cuda", 0))
        {
            baseline = await unshardedEngine.Video.GenerateAsync(spec, request);
        }
        _output.WriteLine($"  baseline: {baseline.Frames.Count} frames, {sw.Elapsed.TotalSeconds:F1}s");
        AssertCoherent(baseline, "baseline");
        AssertAudioCoherent(baseline.Audio, "baseline");

        sw.Restart();
        _output.WriteLine("[2/3] SAME-DEVICE split (both shard backends resolve to ordinal 0 — proves the block-range " +
            "hand-off math itself, with no physical cross-device boundary at all)...");
        PlacementConfig sameDevicePlacement = new PlacementConfig { ShardDevices = ["cuda:0", "cuda:0"], EnableDitSharding = true };
        VideoGenerationResult sameDeviceSharded;
        using (InferenceEngine sameDeviceEngine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = sameDevicePlacement }))
        {
            sameDeviceSharded = await sameDeviceEngine.Video.GenerateAsync(spec, request);
        }
        _output.WriteLine($"  same-device sharded: {sameDeviceSharded.Frames.Count} frames, {sw.Elapsed.TotalSeconds:F1}s");
        AssertCoherent(sameDeviceSharded, "same-device sharded");
        AssertAudioCoherent(sameDeviceSharded.Audio, "same-device sharded");
        Assert.Equal(baseline.Frames.Count, sameDeviceSharded.Frames.Count);
        double sameDeviceSsim = MeanSsim(baseline.Frames, sameDeviceSharded.Frames);
        _output.WriteLine($"  mean SSIM(baseline, same-device sharded) = {sameDeviceSsim:F4}");
        Assert.True(sameDeviceSsim > 0.99, $"same-device split diverged from the unsharded baseline (SSIM={sameDeviceSsim:F4}) " +
            "— the block-range hand-off (video/audio/text/temb CopyFromPeer) has a real defect, since this case has " +
            "no physical cross-device boundary and no fp8-path difference to blame.");

        sw.Restart();
        _output.WriteLine("[3/3] CROSS-DEVICE split (50-block loop, cuda:0 + cuda:1 — the real 2-GPU pooling case)...");
        long[] baselineMib = canSampleVram ? QueryUsedMib() : [];
        long[] peakMib = [.. baselineMib];
        using CancellationTokenSource samplerCts = new();
        Task sampler = canSampleVram
            ? Task.Run(async () =>
            {
                while (!samplerCts.IsCancellationRequested)
                {
                    long[] now = QueryUsedMib();
                    for (int i = 0; i < peakMib.Length && i < now.Length; i++) peakMib[i] = Math.Max(peakMib[i], now[i]);
                    try { await Task.Delay(500, samplerCts.Token); } catch (OperationCanceledException) { }
                }
            })
            : Task.CompletedTask;

        PlacementConfig crossDevicePlacement = new PlacementConfig { ShardDevices = ["cuda:0", "cuda:1"], EnableDitSharding = true };
        VideoGenerationResult crossDeviceSharded;
        using (InferenceEngine crossDeviceEngine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = crossDevicePlacement }))
        {
            crossDeviceSharded = await crossDeviceEngine.Video.GenerateAsync(spec, request);
        }
        samplerCts.Cancel();
        await sampler;
        _output.WriteLine($"  cross-device sharded: {crossDeviceSharded.Frames.Count} frames, {sw.Elapsed.TotalSeconds:F1}s");
        AssertCoherent(crossDeviceSharded, "cross-device sharded");
        AssertAudioCoherent(crossDeviceSharded.Audio, "cross-device sharded");

        if (canSampleVram)
        {
            long riseOrdinal0 = peakMib[smiIndexForOrdinal0] - baselineMib[smiIndexForOrdinal0];
            long riseOrdinal1 = peakMib[smiIndexForOrdinal1] - baselineMib[smiIndexForOrdinal1];
            _output.WriteLine($"  Peak VRAM rise during cross-device generation — ordinal 0: +{riseOrdinal0} MiB, ordinal 1: +{riseOrdinal1} MiB.");
            Assert.True(riseOrdinal0 > 3000,
                $"ordinal 0 (primary, blocks [0,34)) rose only {riseOrdinal0} MiB — expected several GB for its split share.");
            Assert.True(riseOrdinal1 > 1500,
                $"ordinal 1 (shard, blocks [34,50)) rose only {riseOrdinal1} MiB — the split share never landed there.");
            Assert.True(riseOrdinal0 < 18000,
                $"ordinal 0 rose {riseOrdinal0} MiB — that is close to the WHOLE ~19.5 GB DiT, not just its split share.");
        }
        else
        {
            _output.WriteLine("nvidia-smi unavailable — VRAM pooling assertion skipped (coherence + audio checks still ran).");
        }

        // REAL gate again as of 2026-08-06 (was an informational 0.05 floor while the mosaic bug lived): the
        // fp8 input-scale fold fix in CudaBackend.LinearImpl's fallback measured 0.9597 here at defaults.
        // Remaining divergence is the genuine SM 8.6 non-native-fp8 regime (see class remarks); 0.90 leaves
        // margin for that while catching any regression of the scale fold or the boundary hand-off.
        Assert.Equal(baseline.Frames.Count, crossDeviceSharded.Frames.Count);
        double crossDeviceSsim = MeanSsim(baseline.Frames, crossDeviceSharded.Frames);
        _output.WriteLine($"  mean SSIM(baseline, cross-device sharded) = {crossDeviceSsim:F4} (GATED > 0.90; measured 0.9597 at defaults)");
        Assert.True(crossDeviceSsim > 0.90, $"cross-device sharded output regressed (SSIM={crossDeviceSsim:F4}, " +
            "measured 0.9597 when this gate was restored) — check the LinearImpl fp8 input-scale fold " +
            "(the 2026-08-06 mosaic root cause) and the block-range hand-off.");
    }

    private static void AssertCoherent(VideoGenerationResult result, string label)
    {
        Assert.True(result.Frames.Count > 0, $"{label}: no frames decoded");
        foreach (VideoFrame frame in result.Frames)
        {
            int nonZero = frame.Rgb.Count(b => b != 0), nonFF = frame.Rgb.Count(b => b != 255);
            Assert.True(nonZero > frame.Rgb.Length * 0.1, $"{label}: frame {frame.Index} is all black");
            Assert.True(nonFF > frame.Rgb.Length * 0.1, $"{label}: frame {frame.Index} is all white");
        }
    }

    private static void AssertAudioCoherent(AudioBuffer? audio, string label)
    {
        Assert.True(audio is not null && !audio.IsEmpty, $"{label}: expected a non-empty soundtrack (H3 emits one with every clip)");
        foreach (float[] channel in audio!.Channels)
        {
            foreach (float sample in channel)
            {
                Assert.True(float.IsFinite(sample), $"{label}: non-finite audio sample");
            }
        }
    }

    private static double MeanSsim(IReadOnlyList<VideoFrame> a, IReadOnlyList<VideoFrame> b)
    {
        double sum = 0;
        for (int i = 0; i < a.Count; i++)
        {
            sum += Ssim.Compute(a[i].Rgb, b[i].Rgb, a[i].Width, a[i].Height);
        }
        return sum / a.Count;
    }

    private static (double free0Gb, long total0Bytes, double free1Gb, long total1Bytes) ProbeFreeGb(string ptxDir)
    {
        using CudaBackend probe0 = new(deviceOrdinal: 0, ptxDir: ptxDir);
        using CudaBackend probe1 = new(deviceOrdinal: 1, ptxDir: ptxDir);
        (nuint free0, nuint total0) = probe0.Context.GetMemoryInfo();
        (nuint free1, nuint total1) = probe1.Context.GetMemoryInfo();
        return (free0 / (1024.0 * 1024.0 * 1024.0), (long)total0, free1 / (1024.0 * 1024.0 * 1024.0), (long)total1);
    }

    /// <summary>Index into the <c>nvidia-smi</c> row list whose reported total capacity is closest to
    /// <paramref name="cudaTotalBytes"/>, or -1 when nvidia-smi is unavailable.</summary>
    private static int ClosestByTotal(long[] nvidiaSmiTotalMib, long cudaTotalBytes)
    {
        if (nvidiaSmiTotalMib.Length == 0)
        {
            return -1;
        }
        long cudaTotalMib = cudaTotalBytes / (1024 * 1024);
        int best = 0;
        long bestDiff = long.MaxValue;
        for (int i = 0; i < nvidiaSmiTotalMib.Length; i++)
        {
            long diff = Math.Abs(nvidiaSmiTotalMib[i] - cudaTotalMib);
            if (diff < bestDiff)
            {
                bestDiff = diff;
                best = i;
            }
        }
        return best;
    }

    /// <summary>Per-card total VRAM in MiB from <c>nvidia-smi</c>, or empty when unavailable.</summary>
    private static long[] QueryTotalMib() => QueryMib("--query-gpu=memory.total");

    /// <summary>Per-card used VRAM in MiB from <c>nvidia-smi</c>, or empty when unavailable.</summary>
    private static long[] QueryUsedMib() => QueryMib("--query-gpu=memory.used");

    private static long[] QueryMib(string query)
    {
        try
        {
            using Process p = Process.Start(new ProcessStartInfo("nvidia-smi", $"{query} --format=csv,noheader,nounits")
            { RedirectStandardOutput = true, UseShellExecute = false })!;
            string output = p.StandardOutput.ReadToEnd();
            p.WaitForExit(5000);
            return [.. output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(line => long.TryParse(line, out long v) ? v : 0)];
        }
        catch (Exception)
        {
            return [];
        }
    }
}
