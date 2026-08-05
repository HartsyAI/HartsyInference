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
/// <para><b>The sibling models' "SSIM > 0.75 vs baseline" gate does NOT apply to MiniMax-H3 and is deliberately not
/// used here — two control experiments proved why.</b> (1) Splitting <c>ForwardSharded</c> across two backends that
/// are BOTH ordinal 0 (no physical cross-device boundary at all) reproduces the baseline at SSIM 1.0000 — the
/// block-range split math is exact. (2) Running the model with NO sharding whatsoever, once entirely on ordinal 0
/// (native fp8 GEMM, SM 8.9) and once entirely on ordinal 1 (fp8 dequant path, SM 8.6), gives SSIM ≈ 0.14 —
/// matching the ~0.17 this test's real cross-device SHARDED run produces almost exactly. MiniMax-H3's fp8 pruned
/// checkpoint runs residual magnitudes into the millions (documented in <c>MiniMaxH3Recipe</c>: <c>condition_proj</c>
/// overflows 2 of 5376 channels to inf pre-block-0, the residual reaches ~2.7e6 by the last block), and the SM
/// 8.6 dequant GEMM's different rounding/accumulation order gets amplified into a materially different image —
/// a property of this checkpoint's numerics on non-native-fp8 hardware, present with or without sharding. This is
/// therefore gated in two tiers: a tight same-device parity check (proves the sharding math), and a
/// coherence + VRAM-pooling check for the real cross-device run (proves the practical pooling win), with the
/// cross-device SSIM only logged, not asserted tight.</para></remarks>
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

        // Informational only, deliberately not gated at the sibling models' 0.75: the cross-device fp8 dequant path
        // legitimately produces a materially different image for THIS checkpoint (see class remarks) — a >0.75 bar
        // here would either be unreachable for a correct implementation or would mask nothing if lowered blindly.
        // A loose floor still catches true garbage (a black frame, a NaN cascade that decoded to noise, etc.).
        Assert.Equal(baseline.Frames.Count, crossDeviceSharded.Frames.Count);
        double crossDeviceSsim = MeanSsim(baseline.Frames, crossDeviceSharded.Frames);
        _output.WriteLine($"  mean SSIM(baseline, cross-device sharded) = {crossDeviceSsim:F4} (informational — see class remarks)");
        Assert.True(crossDeviceSsim > 0.05, $"cross-device sharded output is not just fp8-divergent but incoherent " +
            $"(SSIM={crossDeviceSsim:F4}) — this is below even the fp8-dequant-path control band (~0.14).");
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
