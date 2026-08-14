using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Temporal chunking of the LTX-2.5 diffusion decoder must be an allocation strategy, not an approximation:
/// a chunked decode has to reproduce an un-chunked one exactly. It is the only thing making 768×512×97f fit, and its
/// failure mode — the neighborhood-attention window sliding inward at a chunk edge instead of at the clip edge — is a
/// seam every N frames that no shape or dtype check would catch.
///
/// <para>Runs on a synthetic 8-channel config with random weights so it needs neither a GPU nor a checkpoint. The
/// geometry is chosen so the budget forces multiple chunks at both a halo-1 deterministic stage and the halo-2
/// stage-5 trunk, including the first and last chunks where the window has to be extended to the kernel width.</para></summary>
public sealed unsafe class LtxVideo25TemporalChunkingTests
{
    private const int LatentFrames = 4, LatentHeight = 2, LatentWidth = 3;

    private static LtxVideo25DiffusionDecoderConfig Config(long chunkBytes) => new LtxVideo25DiffusionDecoderConfig
    {
        InChannels = 8,
        OutChannels = 3,
        PatchSize = 2,
        HeadDim = 16,
        StageChannels = [128, 64, 32, 32, 16],
        StageDepths = [1, 1, 1, 1, 2],
        StageKernels = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (5, 5, 5)],
        Stage5Kernel = (5, 5, 5),
        TimestepEmbedDim = 32,
        TimestepFreqDim = 16,
        ChunkWorkspaceBytes = chunkBytes,
    };

    private static Tensor Random(Random rng, params long[] dims) => Random(rng, new TensorShape(dims));

    private static Tensor Random(Random rng, TensorShape shape)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.25f;
        return t;
    }

    /// <summary>Every key <see cref="LtxVideo25DiffusionDecoder.LoadWeights"/> reads, at synthetic scale.</summary>
    private static Dictionary<string, Tensor> Weights(LtxVideo25DiffusionDecoderConfig config, Random rng)
    {
        Dictionary<string, Tensor> w = [];
        int stage0 = config.StageChannels[0];
        w["decoder.conv_in.weight"] = Random(rng, stage0, config.InChannels);
        w["decoder.conv_in.bias"] = Random(rng, stage0);
        void Block(string prefix, int dim, bool diffusion, int contextChannels)
        {
            int hidden = LtxVideo25DiffusionDecoderConfig.SwiGluHidden(dim);
            w[$"{prefix}.norm1.weight"] = Random(rng, dim);
            w[$"{prefix}.norm2.weight"] = Random(rng, dim);
            w[$"{prefix}.attn.qkv.weight"] = Random(rng, 3L * dim, dim);
            w[$"{prefix}.attn.qkv.bias"] = Random(rng, 3L * dim);
            w[$"{prefix}.attn.proj.weight"] = Random(rng, dim, dim);
            w[$"{prefix}.attn.proj.bias"] = Random(rng, dim);
            w[$"{prefix}.attn.q_norm.weight"] = Random(rng, config.HeadDim);
            w[$"{prefix}.attn.k_norm.weight"] = Random(rng, config.HeadDim);
            w[$"{prefix}.mlp.w_gate.weight"] = Random(rng, hidden, dim);
            w[$"{prefix}.mlp.w_up.weight"] = Random(rng, hidden, dim);
            w[$"{prefix}.mlp.w_down.weight"] = Random(rng, dim, hidden);
            if (!diffusion) return;
            w[$"{prefix}.context_proj.weight"] = Random(rng, dim, contextChannels);
            w[$"{prefix}.context_proj.bias"] = Random(rng, dim);
            w[$"{prefix}.scale_shift_table"] = Random(rng, 7, dim);
        }
        for (int stage = 0; stage < config.Upsamples.Length; stage++)
        {
            for (int block = 0; block < config.StageDepths[stage]; block++)
                Block($"decoder.det_stages.{stage}.{block}", config.StageChannels[stage], diffusion: false, 0);
            ((int T, int H, int W) stride, int reduction) = config.Upsamples[stage];
            int projChannels = stride.T * stride.H * stride.W * config.StageChannels[stage] / reduction;
            w[$"decoder.upsamples.{stage}.proj.weight"] = Random(rng, projChannels, config.StageChannels[stage]);
            w[$"decoder.upsamples.{stage}.proj.bias"] = Random(rng, projChannels);
        }
        w["decoder.t_embedder.mlp.0.weight"] = Random(rng, config.TimestepEmbedDim, config.TimestepFreqDim);
        w["decoder.t_embedder.mlp.0.bias"] = Random(rng, config.TimestepEmbedDim);
        w["decoder.t_embedder.mlp.2.weight"] = Random(rng, config.TimestepEmbedDim, config.TimestepEmbedDim);
        w["decoder.t_embedder.mlp.2.bias"] = Random(rng, config.TimestepEmbedDim);
        int stage5 = config.StageChannels[^1];
        int packed = config.OutChannels * config.PatchSize * config.PatchSize;
        w["decoder.conv_in_x_t.weight"] = Random(rng, stage5, packed);
        w["decoder.conv_in_x_t.bias"] = Random(rng, stage5);
        w["decoder.shared_adaln.proj.weight"] = Random(rng, 7L * stage5, config.TimestepEmbedDim);
        w["decoder.shared_adaln.proj.bias"] = Random(rng, 7L * stage5);
        for (int block = 0; block < config.StageDepths[^1]; block++)
            Block($"decoder.diff_blocks.{block}", stage5, diffusion: true, stage5);
        w["decoder.norm_out.weight"] = Random(rng, stage5);
        w["decoder.conv_out.weight"] = Random(rng, packed, stage5);
        w["decoder.conv_out.bias"] = Random(rng, packed);
        return w;
    }

    private static Tensor Decode(long chunkBytes, IBackend backend, Tensor latent, Tensor noise)
    {
        LtxVideo25DiffusionDecoderConfig config = Config(chunkBytes);
        using LtxVideo25DiffusionDecoder decoder = new LtxVideo25DiffusionDecoder(config);
        decoder.LoadWeights(Weights(config, new Random(1234)));
        return decoder.Decode(backend, latent, noise);
    }

    [Fact]
    public void ChunkedDecodeMatchesUnchunkedExactly()
    {
        using CpuBackend backend = new CpuBackend();
        Random rng = new Random(7);
        using Tensor latent = Random(rng, 1, 8, LatentFrames, LatentHeight, LatentWidth);
        using LtxVideo25DiffusionDecoder shape = new LtxVideo25DiffusionDecoder(Config(0));
        using Tensor noise = Random(rng, shape.NoiseShape(LatentFrames, LatentHeight, LatentWidth));

        // 1 GiB leaves every pass un-chunked at this geometry; 1 MiB forces many chunks at stage 3 and stage 5.
        // Asserted, not assumed: a budget that silently fit would make this test compare two identical decodes.
        int stage5Frames = 8 * LatentFrames - 7, stage5Plane = LatentHeight * 8 * (LatentWidth * 8);
        long stage5PerToken = 7L * 16 * sizeof(float);
        Assert.Equal(stage5Frames,
            LtxVideo25TemporalChunks.AttentionFramesPerChunk(1L << 30, stage5Plane, stage5PerToken, stage5Frames, 5));
        Assert.True(
            LtxVideo25TemporalChunks.AttentionFramesPerChunk(1L << 20, stage5Plane, stage5PerToken, stage5Frames, 5) < stage5Frames);

        using Tensor whole = Decode(1L << 30, backend, latent, noise);
        using Tensor chunked = Decode(1L << 20, backend, latent, noise);

        Assert.Equal(whole.Shape, chunked.Shape);
        float* a = (float*)whole.DataPointer, b = (float*)chunked.DataPointer;
        double worst = 0;
        long worstIndex = -1;
        for (long i = 0; i < whole.ElementCount; i++)
        {
            double d = Math.Abs((double)a[i] - b[i]);
            if (d > worst) { worst = d; worstIndex = i; }
        }
        Assert.True(worst == 0.0,
            $"chunked decode differs from the un-chunked one by {worst:E3} at element {worstIndex} of {whole.ElementCount}");
    }

    /// <summary>The same invariant on CUDA with the shipped weights: the device row gather/scatter and the rope
    /// tables' global frame offset are what the CPU path cannot exercise, and 512×320×25f is the largest geometry
    /// where the un-chunked arm still fits. Measures exactly zero; the tolerance is only so a future cuBLAS kernel
    /// choice that varies with the GEMM's row count cannot turn this into a flake.</summary>
    [Fact]
    [Trait("Category", "Integration")]
    public void ChunkedDecodeMatchesUnchunkedOnCudaWithRealWeights()
    {
        string ckpt = TestPaths.LtxVideo2.VideoVae25;
        if (!File.Exists(ckpt)) return;                      // tier-lint: guarded
        (LtxVideo2CheckpointConverter.ConvertedWeights weights, SafeTensorsLoader loader) =
            LtxVideo2CheckpointConverter.LoadAndConvert(ckpt);
        using SafeTensorsLoader _loader = loader;
        Dictionary<string, Tensor> f32 = [];
        foreach ((string key, Tensor t) in weights.VaeDiffusionDecoder)
            f32[key] = t.DType == DType.F32 ? t : t.CastTo(DType.F32);

        using CudaBackend backend = new CudaBackend(0, Path.Combine(AppContext.BaseDirectory, "Ptx"));
        Random rng = new Random(11);
        using Tensor latent = Random(rng, 1, 128, 4, 10, 16);
        Tensor Run(long chunkBytes)
        {
            using LtxVideo25DiffusionDecoder decoder =
                new LtxVideo25DiffusionDecoder(new LtxVideo25DiffusionDecoderConfig { ChunkWorkspaceBytes = chunkBytes });
            decoder.LoadWeights(f32);
            using Tensor noise = Random(new Random(3), decoder.NoiseShape(4, 10, 16));
            return decoder.Decode(backend, latent, noise);
        }

        using Tensor whole = Run(8L << 30);
        using Tensor chunked = Run(1L << 30);
        float* a = (float*)whole.DataPointer, b = (float*)chunked.DataPointer;
        double worst = 0;
        for (long i = 0; i < whole.ElementCount; i++) worst = Math.Max(worst, Math.Abs((double)a[i] - b[i]));
        Assert.True(worst <= 1e-5, $"chunked CUDA decode differs from the un-chunked one by {worst:E3}");
    }

    /// <summary>The padded window must never be narrower than the kernel: <c>Na3d</c> collapses to
    /// <c>min(kernel, frames)</c>, so a short first or last window would change every window inside it — the failure
    /// that puts a seam at the clip's own start and end rather than at a chunk boundary.</summary>
    [Theory]
    [InlineData(97, 11, 1)]
    [InlineData(97, 11, 4)]
    [InlineData(97, 11, 20)]
    [InlineData(25, 3, 1)]
    [InlineData(7, 11, 3)]
    public void PaddedWindowsReproduceTheUntiledWindowStarts(int frames, int kernel, int step)
    {
        int expectedKernel = Math.Min(kernel, frames);
        for (int start = 0; start < frames; start += step)
        {
            int end = Math.Min(start + step, frames);
            (int windowStart, int windowFrames) = LtxVideo25TemporalChunks.Window(start, end, frames, kernel);
            Assert.True(windowFrames >= expectedKernel);
            Assert.True(windowStart >= 0 && windowStart + windowFrames <= frames);
            Assert.True(windowStart <= start && windowStart + windowFrames >= end);
            // A chunk's write is deferred only until the next chunk's window has been read, which bounds how far
            // back a window may reach to one kernel.
            Assert.True(start - windowStart <= expectedKernel);
            for (int frame = start; frame < end; frame++)
            {
                Assert.Equal(IBackend.Na3dWindowStart(frame, frames, expectedKernel),
                    windowStart + IBackend.Na3dWindowStart(frame - windowStart, windowFrames, expectedKernel));
            }
        }
    }
}
