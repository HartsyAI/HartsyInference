using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Schedulers;
using SharpInference.Diffusion.Utilities;
using SharpInference.Interactive.Camera;
using SharpInference.Interactive.Memory;
using SharpInference.Video;

namespace SharpInference.Interactive.Pipelines;

/// <summary>Matrix-Game 3.0 (Skywork, Apache-2.0) image-to-world pipeline — the canned-action segment loop matching
/// <c>pipeline/inference_pipeline.py</c>. Per segment: assemble [5 FOV-retrieved memory ‖ 4 clean past-overlap ‖ noisy
/// current] latents, denoise with FlowUniPC + 2-way text CFG (the null pass zeroes the memory slots, mirroring the
/// reference's memory-disabled neg path), decode through the (reused) <see cref="Wan22VaeDecoder"/>, and roll the
/// history buffer + camera trajectory forward. Actions arrive as RGB-rate keyboard one-hot [6] + mouse Δ [2] rows;
/// camera poses integrate from them (<see cref="Se3Math.IntegrateActions"/>) and feed the Plücker tokens + FOV memory
/// retrieval.
///
/// <para>Takes pre-computed umT5-XXL features and a pre-encoded, latent-normalized seed-image latent
/// <c>[1, 48, 1, H/16, W/16]</c> (the Wan2.2 VAE <i>encoder</i> is not built yet — same caveat as Wan I2V).
/// <b>Status: built, first-run validation pending</b> — segment-boundary VAE state, Plücker layout/injection,
/// action-integrator step sizes, and the memory RoPE indices are validation-gated. Segments after the first decode
/// independently ((T−1)·4+1 frames instead of the reference's cache-continued T·4) until cross-segment VAE streaming
/// state lands.</para></summary>
public sealed unsafe class MatrixGame3Pipeline : DiffusionPipelineBase
{
    private readonly MatrixGame3Transformer _transformer;
    private readonly Wan22VaeDecoder _vae;
    private readonly Wan22VaeEncoder? _encoder;
    private readonly MatrixGame3Config _config;

    /// <summary>World-units moved per held movement key per frame (validation-gated default).</summary>
    public float MoveStepPerFrame { get; init; } = 0.05f;

    /// <summary>Degrees of yaw/pitch per unit of mouse delta (validation-gated default; the canned bench uses ±0.1 deltas).</summary>
    public float RotateDegreesPerUnit { get; init; } = 90f;

    /// <summary><paramref name="encoder"/> is optional and only needed to encode an RGB seed image
    /// (<see cref="EncodeSeedImage"/>); it loads from the same <c>wan22_vae.safetensors</c> dict as the decoder.</summary>
    public MatrixGame3Pipeline(IBackend backend, MatrixGame3Transformer transformer, Wan22VaeDecoder vae, MatrixGame3Config config,
        Wan22VaeEncoder? encoder = null)
        : base(backend)
    {
        _transformer = transformer;
        _vae = vae;
        _config = config;
        _encoder = encoder;
    }

    /// <summary>Encodes an interleaved-RGB24 seed image to the normalized latent <see cref="GenerateFromEmbeddings"/>
    /// bootstraps from. The caller owns (disposes) the returned tensor. Requires a <see cref="Wan22VaeEncoder"/>.</summary>
    public Tensor EncodeSeedImage(ReadOnlySpan<byte> rgb24, int width, int height)
    {
        ThrowIfDisposed();
        if (_encoder is null)
            throw new InvalidOperationException("Seed-image encoding needs a Wan22VaeEncoder — construct the pipeline with one (it loads from the same VAE weights).");
        return _encoder.EncodeRgbFrame(Backend, rgb24, width, height);
    }

    /// <summary>RGB frames produced by <paramref name="numSegments"/> segments (first 57, then (T−1)·4+1 per segment
    /// under per-segment decode).</summary>
    public int TotalFrames(int numSegments) =>
        ((_config.FirstSegmentLatents - 1) * _config.VaeTemporalCompression + 1)
        + (numSegments - 1) * ((_config.SegmentLatents - 1) * _config.VaeTemporalCompression + 1);

    /// <summary>RGB-rate action rows required for <paramref name="numSegments"/> segments.</summary>
    public int RequiredActionFrames(int numSegments) =>
        _config.FirstSegmentLatents * _config.VaeTemporalCompression
        + (numSegments - 1) * _config.SegmentLatents * _config.VaeTemporalCompression;

    /// <summary>Generates a streamed world rollout. <paramref name="keyboard"/>/<paramref name="mouse"/> are RGB-rate
    /// rows (≥ <see cref="RequiredActionFrames"/>); <paramref name="seedImageLatent"/> is the VAE-encoded + normalized
    /// input image. Returns one interleaved-RGB <c>byte[]</c> per frame.</summary>
    public (byte[][] frames, int width, int height, int seed) GenerateFromEmbeddings(
        Tensor promptEmbeds, Tensor negativeEmbeds, Tensor seedImageLatent,
        float[][] keyboard, float[][] mouse, int width, int height, int numSegments,
        int steps = 0, float guidanceScale = -1f, int? seed = null,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        int sp = _config.VaeSpatialCompression;
        if (width % (sp * _config.PatchSize.W) != 0 || height % (sp * _config.PatchSize.H) != 0)
            throw new ArgumentException($"Width/height must be divisible by {sp * _config.PatchSize.W} for Matrix-Game 3.");
        if (numSegments < 1) throw new ArgumentOutOfRangeException(nameof(numSegments));
        int hLat = height / sp, wLat = width / sp;
        if (seedImageLatent.Shape.Rank != 5 || seedImageLatent.Shape[1] != _config.InChannels
            || seedImageLatent.Shape[2] != 1 || seedImageLatent.Shape[3] != hLat || seedImageLatent.Shape[4] != wLat)
            throw new ArgumentException($"seedImageLatent must be [1,{_config.InChannels},1,{hLat},{wLat}].", nameof(seedImageLatent));
        int requiredActions = RequiredActionFrames(numSegments);
        if (keyboard.Length < requiredActions || mouse.Length < requiredActions)
            throw new ArgumentException($"Need ≥{requiredActions} RGB-rate action rows; got kbd {keyboard.Length} / mouse {mouse.Length}.");

        int actualSeed = seed ?? SeedGenerator.RandomSeed();
        if (steps <= 0) steps = _config.StepsBase;
        if (guidanceScale < 0) guidanceScale = _config.GuidanceScale;
        int gh = hLat / _config.PatchSize.H, gw = wLat / _config.PatchSize.W;
        int pixelBlock = sp * _config.PatchSize.H;
        float fx = width / 2f, fy = width / 2f, cxI = width / 2f, cyI = height / 2f;

        Logs.Info($"Matrix-Game 3: {numSegments} segment(s) {width}x{height}, {steps} steps/segment, cfg={guidanceScale}, seed={actualSeed}");
        Logs.Warning("Matrix-Game 3 pipeline is first-run-validation pending — numerics unverified vs the reference.");

        Backend.PreloadWeights(_transformer.EnumerateWeights());
        FlowUniPCMultistepScheduler scheduler = new();
        using FrameHistoryBuffer history = new(capacity: 64);
        FrustumOverlapSelector selector = new(fx, fy, cxI, cyI, width, height);

        // Integrate the full action plan into per-RGB-frame camera poses.
        float[] startPose = new float[16];
        Se3Math.Identity(startPose);
        float[][] poses = Se3Math.IntegrateActions(startPose, keyboard[..requiredActions], mouse[..requiredActions],
            MoveStepPerFrame, RotateDegreesPerUnit);

        List<byte[]> allFrames = new(TotalFrames(numSegments));
        Tensor pastLatents = RepeatFrames(seedImageLatent, _config.PastFrames);
        long globalLatentIndex = 0;
        int rgbCursor = 0;
        int progressStep = 0, progressTotal = numSegments * steps;

        try
        {
            for (int seg = 0; seg < numSegments; seg++)
            {
                int tLat = seg == 0 ? _config.FirstSegmentLatents : _config.SegmentLatents;
                int rgbThisSegment = tLat * _config.VaeTemporalCompression;
                int mem = _config.MemorySlots, past = _config.PastFrames;
                int tTotal = mem + past + tLat;

                // Memory: bootstrap = seed image; later = FOV-retrieved history latents.
                Tensor memLatents;
                long[] memIndices = new long[mem];
                if (history.Count < mem)
                {
                    memLatents = RepeatFrames(seedImageLatent, mem);
                }
                else
                {
                    int[] candidates = CandidateIndices(history.Count, mem);
                    float[][] candidatePoses = new float[candidates.Length][];
                    for (int i = 0; i < candidates.Length; i++) candidatePoses[i] = history[candidates[i]].Pose;
                    float[] currentPose = poses[Math.Min(rgbCursor + rgbThisSegment - 1, poses.Length - 1)];
                    int[] picked = selector.Select(candidatePoses, currentPose, mem);
                    Tensor[] frames = new Tensor[mem];
                    for (int i = 0; i < mem; i++)
                    {
                        FrameHistoryBuffer.Entry e = history[candidates[picked[i]]];
                        frames[i] = e.Latent;
                        memIndices[i] = e.FrameIndex;
                    }
                    memLatents = ConcatFrames(frames, hLat, wLat);
                }

                // Per-frame RoPE indices: memory keeps historical positions, past+current continue globally.
                int[] ropeIdx = new int[tTotal];
                for (int i = 0; i < mem; i++) ropeIdx[i] = (int)Math.Min(memIndices[i], _config.RopeMaxSeqLen - 1);
                for (int i = 0; i < past + tLat; i++)
                    ropeIdx[mem + i] = (int)Math.Min(Math.Max(globalLatentIndex - past, 0) + i, _config.RopeMaxSeqLen - 1);

                // Camera poses per latent frame of this segment (pose at the frame's last RGB sample).
                float[]?[] framePoses = new float[]?[tTotal];
                for (int i = 0; i < tLat; i++)
                {
                    int rgbIdx = Math.Min(rgbCursor + (i + 1) * _config.VaeTemporalCompression - 1, poses.Length - 1);
                    framePoses[mem + past + i] = poses[rgbIdx];
                }
                Tensor plucker = MatrixGame3PluckerTokens.Build(framePoses, fx, fy, cxI, cyI, gh, gw, pixelBlock);

                // Action rows covering the non-memory frames: past overlap repeats the earliest needed rows.
                int pastRgb = past * _config.VaeTemporalCompression;
                Tensor mouseRows = ActionRows(mouse, rgbCursor - pastRgb, pastRgb + rgbThisSegment, 2);
                Tensor kbdRows = ActionRows(keyboard, rgbCursor - pastRgb, pastRgb + rgbThisSegment, 6);

                Tensor current = SeedGenerator.CreateNoise(new TensorShape([1L, _config.InChannels, tLat, hLat, wLat]), actualSeed + seg);
                scheduler.SetTimesteps(steps, _config.SampleShift);
                float[] frameTs = new float[tTotal];

                for (int k = 0; k < steps; k++)
                {
                    Stopwatch sw = Stopwatch.StartNew();
                    float ts = scheduler.Timesteps[k];
                    for (int i = 0; i < tTotal; i++) frameTs[i] = i < mem + past ? 0f : ts;

                    Tensor input = ConcatFrames([memLatents, pastLatents, current], hLat, wLat);
                    Tensor vCond = _transformer.Forward(Backend, input, promptEmbeds, frameTs, ropeIdx, mem, tLat,
                        mouseRows, kbdRows, plucker);
                    ZeroLeadingFrames(input, mem, hLat, wLat);   // null path disables memory
                    Tensor vUncond = _transformer.Forward(Backend, input, negativeEmbeds, frameTs, ropeIdx, mem, tLat,
                        mouseRows, kbdRows, plucker);
                    input.Dispose();

                    ApplyCfgInPlace(vCond, vUncond, guidanceScale);
                    vUncond.Dispose();
                    scheduler.Step(current, vCond);
                    vCond.Dispose();
                    sw.Stop();
                    onProgress?.Invoke(new GenerationProgress(++progressStep, progressTotal, sw.Elapsed.TotalMilliseconds));
                }

                // Decode + roll state forward. Decode denormalizes its input in place, so hand it a copy —
                // `current` must stay in normalized z-space for the history buffer and the past overlap.
                Tensor decodeInput = CloneTensor(current);
                Tensor rgb;
                try { rgb = _vae.Decode(Backend, decodeInput); }
                finally { decodeInput.Dispose(); }
                int decoded = (int)rgb.Shape[2];
                for (int i = 0; i < decoded; i++) allFrames.Add(VideoRgbFrames.ExtractFrame(rgb, i));
                rgb.Dispose();

                for (int i = 0; i < tLat; i++)
                {
                    Tensor frame = SliceFrame(current, i, hLat, wLat);
                    int rgbIdx = Math.Min(rgbCursor + (i + 1) * _config.VaeTemporalCompression - 1, poses.Length - 1);
                    history.Add(frame, poses[rgbIdx], globalLatentIndex + i);
                    frame.Dispose();
                }
                pastLatents.Dispose();
                pastLatents = LastFrames(current, past, hLat, wLat);
                globalLatentIndex += tLat;
                rgbCursor += rgbThisSegment;

                memLatents.Dispose();
                plucker.Dispose();
                mouseRows.Dispose();
                kbdRows.Dispose();
                current.Dispose();
            }
        }
        finally
        {
            pastLatents.Dispose();
            Backend.Sync();
            Backend.FreeWeights(_transformer.EnumerateWeights());
        }

        Logs.Info($"Matrix-Game 3 rollout complete ({allFrames.Count} frames, seed={actualSeed})");
        return (allFrames.ToArray(), width, height, actualSeed);
    }

    /// <summary>Memory candidate window: stride-8 over the rolling buffer, mirroring the reference's <c>range(1, 34, 8)</c>
    /// subsample, padded with the most recent unused entries when the stride yields fewer than <paramref name="minCount"/>.</summary>
    private static int[] CandidateIndices(int historyCount, int minCount)
    {
        SortedSet<int> idx = new();
        for (int back = 1; back <= Math.Min(33, historyCount); back += 8)
            idx.Add(historyCount - back);
        for (int recent = historyCount - 1; recent >= 0 && idx.Count < minCount; recent--)
            idx.Add(recent);
        return [.. idx];
    }

    private Tensor RepeatFrames(Tensor singleFrame, int count)
    {
        int h = (int)singleFrame.Shape[3], w = (int)singleFrame.Shape[4];
        Tensor o = new Tensor(new TensorShape([1L, _config.InChannels, count, h, w]), DType.F32);
        float* src = (float*)singleFrame.DataPointer;
        float* dst = (float*)o.DataPointer;
        long frame = (long)h * w;
        for (int c = 0; c < _config.InChannels; c++)
            for (int f = 0; f < count; f++)
                Buffer.MemoryCopy(src + c * frame, dst + ((long)c * count + f) * frame, frame * 4, frame * 4);
        return o;
    }

    /// <summary>Concatenates latent clips <c>[1, C, T_i, H, W]</c> along the temporal axis.</summary>
    private Tensor ConcatFrames(IReadOnlyList<Tensor> clips, int h, int w)
    {
        int c = _config.InChannels;
        int tTotal = 0;
        foreach (Tensor clip in clips) tTotal += (int)clip.Shape[2];
        Tensor o = new Tensor(new TensorShape([1L, c, tTotal, h, w]), DType.F32);
        float* dst = (float*)o.DataPointer;
        long frame = (long)h * w;
        for (int ci = 0; ci < c; ci++)
        {
            int fOut = 0;
            foreach (Tensor clip in clips)
            {
                int tc = (int)clip.Shape[2];
                float* src = (float*)clip.DataPointer;
                Buffer.MemoryCopy(src + (long)ci * tc * frame, dst + ((long)ci * tTotal + fOut) * frame, (long)tc * frame * 4, (long)tc * frame * 4);
                fOut += tc;
            }
        }
        return o;
    }

    private Tensor SliceFrame(Tensor clip, int frameIndex, int h, int w)
    {
        int c = _config.InChannels, t = (int)clip.Shape[2];
        Tensor o = new Tensor(new TensorShape([1L, c, 1, h, w]), DType.F32);
        float* src = (float*)clip.DataPointer;
        float* dst = (float*)o.DataPointer;
        long frame = (long)h * w;
        for (int ci = 0; ci < c; ci++)
            Buffer.MemoryCopy(src + ((long)ci * t + frameIndex) * frame, dst + (long)ci * frame, frame * 4, frame * 4);
        return o;
    }

    private Tensor LastFrames(Tensor clip, int count, int h, int w)
    {
        int c = _config.InChannels, t = (int)clip.Shape[2];
        Tensor o = new Tensor(new TensorShape([1L, c, count, h, w]), DType.F32);
        float* src = (float*)clip.DataPointer;
        float* dst = (float*)o.DataPointer;
        long frame = (long)h * w;
        for (int ci = 0; ci < c; ci++)
            Buffer.MemoryCopy(src + ((long)ci * t + (t - count)) * frame, dst + (long)ci * count * frame, (long)count * frame * 4, (long)count * frame * 4);
        return o;
    }

    private void ZeroLeadingFrames(Tensor clip, int frames, int h, int w)
    {
        int c = _config.InChannels, t = (int)clip.Shape[2];
        float* p = (float*)clip.DataPointer;
        long frame = (long)h * w;
        for (int ci = 0; ci < c; ci++)
            new Span<float>(p + (long)ci * t * frame, (int)(frames * frame)).Clear();
    }

    private static Tensor CloneTensor(Tensor t)
    {
        Tensor o = new Tensor(t.Shape, DType.F32);
        long bytes = t.Shape.ElementCount * 4;
        Buffer.MemoryCopy((float*)t.DataPointer, (float*)o.DataPointer, bytes, bytes);
        return o;
    }

    /// <summary>RGB-rate action rows <c>[count, dim]</c> starting at <paramref name="start"/> (negative indices clamp to row 0).</summary>
    private static Tensor ActionRows(float[][] rows, int start, int count, int dim)
    {
        Tensor o = new Tensor(new TensorShape(count, dim), DType.F32);
        float* p = (float*)o.DataPointer;
        for (int i = 0; i < count; i++)
        {
            float[] row = rows[Math.Clamp(start + i, 0, rows.Length - 1)];
            for (int d = 0; d < dim; d++) p[(long)i * dim + d] = row[d];
        }
        return o;
    }

    private static void ApplyCfgInPlace(Tensor cond, Tensor uncond, float scale)
    {
        long n = cond.Shape.ElementCount;
        float* cp = (float*)cond.DataPointer;
        float* up = (float*)uncond.DataPointer;
        for (long i = 0; i < n; i++) cp[i] = up[i] + scale * (cp[i] - up[i]);
    }
}
