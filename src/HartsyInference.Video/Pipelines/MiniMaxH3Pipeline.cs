using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Video.Pipelines;

/// <summary>One MiniMax-H3 generation: a flow-match loop over the packed audio+video sequence, then the two VAEs.
/// The sampler integrates the video sigma only — the audio stream rides its own shifted schedule and has its velocity
/// rescaled by the schedule map's derivative, so a single Euler integrator solves both streams correctly.</summary>
public sealed unsafe class MiniMaxH3Pipeline : DiffusionPipelineBase
{
    private readonly MiniMaxH3Transformer _transformer;
    private readonly MiniMaxH3VideoVaeDecoder _videoVae;
    private readonly MiniMaxH3AudioVaeDecoder? _audioVae;
    private readonly MiniMaxH3Config _config;

    public MiniMaxH3Pipeline(IBackend backend, MiniMaxH3Transformer transformer, MiniMaxH3VideoVaeDecoder videoVae,
        MiniMaxH3AudioVaeDecoder? audioVae) : base(backend)
    {
        _transformer = transformer;
        _videoVae = videoVae;
        _audioVae = audioVae;
        _config = transformer.Config;
    }

    /// <summary>Decoded output: RGB frames plus the jointly generated stereo soundtrack.</summary>
    public readonly record struct Result(byte[][] Frames, int Width, int Height, int Seed,
        float[][]? Audio, int AudioSampleRate);

    /// <summary>Runs the denoise loop and both decodes. <paramref name="textStates"/> is the Qwen3-VL hidden state;
    /// <paramref name="textTagRuns"/> carries the per-token modality tags so vision pads inside the text span
    /// modulate as video.</summary>
    public Result Generate(Tensor textStates, MiniMaxH3GenerationRequest request,
        IReadOnlyList<(int Start, int Stop, int Tag)>? textTagRuns = null, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(textStates);
        ArgumentNullException.ThrowIfNull(request);

        int latentT = request.LatentFrames, latentH = request.Height / 16, latentW = request.Width / 16;
        if (latentH * 16 != request.Height || latentW * 16 != request.Width)
        {
            // 16x spatial compression; the caller snaps geometry before reaching here.
            Logs.Warning($"[MiniMaxH3] {request.Width}x{request.Height} is not a multiple of 16.");
        }
        int audioT = request.AudioLatentFrames;
        int textLen = (int)textStates.Shape[0];
        int seed = request.Seed;

        MiniMaxH3PackedLayout layout = new MiniMaxH3PackedLayout(textLen, latentT, latentH, latentW, audioT);
        int frameRows = (latentH / _config.PatchH) * (latentW / _config.PatchW);
        int videoRowCount = latentT * frameRows;
        int audioRowCount = audioT * 2;

        Tensor videoLat = SeedGenerator.CreateNoise(new TensorShape(videoRowCount, _config.VideoPatchDim), seed);
        Tensor audioLat = SeedGenerator.CreateNoise(new TensorShape(audioRowCount, _config.AudioLatentsDim), seed ^ 0x5D2B);
        (Tensor cos, Tensor sin) = MiniMaxH3Rope.BuildTables(layout.PositionIds, _transformer.RopeInvFreq(), _config.AttentionHeadDim);

        double shiftV = request.SigmaShiftVideo, shiftA = request.SigmaShiftAudio;
        double[] sigmas = MiniMaxH3Schedule.VideoSigmas(request.Steps, shiftV);

        try
        {
            for (int step = 0; step < request.Steps; step++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                double sigma = sigmas[step];
                double dSigma = sigmas[step + 1] - sigma;
                (float tVideo, float tAudio) = MiniMaxH3Schedule.Timesteps(sigma, shiftV, shiftA);
                float[] uniqueT = [tVideo, tAudio];
                Dictionary<MiniMaxH3SegmentKind, int> rowOf = new Dictionary<MiniMaxH3SegmentKind, int>
                {
                    [MiniMaxH3SegmentKind.Text] = 0,
                    [MiniMaxH3SegmentKind.Video] = 0,
                    [MiniMaxH3SegmentKind.Cond] = 0,
                    [MiniMaxH3SegmentKind.RefImage] = 0,
                    [MiniMaxH3SegmentKind.Audio] = 1,
                    [MiniMaxH3SegmentKind.RefAudio] = 1,
                };

                if (step == 0)
                {
                    Probe("video latent (noise, pre-step)", videoLat);
                    Probe("audio latent (noise, pre-step)", audioLat);
                }
                (Tensor vVideo, Tensor vAudio) = _transformer.Forward(
                    Backend, layout, videoLat, audioLat, textStates, cos, sin, uniqueT, rowOf, textTagRuns);
                try
                {
                    // Both heads return the flow velocity; the audio one is integrated over the video sigma, so it
                    // takes the schedule map's derivative as an extra factor.
                    float slope = (float)MiniMaxH3Schedule.ShiftSlope(sigma, shiftV, shiftA);
                    if (step == 0 || step == request.Steps / 2 || step == request.Steps - 1)
                    {
                        Probe($"DiT velocity (video, step {step})", vVideo);
                        Probe($"DiT velocity (audio, step {step}, sigmaV={sigma:F4} tA={tAudio:F4} slope={slope:F4})", vAudio);
                    }
                    EulerStep(videoLat, vVideo, (float)-dSigma);
                    EulerStep(audioLat, vAudio, (float)(-dSigma * slope));
                }
                finally
                {
                    vVideo.Dispose();
                    vAudio.Dispose();
                }
                Backend.Sync();
                sw.Stop();
                Logs.Info($"[minimax-h3] step {step + 1}/{request.Steps}: {sw.ElapsedMilliseconds} ms");
                onProgress?.Invoke(new GenerationProgress(step + 1, request.Steps, sw.Elapsed.TotalMilliseconds));
            }

            Probe("video latent (final)", videoLat);
            Probe("audio latent (final)", audioLat);
            Dump("video_latent_final", videoLat);
            Dump("audio_latent_final", audioLat);
            Tensor videoLatent = MiniMaxH3Latents.UnpackVideo(videoLat, latentT, latentH, latentW, _config);
            Tensor rgb;
            try
            {
                rgb = _videoVae.Decode(Backend, videoLatent);
            }
            finally
            {
                videoLatent.Dispose();
            }

            byte[][] frames;
            int outW, outH;
            try
            {
                outH = (int)rgb.Shape[3];
                outW = (int)rgb.Shape[4];
                int f = (int)rgb.Shape[2];
                frames = new byte[f][];
                for (int i = 0; i < f; i++)
                {
                    frames[i] = VideoRgbFrames.ExtractFrame(rgb, i);
                }
            }
            finally
            {
                rgb.Dispose();
            }

            float[][]? audio = null;
            int sampleRate = 0;
            if (_audioVae is not null)
            {
                Tensor audioLatent = MiniMaxH3Latents.UnpackAudio(audioLat, audioT, _config);
                Tensor wave = _audioVae.Decode(Backend, audioLatent);
                try
                {
                    sampleRate = _audioVae.SampleRate;
                    int ch = (int)wave.Shape[1], samples = (int)wave.Shape[2];
                    audio = new float[ch][];
                    float* wp = (float*)wave.DataPointer;
                    for (int c = 0; c < ch; c++)
                    {
                        audio[c] = new float[samples];
                        for (int i = 0; i < samples; i++) audio[c][i] = wp[(long)c * samples + i];
                    }
                }
                finally
                {
                    audioLatent.Dispose();
                    wave.Dispose();
                }
            }
            return new Result(frames, outW, outH, seed, audio, sampleRate);
        }
        finally
        {
            videoLat.Dispose();
            audioLat.Dispose();
            cos.Dispose();
            sin.Dispose();
        }
    }

    /// <summary>Logs min/max/mean/rms under <c>HARTSY_H3_PROBE=1</c>; no-op otherwise.</summary>
    private static void Probe(string label, Tensor t)
    {
        if (Environment.GetEnvironmentVariable("HARTSY_H3_PROBE") != "1")
        {
            return;
        }
        Tensor f = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        float* p = (float*)f.DataPointer;
        long n = f.ElementCount;
        float mn = float.MaxValue, mx = float.MinValue;
        double sum = 0, sq = 0;
        long bad = 0;
        for (long i = 0; i < n; i++)
        {
            float v = p[i];
            if (!float.IsFinite(v)) { bad++; continue; }
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v; sq += (double)v * v;
        }
        Logs.Warning($"[h3-probe] {label}: min={mn:F4} max={mx:F4} mean={sum / n:F4} rms={Math.Sqrt(sq / n):F4} nonfinite={bad} n={n}");
        if (!ReferenceEquals(f, t)) f.Dispose();
    }

    /// <summary>Writes the raw F32 tensor to <c>$HARTSY_H3_DUMP/&lt;name&gt;.bin</c> for reference comparison; no-op
    /// when the variable is unset.</summary>
    private static void Dump(string name, Tensor t)
    {
        string? dir = Environment.GetEnvironmentVariable("HARTSY_H3_DUMP");
        if (string.IsNullOrEmpty(dir))
        {
            return;
        }
        Directory.CreateDirectory(dir);
        Tensor f = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        using (FileStream fs = File.Create(Path.Combine(dir, name + ".bin")))
        {
            fs.Write(new ReadOnlySpan<byte>((void*)f.DataPointer, checked((int)(f.ElementCount * 4))));
        }
        Logs.Warning($"[h3-dump] {name} {f.Shape} -> {dir}");
        if (!ReferenceEquals(f, t)) { f.Dispose(); }
    }

    /// <summary><c>z += v * delta</c> in place.</summary>
    private void EulerStep(Tensor z, Tensor velocity, float delta)
    {
        Backend.CfgEulerStep(z, velocity, velocity, 1f, delta);
    }

    protected override void DisposeCore()
    {
        _transformer.Dispose();
    }
}
