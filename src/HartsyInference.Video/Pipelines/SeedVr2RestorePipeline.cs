using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Video.Pipelines;

/// <summary>SeedVR2 one-step video restoration. Per chunk: preprocess (bicubic area-resize IS the upscale —
/// the DiT only restores detail at target resolution) → VAE-encode → sample posterior → build the
/// 33-channel conditioning (noise‖latent‖mask=1) → ONE NaDiT forward at t=1000 (no CFG; cfg_scale 1.0 —
/// the negative embedding is loaded upstream but unused) → x₀ = noise − v → VAE-decode → optional wavelet
/// transplant → overlap cross-fade. Components are shared resources owned by the caller
/// (<see cref="DiffusionPipelineBase"/> convention).</summary>
public sealed class SeedVr2RestorePipeline : DiffusionPipelineBase
{
    private const float Timestep = 1000f;

    private readonly SeedVr2Dit _dit;
    private readonly SeedVr2VaeEncoder _encoder;
    private readonly SeedVr2VaeDecoder _decoder;
    private readonly Tensor _posEmb;

    /// <summary>Parity hook: overrides RNG draws so E2E tests can inject the Python reference's saved
    /// noises (C# Box-Muller ≠ torch RNG — seeds can never match across runtimes). Kind is
    /// <c>"posterior"</c> or <c>"init"</c>; layout must be [cells,16] cell-major. Null = SeedGenerator.</summary>
    public Func<string, int, TensorShape, Tensor>? NoiseHook { get; set; }

    /// <summary>Wires shared components; <paramref name="posEmb"/> is the frozen positive embedding
    /// <c>[Lt,5120]</c> as F32 (cast from the shipped bf16 at load).</summary>
    public SeedVr2RestorePipeline(IBackend backend, SeedVr2Dit dit, SeedVr2VaeEncoder encoder,
        SeedVr2VaeDecoder decoder, Tensor posEmb)
        : base(backend)
    {
        _dit = dit;
        _encoder = encoder;
        _decoder = decoder;
        _posEmb = posEmb;
    }

    /// <summary>Restores RGB24 frames; returns restored RGB24 frames at the area-normalized target size.
    /// Stills (<c>frames.Count == 1</c>) flow through the same path with T=1.</summary>
    public (List<byte[]> Frames, int Width, int Height) Restore(IReadOnlyList<byte[]> frames, int width,
        int height, SeedVr2RestoreOptions options, Action<Core.Pipelines.GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        long startTicks = Environment.TickCount64;

        SeedVr2Preprocess.Result pre = SeedVr2Preprocess.Run(frames, width, height, options.TargetArea);
        int outH = pre.Height, outW = pre.Width;

        int chunkLen = Math.Min(SeedVr2Preprocess.PaddedFrameCount(Math.Max(options.ClipFrames, 1)), pre.Frames);
        int overlap = Math.Min(options.Overlap, Math.Max(chunkLen - 1, 0));
        int step = Math.Max(chunkLen - overlap, 1);

        float[] output = new float[3L * pre.Frames * outH * outW];
        float[] weight = new float[pre.Frames];
        int totalChunks = pre.Frames <= chunkLen ? 1 : (pre.Frames - chunkLen + step - 1) / step + 1;
        int chunkIndex = 0;

        for (int start = 0; start < pre.Frames; start += step)
        {
            int begin = Math.Min(start, Math.Max(pre.Frames - chunkLen, 0));
            int len = Math.Min(chunkLen, pre.Frames - begin);
            len = SeedVr2Preprocess.PaddedFrameCount(len) == len ? len : len - (len - 1) % 4;

            float[] chunk = RestoreChunk(pre, begin, len, options, chunkIndex);
            for (int f = 0; f < len; f++)
            {
                int dst = begin + f;
                // Linear cross-fade over overlapping frames (uniform accumulate + ramp weight).
                float ramp = weight[dst] > 0 && f < overlap ? (f + 1f) / (overlap + 1f) : 1f;
                for (int c = 0; c < 3; c++)
                {
                    long srcPlane = ((long)c * len + f) * outH * outW;
                    long dstPlane = ((long)c * pre.Frames + dst) * outH * outW;
                    if (weight[dst] == 0)
                        Array.Copy(chunk, srcPlane, output, dstPlane, (long)outH * outW);
                    else
                        for (long i = 0; i < (long)outH * outW; i++)
                            output[dstPlane + i] = output[dstPlane + i] * (1f - ramp) + chunk[srcPlane + i] * ramp;
                }
                weight[dst] = 1f;
            }
            chunkIndex++;
            onProgress?.Invoke(new Core.Pipelines.GenerationProgress(chunkIndex, totalChunks,
                Environment.TickCount64 - startTicks));
            if (begin + len >= pre.Frames)
                break;
        }

        if (options.Strength < 1f)
            for (int f = 0; f < pre.OriginalFrames; f++)
                TransplantFrame(output, pre.Data, pre.Frames, f, outH, outW, options.Strength);

        List<byte[]> result = new List<byte[]>(pre.OriginalFrames);
        for (int f = 0; f < pre.OriginalFrames; f++)
        {
            byte[] rgb = new byte[outH * outW * 3];
            for (int c = 0; c < 3; c++)
            {
                long plane = ((long)c * pre.Frames + f) * outH * outW;
                for (int i = 0; i < outH * outW; i++)
                {
                    float v = (Math.Clamp(output[plane + i], -1f, 1f) * 0.5f + 0.5f) * 255f;
                    rgb[i * 3 + c] = (byte)Math.Clamp(MathF.Round(v), 0f, 255f);
                }
            }
            result.Add(rgb);
        }
        return (result, outW, outH);
    }

    private float[] RestoreChunk(SeedVr2Preprocess.Result pre, int begin, int len,
        SeedVr2RestoreOptions options, int chunkIndex)
    {
        int h = pre.Height, w = pre.Width;
        float scaling = _decoder.Config.ScalingFactor;

        Tensor pixels = new Tensor(new TensorShape([1L, 3, len, h, w]), DType.F32);
        Span<float> px = pixels.AsSpan<float>();
        for (int c = 0; c < 3; c++)
            pre.Data.AsSpan((int)(((long)c * pre.Frames + begin) * h * w), len * h * w)
                .CopyTo(px.Slice(c * len * h * w, len * h * w));

        // Phase-staged weight residency: at 720p-area the fp32 activations alone reach many GB, so only one
        // component's weights stay on-device at a time (DiT 13.6 GB f32 + VAE + activations exceed 24 GB).
        Backend.PreloadWeights(_encoder.EnumerateWeights());
        (Tensor mean, Tensor logvar) = _encoder.Encode(Backend, pixels);
        Backend.Sync();
        Backend.FreeWeights(_encoder.EnumerateWeights());
        Backend.TrimMemoryPool();
        pixels.Dispose();

        int lt = (int)mean.Shape[2], lh = (int)mean.Shape[3], lw = (int)mean.Shape[4];
        int cells = lt * lh * lw;
        const int C = 16;

        // Posterior sample (use_sample: True) then ×scaling, laid out channels-last [t',h',w',16].
        Tensor noiseA = NoiseHook?.Invoke("posterior", chunkIndex, new TensorShape(cells, C))
            ?? SeedGenerator.CreateNoise(new TensorShape(cells, C), options.Seed * 977 + chunkIndex * 131);
        Tensor noiseB = NoiseHook?.Invoke("init", chunkIndex, new TensorShape(cells, C))
            ?? SeedGenerator.CreateNoise(new TensorShape(cells, C), options.Seed * 977 + chunkIndex * 131 + 7);
        ReadOnlySpan<float> m = mean.AsSpan<float>(), lv = logvar.AsSpan<float>();
        ReadOnlySpan<float> epsA = noiseA.AsSpan<float>(), initNoise = noiseB.AsSpan<float>();

        Tensor ditIn = new Tensor(new TensorShape([(long)lt, lh, lw, 33]), DType.F32);
        Span<float> di = ditIn.AsSpan<float>();
        for (int cell = 0; cell < cells; cell++)
        {
            int t0 = cell / (lh * lw), rem = cell % (lh * lw);
            for (int c = 0; c < C; c++)
            {
                int chw = (c * lt + t0) * lh * lw + rem;
                float std = MathF.Exp(0.5f * Math.Clamp(lv[chw], -30f, 20f));
                float z = (m[chw] + std * epsA[cell * C + c]) * scaling;
                di[cell * 33 + c] = initNoise[cell * C + c];
                di[cell * 33 + C + c] = z;
            }
            di[cell * 33 + 32] = 1f;
        }
        mean.Dispose(); logvar.Dispose(); noiseA.Dispose();

        Backend.PreloadWeights(_dit.EnumerateWeights());
        Tensor v = _dit.Forward(Backend, ditIn, _posEmb, Timestep);
        Backend.Sync();
        Backend.FreeWeights(_dit.EnumerateWeights());
        Backend.TrimMemoryPool();
        ditIn.Dispose();

        // One Euler step from σ=1: x₀ = x_t − v with x_t = the init noise; then ÷scaling, channels-first.
        Tensor latent = new Tensor(new TensorShape([1L, C, lt, lh, lw]), DType.F32);
        Span<float> ls = latent.AsSpan<float>();
        ReadOnlySpan<float> vs = v.AsSpan<float>();
        for (int cell = 0; cell < cells; cell++)
        {
            int t0 = cell / (lh * lw), rem = cell % (lh * lw);
            for (int c = 0; c < C; c++)
                ls[(c * lt + t0) * lh * lw + rem] = (initNoise[cell * C + c] - vs[cell * C + c]) / scaling;
        }
        v.Dispose(); noiseB.Dispose();

        Backend.PreloadWeights(_decoder.EnumerateWeights());
        Tensor decoded = _decoder.Decode(Backend, latent);
        Backend.Sync();
        Backend.FreeWeights(_decoder.EnumerateWeights());
        Backend.TrimMemoryPool();
        latent.Dispose();
        float[] chunk = decoded.AsSpan<float>().ToArray();
        decoded.Dispose();
        return chunk;
    }

    private static void TransplantFrame(float[] output, float[] reference, int totalFrames, int frame,
        int h, int w, float strength)
    {
        float[] model = new float[3 * h * w];
        float[] input = new float[3 * h * w];
        for (int c = 0; c < 3; c++)
        {
            long plane = ((long)c * totalFrames + frame) * h * w;
            Array.Copy(output, plane, model, c * h * w, h * w);
            Array.Copy(reference, plane, input, c * h * w, h * w);
        }
        SeedVr2ColorFix.Transplant(model, input, 3, h, w, strength);
        for (int c = 0; c < 3; c++)
            Array.Copy(model, c * h * w, output, ((long)c * totalFrames + frame) * h * w, h * w);
    }
}
