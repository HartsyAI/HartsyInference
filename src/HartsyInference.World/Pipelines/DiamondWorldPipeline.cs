using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.Diamond;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.World.Pipelines;

/// <summary>DIAMOND (Eloi Alonso et al., MIT) action-conditioned world model — a genuinely per-frame
/// interactive pipeline: each call denoises exactly one next frame from a rolling window of the last
/// <see cref="DiamondConfig.NumStepsConditioning"/> frames + actions (no fixed-length rollout, unlike
/// <see cref="OasisPipeline"/>'s one-shot batch). <see cref="DiamondSampler"/>/<see cref="DiamondDenoiser"/> are
/// bit-exact vs the reference in isolation (see <c>DiamondParityTests</c>); this wrapper adds the pieces the
/// reference's <c>play.py</c> owns instead of the model: RGB↔tensor conversion, initial-noise sampling, and
/// history-buffer bootstrap from a single seed frame.
/// <para><b>Initial-noise convention assumed, not independently parity-verified</b>: standard EDM/Karras
/// practice draws <c>x_T ~ N(0, sigma_max²·I)</c>; <c>DiamondParityTests</c> feeds a fixed reference-dumped
/// <c>xInit</c> instead of sampling one, so this convention (used here) has not been diffed against the
/// reference's own sampler entry point.</para></summary>
public sealed unsafe class DiamondWorldPipeline : DiffusionPipelineBase
{
    private readonly DiamondDenoiser _denoiser;
    private readonly DiamondSampler _sampler;
    private readonly List<IDisposable> _ownedLoaders = [];

    /// <summary>The resolved config (image size, action count, conditioning window).</summary>
    public DiamondConfig Config { get; }

    public DiamondWorldPipeline(IBackend backend, DiamondDenoiser denoiser) : base(backend)
    {
        _denoiser = denoiser;
        _sampler = new DiamondSampler(denoiser);
        Config = denoiser.Config;
    }

    /// <summary>Loads a DIAMOND denoiser from a checkpoint file. Accepts either a full <c>eloialonso/diamond</c>
    /// agent dump (keys prefixed <c>denoiser.inner_model.</c>) or an already-stripped inner-model-only dict
    /// (bare keys) — detected from whichever prefix the first expected key matches.</summary>
    public static DiamondWorldPipeline LoadFromPath(IBackend backend, string modelPath, DiamondConfig? cfg = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"DIAMOND checkpoint not found: '{modelPath}'.");
        DiamondConfig config = cfg ?? DiamondConfig.Atari(4);

        SafeTensorsLoader loader = new();
        loader.Load(modelPath);
        IReadOnlyDictionary<string, Tensor> all = loader.GetAllTensors();
        string prefix = all.ContainsKey("denoiser.inner_model.noise_emb.weight") ? "denoiser.inner_model"
            : all.ContainsKey("inner_model.noise_emb.weight") ? "inner_model"
            : "";

        DiamondDenoiser denoiser = new(config);
        denoiser.LoadWeights(all, prefix);

        DiamondWorldPipeline pipeline = new(backend, denoiser);
        pipeline._ownedLoaders.Add(loader);
        return pipeline;
    }

    /// <summary>Builds the initial <c>[1, C·K, H, W]</c> history tensor by replicating a single seed frame across
    /// all <see cref="DiamondConfig.NumStepsConditioning"/> conditioning slots (the model has no other way to
    /// bootstrap from one image — there is no "shorter" real history to feed it).</summary>
    public Tensor EncodeInitialHistory(ReadOnlySpan<byte> seedFrameRgb)
    {
        int size = Config.ImgSize, c = Config.ImgChannels, k = Config.NumStepsConditioning;
        long expected = (long)size * size * c;
        if (seedFrameRgb.Length != expected)
            throw new ArgumentException($"Seed frame must be exactly {size}x{size} RGB ({expected} bytes); got {seedFrameRgb.Length}.", nameof(seedFrameRgb));

        Tensor history = new(new TensorShape(1, c * k, size, size), DType.F32);
        float* dst = (float*)history.DataPointer;
        long frameLen = (long)c * size * size;
        Tensor frame = EncodeFrame(seedFrameRgb);
        float* src = (float*)frame.DataPointer;
        for (int i = 0; i < k; i++)
        {
            Buffer.MemoryCopy(src, dst + i * frameLen, frameLen * sizeof(float), frameLen * sizeof(float));
        }
        frame.Dispose();
        return history;
    }

    /// <summary>Encodes one HWC rgb24 frame to a <c>[1,C,H,W]</c> tensor in <c>[-1,1]</c>.</summary>
    public Tensor EncodeFrame(ReadOnlySpan<byte> rgb)
    {
        int size = Config.ImgSize, c = Config.ImgChannels;
        long expected = (long)size * size * c;
        if (rgb.Length != expected)
            throw new ArgumentException($"Frame must be exactly {size}x{size} RGB ({expected} bytes); got {rgb.Length}.", nameof(rgb));

        Tensor t = new(new TensorShape(1, c, size, size), DType.F32);
        float* dst = (float*)t.DataPointer;
        int plane = size * size;
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                int px = (y * size + x) * c;
                int p = y * size + x;
                for (int ch = 0; ch < c; ch++) dst[ch * plane + p] = rgb[px + ch] / 127.5f - 1f;
            }
        }
        return t;
    }

    /// <summary>Decodes a <c>[1,C,H,W]</c> tensor in <c>[-1,1]</c> back to HWC rgb24.</summary>
    public byte[] DecodeFrame(Tensor frame)
    {
        int size = Config.ImgSize, c = Config.ImgChannels;
        byte[] rgb = new byte[(long)size * size * c];
        float* src = (float*)frame.DataPointer;
        int plane = size * size;
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                int p = y * size + x;
                int px = p * c;
                for (int ch = 0; ch < c; ch++)
                    rgb[px + ch] = (byte)Math.Clamp((int)MathF.Round((src[ch * plane + p] + 1f) * 127.5f), 0, 255);
            }
        }
        return rgb;
    }

    /// <summary>Denoises exactly one new frame from <paramref name="history"/> (<c>[1,C·K,H,W]</c>) and the
    /// aligned <paramref name="actions"/> (length K). Does not mutate <paramref name="history"/>; the caller
    /// rolls its own buffer (see <c>DiamondGenPerfTests.Roll</c> for the reference shift pattern).</summary>
    public Tensor GenerateNextFrame(Tensor history, ReadOnlySpan<int> actions, int? seed = null)
    {
        ThrowIfDisposed();
        Backend.PreloadWeights(_denoiser.EnumerateWeights());
        Tensor xInit = SeedGenerator.CreateNoise(new TensorShape(1, Config.ImgChannels, Config.ImgSize, Config.ImgSize), seed ?? SeedGenerator.RandomSeed());
        Tensor scaledInit = new(xInit.Shape, DType.F32);
        Backend.Scale(scaledInit, xInit, Config.SigmaMax);
        xInit.Dispose();
        Tensor next = _sampler.Sample(Backend, scaledInit, history, actions);
        scaledInit.Dispose();
        Backend.FreeWeights(_denoiser.EnumerateWeights());
        return next;
    }

    /// <inheritdoc/>
    protected override void DisposeCore()
    {
        foreach (IDisposable d in _ownedLoaders) d.Dispose();
        _ownedLoaders.Clear();
    }
}
