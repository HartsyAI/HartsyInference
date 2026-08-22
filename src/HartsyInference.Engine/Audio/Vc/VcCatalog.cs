using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Hubert;
using HartsyInference.Audio.Models.Rvc;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Audio;

/// <summary>The voice-conversion model registry. RVC re-voices a source clip with a user-placed trained voice model using the engine's HuBERT/ContentVec content encoder plus YIN pitch; OpenVoice does tone-color transfer.</summary>
internal static class VcCatalog
{
    /// <summary>Category folder RVC voice models are placed in: <c>{models}/audio/clone/rvc</c>.</summary>
    internal const string RvcCategory = "clone";

    /// <summary>Model-prefix folder RVC voice models are placed in.</summary>
    internal const string RvcPrefix = "rvc";

    private const string ContentVecFile = "contentvec.safetensors";
    // There is no upstream "contentvec.safetensors"; the canonical artifact is the HF-transformers HubertModel at
    // lengyue233/content-vec-best (MIT) whose keys match the engine's Hubert layout, so this is a passthrough.
    private const string ContentVecRepo = "lengyue233/content-vec-best";
    private const string ContentVecSourceFile = "pytorch_model.bin";

    private const string RmvpeFile = "rmvpe.safetensors";
    // RVC-WebUI's own model host; rmvpe.pt is a flat (or single-envelope-wrapped) state dict whose keys already
    // match RvcRmvpe.LoadWeights's layout, so this is a passthrough repack too.
    private const string RmvpeRepo = "lj1995/VoiceConversionWebUI";
    private const string RmvpeSourceFile = "rmvpe.pt";

    // RVC voice models ship as native PyTorch .pth (the training output); a .safetensors a power user dropped in is
    // also accepted. Search order per variant name.
    private static readonly string[] RvcExtensions = ["", ".safetensors", ".pth", ".pt"];

    /// <summary>Resolves a catalog id to its descriptor, or throws naming what is available.</summary>
    internal static VcModelDescriptor Resolve(string id)
    {
        if (Registry.TryGetValue(id ?? string.Empty, out VcModelDescriptor? descriptor))
        {
            return descriptor;
        }
        throw new NotSupportedException(
            $"No voice-conversion model '{id}' is registered. Available: {string.Join(", ", Registry.Keys)} "
            + "(pass the RVC voice name as 'rvc:myvoice').");
    }

    /// <summary>RVC v2 — a user-placed voice model plus the shared ContentVec encoder; re-voices the source audio in the model's voice (speaker id 0). The HuBERT content encoder and YIN F0 both run on 16 kHz source.</summary>
    internal static VcModelDescriptor Rvc { get; } = new VcModelDescriptor
    {
        ManagesOwnWeights = false,
        InputSampleRate = 16_000,
        CacheKey = ResolveRvcModel,
        LoadAsync = (selector, cancel) => LoadRvcAsync(ResolveRvcModel(selector), cancel),
    };

    private static Dictionary<string, VcModelDescriptor>? _registry;

    // Built on first use, not in a static initializer: the descriptor properties would still be null if the map were
    // materialized while this type's static initializers were still running.
    private static Dictionary<string, VcModelDescriptor> Registry => _registry ??= new(StringComparer.OrdinalIgnoreCase)
    {
        ["rvc"] = Rvc,
        ["openvoice"] = OpenVoiceModel.Descriptor,
    };

    /// <summary>Resolves the RVC voice checkpoint: an explicit local path wins, else the variant name under the RVC weights folder with the accepted extensions.</summary>
    private static string ResolveRvcModel(AudioModelSelector selector)
    {
        if (!string.IsNullOrEmpty(selector.LocalPath) && File.Exists(selector.LocalPath))
        {
            return selector.LocalPath;
        }
        string directory = AudioModelRoot.WeightsDirectory(RvcCategory, RvcPrefix);
        string variant = (selector.Variant ?? string.Empty).Trim();
        foreach (string extension in RvcExtensions)
        {
            string candidate = Path.Combine(directory, variant + extension);
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }
        return Path.Combine(directory, variant);   // not-found path, used in the message below
    }

    private static async Task<IVcRunner> LoadRvcAsync(string rvcModelPath, CancellationToken cancel)
    {
        if (!File.Exists(rvcModelPath))
        {
            throw new FileNotFoundException(
                $"RVC voice model not found: '{rvcModelPath}'. Place the RVC voice model (.pth or .safetensors) there.", rvcModelPath);
        }
        string contentVec = AudioModelRoot.SharedFile(ContentVecFile);
        await EnsureContentVecAsync(contentVec, cancel).ConfigureAwait(false);
        string rmvpePath = AudioModelRoot.SharedFile(RmvpeFile);
        await EnsureRmvpeAsync(rmvpePath, cancel).ConfigureAwait(false);

        Hubert hubert = new Hubert(HubertConfig.ChineseHubertBase);
        SafeTensorsLoader hubertLoader = new SafeTensorsLoader();
        hubertLoader.Load(contentVec);
        hubert.LoadWeights(hubertLoader.GetAllTensors());

        RvcRmvpe rmvpe = new RvcRmvpe(RvcRmvpeConfig.Default);
        SafeTensorsLoader rmvpeLoader = new SafeTensorsLoader();
        rmvpeLoader.Load(rmvpePath);
        rmvpe.LoadWeights(rmvpeLoader.GetAllTensors());

        (IReadOnlyDictionary<string, Tensor> rvcTensors, IDisposable rvcLoader) = AudioCheckpoints.LoadFile(rvcModelPath);
        RvcConfig config = DetectRvcConfig(rvcTensors);
        RvcPipeline rvc = new RvcPipeline(config);
        rvc.LoadWeights(rvcTensors);

        Logs.Info($"[Audio][RVC] Loaded voice '{Path.GetFileName(rvcModelPath)}' ({rvc.SampleRate} Hz; ContentVec + RMVPE/YIN F0).");
        // Loaders stay alive for the runner's lifetime (the model tensors reference them). RVC carries the target
        // voice in its trained weights, so the target argument is unused.
        return new VcRunner(rvc.SampleRate,
            (backend, source, _, request) => ConvertRvc(backend, hubert, rmvpe, rvc, source, request.PitchShift, request.F0Method),
            hubertLoader, rvcLoader, rmvpeLoader, rvc, rmvpe);
    }

    /// <summary>Ensures the shared ContentVec encoder exists as <c>contentvec.safetensors</c>: on first use the HF-transformers ContentVec pickle is fetched and re-saved as safetensors (a straight passthrough — its keys already match the engine's Hubert layout).</summary>
    private static async Task EnsureContentVecAsync(string contentVecPath, CancellationToken cancel)
    {
        if (File.Exists(contentVecPath))
        {
            return;
        }
        Logs.Info($"[Audio][RVC] ContentVec encoder missing — fetching {ContentVecRepo} and converting to {ContentVecFile}...");
        string binPath = await AudioModelCache.GetAsync(ContentVecRepo, ContentVecSourceFile, category: "clone", ct: cancel).ConfigureAwait(false);
        // Straight passthrough — ContentVec's keys already match the engine's Hubert layout.
        PickleCheckpointRepacker.Repack(binPath, contentVecPath);
        Logs.Info($"[Audio][RVC] {ContentVecFile} ready.");
    }

    /// <summary>Ensures the shared RMVPE pitch extractor exists as <c>rmvpe.safetensors</c>: on first use the RVC-WebUI pickle is fetched and re-saved as safetensors (a straight passthrough — its keys already match <see cref="RvcRmvpe"/>'s layout).</summary>
    private static async Task EnsureRmvpeAsync(string rmvpePath, CancellationToken cancel)
    {
        if (File.Exists(rmvpePath))
        {
            return;
        }
        Logs.Info($"[Audio][RVC] RMVPE pitch extractor missing — fetching {RmvpeRepo} and converting to {RmvpeFile}...");
        string ptPath = await AudioModelCache.GetAsync(RmvpeRepo, RmvpeSourceFile, category: "clone", ct: cancel).ConfigureAwait(false);
        PickleCheckpointRepacker.Repack(ptPath, rmvpePath);
        Logs.Info($"[Audio][RVC] {RmvpeFile} ready.");
    }

    /// <summary>Picks the RVC config from the first upsample-kernel width: the ConvTranspose1d kernel is 24 for 48 kHz and 16 for 40 kHz. weight-norm models store <c>weight_v</c>; pre-fused models store <c>weight</c>.</summary>
    private static RvcConfig DetectRvcConfig(IReadOnlyDictionary<string, Tensor> weights)
    {
        Tensor? upsample0 = weights.TryGetValue("dec.ups.0.weight_v", out Tensor? weightV) ? weightV
            : weights.TryGetValue("dec.ups.0.weight", out Tensor? weight) ? weight : null;
        int kernel = upsample0 is not null ? (int)upsample0.Shape[upsample0.Shape.Rank - 1] : 16;
        return kernel >= 20 ? RvcConfig.V2_48k : RvcConfig.V2_40k;
    }

    private static float[] ConvertRvc(IBackend backend, Hubert hubert, RvcRmvpe rmvpe, RvcPipeline rvc,
        float[] source16k, double pitchSemitones, string? f0Method)
    {
        int pcmLength = source16k.Length;
        Tensor pcm = new Tensor(new TensorShape(1, 1, pcmLength), DType.F32);
        source16k.AsSpan().CopyTo(pcm.AsSpan<float>());
        Tensor content = hubert.Forward(backend, pcm, pcmLength);   // [1, 768, T] at the 50 Hz HuBERT frame rate
        pcm.Dispose();
        // RVC upsamples the HuBERT features 2x (nearest) → 100 Hz, so the NSF decoder's upsample product maps back to
        // the true output rate; skipping this halves the output duration. F0 is sampled on the matching 100 Hz grid.
        Tensor content2x = Interpolate2xNearest(content);
        content.Dispose();
        try
        {
            int contentFrames = (int)content2x.Shape[2];
            // Both estimators run at hop 160 (100 Hz on 16 kHz audio), matching content2x's grid.
            float[] f0 = string.Equals(f0Method, "yin", StringComparison.OrdinalIgnoreCase)
                ? F0Estimator.EstimateYin(source16k, 16_000, hopSize: 160)
                : rmvpe.ExtractF0(backend, source16k);
            if (pitchSemitones != 0d)
            {
                f0 = RvcPitch.Shift(f0, (float)pitchSemitones);
            }
            return rvc.Convert(backend, content2x, AlignF0(f0, contentFrames), sid: 0);
        }
        finally
        {
            content2x.Dispose();
        }
    }

    /// <summary>Nearest-neighbour 2x upsample of the content along time (<c>[1, C, T] → [1, C, 2T]</c>), matching RVC's interpolate before the synthesizer.</summary>
    private static Tensor Interpolate2xNearest(Tensor content)
    {
        int channels = (int)content.Shape[1];
        int frames = (int)content.Shape[2];
        ReadOnlySpan<float> source = content.AsSpan<float>();
        Tensor output = new Tensor(new TensorShape(1, channels, 2 * frames), DType.F32);
        Span<float> destination = output.AsSpan<float>();
        for (int channel = 0; channel < channels; channel++)
        {
            int sourceBase = channel * frames;
            int destinationBase = channel * 2 * frames;
            for (int i = 0; i < frames; i++)
            {
                float value = source[sourceBase + i];
                destination[destinationBase + 2 * i] = value;
                destination[destinationBase + 2 * i + 1] = value;
            }
        }
        return output;
    }

    /// <summary>RVC requires <c>f0.Length == content T</c>; trim or zero-pad (trailing zeros read as unvoiced).</summary>
    private static float[] AlignF0(float[] f0, int targetLength)
    {
        if (f0.Length == targetLength)
        {
            return f0;
        }
        float[] aligned = new float[targetLength];
        Array.Copy(f0, aligned, Math.Min(f0.Length, targetLength));
        return aligned;
    }
}
