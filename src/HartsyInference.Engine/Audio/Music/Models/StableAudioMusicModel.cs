using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Codecs.Oobleck;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Audio;

/// <summary>Stable Audio Open Small (<c>stabilityai/stable-audio-open-small</c>, via the ungated <c>FastVideo/stable-audio-open-small-Diffusers</c> repack) — T5-base prompt encode → rectified-flow DiT ping-pong denoise (8 steps, no CFG, ARC-distilled) → Oobleck VAE decode → 44.1 kHz stereo, up to ~11.89 s. DiT/VAE/timing-conditioner components are individually real-weight parity-verified (cosine 1.0 each); the composed pipeline is structurally wired but not yet validated end-to-end against a Python reference.</summary>
internal static class StableAudioMusicModel
{
    private const string Repo = "FastVideo/stable-audio-open-small-Diffusers";
    private const string T5Repo = "google-t5/t5-base";
    private const int T5MaxTokens = 64;

    internal static MusicModelDescriptor Descriptor { get; } = new MusicModelDescriptor
    {
        ManagesOwnWeights = true,
        CacheKey = _ => Repo,
        LoadAsync = LoadAsync,
    };

    private static async Task<IMusicRunner> LoadAsync(MusicLoadContext context, AudioModelSelector selector, CancellationToken cancel)
    {
        Task<string> ditPath = AudioModelCache.GetAsync(Repo, "transformer/diffusion_pytorch_model.safetensors", category: "music", ct: cancel);
        Task<string> vaePath = AudioModelCache.GetAsync(Repo, "vae/diffusion_pytorch_model.safetensors", category: "music", ct: cancel);
        Task<string> condPath = AudioModelCache.GetAsync(Repo, "conditioner/diffusion_pytorch_model.safetensors", category: "music", ct: cancel);
        Task<string> t5Path = AudioModelCache.GetAsync(T5Repo, "model.safetensors", category: "music", ct: cancel);
        await Task.WhenAll(ditPath, vaePath, condPath, t5Path).ConfigureAwait(false);

        StableAudioDitConfig config = StableAudioDitConfig.OpenSmall;

        SafeTensorsLoader ditLoader = new();
        ditLoader.Load(ditPath.Result);
        StableAudioDit dit = new(config);
        dit.LoadWeights(ditLoader.GetAllTensors());

        SafeTensorsLoader vaeLoader = new();
        vaeLoader.Load(vaePath.Result);
        OobleckConfig vaeConfig = OobleckConfig.StableAudioOpen;
        Dictionary<string, Tensor> vaeWeights = OobleckKeyRemap.ToFlatSequentialLayout(vaeLoader.GetAllTensors(), vaeConfig);
        OobleckVae vae = new(vaeConfig);
        vae.LoadWeights(vaeWeights);

        SafeTensorsLoader condLoader = new();
        condLoader.Load(condPath.Result);
        StableAudioNumberEmbedder timing = new(minVal: 0f, maxVal: (float)config.TimingMaxSeconds);
        timing.LoadWeights(condLoader.GetAllTensors(), "conditioners.seconds_total");

        SafeTensorsLoader t5Loader = new();
        t5Loader.Load(t5Path.Result);
        T5TextEncoder textEncoder = new(T5TextEncoderConfig.T5Base);
        textEncoder.LoadWeights(t5Loader.GetAllTensors());
        T5Tokenizer tokenizer = new(maxLength: T5MaxTokens);

        StableAudioPipeline pipeline = new(context.Backend, dit, vae, timing, config);
        Logs.Info($"[Audio][Stable Audio] Loaded Open Small (44.1 kHz stereo, up to {config.MaxLatentTokens * (double)config.VaeDownsample / config.SampleRate:0.0}s).");

        MusicAudio Synth(IBackend device, MusicRequest request, CancellationToken ct)
        {
            ct.ThrowIfCancellationRequested();
            int[] promptIds = tokenizer.Encode(request.Prompt);
            Tensor textEmbeds = textEncoder.Encode(device, [promptIds]);
            device.Sync();
            device.FreeWeights(textEncoder.EnumerateWeights());
            try
            {
                (float[] left, float[] right, int _, int _) = pipeline.Generate(
                    textEmbeds, request.Duration.Filled(), steps: request.InferSteps, seed: request.Seed);
                return MusicAudio.Stereo(left, right);
            }
            finally
            {
                textEmbeds.Dispose();
            }
        }

        return new MusicRunner(config.SampleRate, Synth, pipeline, dit, textEncoder, tokenizer, ditLoader, vaeLoader, condLoader, t5Loader);
    }

    /// <summary>Comfy-Org's single-file repack of <c>stabilityai/stable-audio-open-1.0</c> (stable-audio-tools keys: the DiT
    /// under <c>model.model.</c>, the Oobleck VAE under <c>pretransform.model.</c>, T5-base and the two timing conditioners
    /// under <c>conditioner.conditioners.</c>).</summary>
    internal const string Open1Repo = "Comfy-Org/stable-audio-open-1.0";

    /// <summary>The all-in-one checkpoint's file name in <see cref="Open1Repo"/>.</summary>
    internal const string Open1File = "stable-audio-open-1.0.safetensors";

    /// <summary>The official conditioner's T5 window (model_config <c>max_length</c>).</summary>
    internal const int Open1T5MaxTokens = 128;

    /// <summary>Stable Audio Open 1.0: T5-base prompt → v-prediction EDM DiT with CFG over DPM++ 3M SDE (100 steps, CFG 7 by
    /// default) → Oobleck VAE → 44.1 kHz stereo, up to about 47.55 s. Built from the stable-audio-tools source and
    /// ComfyUI's loader; not yet run on real weights.</summary>
    internal static MusicModelDescriptor Open1Descriptor { get; } = new MusicModelDescriptor
    {
        ManagesOwnWeights = true,
        CacheKey = selector => string.IsNullOrEmpty(selector.LocalPath) ? Open1Repo : selector.LocalPath,
        LoadAsync = LoadOpen1Async,
    };

    private static async Task<IMusicRunner> LoadOpen1Async(MusicLoadContext context, AudioModelSelector selector, CancellationToken cancel)
    {
        string path = !string.IsNullOrEmpty(selector.LocalPath) ? selector.LocalPath
            : await AudioModelCache.GetAsync(Open1Repo, Open1File, category: "music", ct: cancel).ConfigureAwait(false);
        SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> all = loader.GetAllTensors();

        Dictionary<string, Tensor> ditWeights = Strip(all, "model.model.");
        StableAudioDitConfig config = StableAudioDitConfig.FromWeights(ditWeights);
        if (!config.HasSecondsStart)
            throw new NotSupportedException($"'{path}' is a Stable Audio Open Small DiT; use the stableaudio model for it.");
        StableAudioDit dit = new(config);
        dit.LoadWeights(ditWeights);

        OobleckConfig vaeConfig = OobleckConfig.StableAudioOpen;
        OobleckVae vae = new(vaeConfig);
        vae.LoadWeights(OobleckKeyRemap.ToFlatSequentialLayout(Strip(all, "pretransform.model."), vaeConfig));

        StableAudioNumberEmbedder total = new(minVal: 0f, maxVal: (float)config.TimingMaxSeconds);
        total.LoadWeights(all, "conditioner.conditioners.seconds_total");
        StableAudioNumberEmbedder start = new(minVal: 0f, maxVal: (float)config.TimingMaxSeconds);
        start.LoadWeights(all, "conditioner.conditioners.seconds_start");

        T5TextEncoder textEncoder = new(T5TextEncoderConfig.T5Base);
        textEncoder.LoadWeights(SelectOpen1T5Weights(all));
        T5Tokenizer tokenizer = new(maxLength: Open1T5MaxTokens);

        StableAudioPipeline pipeline = new(context.Backend, dit, vae, total, config, start);
        Logs.Info($"[Audio][Stable Audio] Loaded Open 1.0 (44.1 kHz stereo, up to {config.MaxLatentTokens * (double)config.VaeDownsample / config.SampleRate:0.0}s).");

        MusicAudio Synth(IBackend device, MusicRequest request, CancellationToken ct)
        {
            ct.ThrowIfCancellationRequested();
            Tensor textEmbeds = textEncoder.Encode(device, [PromptIds(tokenizer, request.Prompt, Open1T5MaxTokens)]);
            device.Sync();
            device.FreeWeights(textEncoder.EnumerateWeights());
            try
            {
                (float[] left, float[] right, int _, int _) = pipeline.GenerateVPrediction(
                    textEmbeds, secondsStart: 0, request.Duration.Filled(), request.InferSteps.Filled(),
                    (float)request.CfgScale.Filled(), request.Seed);
                return MusicAudio.Stereo(left, right);
            }
            finally
            {
                textEmbeds.Dispose();
            }
        }

        return new MusicRunner(config.SampleRate, Synth, pipeline, dit, textEncoder, tokenizer, loader);
    }

    internal static Dictionary<string, Tensor> SelectOpen1T5Weights(IReadOnlyDictionary<string, Tensor> all)
    {
        Dictionary<string, Tensor> conditioner = Strip(all, "conditioner.conditioners.prompt.model.");
        if (conditioner.ContainsKey("shared.weight") || conditioner.ContainsKey("encoder.embed_tokens.weight"))
            return conditioner;
        return Strip(all, "text_encoders.t5base.transformer.");
    }

    /// <summary>The prompt's T5 ids without padding: up to <paramref name="maxTokens"/> − 1 pieces, then EOS. The official
    /// conditioner pads to <paramref name="maxTokens"/> and masks the padding in the encoder and in cross-attention, which
    /// gives exactly what these unpadded ids give.</summary>
    internal static int[] PromptIds(T5Tokenizer tokenizer, string prompt, int maxTokens)
    {
        IReadOnlyList<int> pieces = tokenizer.EncodeRaw(prompt);
        int count = Math.Min(pieces.Count, maxTokens - 1);
        int[] ids = new int[count + 1];
        for (int i = 0; i < count; i++)
            ids[i] = pieces[i];
        ids[count] = T5Tokenizer.EosTokenId;
        return ids;
    }

    /// <summary>The entries under <paramref name="prefix"/>, with the prefix removed.</summary>
    private static Dictionary<string, Tensor> Strip(IReadOnlyDictionary<string, Tensor> all, string prefix) =>
        all.Where(kv => kv.Key.StartsWith(prefix, StringComparison.Ordinal))
            .ToDictionary(kv => kv.Key[prefix.Length..], kv => kv.Value, StringComparer.Ordinal);
}

