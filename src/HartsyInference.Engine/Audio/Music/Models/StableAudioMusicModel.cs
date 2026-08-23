using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Codecs.Oobleck;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
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
                    textEmbeds, request.Duration, steps: request.InferSteps, seed: request.Seed);
                return MusicAudio.Stereo(left, right);
            }
            finally
            {
                textEmbeds.Dispose();
            }
        }

        return new MusicRunner(config.SampleRate, Synth, pipeline, dit, textEncoder, tokenizer, ditLoader, vaeLoader, condLoader, t5Loader);
    }
}
