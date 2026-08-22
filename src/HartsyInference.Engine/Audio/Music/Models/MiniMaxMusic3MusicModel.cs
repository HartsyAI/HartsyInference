using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Audio;

/// <summary>MiniMax Music 3 — lyrics- and caption-conditioned song generation up to six minutes, 44.1 kHz stereo.
/// An 8B Qwen3 emits one semantic RVQ code per 25 Hz frame while a 0.6B depth decoder fills in the seven residual
/// codebooks; the two models' hidden states (not their discrete codes) condition a flow-matching transformer whose
/// latents a DAC-style vocoder turns into audio.
///
/// <para>Only the diffusers-format subfolders are fetched. The repo also carries the SGLang-native format
/// (<c>qwen_7B/</c>, <c>flowmatching_vae.pth</c>, <c>dav.pth</c>) — another 28 GB the engine never reads.</para>
///
/// <para>The two stages never need to be resident together, so the language model's weights are freed before the
/// flow stage is preloaded. That is what lets the quantized variants run on a 12 GB card.</para></summary>
internal static class MiniMaxMusic3MusicModel
{
    private const string DefaultRepo = "MiniMaxAI/MiniMax-Music3";
    private const string Category = "music";

    /// <summary>A user-selected checkpoint (<see cref="AudioModelSelector.LocalPath"/>, the Comfy-Org single-file DiT) is honored for the flow transformer + condition encoder — it participates in the cache key so switching checkpoints never serves the previous file's runner. The language model, depth decoder, vocoder, and tokenizer always come from the official repo (the DiT file does not carry them).</summary>
    internal static MusicModelDescriptor Descriptor { get; } = new MusicModelDescriptor
    {
        ManagesOwnWeights = true,
        CacheKey = selector =>
            $"{ResolveRepo(selector.Variant)}|{ResolveQuant(selector.Variant) ?? "bf16"}|{selector.LocalPath ?? ""}",
        LoadAsync = (context, selector, cancel) =>
            LoadAsync(context, ResolveRepo(selector.Variant), ResolveQuant(selector.Variant), selector.LocalPath, cancel),
    };

    /// <summary>A bare id is the released checkpoint; an <c>org/repo</c> variant passes through so a future official release needs no code change. The precision suffix does not affect repo selection.</summary>
    private static string ResolveRepo(string variant)
    {
        string id = (variant ?? string.Empty).Trim();
        return id.Contains('/', StringComparison.Ordinal) ? id.Split(':')[0] : DefaultRepo;
    }

    /// <summary>Maps a variant's precision suffix to a quant mode for the global language model: <c>-q8</c>/<c>:q8</c> → q8_0, <c>-q4</c>/<c>:q4</c> → q4_k, else null (the checkpoint's BF16).</summary>
    private static string? ResolveQuant(string variant)
    {
        string lower = (variant ?? string.Empty).Trim().ToLowerInvariant();
        if (lower.EndsWith("q8", StringComparison.Ordinal) || lower.EndsWith("q8_0", StringComparison.Ordinal))
        {
            return "q8_0";
        }
        return lower.EndsWith("q4", StringComparison.Ordinal) || lower.EndsWith("q4_k", StringComparison.Ordinal) ? "q4_k" : null;
    }

    private static async Task<IMusicRunner> LoadAsync(MusicLoadContext context, string repo, string? quant, string? localPath, CancellationToken cancel)
    {
        string tokenizerPath = await AudioModelCache.GetAsync(repo, "tokenizer/tokenizer.json", Category, ct: cancel).ConfigureAwait(false);
        (IReadOnlyDictionary<string, Tensor> languageWeights, IDisposable[] languageLoaders) =
            await AudioCheckpoints.LoadSubfolderAsync(repo, "language_model", Category, cancel).ConfigureAwait(false);
        (IReadOnlyDictionary<string, Tensor> depthWeights, IDisposable[] depthLoaders) =
            await AudioCheckpoints.LoadSubfolderAsync(repo, "rvq_depth_decoder", Category, cancel).ConfigureAwait(false);

        // A user-selected checkpoint carries the flow DiT + condition encoder (the Comfy-Org single-file
        // layout); anything else it might carry is not usable, so refuse loudly rather than silently
        // generating with substituted weights.
        IReadOnlyDictionary<string, Tensor> conditionWeights;
        IReadOnlyDictionary<string, Tensor> ditWeights;
        IDisposable[] conditionLoaders;
        IDisposable[] ditLoaders;
        if (!string.IsNullOrEmpty(localPath) && File.Exists(localPath))
        {
            SafeTensorsLoader ditFile = new SafeTensorsLoader();
            try
            {
                ditFile.Load(localPath);
                if (!MiniMaxMusic3CheckpointConverter.IsComfyDit(ditFile.Descriptors))
                {
                    throw new NotSupportedException(
                        $"'{Path.GetFileName(localPath)}' is not a MiniMax-Music-3 DiT checkpoint the engine can load. "
                        + "Supported: the Comfy-Org single-file DiT (diffusion_models/minimax_music3_dit_*.safetensors). "
                        + "Remove the file to use the official MiniMaxAI/MiniMax-Music3 weights instead.");
                }
                (Dictionary<string, Tensor> convertedDit, Dictionary<string, Tensor> convertedCondition) =
                    MiniMaxMusic3CheckpointConverter.ConvertComfyDit(ditFile.GetAllTensors());
                ditWeights = convertedDit;
                conditionWeights = convertedCondition;
                ditLoaders = [ditFile];
                conditionLoaders = [];
                Logs.Info($"[Audio][MiniMaxMusic3] Flow DiT + condition encoder from selected checkpoint "
                    + $"'{Path.GetFileName(localPath)}'; language model, depth decoder, vocoder from {repo}.");
            }
            catch
            {
                ditFile.Dispose();
                throw;
            }
        }
        else
        {
            (conditionWeights, conditionLoaders) =
                await AudioCheckpoints.LoadSubfolderAsync(repo, "condition_encoder", Category, cancel).ConfigureAwait(false);
            (ditWeights, ditLoaders) =
                await AudioCheckpoints.LoadSubfolderAsync(repo, "transformer", Category, cancel).ConfigureAwait(false);
        }
        (IReadOnlyDictionary<string, Tensor> vocoderWeights, IDisposable[] vocoderLoaders) =
            await AudioCheckpoints.LoadSubfolderAsync(repo, "vocoder", Category, cancel).ConfigureAwait(false);

        IReadOnlyDictionary<string, Tensor> preparedLanguage =
            MiniMaxMusic3WeightPolicy.PrepareLanguageModel(languageWeights, repo, quant, out IDisposable? quantCache);
        // HARTSY_MM3_DEPTH_QUANT=0 leaves the depth decoder at checkpoint precision, so its share of the
        // autoregressive stage can be measured independently of the language model's.
        string? depthQuant = EnvSwitch.IsEnabled("HARTSY_MM3_DEPTH_QUANT", defaultOn: true) ? quant : null;
        IReadOnlyDictionary<string, Tensor> preparedDepth =
            MiniMaxMusic3WeightPolicy.PrepareDepthDecoder(depthWeights, repo, depthQuant, out IDisposable? depthQuantCache);
        IReadOnlyDictionary<string, Tensor> preparedDit =
            MiniMaxMusic3WeightPolicy.PrepareTransformer(ditWeights, quant, out IDisposable? ditCast);
        ModelAssets.Lora.LoraStack? loras = ApplyLoras(context, ref preparedLanguage, ref preparedDit);

        // The quantized variants are the small-card path, where an F32 KV cache is the difference between
        // a five-minute song fitting and not.
        MiniMaxMusic3GlobalLm languageModel = new MiniMaxMusic3GlobalLm(halfPrecisionKv: quant is not null && context.Backend.Device.IsCuda);
        languageModel.LoadWeights(preparedLanguage);
        MiniMaxMusic3DepthDecoder depthDecoder = new MiniMaxMusic3DepthDecoder();
        depthDecoder.LoadWeights(preparedDepth);
        MiniMaxMusic3ConditionEncoder conditionEncoder = new MiniMaxMusic3ConditionEncoder();
        conditionEncoder.LoadWeights(conditionWeights);
        MiniMaxMusic3Dit dit = new MiniMaxMusic3Dit();
        dit.LoadWeights(preparedDit);
        MiniMaxMusic3Vocoder vocoder = new MiniMaxMusic3Vocoder();
        vocoder.LoadWeights(vocoderWeights);

        using FileStream tokenizerJson = File.OpenRead(tokenizerPath);
        GgufTokenizer tokenizer = HfTokenizerJson.LoadByteLevelBpe(tokenizerJson);

        MiniMaxMusic3ArPipeline autoregressive = new MiniMaxMusic3ArPipeline(languageModel, depthDecoder);
        MiniMaxMusic3FlowPipeline flow = new MiniMaxMusic3FlowPipeline(context.Backend, conditionEncoder, dit);
        Logs.Info($"[Audio][MiniMaxMusic3] Loaded {repo} ({(quant is null ? "bf16" : quant)} language model, "
            + $"{vocoder.SampleRate} Hz stereo).");

        MusicAudio Synth(IBackend backend, MusicRequest request, CancellationToken ct)
        {
            ct.ThrowIfCancellationRequested();
            // ACE-Step's mapping, which the CLI and the request DTO already speak: the style/description rides in
            // Genre and the lyrics in Prompt.
            string caption = string.IsNullOrWhiteSpace(request.Genre)
                ? "A warm instrumental track."
                : request.Genre;
            string lyrics = string.IsNullOrWhiteSpace(request.Prompt) ? "[instrumental]" : request.Prompt;
            (int[] conditional, int[] unconditional) = MiniMaxMusic3Prompt.Tokenize(tokenizer, caption, lyrics);

            int frames = (int)Math.Clamp(
                Math.Round(request.Duration * MiniMaxMusic3ArPipeline.FrameRate), 1, MiniMaxMusic3ArPipeline.MaxAudioFrames);
            int steps = request.InferSteps is > 0 ? request.InferSteps.Value : MiniMaxMusic3FlowPipeline.DefaultSteps;
            float cfgScale = request.CfgScale is > 0
                ? (float)request.CfgScale.Value
                : MiniMaxMusic3FlowPipeline.DefaultCfgScale;

            long autoregressiveStart = Environment.TickCount64;
            float[] frameHiddens;
            int produced;
            // Preloads INSIDE the guarded region: an OOM mid-preload otherwise leaks every already-uploaded
            // batch (the finally would never run), leaving the card full for all later generations.
            try
            {
                backend.PreloadWeights(languageModel.EnumerateWeights());
                backend.PreloadWeights(depthDecoder.EnumerateWeights());
                (frameHiddens, produced) = autoregressive.Generate(backend, conditional, unconditional, frames,
                    request.Seed, OnFrame, ct);
            }
            finally
            {
                backend.Sync();
                backend.FreeWeights(languageModel.EnumerateWeights());
                backend.FreeWeights(depthDecoder.EnumerateWeights());
            }
            long autoregressiveMs = Environment.TickCount64 - autoregressiveStart;
            Logs.Info($"[Audio][MiniMaxMusic3] Autoregressive stage produced {produced} frames "
                + $"({produced / MiniMaxMusic3ArPipeline.FrameRate:0.0}s of audio) in {autoregressiveMs / 1000.0:0.0}s "
                + $"({autoregressiveMs / (double)Math.Max(1, produced):0.0} ms/frame); flow-matching {steps} steps per window.");
            long flowStart = Environment.TickCount64;

            Tensor[] chunks;
            try
            {
                backend.PreloadWeights(conditionEncoder.EnumerateWeights());
                backend.PreloadWeights(dit.EnumerateWeights());
                chunks = flow.Denoise(frameHiddens, produced, steps, cfgScale, request.Seed, OnStep, ct);
            }
            finally
            {
                backend.Sync();
                backend.FreeWeights(dit.EnumerateWeights());
                backend.FreeWeights(conditionEncoder.EnumerateWeights());
            }

            long flowMs = Environment.TickCount64 - flowStart;
            long vocodeStart = Environment.TickCount64;
            Tensor[] waveforms = new Tensor[chunks.Length];
            try
            {
                backend.PreloadWeights(vocoder.EnumerateWeights());
                for (int i = 0; i < chunks.Length; i++)
                {
                    ct.ThrowIfCancellationRequested();
                    waveforms[i] = vocoder.Decode(backend, chunks[i]);
                }
                (float[] left, float[] right) = MiniMaxMusic3FlowPipeline.Stitch(waveforms, MiniMaxMusic3Vocoder.LatentHopLength);
                Logs.Info($"[Audio][MiniMaxMusic3] Stage timing — autoregressive {autoregressiveMs / 1000.0:0.0}s, "
                    + $"flow {flowMs / 1000.0:0.0}s over {chunks.Length} window(s), "
                    + $"vocoder {(Environment.TickCount64 - vocodeStart) / 1000.0:0.0}s.");
                return MusicAudio.Stereo(left, right);
            }
            finally
            {
                foreach (Tensor waveform in waveforms)
                {
                    waveform?.Dispose();
                }
                foreach (Tensor chunk in chunks)
                {
                    chunk.Dispose();
                }
                backend.Sync();
                backend.FreeWeights(vocoder.EnumerateWeights());
            }
        }

        IDisposable?[] keep =
        [
            autoregressive, flow, languageModel, depthDecoder, conditionEncoder, dit, vocoder, quantCache, depthQuantCache, ditCast, loras,
            .. languageLoaders, .. depthLoaders, .. conditionLoaders, .. ditLoaders, .. vocoderLoaders,
        ];
        return new MusicRunner(vocoder.SampleRate, Synth, keep);
    }

    /// <summary>Merges the request's LoRAs into the language model and the transformer before either is loaded, the same weight-space merge the diffusion pipelines use. Both dictionaries are offered to the stack under <see cref="ModelAssets.Lora.LoraTarget.Transformer"/> and each layer lands wherever its key exists — no MiniMax Music 3 LoRAs have been published yet, so which component a real one targets is not yet known.</summary>
    private static ModelAssets.Lora.LoraStack? ApplyLoras(MusicLoadContext context,
        ref IReadOnlyDictionary<string, Tensor> languageWeights, ref IReadOnlyDictionary<string, Tensor> ditWeights)
    {
        if (context.Loras is null or { Entries.Count: 0 })
        {
            return null;
        }
        ModelAssets.Lora.LoraStack stack = new ModelAssets.Lora.LoraStack();
        foreach (LoraEntry lora in context.Loras.Entries)
        {
            stack.AddFromPath(lora.Model, (float)lora.Weight);
        }
        Dictionary<string, Tensor> language = new Dictionary<string, Tensor>(languageWeights, StringComparer.Ordinal);
        Dictionary<string, Tensor> transformer = new Dictionary<string, Tensor>(ditWeights, StringComparer.Ordinal);
        int merged = stack.ApplyTo(language, ModelAssets.Lora.LoraTarget.Transformer, context.Backend)
            + stack.ApplyTo(transformer, ModelAssets.Lora.LoraTarget.Transformer, context.Backend);
        if (merged == 0)
        {
            Logs.Warning($"[Audio][MiniMaxMusic3] None of the {context.Loras.Entries.Count} supplied LoRA(s) matched a weight "
                + "in the language model or the transformer — the generation will be unchanged.");
        }
        else
        {
            Logs.Info($"[Audio][MiniMaxMusic3] Merged {merged} weight(s) from {context.Loras.Entries.Count} LoRA(s).");
        }
        languageWeights = language;
        ditWeights = transformer;
        return stack;
    }

    /// <summary>The autoregressive stage is the long pole — silence in the log reads as a hang.</summary>
    private static void OnFrame(int done, int total)
    {
        if (done <= 3 || done % 100 == 0 || done == total)
        {
            Logs.Info($"[Audio][MiniMaxMusic3] Frame {done}/{total} ({done / MiniMaxMusic3ArPipeline.FrameRate:0.0}s of audio).");
        }
    }

    private static void OnStep(int done, int total)
    {
        if (done <= 2 || done % 10 == 0 || done == total)
        {
            Logs.Info($"[Audio][MiniMaxMusic3] Flow-matching step {done}/{total}.");
        }
    }
}
