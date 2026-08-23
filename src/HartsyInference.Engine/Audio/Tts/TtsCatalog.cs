using System.IO.Compression;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Models.Bark;
using HartsyInference.Audio.Models.Codecs.EnCodec;
using HartsyInference.Audio.Models.Codecs.Mimi;
using HartsyInference.Audio.Models.Csm;
using HartsyInference.Audio.Models.Dia;
using HartsyInference.Audio.Models.Orpheus;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Audio;

/// <summary>The text-to-speech model registry: catalog id → descriptor, plus the shared runner cache. Adding a model is a descriptor, not a class.</summary>
internal static class TtsCatalog
{
    /// <summary>Public-domain CMU Pronouncing Dictionary — the English G2P source, fetched on first use.</summary>
    private const string CmudictUrl = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict";

    /// <summary>Resolves a catalog id to its descriptor, or throws naming what is available.</summary>
    internal static TtsModelDescriptor Resolve(string id)
    {
        if (Registry.TryGetValue(id ?? string.Empty, out TtsModelDescriptor? descriptor))
        {
            return descriptor;
        }
        throw new NotSupportedException(
            $"No speech model '{id}' is registered. Available: {string.Join(", ", Registry.Keys)} "
            + "(pass a variant as 'id:variant', e.g. 'kokoro:af_heart').");
    }

    private static Dictionary<string, TtsModelDescriptor>? _registry;

    // Built on first use, not in a static initializer: the descriptor properties below would still be null if the
    // map were materialized while this type's static initializers were still running.
    private static Dictionary<string, TtsModelDescriptor> Registry => _registry ??= new(StringComparer.OrdinalIgnoreCase)
    {
        ["vibevoice"] = VibeVoice,
        ["kokoro"] = Kokoro,
        ["bark"] = BarkTtsModel.Descriptor,
        ["dia"] = DiaTtsModel.Descriptor,
        ["orpheus"] = OrpheusTtsModel.Descriptor,
        ["csm"] = CsmTtsModel.Descriptor,
        ["neutts"] = NeuTtsModel.Descriptor,
        ["fishspeech"] = FishSpeechModel.Descriptor,
        ["cosyvoice"] = CosyVoiceModel.Descriptor,
        ["f5"] = F5TtsModel.Descriptor,
        ["qwen3tts"] = Qwen3TtsModel.Descriptor,
        ["chatterbox"] = ChatterboxModel.Descriptor,
        ["kyutaitts"] = KyutaiTtsModel.Descriptor,
        ["piper"] = PiperModel.Descriptor,
        ["melotts"] = MeloTtsModel.Descriptor,
        ["sparktts"] = SparkTtsModel.Descriptor,
        ["pockettts"] = PocketTtsModel.Descriptor,
        ["styletts2"] = StyleTts2Model.Descriptor,
        ["zonos"] = ZonosModel.Descriptor,
        ["gptsovits"] = GptSoVitsModel.Descriptor,
        ["zipvoice"] = ZipVoiceModel.Descriptor,
    };

    /// <summary>VibeVoice — long-form multi-speaker synthesis (default, 1.5B) or the single-speaker low-latency Realtime-0.5B variant (<c>vibevoice:realtime</c>). Both need a 24 kHz voice reference.</summary>
    internal static TtsModelDescriptor VibeVoice { get; } = new TtsModelDescriptor
    {
        ResolveRepo = variant => IsRealtimeVariant(variant) ? "microsoft/VibeVoice-Realtime-0.5B" : "microsoft/VibeVoice-1.5B",
        LoadAsync = async (_, variant, cancel) =>
        {
            if (IsRealtimeVariant(variant))
            {
                VibeVoiceStreamingPipeline pipeline = await VibeVoiceStreamingPipeline.LoadAsync(cancel).ConfigureAwait(false);
                float[] SynthRt(IBackend backend, TtsJob job)
                {
                    if (job.ReferenceWavPath is null)
                        throw new InvalidOperationException(
                            "VibeVoice-Realtime needs a voice reference — supply a short WAV clip as the request's reference.");
                    return pipeline.Synthesize(backend, job.Text, job.ReferenceWavPath, cfgScale: job.CfgScale is > 0 ? (float)job.CfgScale.Value : 1.5f,
                        diffusionSteps: job.NfeStep, seed: job.Seed);
                }
                IAsyncEnumerable<AudioChunk> StreamRt(IBackend backend, TtsJob job, CancellationToken cancel)
                {
                    if (job.ReferenceWavPath is null)
                        throw new InvalidOperationException(
                            "VibeVoice-Realtime needs a voice reference — supply a short WAV clip as the request's reference.");
                    return pipeline.SynthesizeStream(backend, job.Text, job.ReferenceWavPath, maxNewFrames: 600,
                        cfgScale: job.CfgScale is > 0 ? (float)job.CfgScale.Value : 1.5f, diffusionSteps: job.NfeStep,
                        seed: job.Seed, cancel);
                }
                return new StreamingTtsRunner(VibeVoiceStreamingPipeline.SampleRate, SynthRt, StreamRt, pipeline);
            }

            VibeVoicePipeline pipeline15 = await VibeVoicePipeline.LoadAsync(cancel).ConfigureAwait(false);
            float[] Synth(IBackend backend, TtsJob job)
            {
                if (job.ReferenceWavPath is null)
                {
                    throw new InvalidOperationException(
                        "VibeVoice needs a voice reference — supply a short WAV clip as the request's reference.");
                }
                return pipeline15.Synthesize(backend, [job.Text], [job.ReferenceWavPath], maxNewTokens: 1024,
                    progress: null, temperature: 0.95f, topP: 0.95f, seed: job.Seed,
                    cfgScale: job.CfgScale, diffusionSteps: job.NfeStep);
            }
            IAsyncEnumerable<AudioChunk> Stream(IBackend backend, TtsJob job, CancellationToken cancel)
            {
                if (job.ReferenceWavPath is null)
                {
                    throw new InvalidOperationException(
                        "VibeVoice needs a voice reference — supply a short WAV clip as the request's reference.");
                }
                return pipeline15.SynthesizeStream(backend, [job.Text], [job.ReferenceWavPath], maxNewTokens: 1024,
                    progress: null, temperature: 0.95f, topP: 0.95f, seed: job.Seed,
                    cfgScale: job.CfgScale, diffusionSteps: job.NfeStep, cancel);
            }
            return new StreamingTtsRunner(VibeVoicePipeline.SampleRate, Synth, Stream, pipeline15);
        },
    };

    /// <summary>Matches the <c>vibevoice:realtime</c> / <c>vibevoice:realtime-0.5b</c> variant token.</summary>
    private static bool IsRealtimeVariant(string variant) =>
        (variant ?? string.Empty).Contains("realtime", StringComparison.OrdinalIgnoreCase);

    /// <summary>Kokoro-82M — fast CPU-capable TTS at 24 kHz over the engine's English G2P, backed by the CMU dictionary; voice packs are fetched per voice (default <c>af_heart</c>).</summary>
    internal static TtsModelDescriptor Kokoro { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => "hexgrad/Kokoro-82M",
        LoadAsync = async (_, _, cancel) =>
        {
            string cmudict = AudioModelRoot.SharedFile("cmudict.dict");
            if (!File.Exists(cmudict))
            {
                Logs.Info("[Audio][Kokoro] Downloading the public-domain CMU Pronouncing Dictionary (cmudict.dict)...");
                await AudioFileFetcher.EnsureAsync(CmudictUrl, cmudict, cancel).ConfigureAwait(false);
                Logs.Info("[Audio][Kokoro] CMU dictionary ready.");
            }
            EnglishG2P g2p = new EnglishG2P(cmudict);
            KokoroPipeline pipeline = await KokoroPipeline.LoadAsync(cancel).ConfigureAwait(false);
            await EnsureKokoroVoiceAsync("af_heart", cancel).ConfigureAwait(false);
            return new TtsRunner(24_000, (backend, job) =>
            {
                string voice = string.IsNullOrEmpty(job.Voice) ? "af_heart" : job.Voice;
                EnsureKokoroVoiceAsync(voice, CancellationToken.None).GetAwaiter().GetResult();
                float speed = job.Speed.HasValue ? (float)job.Speed.Value : 1f;
                return pipeline.Synthesize(backend, g2p.ToIpa(job.Text), voiceName: voice, speed: speed);
            }, pipeline);
        },
    };

    /// <summary>Ensures a Kokoro voice pack exists as the raw-float32 <c>.bin</c> the engine reads. The HF repo ships each voice as a torch-saved <c>.pt</c> whose single contiguous f32 storage at <c>*/data/0</c> is that payload.</summary>
    private static async Task EnsureKokoroVoiceAsync(string voiceName, CancellationToken cancel)
    {
        string repoDir = AudioModelCache.GetRepoDirectory("hexgrad/Kokoro-82M", "tts");
        string binPath = Path.Combine(repoDir, "voices", $"{voiceName}.bin");
        if (File.Exists(binPath))
        {
            return;
        }
        Logs.Info($"[Audio][Kokoro] Fetching voice pack '{voiceName}'...");
        string ptPath = await AudioModelCache.GetAsync("hexgrad/Kokoro-82M", $"voices/{voiceName}.pt", category: "tts", ct: cancel).ConfigureAwait(false);
        Directory.CreateDirectory(Path.GetDirectoryName(binPath)!);
        using ZipArchive zip = ZipFile.OpenRead(ptPath);
        ZipArchiveEntry storage = zip.Entries.FirstOrDefault(e => e.FullName.Replace('\\', '/').EndsWith("/data/0", StringComparison.Ordinal))
            ?? throw new InvalidOperationException($"Unexpected Kokoro voice format in '{ptPath}' — no tensor storage entry.");
        string tempPath = binPath + ".tmp";
        using (Stream source = storage.Open())
        using (FileStream destination = File.Create(tempPath))
        {
            await source.CopyToAsync(destination, cancel).ConfigureAwait(false);
        }
        File.Move(tempPath, binPath, overwrite: true);
        Logs.Info($"[Audio][Kokoro] Voice pack '{voiceName}' ready.");
    }
}
