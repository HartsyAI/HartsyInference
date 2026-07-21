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
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Audio;

/// <summary>The text-to-speech model registry: catalog id → descriptor, plus the shared runner cache. Adding a model
/// is a descriptor, not a class.</summary>
internal static class TtsCatalog
{
    /// <summary>Resident speech pipelines, keyed by resolved repo.</summary>
    internal static readonly AudioRunnerCache<ITtsRunner> Cache = new AudioRunnerCache<ITtsRunner>();

    /// <summary>Public-domain CMU Pronouncing Dictionary — the English G2P source, fetched on first use.</summary>
    private const string CmudictUrl = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict";

    /// <summary>The registered catalog ids, for error messages.</summary>
    internal static IReadOnlyCollection<string> RegisteredIds => Registry.Keys;

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
        ["bark"] = Bark,
        ["dia"] = Dia,
        ["orpheus"] = Orpheus,
        ["csm"] = Csm,
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

    /// <summary>VibeVoice 1.5B — long-form multi-speaker synthesis; built-in tokenizer, requires a 24 kHz reference.</summary>
    internal static TtsModelDescriptor VibeVoice { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => "microsoft/VibeVoice-1.5B",
        LoadAsync = async (_, cancel) =>
        {
            VibeVoicePipeline pipeline = await VibeVoicePipeline.LoadAsync(cancel).ConfigureAwait(false);
            return new TtsRunner(24_000, (backend, job) =>
            {
                if (job.ReferenceWavPath is null)
                {
                    throw new InvalidOperationException(
                        "VibeVoice needs a voice reference — supply a short WAV clip as the request's reference.");
                }
                return pipeline.Synthesize(backend, [job.Text], [job.ReferenceWavPath], maxNewTokens: 1024);
            }, pipeline);
        },
    };

    /// <summary>Kokoro-82M — fast CPU-capable TTS at 24 kHz over the engine's English G2P, backed by the CMU
    /// dictionary; voice packs are fetched per voice (default <c>af_heart</c>).</summary>
    internal static TtsModelDescriptor Kokoro { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => "hexgrad/Kokoro-82M",
        LoadAsync = async (_, cancel) =>
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

    /// <summary>Bark (suno/bark) — 3-stage GPT cascade + EnCodec 24 kHz, text via the multilingual BERT WordPiece
    /// tokenizer plus Bark's text-encoding offset.</summary>
    internal static TtsModelDescriptor Bark { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => "suno/bark",
        LoadAsync = async (_, cancel) =>
        {
            string vocabPath = await AudioModelCache.GetAsync("google-bert/bert-base-multilingual-cased", "vocab.txt", ct: cancel).ConfigureAwait(false);
            (IReadOnlyDictionary<string, Tensor> dict, IDisposable loader) = await LoadBarkWeightsAsync(cancel).ConfigureAwait(false);

            BarkConfig config = BarkConfig.Full;
            BarkCausalStage semantic = new BarkCausalStage(config.Stage, config.SemanticInputVocab, config.SemanticOutputVocab);
            semantic.LoadWeights(dict, "semantic");
            BarkCausalStage coarse = new BarkCausalStage(config.Stage, config.CoarseVocab, config.CoarseVocab);
            coarse.LoadWeights(dict, "coarse_acoustics");
            BarkFineModel fine = new BarkFineModel(config);
            fine.LoadWeights(dict, "fine_acoustics");

            Dictionary<string, Tensor> codecRaw = new Dictionary<string, Tensor>(StringComparer.Ordinal);
            foreach (KeyValuePair<string, Tensor> entry in dict)
            {
                if (entry.Key.StartsWith("codec_model.", StringComparison.Ordinal))
                {
                    codecRaw[entry.Key["codec_model.".Length..]] = entry.Value;
                }
            }
            EnCodec encodec = new EnCodec(EnCodecConfig.EnCodec24kHz);
            encodec.LoadWeights(MusicGenCheckpointConverter.ConvertEnCodec(codecRaw));

            BarkPipeline pipeline = new BarkPipeline(config, semantic, coarse, fine, encodec);
            BertWordPieceTokenizer bert = new BertWordPieceTokenizer(vocabPath, lowerCase: false);
            Logs.Info("[Audio][Bark] Loaded suno/bark (3-stage GPT + EnCodec 24 kHz).");

            // Loader kept alive: the F32 stage weights reference its tensors.
            return new TtsRunner(config.SampleRate,
                (backend, job) => pipeline.Synthesize(backend, AudioTextFrontend.BarkText(bert, job.Text, config.TextEncodingOffset), job.Seed),
                pipeline, loader);
        },
    };

    /// <summary>Dia-1.6B (0626 release) — byte-level two-speaker dialogue TTS at 44.1 kHz with the Descript DAC codec.</summary>
    internal static TtsModelDescriptor Dia { get; } = new TtsModelDescriptor
    {
        // The 0626 release, NOT the original Dia-1.6B: same architecture/keys/shapes, but the original checkpoint's
        // weights degenerate through the engine while 0626 produces the full multi-turn dialogue word-correct.
        ResolveRepo = _ => "nari-labs/Dia-1.6B-0626",
        LoadAsync = async (_, cancel) =>
        {
            string modelPath = await AudioModelCache.GetAsync("nari-labs/Dia-1.6B-0626", "pytorch_model.bin", ct: cancel).ConfigureAwait(false);
            // The canonical descript .pth has the layout the engine expects (the HF safetensors mirrors are reshaped).
            string dacPath = await AudioModelCache.GetAsync("descript/descript-audio-codec", "weights.pth", ct: cancel).ConfigureAwait(false);
            // 0626 is a flat pickle state_dict; non-recursive flatten keeps the encoder./decoder. prefixes intact.
            PytorchPickleLoader modelLoader = new PytorchPickleLoader();
            modelLoader.Load(modelPath, recursiveFlatten: false);
            PytorchPickleLoader dacLoader = new PytorchPickleLoader();
            dacLoader.Load(dacPath);

            DiaPipeline pipeline = new DiaPipeline(DiaConfig.Dia1_6B);
            pipeline.LoadWeights(modelLoader.GetAllTensors(), dacLoader.GetAllTensors());
            Logs.Info("[Audio][Dia] Loaded nari-labs/Dia-1.6B-0626 (byte-level dialogue TTS, 44.1 kHz).");

            return new TtsRunner(44_100,
                (backend, job) =>
                {
                    // Dia was trained on [S1]/[S2]-tagged dialogue; untagged text degenerates into repetition loops.
                    string text = job.Text.Contains("[S", StringComparison.Ordinal) ? job.Text : $"[S1] {job.Text}";
                    // The checkpoint degenerates to silence on short prompts (upstream behaves the same).
                    if (text.Length < 120)
                    {
                        Logs.Warning($"[Audio][Dia] Prompt is very short ({text.Length} chars) — Dia-1.6B tends to produce "
                            + "silence below ~2 sentences (upstream behaves the same). Use longer dialogue-style text.");
                    }
                    return pipeline.Generate(backend, AudioTextFrontend.DiaBytes(text), seed: job.Seed);
                },
                pipeline, modelLoader, dacLoader);
        },
    };

    /// <summary>Orpheus TTS — Llama-3.2-3B LM + SNAC 24 kHz, from a non-gated mirror of the license-gated release.</summary>
    internal static TtsModelDescriptor Orpheus { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => "unsloth/orpheus-3b-0.1-ft",
        LoadAsync = async (_, cancel) =>
        {
            (IReadOnlyDictionary<string, Tensor> backbone, IDisposable[] backboneLoaders) =
                await AudioCheckpoints.LoadAsync("unsloth/orpheus-3b-0.1-ft", cancel).ConfigureAwait(false);
            (IReadOnlyDictionary<string, Tensor> snac, IDisposable[] snacLoaders) =
                await AudioCheckpoints.LoadAsync("hubertsiuzdak/snac_24khz", cancel).ConfigureAwait(false);
            OrpheusPipeline pipeline = new OrpheusPipeline(OrpheusConfig.Orpheus3B);
            pipeline.LoadWeights(backbone, snac);
            Logs.Info("[Audio][Orpheus] Loaded unsloth/orpheus-3b-0.1-ft (Llama-3.2-3B + SNAC 24 kHz).");
            IDisposable?[] keep = [pipeline, .. backboneLoaders, .. snacLoaders];
            return new TtsRunner(pipeline.SampleRate,
                (backend, job) => pipeline.Synthesize(backend, AudioTextFrontend.OrpheusText(job.Text), seed: job.Seed), keep);
        },
    };

    /// <summary>Sesame CSM-1B — dual-transformer conversational TTS + Mimi 24 kHz, from a non-gated mirror that keeps
    /// the original-format key layout (not the transformers re-export, whose keys the engine loader would not match).</summary>
    internal static TtsModelDescriptor Csm { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => "nielsr/csm-1b",
        LoadAsync = async (_, cancel) =>
        {
            (IReadOnlyDictionary<string, Tensor> modelDict, IDisposable[] modelLoaders) =
                await AudioCheckpoints.LoadAsync("nielsr/csm-1b", cancel).ConfigureAwait(false);
            (IReadOnlyDictionary<string, Tensor> mimiDict, IDisposable[] mimiLoaders) =
                await AudioCheckpoints.LoadAsync("kyutai/mimi", cancel).ConfigureAwait(false);
            CsmModel model = new CsmModel(CsmConfig.V1B);
            model.LoadWeights(modelDict);
            Mimi mimi = new Mimi(MimiConfig.Mimi24kHz);
            mimi.LoadWeights(mimiDict);
            CsmPipeline pipeline = new CsmPipeline(CsmConfig.V1B, model, mimi);
            Logs.Info("[Audio][CSM] Loaded nielsr/csm-1b (dual-transformer + Mimi 24 kHz).");
            IDisposable?[] keep = [pipeline, .. modelLoaders, .. mimiLoaders];
            return new TtsRunner(24_000,
                (backend, job) => pipeline.Synthesize(backend, AudioTextFrontend.CsmText(job.Text), seed: job.Seed), keep);
        },
    };

    /// <summary>Ensures a Kokoro voice pack exists as the raw-float32 <c>.bin</c> the engine reads. The HF repo ships
    /// each voice as a torch-saved <c>.pt</c> whose single contiguous f32 storage at <c>*/data/0</c> is that payload.</summary>
    private static async Task EnsureKokoroVoiceAsync(string voiceName, CancellationToken cancel)
    {
        string repoDir = AudioModelCache.GetRepoDirectory("hexgrad/Kokoro-82M");
        string binPath = Path.Combine(repoDir, "voices", $"{voiceName}.bin");
        if (File.Exists(binPath))
        {
            return;
        }
        Logs.Info($"[Audio][Kokoro] Fetching voice pack '{voiceName}'...");
        string ptPath = await AudioModelCache.GetAsync("hexgrad/Kokoro-82M", $"voices/{voiceName}.pt", ct: cancel).ConfigureAwait(false);
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

    /// <summary>Loads the Bark transformers checkpoint — safetensors preferred, pickle fallback.</summary>
    private static async Task<(IReadOnlyDictionary<string, Tensor> Dict, IDisposable Loader)> LoadBarkWeightsAsync(CancellationToken cancel)
    {
        try
        {
            string path = await AudioModelCache.GetAsync("suno/bark", "model.safetensors", ct: cancel).ConfigureAwait(false);
            SafeTensorsLoader loader = new SafeTensorsLoader();
            loader.Load(path);
            return (loader.GetAllTensors(), loader);
        }
        catch (FileNotFoundException ex)
        {
            Logs.Debug($"[Audio][Bark] No model.safetensors ({ex.Message}); loading pytorch_model.bin.");
            string path = await AudioModelCache.GetAsync("suno/bark", "pytorch_model.bin", ct: cancel).ConfigureAwait(false);
            PytorchPickleLoader loader = new PytorchPickleLoader();
            loader.Load(path);
            return (loader.GetAllTensors(), loader);
        }
    }
}
