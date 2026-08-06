using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Codecs.Mimi;
using HartsyInference.Audio.Models.Kyutai;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Audio;

/// <summary>Kyutai TTS (kyutai/tts-1.6b-en_fr) — a Moshi delayed-streams model: the Helium backbone steps once per
/// 12.5 Hz frame over summed text + previous-audio embeddings while the depformer emits the 32 Mimi codebooks, which
/// Mimi decodes to 24 kHz. Voice is a pre-embedded speaker from <c>kyutai/tts-voices</c>, not a raw reference clip.
///
/// <para>Synthesis mirrors moshi's <c>script_to_entries</c>: one entry per word, SentencePiece-tokenized, with
/// per-word articulation padding and the Main speaker token on the first word; the CFG-distilled guidance
/// coefficient (2.0) is a conditioning input, not a second forward.</para></summary>
internal static class KyutaiTtsModel
{
    private const string Repo = "kyutai/tts-1.6b-en_fr";
    private const string VoicesRepo = "kyutai/tts-voices";
    // Checkpoint-pinned asset names for this repo revision (the DSM release uses hashed filenames). The
    // '1e68beda@240' tag pins the voice-embedding version to the backbone it was trained against.
    private const string BackboneFile = "dsm_tts_1e68beda@240.safetensors";
    private const string MimiFile = "tokenizer-e351c8d8-checkpoint125.safetensors";
    private const string SpmFile = "tokenizer_spm_8k_en_fr_audio.model";
    private const string VoiceSuffix = ".1e68beda@240.safetensors";
    private const string DefaultVoice = "expresso/ex03-ex01_happy_001_channel1_334s.wav";

    /// <summary>The Kyutai TTS descriptor.</summary>
    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => Repo,
        LoadAsync = async (_, _, cancel) =>
        {
            string dsmPath = await AudioModelCache.GetAsync(Repo, BackboneFile, category: "tts", ct: cancel).ConfigureAwait(false);
            string mimiPath = await AudioModelCache.GetAsync(Repo, MimiFile, category: "tts", ct: cancel).ConfigureAwait(false);
            string spmPath = await AudioModelCache.GetAsync(Repo, SpmFile, category: "tts", ct: cancel).ConfigureAwait(false);
            await EnsureVoiceAsync(DefaultVoice, cancel).ConfigureAwait(false);

            Session session = Session.Load(dsmPath, mimiPath, spmPath);
            Logs.Info("[Audio][Kyutai-TTS] Loaded kyutai/tts-1.6b-en_fr (Helium backbone + Mimi DSM, 24 kHz).");
            return new TtsRunner(session.SampleRate, session.Synthesize, session);
        },
    };

    /// <summary>Ensures a tts-voices speaker embedding is present, mapping the voice name (a repo-relative path) to
    /// its versioned safetensors filename.</summary>
    private static async Task<string> EnsureVoiceAsync(string voiceName, CancellationToken cancel)
    {
        string file = voiceName.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase) ? voiceName : voiceName + VoiceSuffix;
        return await AudioModelCache.GetAsync(VoicesRepo, file, category: "tts", ct: cancel).ConfigureAwait(false);
    }

    /// <summary>A loaded Kyutai TTS model: generator + Mimi codec + SentencePiece tokenizer, plus a cache of
    /// transposed voice embeddings. Owns the weight loaders and disposes everything together.</summary>
    private sealed unsafe class Session : IDisposable
    {
        private readonly MoshiTtsGenerator _generator;
        private readonly Mimi _codec;
        private readonly KyutaiSttTokenizer _tokenizer;
        private readonly IDisposable[] _loaders;
        private readonly Dictionary<string, Tensor> _voices = new(StringComparer.Ordinal);
        private readonly object _voiceLock = new();
        private int _disposed;

        private Session(MoshiTtsGenerator generator, Mimi codec, KyutaiSttTokenizer tokenizer, IDisposable[] loaders)
        {
            _generator = generator;
            _codec = codec;
            _tokenizer = tokenizer;
            _loaders = loaders;
        }

        public int SampleRate => 24_000;

        public static Session Load(string dsmPath, string mimiPath, string spmPath)
        {
            SafeTensorsLoader dsm = new SafeTensorsLoader();
            dsm.Load(dsmPath);
            SafeTensorsLoader mimi = new SafeTensorsLoader();
            mimi.Load(mimiPath);

            MoshiTtsGenerator generator = new MoshiTtsGenerator();
            generator.LoadWeights(dsm.GetAllTensors());
            generator.SetZeroToken(-1);   // moshi ScaledEmbedding zero token → zero contribution
            Mimi codec = new Mimi(MimiConfig.Mimi24kHzDsm);
            codec.LoadWeights(mimi.GetAllTensors());
            KyutaiSttTokenizer tokenizer = new KyutaiSttTokenizer(spmPath);
            return new Session(generator, codec, tokenizer, [dsm, mimi]);
        }

        public float[] Synthesize(IBackend backend, TtsJob job)
        {
            if (string.IsNullOrWhiteSpace(job.Text))
            {
                return [];
            }
            Tensor voice = GetVoice(string.IsNullOrEmpty(job.Voice) ? DefaultVoice : job.Voice);

            // moshi script_to_entries with padding_between=1: one entry per word, the Main speaker token on the
            // first word, and forced per-word articulation padding so the model paces words as trained.
            const int paddingBetween = 1;
            List<KyutaiTextScheduler.Entry> entries = [];
            bool first = true;
            foreach (string word in job.Text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            {
                List<int> ids = [.. _tokenizer.Encode(word)];
                if (ids.Count == 0)
                {
                    continue;
                }
                if (first)
                {
                    ids.Insert(0, KyutaiTextScheduler.Main);
                    first = false;
                }
                int padding = Math.Max(0, paddingBetween + ids.Count - 1);
                entries.Add(new KyutaiTextScheduler.Entry(ids, word, padding));
            }
            if (entries.Count == 0)
            {
                return [];
            }

            // Frame budget: the 16-frame text lead + per-word tokens/padding + tail, capped at the 500-position budget.
            int maxFrames = 16 + 8;
            foreach (KyutaiTextScheduler.Entry entry in entries)
            {
                maxFrames += entry.Tokens.Count + entry.Padding + 2;
            }
            maxFrames = Math.Min(499, maxFrames);

            using Tensor cross = _generator.Conditioner.ComputeCross(backend, voice);
            // CFG-distilled model: guidance is a conditioning input and moshi's TTS default is 2.0. A coefficient of
            // 1.0 means no guidance, so the text is not enforced.
            using Tensor sumCond = _generator.Conditioner.ComputeSum(backend, MoshiConditioner.CfgBin(2.0f));

            KyutaiTextScheduler scheduler = new KyutaiTextScheduler(secondStreamAhead: 2, maxPadding: 8, initialPadding: 2);
            int[,] codes = _generator.Generate(backend, scheduler, entries, cross, sumCond,
                maxFrames: maxFrames, audioTemp: 0.6f, seed: job.Seed);
            int frames = codes.GetLength(1);
            if (frames == 0)
            {
                Logs.Warning("[Audio][Kyutai-TTS] Generation produced no audio frames.");
                return [];
            }

            Tensor codeTensor = new Tensor(new TensorShape(1, MoshiTtsGenerator.NumCodebooks, frames), DType.I32);
            int* codePtr = (int*)codeTensor.DataPointer;
            for (int k = 0; k < MoshiTtsGenerator.NumCodebooks; k++)
            {
                for (int f = 0; f < frames; f++)
                {
                    codePtr[(long)k * frames + f] = codes[k, f];
                }
            }
            using Tensor audioTensor = _codec.Decode(backend, codeTensor, batch: 1, tFrames: frames);
            codeTensor.Dispose();

            int samples = (int)audioTensor.Shape[audioTensor.Shape.Rank - 1];
            float[] audio = new float[samples];
            fixed (float* destination = audio)
            {
                Buffer.MemoryCopy((void*)audioTensor.DataPointer, destination, (long)samples * 4, (long)samples * 4);
            }
            Logs.Info($"[Audio][Kyutai-TTS] {entries.Count} words → {frames} frames → {samples / (double)SampleRate:0.0}s.");
            return audio;
        }

        /// <summary>Loads (and caches) a voice embedding transposed to the conditioner's [1,T,512] layout; the
        /// tts-voices files ship <c>speaker_wavs</c> channels-first.</summary>
        private Tensor GetVoice(string voiceName)
        {
            lock (_voiceLock)
            {
                if (_voices.TryGetValue(voiceName, out Tensor? cached))
                {
                    return cached;
                }
                string path = EnsureVoiceAsync(voiceName, CancellationToken.None).GetAwaiter().GetResult();
                using SafeTensorsLoader loader = new SafeTensorsLoader();
                loader.Load(path);
                Tensor speakerWavs = loader.GetAllTensors()["speaker_wavs"];
                int speakerDim = (int)speakerWavs.Shape[1];
                int frames = (int)speakerWavs.Shape[2];
                Tensor voice = new Tensor(new TensorShape(1, frames, speakerDim), DType.F32);
                float* destination = (float*)voice.DataPointer;
                float* source = (float*)speakerWavs.DataPointer;
                for (int c = 0; c < speakerDim; c++)
                {
                    for (int t = 0; t < frames; t++)
                    {
                        destination[(long)t * speakerDim + c] = source[(long)c * frames + t];
                    }
                }
                _voices[voiceName] = voice;
                return voice;
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }
            _generator.Dispose();
            _tokenizer.Dispose();
            lock (_voiceLock)
            {
                foreach (Tensor voice in _voices.Values)
                {
                    voice.Dispose();
                }
                _voices.Clear();
            }
            foreach (IDisposable loader in _loaders)
            {
                loader?.Dispose();
            }
        }
    }
}
