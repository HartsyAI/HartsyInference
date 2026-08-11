using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Chatterbox;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Audio;

/// <summary>Chatterbox (ResembleAI/chatterbox) — T3 LM → CosyVoice2 S3Gen flow → HiFTNet, 24 kHz. Merges
/// <c>t3_cfg.safetensors</c> (under <c>t3.</c>), <c>s3gen.safetensors</c> (under <c>s3gen.</c>), and
/// <c>ve.safetensors</c> (under <c>ve.</c>). The default voice runs off the precomputed conditionals in
/// <c>conds.pt</c>; a reference clip switches to zero-shot cloning, where the pipeline derives all
/// conditioning (voice-encoder embedding, T3 cond tokens, S3Gen reference dict) from the raw audio.</summary>
internal static class ChatterboxModel
{
    private const string Repo = "ResembleAI/chatterbox";

    /// <summary>The Chatterbox descriptor.</summary>
    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => Repo,
        LoadAsync = async (_, _, cancel) =>
        {
            string t3Path = await AudioModelCache.GetAsync(Repo, "t3_cfg.safetensors", category: "tts", ct: cancel).ConfigureAwait(false);
            string s3Path = await AudioModelCache.GetAsync(Repo, "s3gen.safetensors", category: "tts", ct: cancel).ConfigureAwait(false);
            string vePath = await AudioModelCache.GetAsync(Repo, "ve.safetensors", category: "tts", ct: cancel).ConfigureAwait(false);
            string tokenizerPath = await AudioModelCache.GetAsync(Repo, "tokenizer.json", category: "tts", ct: cancel).ConfigureAwait(false);
            string condsPath = await AudioModelCache.GetAsync(Repo, "conds.pt", category: "tts", ct: cancel).ConfigureAwait(false);

            // Merge the three checkpoints under the prefixes ChatterboxPipeline.LoadWeights validates.
            Dictionary<string, Tensor> merged = new Dictionary<string, Tensor>(StringComparer.Ordinal);
            SafeTensorsLoader t3Loader = new SafeTensorsLoader();
            t3Loader.Load(t3Path);
            foreach (KeyValuePair<string, Tensor> entry in t3Loader.GetAllTensors())
            {
                merged["t3." + entry.Key] = entry.Value;
            }
            SafeTensorsLoader s3Loader = new SafeTensorsLoader();
            s3Loader.Load(s3Path);
            foreach (KeyValuePair<string, Tensor> entry in s3Loader.GetAllTensors())
            {
                merged["s3gen." + entry.Key] = entry.Value;
            }
            SafeTensorsLoader veLoader = new SafeTensorsLoader();
            veLoader.Load(vePath);
            foreach (KeyValuePair<string, Tensor> entry in veLoader.GetAllTensors())
            {
                merged["ve." + entry.Key] = entry.Value;
            }

            ChatterboxConfig config = ChatterboxConfig.Default;
            CosyVoiceConfig cosy = CosyVoiceConfig.V2_0_5B;   // Chatterbox S3Gen == CosyVoice2-0.5B
            ChatterboxT3 t3 = new ChatterboxT3(config);
            CosyVoiceFlow flow = new CosyVoiceFlow(cosy);
            HiFTNetVocoder vocoder = new HiFTNetVocoder(cosy.Hift);
            CamPlusSpeakerEncoder speakerEncoder = new CamPlusSpeakerEncoder(cosy.Flow.SpeakerEmbedDim);
            S3Tokenizer s3Tokenizer = new S3Tokenizer();
            ChatterboxVoiceEncoder voiceEncoder = new ChatterboxVoiceEncoder();
            ChatterboxPipeline pipeline = new ChatterboxPipeline(config, t3, flow, vocoder, speakerEncoder, s3Tokenizer, voiceEncoder);
            pipeline.LoadWeights(merged);   // key-validation gate: throws on any missing/renamed key.

            // Precomputed default-voice conditionals. The pickle loader materializes into owned memory, so the three
            // tensors stay valid as long as it is kept alive.
            PytorchPickleLoader condsLoader = new PytorchPickleLoader();
            condsLoader.Load(condsPath, recursiveFlatten: true);
            IReadOnlyDictionary<string, Tensor> conds = condsLoader.GetAllTensors();
            Tensor referenceSpeaker = Flatten(conds["t3.speaker_emb"], config.SpeakerEmbedDim);
            Tensor flowSpeaker = conds["gen.embedding"];
            int[] promptSpeechTokens = ToInts(conds["t3.cond_prompt_speech_tokens"]);

            using Stream tokenizerStream = File.OpenRead(tokenizerPath);
            ChatterboxEnTokenizer tokenizer = new ChatterboxEnTokenizer(tokenizerStream);
            Logs.Info("[Audio][Chatterbox] Loaded ResembleAI/chatterbox (T3 + S3Gen + HiFTNet, 24 kHz, default voice).");

            IDisposable?[] keep = [pipeline, t3, flow, vocoder, speakerEncoder, s3Tokenizer, voiceEncoder,
                t3Loader, s3Loader, veLoader, condsLoader, referenceSpeaker];
            return new TtsRunner(config.SampleRate, (backend, job) =>
            {
                int[] textTokens = tokenizer.EncodeWithStartStop(job.Text);
                float exaggeration = job.Exaggeration.HasValue ? (float)job.Exaggeration.Value : config.Exaggeration;
                float cfgWeight = job.CfgScale.HasValue ? (float)job.CfgScale.Value : config.CfgWeight;
                if (job.ReferenceMono24k is not null && job.ReferenceMono24k.Length > 0)
                {
                    return pipeline.Synthesize(backend, textTokens, refSpeakerEmbed: null, exaggeration, job.Seed,
                        referenceAudio: job.ReferenceMono24k, referenceSampleRate: 24_000, cfgWeight: cfgWeight);
                }
                return pipeline.Synthesize(backend, textTokens, referenceSpeaker, exaggeration, job.Seed,
                    flowSpeakerEmbed: flowSpeaker, t3PromptSpeechTokens: promptSpeechTokens, cfgWeight: cfgWeight);
            }, keep);
        },
    };

    /// <summary>Copies the first <paramref name="count"/> floats into a fresh owned tensor (the T3 voice-encoder
    /// embedding ships as <c>[1, 256]</c>; the pipeline takes the flat <c>[256]</c>).</summary>
    private static Tensor Flatten(Tensor source, int count)
    {
        Tensor flat = new Tensor(new TensorShape(count), DType.F32);
        source.AsSpan<float>()[..count].CopyTo(flat.AsSpan<float>());
        return flat;
    }

    /// <summary>Reads an int64 token tensor as the <c>int[]</c> the T3 takes.</summary>
    private static int[] ToInts(Tensor source)
    {
        ReadOnlySpan<long> values = source.AsSpan<long>();
        int[] result = new int[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = (int)values[i];
        }
        return result;
    }
}
