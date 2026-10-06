using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.FishAudio;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Audio;

/// <summary>Fish Audio S2 Pro (<c>fishaudio/s2-pro</c>) — a Dual-AR text-to-speech model (36-layer Qwen3-style slow
/// transformer plus a 4-layer fast transformer over 10 codebooks) decoded by the ModifiedDAC codec to 44.1 kHz mono.
/// Inline <c>[tag]</c> controls and <c>&lt;|speaker:N|&gt;</c> turns pass through in the text. The weights ship under the
/// Fish Audio Research License (non-commercial); voice cloning from a reference clip needs the codec encoder, which is
/// not ported yet.</summary>
internal static class FishAudioS2Model
{
    private const string Repo = "fishaudio/s2-pro";
    private const string TokenizerFile = "tokenizer.json";
    private const string CodecFile = "codec.pth";
    private static readonly string[] WeightPrefixes = ["text_model.", "audio_decoder."];

    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = variant => (variant ?? string.Empty).Contains('/', StringComparison.Ordinal) ? variant! : Repo,
        ResolveFiles = async (_, cancel) =>
        {
            // Companions first, the weights last: the weights are the artifact that marks the model installed.
            List<AudioModelFile> files = [new(TokenizerFile), new(CodecFile)];
            files.AddRange(await AudioCheckpoints.ResolveCheckpointFilesAsync(Repo, "tts", cancel, WeightPrefixes).ConfigureAwait(false));
            return files;
        },
        LoadAsync = async (_, _, cancel) =>
        {
            string tokenizerPath = await AudioModelCache.GetAsync(Repo, TokenizerFile, category: "tts", ct: cancel).ConfigureAwait(false);
            string codecPath = await AudioModelCache.GetAsync(Repo, CodecFile, category: "tts", ct: cancel).ConfigureAwait(false);
            (IReadOnlyDictionary<string, Tensor> model, IDisposable[] modelLoaders) =
                await AudioCheckpoints.LoadAsync(Repo, "tts", cancel, WeightPrefixes).ConfigureAwait(false);

            AnyFormatCheckpointLoader codecLoader = new();
            codecLoader.Load(codecPath);
            Dictionary<string, Tensor> codec = codecLoader.GetAllTensors();

            using FileStream json = File.OpenRead(tokenizerPath);
            GgufTokenizer tokenizer = HfTokenizerJson.LoadByteLevelBpe(json);
            FishAudioS2Pipeline pipeline = new(FishAudioS2Config.S2Pro, ModifiedDacConfig.S2,
                text => tokenizer.Encode(text, addSpecial: true));
            pipeline.LoadWeights(model, codec);
            Logs.Info("[Audio][FishAudioS2] Loaded fishaudio/s2-pro (Dual-AR 4B + ModifiedDAC, 44.1 kHz).");

            IDisposable?[] keep = [pipeline, codecLoader, .. modelLoaders];
            return new TtsRunner(pipeline.SampleRate, (backend, job) => Synthesize(pipeline, backend, job), keep);
        },
    };

    private static float[] Synthesize(FishAudioS2Pipeline pipeline, IBackend backend, TtsJob job)
    {
        if (job.Reference is not null || job.ReferenceWavPath is not null)
        {
            throw new NotSupportedException(
                "Fish Audio S2 voice cloning needs the codec encoder to turn the reference clip into codes, which is not "
                + "ported yet. Omit the reference to use the model's default voice.");
        }
        return pipeline.Synthesize(backend, new FishAudioS2Pipeline.Request
        {
            Text = job.Text,
            Temperature = job.Temperature is { } t ? (float)t : null,
            TopP = job.TopP is { } p ? (float)p : null,
            TopK = job.TopK,
            MaxFrames = job.MaxTokens,
            Seed = job.Seed,
            Cancel = job.Cancel,
        });
    }
}
