using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Wake;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Onnx;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Audio.Wake;

/// <summary>Options for training a new wake word.</summary>
public sealed record WakeTrainingOptions
{
    /// <summary>The phrase to detect, e.g. "hey hartsy".</summary>
    public required string Phrase { get; init; }

    /// <summary>File-safe name for the head; defaults to the phrase slugified.</summary>
    public string? Name { get; init; }

    /// <summary>TTS model used to synthesize positives.</summary>
    public string SpeechModel { get; init; } = "kokoro";

    /// <summary>Voices to synthesize with. More voices is the single biggest quality lever, because the head
    /// has to generalize past whoever it heard during training.</summary>
    public IReadOnlyList<string> Voices { get; init; } =
        ["af_heart", "af_bella", "af_nicole", "am_michael", "am_adam", "bf_emma", "bm_george", "af_sarah"];

    /// <summary>Speaking rates applied to every voice.</summary>
    public IReadOnlyList<double> Speeds { get; init; } = [0.85, 1.0, 1.15];

    /// <summary>Extra phrases synthesized as negatives. Phonetically close ones matter most — the failure that
    /// annoys people is a wake word that fires on a similar-sounding phrase, not one that misses a stranger.</summary>
    public IReadOnlyList<string> NegativePhrases { get; init; } = [];

    /// <summary>Directory of WAV files used as additional negatives — podcasts, TV, room ambience. This is the
    /// most valuable input a user can supply: synthetic negatives cannot represent a real room.</summary>
    public string? NegativeAudioDirectory { get; init; }

    /// <summary>Training epochs over the assembled dataset.</summary>
    public int Epochs { get; init; } = 120;

    /// <summary>Adam learning rate.</summary>
    public float LearningRate { get; init; } = 1e-3f;

    /// <summary>Hidden width; 128 matches the shipped heads.</summary>
    public int Hidden { get; init; } = 128;

    /// <summary>Where to write the head. Defaults to the wake model root's <c>heads/</c>.</summary>
    public string? OutputDirectory { get; init; }
}

/// <summary>How the trained head performed on held-out data, and the threshold chosen from it.</summary>
public sealed record WakeTrainingResult
{
    public required string Name { get; init; }
    public required string HeadPath { get; init; }
    public required int PositiveWindows { get; init; }
    public required int NegativeWindows { get; init; }
    public required float Recall { get; init; }
    public required float FalseAcceptRate { get; init; }

    /// <summary>False accepts per hour of continuous audio, derived from the per-window rate at 12.5 scored
    /// windows per second. This is the figure the field quotes (openWakeWord targets under 0.5/hour), and it
    /// makes an inadequate negative set obvious in a way a small-looking percentage does not.</summary>
    public required float FalseAcceptsPerHour { get; init; }
    public required float SuggestedThreshold { get; init; }
    public required float FinalLoss { get; init; }
}

/// <summary>Trains a wake word from nothing but its text, using the engine's own TTS for positives.
///
/// <para>This is why the openWakeWord architecture was chosen over an open-vocabulary spotter: the front-end
/// and backbone are frozen, so a new phrase is a ~213k-parameter head that trains on CPU in seconds, and the
/// two expensive ingredients — many synthetic voices, and the backbone that turns them into features — are
/// already in this engine.</para>
///
/// <para>Training features are produced by running <see cref="WakeDetectionPipeline"/> itself, so they cannot
/// drift from what inference sees. Honest expectation: upstream trained its heads against roughly 30,000 hours
/// of negative audio, and a handful of synthetic negatives is not that. Point
/// <see cref="WakeTrainingOptions.NegativeAudioDirectory"/> at real recordings and the false-accept rate
/// improves accordingly — the evaluation below reports it rather than assuming it.</para></summary>
public sealed class WakeTrainingJob
{
    private const int SampleRate = 16_000;

    private readonly IInferenceEngine _engine;
    private readonly string _modelRoot;

    public WakeTrainingJob(IInferenceEngine engine, string modelRoot)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _modelRoot = modelRoot;
    }

    public async Task<WakeTrainingResult> RunAsync(WakeTrainingOptions options, IProgress<string>? progress = null, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (string.IsNullOrWhiteSpace(options.Phrase)) throw new ArgumentException("A wake phrase is required.", nameof(options));

        string name = options.Name ?? Slugify(options.Phrase);
        progress?.Report($"Synthesizing '{options.Phrase}' across {options.Voices.Count} voices x {options.Speeds.Count} speeds");

        List<float[]> positiveClips = await SynthesizeAsync(options.Phrase, options, cancel).ConfigureAwait(false);
        if (positiveClips.Count == 0) throw new HartsyInferenceException($"No positive audio was synthesized for '{options.Phrase}'.");

        List<float[]> negativeClips = [];
        foreach (string phrase in options.NegativePhrases)
            negativeClips.AddRange(await SynthesizeAsync(phrase, options, cancel).ConfigureAwait(false));
        negativeClips.AddRange(LoadNegativeAudio(options.NegativeAudioDirectory, progress));
        if (negativeClips.Count == 0)
            progress?.Report("WARNING: no negative audio — the head will fire readily. Supply --negative-audio for a usable false-accept rate.");

        progress?.Report($"Embedding {positiveClips.Count} positive and {negativeClips.Count} negative clips");
        using CpuBackend backend = new();
        using WakeMelFrontend mel = LoadMel();
        using SpeechEmbeddingModel embedding = LoadEmbedding();

        // Hold out whole CLIPS, not windows. Consecutive windows from one clip overlap by 15 of their 16
        // frames, so a per-window split puts near-duplicates on both sides and reports a recall and
        // false-accept rate that mean nothing.
        (List<float[]> posTrainClips, List<float[]> posTestClips) = Split(positiveClips);
        (List<float[]> negTrainClips, List<float[]> negTestClips) = Split(negativeClips);

        List<float[]> trainPos = ExtractFeatures(backend, mel, embedding, posTrainClips, augment: true);
        List<float[]> testPos = ExtractFeatures(backend, mel, embedding, posTestClips, augment: false);
        List<float[]> trainNeg = ExtractFeatures(backend, mel, embedding, negTrainClips, augment: false);
        List<float[]> testNeg = ExtractFeatures(backend, mel, embedding, negTestClips, augment: false);
        int positives = trainPos.Count + testPos.Count, negatives = trainNeg.Count + testNeg.Count;
        if (trainPos.Count == 0) throw new HartsyInferenceException("Positive clips produced no feature windows; they may be shorter than the 1.3 s warm-up.");

        progress?.Report($"Training on {trainPos.Count} positive / {trainNeg.Count} negative windows for {options.Epochs} epochs");
        WakeHeadTrainer trainer = new(options.Hidden);
        float positiveWeight = trainNeg.Count == 0 ? 1f : Math.Clamp((float)trainNeg.Count / Math.Max(1, trainPos.Count), 1f, 20f);

        List<float[]> batch = [.. trainPos, .. trainNeg];
        List<float> labels = [.. trainPos.Select(static _ => 1f), .. trainNeg.Select(static _ => 0f)];
        float loss = 0f;
        for (int epoch = 0; epoch < options.Epochs; epoch++)
        {
            cancel.ThrowIfCancellationRequested();
            loss = trainer.TrainBatch(batch, labels, options.LearningRate, positiveWeight);
            if (epoch % 20 == 0) progress?.Report($"  epoch {epoch}: loss {loss:F4}");
        }

        (float threshold, float recall, float falseAccepts) = ChooseThreshold(trainer, testPos, testNeg);
        progress?.Report($"Held-out recall {recall:P1}, false accepts {falseAccepts:P2} " +
            $"({falseAccepts * 3600f * SampleRate / WakeDetectionPipeline.ChunkSamples:F0}/hour) at threshold {threshold:F2}");

        string outputDir = options.OutputDirectory ?? Path.Combine(_modelRoot, "heads");
        Directory.CreateDirectory(outputDir);
        string headPath = Path.Combine(outputDir, name + ".safetensors");
        Save(trainer, headPath);
        Logs.Info($"[Audio][Wake] Trained wake word '{name}' → {headPath} (recall {recall:P1}, FA {falseAccepts:P2}).");

        return new WakeTrainingResult
        {
            Name = name,
            HeadPath = headPath,
            PositiveWindows = positives,
            NegativeWindows = negatives,
            Recall = recall,
            FalseAcceptRate = falseAccepts,
            FalseAcceptsPerHour = falseAccepts * 3600f * SampleRate / WakeDetectionPipeline.ChunkSamples,
            SuggestedThreshold = threshold,
            FinalLoss = loss,
        };
    }

    private async Task<List<float[]>> SynthesizeAsync(string text, WakeTrainingOptions options, CancellationToken cancel)
    {
        List<float[]> clips = [];
        ModelSpec spec = Registry.ModelResolver.Resolve(options.SpeechModel, null, Modality.Speech);
        foreach (string voice in options.Voices)
        {
            foreach (double speed in options.Speeds)
            {
                cancel.ThrowIfCancellationRequested();
                try
                {
                    AudioResult result = await _engine.Speech.SynthesizeAsync(spec, new SpeechRequest
                    {
                        Text = text,
                        Voice = voice,
                        Speed = speed,
                    }, cancel).ConfigureAwait(false);
                    clips.Add(DecodeToInt16Scaled(result.Data));
                }
                catch (Exception ex)
                {
                    // One unavailable voice should not abort a training run that has seven others.
                    Logs.Warning($"[Audio][Wake] Voice '{voice}' at speed {speed} failed: {ex.Message}");
                }
            }
        }
        return clips;
    }

    private static IEnumerable<float[]> LoadNegativeAudio(string? directory, IProgress<string>? progress)
    {
        if (string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory)) yield break;
        int count = 0;
        foreach (string file in Directory.EnumerateFiles(directory, "*.wav", SearchOption.AllDirectories))
        {
            float[]? clip = null;
            try
            {
                WavFile.DecodedAudio decoded = WavFile.Read(file);
                float[] mono = decoded.ToMono();
                if (decoded.SampleRate != SampleRate)
                    mono = Resampler.Create(decoded.SampleRate, SampleRate).Resample(mono);
                clip = new float[mono.Length];
                for (int i = 0; i < mono.Length; i++) clip[i] = mono[i] * 32768f;
            }
            catch (Exception ex)
            {
                Logs.Warning($"[Audio][Wake] Skipping negative '{file}': {ex.Message}");
            }
            if (clip is not null) { count++; yield return clip; }
        }
        progress?.Report($"Loaded {count} negative recordings from {directory}");
    }

    /// <summary>Runs clips through the real detection pipeline and collects the feature windows it scores.
    /// <paramref name="augment"/> adds gain and noise variants, which is what stops the head from keying on the
    /// TTS engine's recording conditions instead of the phrase.</summary>
    private static List<float[]> ExtractFeatures(IBackend backend, WakeMelFrontend mel, SpeechEmbeddingModel embedding,
        IReadOnlyList<float[]> clips, bool augment)
    {
        List<float[]> features = [];
        List<WakeDetection> ignored = [];
        Random random = new(4242);

        foreach (float[] clip in clips)
        {
            List<float[]> variants = [clip];
            if (augment)
            {
                variants.Add(Scale(clip, 0.45f));
                variants.Add(Scale(clip, 1.6f));
                variants.Add(AddNoise(clip, 0.02f, random));
                variants.Add(AddNoise(clip, 0.08f, random));
            }
            foreach (float[] variant in variants)
            {
                // Lead-in silence covers the pipeline's warm-up so the phrase itself is always scored.
                float[] padded = new float[SampleRate + variant.Length + SampleRate / 2];
                variant.CopyTo(padded, SampleRate);

                using WakeDetectionPipeline pipeline = new(mel, embedding);
                pipeline.Push(backend, padded, ignored, features);
            }
        }
        return features;
    }

    private static float[] Scale(float[] clip, float gain)
    {
        float[] result = new float[clip.Length];
        for (int i = 0; i < clip.Length; i++) result[i] = Math.Clamp(clip[i] * gain, -32768f, 32767f);
        return result;
    }

    private static float[] AddNoise(float[] clip, float relativeLevel, Random random)
    {
        float peak = 1f;
        foreach (float sample in clip) peak = MathF.Max(peak, MathF.Abs(sample));
        float amplitude = peak * relativeLevel;
        float[] result = new float[clip.Length];
        for (int i = 0; i < clip.Length; i++)
            result[i] = Math.Clamp(clip[i] + (float)(random.NextDouble() * 2 - 1) * amplitude, -32768f, 32767f);
        return result;
    }

    private static (List<float[]> Train, List<float[]> Test) Split(List<float[]> items)
    {
        List<float[]> train = [], test = [];
        for (int i = 0; i < items.Count; i++) (i % 5 == 0 ? test : train).Add(items[i]);
        return (train, test);
    }

    /// <summary>Picks the threshold with the best recall among those keeping held-out false accepts under 1%,
    /// falling back to the fewest false accepts when nothing clears that bar.</summary>
    private static (float Threshold, float Recall, float FalseAccepts) ChooseThreshold(
        WakeHeadTrainer trainer, IReadOnlyList<float[]> positives, IReadOnlyList<float[]> negatives)
    {
        float bestThreshold = 0.5f, bestRecall = 0f, bestFalse = 1f;
        for (float threshold = 0.3f; threshold <= 0.96f; threshold += 0.05f)
        {
            float recall = positives.Count == 0 ? 0f : positives.Count(f => trainer.Predict(f) >= threshold) / (float)positives.Count;
            float falseAccepts = negatives.Count == 0 ? 0f : negatives.Count(f => trainer.Predict(f) >= threshold) / (float)negatives.Count;
            bool better = falseAccepts <= 0.01f
                ? bestFalse > 0.01f || recall > bestRecall
                : bestFalse > 0.01f && falseAccepts < bestFalse;
            if (better) (bestThreshold, bestRecall, bestFalse) = (threshold, recall, falseAccepts);
        }
        return (bestThreshold, bestRecall, bestFalse);
    }

    private static void Save(WakeHeadTrainer trainer, string path)
    {
        Dictionary<string, Tensor> tensors = [];
        try
        {
            foreach ((string tensorName, (int[] shape, float[] data)) in trainer.ExportWeights())
            {
                Tensor t = new(new TensorShape([.. shape.Select(static d => (long)d)]), DType.F32);
                data.CopyTo(t.AsSpan<float>());
                tensors[tensorName] = t;
            }
            SafeTensorsWriter.Save(path, tensors);
        }
        finally
        {
            foreach (Tensor t in tensors.Values) t.Dispose();
        }
    }

    private WakeMelFrontend LoadMel()
    {
        using OnnxWeightLoader loader = new();
        loader.Load(Path.Combine(_modelRoot, "backbone", "melspectrogram.onnx"));
        WakeMelFrontend mel = new();
        mel.LoadWeights(loader.GetAllTensors());
        return mel;
    }

    private SpeechEmbeddingModel LoadEmbedding()
    {
        using OnnxWeightLoader loader = new();
        loader.Load(Path.Combine(_modelRoot, "backbone", "embedding_model.onnx"));
        SpeechEmbeddingModel model = new();
        model.LoadWeights(loader.GetAllTensors());
        return model;
    }

    /// <summary>Decodes synthesized WAV bytes to 16 kHz mono at int16 scale, which is what the wake models
    /// consume — feeding them normalized audio silently shifts every log-mel value.</summary>
    private static float[] DecodeToInt16Scaled(byte[] wav)
    {
        using MemoryStream ms = new(wav);
        WavFile.DecodedAudio decoded = WavFile.Read(ms);
        float[] mono = decoded.ToMono();
        if (decoded.SampleRate != SampleRate)
            mono = Resampler.Create(decoded.SampleRate, SampleRate).Resample(mono);
        float[] result = new float[mono.Length];
        for (int i = 0; i < mono.Length; i++) result[i] = mono[i] * 32768f;
        return result;
    }

    private static string Slugify(string phrase)
    {
        char[] chars = phrase.Trim().ToLowerInvariant()
            .Select(static c => char.IsLetterOrDigit(c) ? c : '_').ToArray();
        return new string(chars).Trim('_');
    }
}
