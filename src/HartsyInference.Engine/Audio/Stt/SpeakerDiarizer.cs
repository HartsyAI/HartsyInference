using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Audio;

/// <summary>Speaker diarization by embedding-and-clustering: cut the clip into spans, embed each with the CAM++
/// speaker encoder (3D-Speaker <c>campplus_cn_common</c>, the same weights CosyVoice2 ships and this repo has
/// validated against ONNX Runtime), then merge spans with agglomerative average-linkage clustering over cosine
/// distance.
///
/// <para><b>Documented limitations — read before trusting the output.</b> (1) There is no VAD in this repo, so
/// silence and non-speech are attributed to whichever speaker they cluster nearest; span boundaries come from
/// Whisper's timestamp tokens when available and from a fixed sliding window otherwise, never from speech activity.
/// (2) <see cref="MergeDistance"/> is an <b>uncalibrated</b> constant — the encoder itself is real-weight verified,
/// but no speaker-diarization error rate has been measured for this threshold on this codebase, so the speaker
/// <i>count</i> is a guess whenever the caller cannot supply one. (3) Weights are never downloaded here; the clip is
/// re-decoded to 16 kHz mono because CAM++ is a 16 kHz Kaldi-fbank model.</para></summary>
internal sealed class SpeakerDiarizer : IDisposable
{
    /// <summary>CAM++ is a 16 kHz model (Kaldi fbank, 25 ms / 10 ms).</summary>
    internal const int SampleRate = 16_000;

    /// <summary>Cosine-distance ceiling below which two clusters merge. Uncalibrated — see the type remarks.</summary>
    private const float MergeDistance = 0.40f;

    /// <summary>Sliding-window length used when no timestamp segments are available.</summary>
    private const double WindowSeconds = 1.5;

    /// <summary>Sliding-window hop used when no timestamp segments are available.</summary>
    private const double HopSeconds = 0.75;

    /// <summary>Spans shorter than this carry too few fbank frames for a stable embedding and are dropped.</summary>
    private const double MinSpanSeconds = 0.4;

    /// <summary>Upper bound on distinct speakers; clustering keeps merging past the threshold to reach it.</summary>
    private const int MaxSpeakers = 8;

    /// <summary>Hard cap on embedded spans so a long clip cannot turn the O(n³) clustering into a hang.</summary>
    private const int MaxSpans = 512;

    private static readonly object _gate = new object();
    private static SpeakerDiarizer? _shared;

    private readonly CamPlusSpeakerEncoder _encoder;
    private readonly IDisposable[] _loaders;
    private readonly KaldiFbankExtractor _fbank = new KaldiFbankExtractor(SampleRate, 80);
    private int _disposed;

    private SpeakerDiarizer(CamPlusSpeakerEncoder encoder, IDisposable[] loaders)
    {
        _encoder = encoder;
        _loaders = loaders;
    }

    /// <summary>The process-wide instance, loaded from disk on first use. Callers run under the audio generation lock, which is what makes the shared (non-thread-safe) encoder safe to reuse.</summary>
    internal static SpeakerDiarizer Shared()
    {
        lock (_gate)
        {
            return _shared ??= Load();
        }
    }

    /// <summary>Attributes speakers to <paramref name="mono16k"/>, cutting on <paramref name="segments"/> when the transcriber produced timestamps and on a fixed sliding window otherwise.</summary>
    internal IReadOnlyList<DiarizedSegment> Diarize(IBackend backend, float[] mono16k, IReadOnlyList<SttSegment>? segments,
        CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(mono16k);
        double totalSeconds = AudioClipCodec.Seconds(mono16k.Length, SampleRate);
        List<SttSegment> spans = BuildSpans(segments, totalSeconds);
        if (spans.Count == 0)
        {
            Logs.Warning($"[Audio][Diarize] {totalSeconds:0.0}s of audio yielded no span of at least {MinSpanSeconds:0.0}s — no speakers emitted.");
            return [];
        }

        List<float[]> embeddings = new List<float[]>(spans.Count);
        List<SttSegment> kept = new List<SttSegment>(spans.Count);
        foreach (SttSegment span in spans)
        {
            cancel.ThrowIfCancellationRequested();
            float[]? embedding = Embed(backend, mono16k, span);
            if (embedding is not null)
            {
                embeddings.Add(embedding);
                kept.Add(span);
            }
        }
        if (embeddings.Count == 0)
        {
            Logs.Warning("[Audio][Diarize] Every span failed to embed — no speakers emitted.");
            return [];
        }

        int[] labels = Cluster(embeddings, cancel);
        Logs.Verbose($"[Audio][Diarize] {kept.Count} spans → {CountDistinct(labels)} speaker(s) "
            + $"(CAM++ 192-d, cosine merge < {MergeDistance:0.00}).");
        return Merge(kept, labels);
    }

    /// <summary>Loads CAM++ from a user-placed checkpoint. Never downloads: diarization is a side capability and a silent multi-GB fetch inside a transcription call is not acceptable.</summary>
    private static SpeakerDiarizer Load()
    {
        string[] names = ["campplus_cn_common.bin", "campplus_cn_common.safetensors", "campplus.safetensors", "s3gen.safetensors"];
        string? path = null;
        foreach (string name in names)
        {
            path = ModelFileLocator.Find(name, Path.Combine("audio", "speaker"), "audio");
            if (path is not null)
            {
                break;
            }
        }
        if (path is null)
        {
            throw new InvalidOperationException(
                "Speaker diarization needs the CAM++ speaker encoder, which is not present and is never auto-downloaded. "
                + $"Place '{names[0]}' (3D-Speaker / funasr campplus_cn_common) in "
                + $"'{AudioModelRoot.WeightsDirectory("speaker", "campplus")}' — a CosyVoice/Chatterbox "
                + "'s3gen.safetensors' under the audio models root works too.");
        }

        IDisposable[] loaders;
        IReadOnlyDictionary<string, Tensor> weights;
        try
        {
            if (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
            {
                SafeTensorsLoader safetensors = new SafeTensorsLoader();
                safetensors.Load(path);
                weights = safetensors.GetAllTensors();
                loaders = [safetensors];
            }
            else
            {
                PytorchPickleLoader pickle = new PytorchPickleLoader();
                pickle.Load(path);
                weights = pickle.GetAllTensors();
                loaders = [pickle];
            }
        }
        catch (Exception ex)
        {
            Logs.Error($"[Audio][Diarize] Failed to read the CAM++ checkpoint '{path}'", ex);
            throw;
        }

        // Standalone campplus checkpoints are unprefixed; the CosyVoice/Chatterbox bundle nests it under speaker_encoder.
        string prefix = weights.ContainsKey("xvector.dense.linear.weight") ? string.Empty : "speaker_encoder";
        CamPlusSpeakerEncoder encoder = new CamPlusSpeakerEncoder(192);
        try
        {
            encoder.LoadWeights(weights, prefix);
        }
        catch (Exception ex)
        {
            Logs.Error($"[Audio][Diarize] '{path}' does not carry CAM++ weights under prefix '{prefix}'", ex);
            encoder.Dispose();
            foreach (IDisposable loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
        Logs.Info($"[Audio][Diarize] Loaded the CAM++ speaker encoder from '{path}'.");
        return new SpeakerDiarizer(encoder, loaders);
    }

    /// <summary>Timestamp spans when the transcriber gave them, else a fixed overlapping window sweep.</summary>
    private static List<SttSegment> BuildSpans(IReadOnlyList<SttSegment>? segments, double totalSeconds)
    {
        List<SttSegment> spans = [];
        if (segments is { Count: > 0 })
        {
            foreach (SttSegment segment in segments)
            {
                if (segment.End - segment.Start >= MinSpanSeconds && segment.Start < totalSeconds)
                {
                    spans.Add(segment with { End = Math.Min(segment.End, totalSeconds) });
                }
            }
            if (spans.Count > 0)
            {
                return spans;
            }
        }
        for (double start = 0d; start + MinSpanSeconds <= totalSeconds && spans.Count < MaxSpans; start += HopSeconds)
        {
            spans.Add(new SttSegment { Text = string.Empty, Start = start, End = Math.Min(start + WindowSeconds, totalSeconds) });
        }
        return spans;
    }

    /// <summary>Embeds one span: Kaldi fbank → cepstral mean normalization → CAM++ → L2-normalized 192-d vector. Returns null when the span has too few frames for the encoder's strided front end.</summary>
    private unsafe float[]? Embed(IBackend backend, float[] mono16k, SttSegment span)
    {
        int from = Math.Clamp((int)(span.Start * SampleRate), 0, mono16k.Length);
        int to = Math.Clamp((int)(span.End * SampleRate), from, mono16k.Length);
        if (to - from < (int)(MinSpanSeconds * SampleRate))
        {
            return null;
        }
        float[,] fbank = _fbank.Compute(new ReadOnlySpan<float>(mono16k, from, to - from));
        int frames = fbank.GetLength(0);
        int bins = fbank.GetLength(1);
        if (frames < 16)
        {
            return null;
        }

        Tensor features = new Tensor(new TensorShape(1, frames, bins), DType.F32);
        try
        {
            float* destination = (float*)features.DataPointer;
            for (int bin = 0; bin < bins; bin++)
            {
                // Cepstral mean normalization, per bin over time — what CosyVoice feeds CAM++.
                double mean = 0d;
                for (int frame = 0; frame < frames; frame++)
                {
                    mean += fbank[frame, bin];
                }
                mean /= frames;
                for (int frame = 0; frame < frames; frame++)
                {
                    destination[(long)frame * bins + bin] = (float)(fbank[frame, bin] - mean);
                }
            }
            Tensor embedding = _encoder.Forward(backend, features);
            try
            {
                int dim = (int)embedding.Shape[embedding.Shape.Rank - 1];
                float[] vector = new float[dim];
                new ReadOnlySpan<float>((float*)embedding.DataPointer, dim).CopyTo(vector);
                Normalize(vector);
                return vector;
            }
            finally
            {
                embedding.Dispose();
            }
        }
        catch (Exception ex)
        {
            Logs.Error($"[Audio][Diarize] CAM++ failed on the span {span.Start:0.00}-{span.End:0.00}s", ex);
            throw;
        }
        finally
        {
            features.Dispose();
        }
    }

    /// <summary>Agglomerative average-linkage clustering over cosine distance: merge the closest pair while it is under <see cref="MergeDistance"/>, or while more than <see cref="MaxSpeakers"/> clusters remain.</summary>
    private static int[] Cluster(List<float[]> embeddings, CancellationToken cancel)
    {
        int count = embeddings.Count;
        List<List<int>> clusters = new List<List<int>>(count);
        for (int i = 0; i < count; i++)
        {
            clusters.Add([i]);
        }
        while (clusters.Count > 1)
        {
            cancel.ThrowIfCancellationRequested();
            float best = float.MaxValue;
            int bestA = -1;
            int bestB = -1;
            for (int a = 0; a < clusters.Count; a++)
            {
                for (int b = a + 1; b < clusters.Count; b++)
                {
                    float distance = AverageDistance(embeddings, clusters[a], clusters[b]);
                    if (distance < best)
                    {
                        best = distance;
                        bestA = a;
                        bestB = b;
                    }
                }
            }
            if (bestA < 0 || (best >= MergeDistance && clusters.Count <= MaxSpeakers))
            {
                break;
            }
            clusters[bestA].AddRange(clusters[bestB]);
            clusters.RemoveAt(bestB);
        }

        int[] labels = new int[count];
        // Number speakers by first appearance so span 0 is always speaker 0.
        clusters.Sort((left, right) => Min(left).CompareTo(Min(right)));
        for (int c = 0; c < clusters.Count; c++)
        {
            foreach (int index in clusters[c])
            {
                labels[index] = c;
            }
        }
        return labels;
    }

    /// <summary>Mean pairwise cosine distance between two clusters (average linkage).</summary>
    private static float AverageDistance(List<float[]> embeddings, List<int> left, List<int> right)
    {
        double total = 0d;
        foreach (int a in left)
        {
            foreach (int b in right)
            {
                float dot = 0f;
                float[] va = embeddings[a];
                float[] vb = embeddings[b];
                for (int i = 0; i < va.Length; i++)
                {
                    dot += va[i] * vb[i];
                }
                total += 1d - dot;
            }
        }
        return (float)(total / (left.Count * right.Count));
    }

    /// <summary>Collapses consecutive same-speaker spans into contiguous diarized segments.</summary>
    private static List<DiarizedSegment> Merge(List<SttSegment> spans, int[] labels)
    {
        List<DiarizedSegment> result = [];
        int start = 0;
        for (int i = 1; i <= spans.Count; i++)
        {
            if (i < spans.Count && labels[i] == labels[start])
            {
                continue;
            }
            string text = string.Join(" ", spans.GetRange(start, i - start).Select(s => s.Text).Where(t => t.Length > 0)).Trim();
            result.Add(new DiarizedSegment
            {
                Speaker = labels[start],
                Start = spans[start].Start,
                End = spans[i - 1].End,
                Text = text.Length == 0 ? null : text,
            });
            start = i;
        }
        return result;
    }

    private static void Normalize(float[] vector)
    {
        double sum = 0d;
        foreach (float value in vector)
        {
            sum += value * value;
        }
        float inverse = (float)(1d / Math.Sqrt(Math.Max(sum, 1e-12d)));
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] *= inverse;
        }
    }

    private static int Min(List<int> indices)
    {
        int min = int.MaxValue;
        foreach (int index in indices)
        {
            min = Math.Min(min, index);
        }
        return min;
    }

    private static int CountDistinct(int[] labels)
    {
        int max = -1;
        foreach (int label in labels)
        {
            max = Math.Max(max, label);
        }
        return max + 1;
    }

    /// <summary>Releases the encoder and its checkpoint loaders.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        _encoder.Dispose();
        foreach (IDisposable loader in _loaders)
        {
            loader.Dispose();
        }
    }
}
