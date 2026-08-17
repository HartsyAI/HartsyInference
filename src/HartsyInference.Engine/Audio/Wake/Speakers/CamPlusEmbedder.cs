using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Features;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Audio.Wake.Speakers;

/// <summary>Turns a mono 16 kHz buffer into a 192-d CAM++ speaker embedding: whole-buffer Kaldi fbank (80 bins) →
/// per-bin cepstral mean normalization → <see cref="CamPlusSpeakerEncoder"/> → L2 normalization.
///
/// <para>The whole-buffer extractor is the right one here: speaker identification runs on a captured utterance after
/// a detection has already fired, never on the live 80 ms stream, so there is nothing incremental to preserve.</para>
///
/// <para>The weight lookup and the fbank/CMN recipe mirror <c>Engine/Audio/Stt/SpeakerDiarizer</c>, which had them
/// first but keeps them private behind a diarization-shaped API. When the two can be edited together, fold the
/// diarizer's <c>Load</c>/<c>Embed</c> onto this type rather than growing a third copy.</para>
///
/// <para>Not thread-safe internally by the encoder's design, so every call is serialized on a private lock; embedding
/// one utterance is burst work at human cadence, so the contention does not matter.</para></summary>
public sealed class CamPlusEmbedder : IDisposable
{
    /// <summary>CAM++ is a 16 kHz model (Kaldi fbank, 25 ms window / 10 ms shift).</summary>
    public const int SampleRate = 16_000;

    /// <summary>Width of the embedding produced by <c>campplus_cn_common</c>.</summary>
    public const int EmbeddingDimension = 192;

    /// <summary>Fewest fbank frames the encoder's strided front end can consume.</summary>
    public const int MinimumFrames = 16;

    /// <summary>Shortest clip that produces an embedding at all — not the shortest that produces a *reliable* one.
    /// Text-independent verification degrades sharply below a couple of seconds; see <see cref="SpeakerVerifier"/>.</summary>
    public const double MinimumSeconds = 0.4;

    /// <summary>Checkpoint file names tried, in order, under the audio models root.</summary>
    private static readonly string[] _weightFileNames =
        ["campplus_cn_common.bin", "campplus_cn_common.safetensors", "campplus.safetensors", "s3gen.safetensors"];

    private readonly object _gate = new object();
    private readonly CamPlusSpeakerEncoder _encoder;
    private readonly IDisposable[] _loaders;
    private readonly KaldiFbankExtractor _fbank = new KaldiFbankExtractor(SampleRate, 80);
    private int _disposed;

    private CamPlusEmbedder(CamPlusSpeakerEncoder encoder, IDisposable[] loaders)
    {
        _encoder = encoder;
        _loaders = loaders;
    }

    /// <summary>Shortest buffer that can be embedded, in samples.</summary>
    public static int MinimumSamples => (int)(MinimumSeconds * SampleRate);

    /// <summary>The CAM++ checkpoint on disk, or null when none is present. Weights are never downloaded: a silent
    /// multi-GB fetch inside a wake-word detection is not acceptable.</summary>
    public static string? LocateWeights()
    {
        foreach (string name in _weightFileNames)
        {
            string? path = ModelFileLocator.Find(name, Path.Combine("audio", "speaker"), "audio");
            if (path is not null)
            {
                return path;
            }
        }
        return null;
    }

    /// <summary>Loads CAM++ from a user-placed checkpoint, throwing an actionable
    /// <see cref="InvalidOperationException"/> naming the expected file and directory when none is present.</summary>
    public static CamPlusEmbedder Load()
    {
        string path = LocateWeights() ?? throw new InvalidOperationException(
            "Speaker identification needs the CAM++ speaker encoder, which is not present and is never auto-downloaded. "
            + $"Place '{_weightFileNames[0]}' (3D-Speaker / funasr campplus_cn_common) in "
            + $"'{AudioModelRoot.WeightsDirectory("speaker", "campplus")}' — a CosyVoice/Chatterbox 's3gen.safetensors' "
            + "under the audio models root works too.");
        return LoadFrom(path);
    }

    /// <summary>Loads CAM++ from an explicit checkpoint path.</summary>
    public static CamPlusEmbedder LoadFrom(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
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
            Logs.Error($"[Wake][Speaker] Failed to read the CAM++ checkpoint '{path}'", ex);
            throw;
        }

        // Standalone campplus checkpoints are unprefixed; the CosyVoice/Chatterbox bundle nests it under speaker_encoder.
        string prefix = weights.ContainsKey("xvector.dense.linear.weight") ? string.Empty : "speaker_encoder";
        CamPlusSpeakerEncoder encoder = new CamPlusSpeakerEncoder(EmbeddingDimension);
        try
        {
            encoder.LoadWeights(weights, prefix);
        }
        catch (Exception ex)
        {
            Logs.Error($"[Wake][Speaker] '{path}' does not carry CAM++ weights under prefix '{prefix}'", ex);
            encoder.Dispose();
            foreach (IDisposable loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
        Logs.Info($"[Wake][Speaker] Loaded the CAM++ speaker encoder from '{path}'.");
        return new CamPlusEmbedder(encoder, loaders);
    }

    /// <summary>L2-normalized 192-d embedding of <paramref name="mono16k"/>, which must be mono 16 kHz. Amplitude
    /// scale does not matter — cepstral mean normalization removes constant gain — so the wake path's int16-scaled
    /// audio and a decoder's ±1 audio both work, as long as one clip is not a mix of the two.</summary>
    public unsafe float[] Embed(IBackend backend, ReadOnlySpan<float> mono16k)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (mono16k.Length < MinimumSamples)
        {
            throw new ArgumentException(
                $"CAM++ needs at least {MinimumSeconds:0.0}s ({MinimumSamples} samples) at {SampleRate} Hz, got {mono16k.Length}.",
                nameof(mono16k));
        }
        lock (_gate)
        {
            float[,] fbank = _fbank.Compute(mono16k);
            int frames = fbank.GetLength(0);
            int bins = fbank.GetLength(1);
            if (frames < MinimumFrames)
            {
                throw new ArgumentException(
                    $"CAM++ needs at least {MinimumFrames} fbank frames, got {frames} from {mono16k.Length} samples.", nameof(mono16k));
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
                    int dimension = (int)embedding.Shape[embedding.Shape.Rank - 1];
                    float[] vector = new float[dimension];
                    new ReadOnlySpan<float>((float*)embedding.DataPointer, dimension).CopyTo(vector);
                    SpeakerEmbeddingMath.NormalizeInPlace(vector);
                    return vector;
                }
                finally
                {
                    embedding.Dispose();
                }
            }
            catch (Exception ex)
            {
                Logs.Error($"[Wake][Speaker] CAM++ failed on a {mono16k.Length / (double)SampleRate:0.00}s clip", ex);
                throw;
            }
            finally
            {
                features.Dispose();
            }
        }
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
