using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Audio.Wake.Speakers;

/// <summary>Turns a mono 16 kHz buffer into a 192-d CAM++ speaker embedding: whole-buffer Kaldi fbank (80 bins) →
/// per-bin cepstral mean normalization → <see cref="CamPlusSpeakerEncoder"/> → L2 normalization.
///
/// <para>The whole-buffer extractor is the right one here: speaker identification runs on a captured utterance after
/// a detection has already fired, never on the live 80 ms stream, so there is nothing incremental to preserve.</para>
///
/// <para>The weight lookup and the fbank/CMN recipe live in <see cref="CamPlusLoader"/>, shared with
/// <c>Engine/Audio/Stt/SpeakerDiarizer</c>; this type contributes the whole-buffer framing and the clip guards.</para>
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

    /// <summary>Shortest clip that produces an embedding at all — not the shortest that produces a *reliable* one. Text-independent verification degrades sharply below a couple of seconds; see <see cref="SpeakerVerifier"/>.</summary>
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

    /// <summary>The CAM++ checkpoint on disk, or null when none is present. Weights are never downloaded: a silent multi-GB fetch inside a wake-word detection is not acceptable.</summary>
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

    /// <summary>Loads CAM++ from a user-placed checkpoint, throwing an actionable <see cref="InvalidOperationException"/> naming the expected file and directory when none is present.</summary>
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
        (CamPlusSpeakerEncoder encoder, IDisposable[] loaders) = CamPlusLoader.Load(path, "[Wake][Speaker]", EmbeddingDimension);
        return new CamPlusEmbedder(encoder, loaders);
    }

    /// <summary>L2-normalized 192-d embedding of <paramref name="mono16k"/>, which must be mono 16 kHz. Amplitude scale does not matter — cepstral mean normalization removes constant gain — so the wake path's int16-scaled audio and a decoder's ±1 audio both work, as long as one clip is not a mix of the two.</summary>
    public float[] Embed(IBackend backend, ReadOnlySpan<float> mono16k)
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
            if (frames < MinimumFrames)
            {
                throw new ArgumentException(
                    $"CAM++ needs at least {MinimumFrames} fbank frames, got {frames} from {mono16k.Length} samples.", nameof(mono16k));
            }

            try
            {
                return CamPlusLoader.Embed(backend, _encoder, fbank);
            }
            catch (Exception ex)
            {
                Logs.Error($"[Wake][Speaker] CAM++ failed on a {mono16k.Length / (double)SampleRate:0.00}s clip", ex);
                throw;
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
