using System.Globalization;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Codecs.EnCodec;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Engine;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.Tokenizers;

namespace HartsyInference.Engine.Dispatch.Handlers;

/// <summary>Text-to-music via MusicGen. Assembles the T5 prompt encoder, the decoder, and the 32 kHz EnCodec from one
/// <c>facebook/musicgen-small</c> checkpoint (validation-pending; heavy — prefer CUDA).</summary>
public sealed class MusicHandler : IModalityHandler
{
    /// <inheritdoc/>
    public Modality Modality => Modality.Music;

    /// <inheritdoc/>
    public IModalityRunner Load(ModelSpec spec, IBackend backend, IProgressSink progress)
    {
        if (spec.LocalPath is null)
        {
            throw new FileNotFoundException(
                "No MusicGen checkpoint found. Pass a musicgen-small model.safetensors via --model-path.");
        }

        bool cuda = backend is CudaBackend;
        string path = spec.LocalPath;

        progress.Stage("Loading MusicGen T5 text encoder …");
        (Dictionary<string, Tensor> teWeights, IDisposable teLoader) = MusicGenCheckpointConverter.LoadTextEncoderAny(path, castToF32: !cuda);
        T5TextEncoder textEncoder = new T5TextEncoder(T5TextEncoderConfig.T5Base);
        textEncoder.LoadWeights(teWeights);

        progress.Stage("Loading MusicGen decoder …");
        (Dictionary<string, Tensor> decWeights, IDisposable decLoader) = MusicGenCheckpointConverter.LoadDecoderAny(path, castToF32: !cuda);
        MusicGenDecoder decoder = new MusicGenDecoder(MusicGenConfig.Small);
        decoder.LoadWeights(decWeights);

        progress.Stage("Loading EnCodec (32 kHz) …");
        (Dictionary<string, Tensor> ecWeights, IDisposable ecLoader) = MusicGenCheckpointConverter.LoadEnCodecAny(path, castToF32: !cuda);
        EnCodec codec = new EnCodec(EnCodecConfig.EnCodec32kHz);
        codec.LoadWeights(ecWeights);

        MusicGenPipeline pipeline = new MusicGenPipeline(MusicGenConfig.Small, decoder, codec);
        T5Tokenizer tokenizer = new T5Tokenizer();
        List<IDisposable> owned = new List<IDisposable> { decoder, teLoader, decLoader, ecLoader };

        string id = spec.Catalog?.Id ?? Path.GetFileNameWithoutExtension(path);
        return new MusicRunner(id, textEncoder, tokenizer, pipeline, MusicGenConfig.Small.CodecSampleRate, backend, owned);
    }

    /// <inheritdoc/>
    public GeneratedArtifact Run(IModalityRunner runner, string prompt, ParamState parameters, IProgressSink progress, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        MusicRunner music = (MusicRunner)runner;

        int[] ids = music.Tokenizer.Encode(prompt);
        using Tensor t5States = music.TextEncoder.Encode(music.Backend, new[] { ids });

        int duration = parameters.GetInt("duration", 10);
        int seed = parameters.GetInt("seed", -1);

        progress.Stage($"Generating {duration}s of music …");
        float[] audio = music.Pipeline.Synthesize(music.Backend, t5States, seconds: duration, seed: seed < 0 ? 0 : seed);

        byte[] wav = ToWav(audio, music.SampleRate);
        double seconds = audio.Length / (double)music.SampleRate;
        GeneratedArtifact artifact = new GeneratedArtifact
        {
            Kind = ArtifactKind.Audio,
            FileBytes = wav,
            Extension = "wav",
            Text = $"{seconds:F1}s of music ({music.SampleRate} Hz)",
        };
        artifact.Meta["model"] = music.ModelId;
        artifact.Meta["seconds"] = seconds.ToString("F1", CultureInfo.InvariantCulture);
        artifact.Meta["sample_rate"] = music.SampleRate.ToString(CultureInfo.InvariantCulture);
        return artifact;
    }

    private static byte[] ToWav(float[] samples, int sampleRate)
    {
        using MemoryStream ms = new MemoryStream();
        WavFile.WriteMono16(ms, samples, sampleRate);
        return ms.ToArray();
    }
}
