using System.Globalization;
using System.Linq;
using HartsyInference.Cli.Infra;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.ModelHandler.TextEncoders;
using HartsyInference.Tokenizers;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Cli.Dispatch.Handlers;

/// <summary>Text-to-video via LTX-Video (validation-pending, CUDA-only). Needs the LTX checkpoint (--model-path), a
/// standalone T5-XXL (aux "text-encoder-path") and its SentencePiece model (aux "tokenizer-path"). Frames are written
/// as a numbered BMP sequence; the generation "prompt" is the video description.</summary>
public sealed class VideoHandler : IModalityHandler
{
    /// <inheritdoc/>
    public Modality Modality => Modality.Video;

    /// <inheritdoc/>
    public IModalityRunner Load(ModelSpec spec, IBackend backend, IProgressSink progress)
    {
        if (backend is not CudaBackend)
            throw new NotSupportedException("Video generation requires the CUDA backend (the LTX stack is bf16 GPU-resident).");
        if (spec.LocalPath is null)
            throw new FileNotFoundException("No LTX checkpoint found. Pass the LTX .safetensors via --model-path.");
        if (!spec.Aux.TryGetValue("text-encoder-path", out string? t5Path))
            throw new ArgumentException("Video needs a T5-XXL encoder via --text-encoder-path.");
        if (!spec.Aux.TryGetValue("tokenizer-path", out string? spiecePath))
            throw new ArgumentException("Video needs the T5 SentencePiece model via --tokenizer-path.");

        progress.Stage("Loading LTX-Video checkpoint …");
        (LtxVideoCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader ckptLoader) = LtxVideoCheckpointConverter.LoadAndConvert(spec.LocalPath);
        LtxVideoConfig config = LtxVideoConfig.V09;
        LtxVideoTransformer transformer = new LtxVideoTransformer(config);
        transformer.LoadWeights(converted.Transformer);
        LtxVideoVaeDecoder vae = new LtxVideoVaeDecoder();
        vae.LoadWeights(converted.Vae);

        progress.Stage("Loading T5-XXL text encoder …");
        SafeTensorsLoader t5Loader = new SafeTensorsLoader();
        t5Loader.Load(t5Path);
        Dictionary<string, Tensor> t5Weights = TextEncoderQuantNormalizer.Normalize(t5Loader.GetAllTensors());
        T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
        t5.LoadWeights(t5Weights);

        T5Tokenizer tokenizer = new T5Tokenizer(spiecePath, maxLength: 128);
        LtxVideoPipeline pipeline = new LtxVideoPipeline(backend, transformer, vae, config);

        string id = spec.Catalog?.Id ?? Path.GetFileNameWithoutExtension(spec.LocalPath);
        return new VideoRunner(id, pipeline, t5, tokenizer, config, backend, new IDisposable[] { ckptLoader, t5Loader });
    }

    /// <inheritdoc/>
    public GeneratedArtifact Run(IModalityRunner runner, string prompt, ParamState parameters, IProgressSink progress, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        VideoRunner video = (VideoRunner)runner;

        string negative = parameters.Get("negative") is { Length: > 0 } n ? n : "blurry, low quality, distorted, watermark";
        int[] promptTokens = video.Tokenizer.Encode(prompt);
        int[] negativeTokens = video.Tokenizer.Encode(negative);
        int[] promptMask = T5Tokenizer.CreateAttentionMask(promptTokens);
        int[] negativeMask = T5Tokenizer.CreateAttentionMask(negativeTokens);

        progress.Stage("Encoding prompt (T5-XXL) …");
        using Tensor batch = video.TextEncoder.Encode(video.Backend, new[] { promptTokens, negativeTokens }, new[] { promptMask, negativeMask });
        using Tensor promptEmbeds = CfgHelper.SliceBatchElementPrefix(batch, 0, promptTokens.Length, promptMask.Sum(), 4096);
        using Tensor negativeEmbeds = CfgHelper.SliceBatchElementPrefix(batch, 1, negativeTokens.Length, negativeMask.Sum(), 4096);

        int steps = parameters.GetInt("steps", 30);
        int numFrames = parameters.GetInt("frames", 25);
        int fps = parameters.GetInt("fps", 25);
        int seedParam = parameters.GetInt("seed", -1);
        TextToImageRequest request = new TextToImageRequest
        {
            Prompt = prompt,
            Width = parameters.GetInt("width", 704),
            Height = parameters.GetInt("height", 480),
            Steps = steps,
            Seed = seedParam < 0 ? null : seedParam,
        };

        progress.BeginSteps("denoise", steps);
        (byte[][] frames, int width, int height, int usedSeed) = video.Pipeline.GenerateFromEmbeddings(
            promptEmbeds, negativeEmbeds, request, numFrames, fps, p => progress.Step(p.Step, $"{p.ElapsedMs:F0}ms"));
        progress.EndSteps();

        string baseDir = parameters.OutputDir ?? RepoPaths.OutputRoot();
        string dir = FrameWriter.WriteFrames(frames, width, height, baseDir, prompt);

        GeneratedArtifact artifact = new GeneratedArtifact
        {
            Kind = ArtifactKind.Video,
            Extension = "png",
            Text = $"{frames.Length} frames ({width}x{height}) → {dir}",
            PreviewRgb = frames.Length > 0 ? frames[0] : null,
            PreviewWidth = width,
            PreviewHeight = height,
        };
        artifact.Meta["model"] = video.ModelId;
        artifact.Meta["frames"] = frames.Length.ToString(CultureInfo.InvariantCulture);
        artifact.Meta["size"] = $"{width}x{height}";
        artifact.Meta["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture);
        return artifact;
    }
}
