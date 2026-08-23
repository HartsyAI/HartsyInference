using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Registry;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Constructs <see cref="DiffusionPipelineBase"/> instances from on-disk model artifacts. Resolves the five concerns the previous scaffolding flagged:
/// <list type="number">
///   <item><b>Model-type detection</b> via <see cref="ModelArchitectureDetector"/> (shared tensor-key signatures).</item>
///   <item><b>Component-file discovery</b> via <see cref="ModelLayoutResolver"/> (single-file / sharded / diffusers).</item>
///   <item><b>Tokenizer</b> — CLIP-family pipelines use the embedded OpenAI vocab/merges, so a caller can
///         tokenize with <c>new ClipTokenizer()</c> with no extra files; per-pipeline construction stays explicit.</item>
///   <item><b>Dtype</b> — weights are cast to F32 owned copies so the mmap loader can be released and the
///         result runs on every backend (CPU requires F32). A future quality profile can pick F16 for CUDA.</item>
///   <item><b>Caching</b> — <see cref="ModelRegistry.LoadAsync"/> handles download + mmap lifetime for servers.</item>
/// </list>
///
/// <para>Automatic construction is wired for <b>SDXL</b> end-to-end (the simplest dual-CLIP family, the
/// reference path every other family is measured against). Other architectures are <i>detected</i> but
/// not yet auto-constructed; <see cref="LoadAuto"/> throws a clear <see cref="NotSupportedException"/>
/// naming the detected family so callers fall back to the per-pipeline constructors. Extending coverage
/// is mechanical: add a case to the switch that drives the matching <c>*CheckpointConverter</c>.</para></summary>
public static class PipelineFactory
{
    /// <summary>Detects the architecture of the checkpoint or diffusers directory at <paramref name="path"/> by reading only tensor headers (no weight data is materialized).</summary>
    public static ModelArchitecture DetectArchitecture(string path)
    {
        ModelLayout layout = ModelLayoutResolver.Resolve(path);
        return ModelArchitectureDetector.DetectFromFile(layout.RepresentativeFile);
    }

    /// <summary>Loads a diffusion pipeline by inspecting the checkpoint at <paramref name="path"/> and dispatching to the matching per-model loader. SDXL is fully supported; other detected families throw <see cref="NotSupportedException"/> with the detected architecture named.</summary>
    /// <param name="path">A safetensors file, a sharded checkpoint directory, or a diffusers-layout directory.</param>
    /// <param name="backend">Compute backend the pipeline will use for inference.</param>
    public static DiffusionPipelineBase LoadAuto(string path, IBackend backend)
    {
        ModelLayout layout = ModelLayoutResolver.Resolve(path);
        ModelArchitecture arch = ModelArchitectureDetector.DetectFromFile(layout.RepresentativeFile);

        return arch switch
        {
            ModelArchitecture.Sdxl => LoadSdxl(layout.RepresentativeFile, backend),
            ModelArchitecture.Unknown => throw new NotSupportedException(
                $"Could not detect the model architecture of '{path}'. Construct the pipeline directly via its constructor."),
            _ => throw new NotSupportedException(
                $"Detected {arch}, but automatic construction is currently implemented for SDXL only. " +
                $"Construct the {arch} pipeline directly via its constructor (see HartsyInference.Diffusion.Tests for the pattern)."),
        };
    }

    /// <summary>Builds an <see cref="SdxlPipeline"/> from a single-file SDXL checkpoint. Loads + converts the LDM checkpoint, casts every component to F32 owned tensors (severing the mmap so the loader can be released), and wires CLIP-L, CLIP-G, the UNet, and the VAE decoder.</summary>
    public static SdxlPipeline LoadSdxl(string checkpointPath, IBackend backend)
    {
        (SdxlCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            SdxlCheckpointConverter.LoadAndConvert(checkpointPath);
        try
        {
            Dictionary<string, Tensor> unetWeights = ToOwnedF32(converted.UNet);
            Dictionary<string, Tensor> clipLWeights = ToOwnedF32(converted.ClipL);
            Dictionary<string, Tensor> clipGWeights = ToOwnedF32(converted.ClipG);
            Dictionary<string, Tensor> vaeWeights = ToOwnedF32(converted.Vae);

            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(clipLWeights, "text_model");

            ClipTextEncoder clipG = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipG);
            clipG.LoadWeights(clipGWeights, "text_model");

            UNet unet = new UNet(UNetConfig.SdxlBase);
            unet.LoadWeights(unetWeights);

            VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Sdxl);
            vaeDecoder.LoadWeights(vaeWeights);

            return new SdxlPipeline(backend, clipL, clipG, unet, vaeDecoder);
        }
        finally
        {
            // Safe: ToOwnedF32 copied every weight out of the mmap.
            loader.Dispose();
        }
    }

    /// <summary>Copies each weight into an owned F32 tensor. <see cref="Tensor.CastTo"/> always allocates (F32→F32 falls back to a memory copy), so the result no longer borrows the safetensors mmap and the source — when it owns its memory — is released to avoid a transient double of resident weights.</summary>
    private static Dictionary<string, Tensor> ToOwnedF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            Tensor f32 = kvp.Value.CastTo(DType.F32);
            if (kvp.Value.OwnsMemory)
            {
                kvp.Value.Dispose();
            }
            result[kvp.Key] = f32;
        }
        return result;
    }
}
