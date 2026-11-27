using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Utonia;

namespace HartsyInference.ThreeD.Models.CoLa3D;

/// <summary>
/// Loads the CoLa3D scene pipeline from a Models root. Expects one decomposer under <c>CoLa3D/</c> (see
/// <see cref="DecomposerFileName"/>), <c>TRELLIS2-ShapeVAE/shape_{enc,dec}_next_dc_f16c32_fp16.safetensors</c> and
/// <c>Utonia/utonia.pth</c> beneath it. Pickle checkpoints go through <see cref="PytorchPickleLoader"/>, which reads tensors
/// without executing pickled code.
/// </summary>
public static class CoLa3DCheckpointConverter
{
    /// <summary>Shape VAE encoder checkpoint.</summary>
    public const string ShapeEncoderFile = "shape_enc_next_dc_f16c32_fp16.safetensors";

    /// <summary>Shape VAE decoder checkpoint.</summary>
    public const string ShapeDecoderFile = "shape_dec_next_dc_f16c32_fp16.safetensors";

    /// <summary>Utonia encoder checkpoint.</summary>
    public const string UtoniaFile = "utonia.pth";

    private const string DecomposerDir = "CoLa3D";
    private const string ShapeVaeDir = "TRELLIS2-ShapeVAE";
    private const string UtoniaDir = "Utonia";
    private const string StateDictPrefix = "state_dict.";

    /// <summary>The q20 decomposer checkpoint for a query source: free, mask or point.</summary>
    public static string DecomposerFileName(CoLa3DConfig.QuerySourceKind source) => source switch
    {
        CoLa3DConfig.QuerySourceKind.Fixed => "cola3d_free_q20.pt",
        CoLa3DConfig.QuerySourceKind.PromptMask => "cola3d_mask_q20.pt",
        CoLa3DConfig.QuerySourceKind.PromptPoint => "cola3d_point_q20.pt",
        _ => throw new ArgumentOutOfRangeException(nameof(source), source, "Unknown query source."),
    };

    /// <summary>Loads and converts all three weight sets for one decomposer variant. Throws when a file is missing or a
    /// checkpoint does not match its key map.</summary>
    public static CoLa3DWeights Load(string modelRoot, CoLa3DConfig.QuerySourceKind source = CoLa3DConfig.QuerySourceKind.Fixed)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelRoot);
        string decomposerFile = DecomposerFileName(source);
        List<IDisposable> loaders = [];
        bool built = false;
        try
        {
            PytorchPickleLoader decomposerLoader = new();
            loaders.Add(decomposerLoader);
            decomposerLoader.Load(Locate(modelRoot, DecomposerDir, decomposerFile), recursiveFlatten: true);
            if (decomposerLoader.SkippedTensorCount != 0)
                throw new HartsyInferenceException($"{decomposerFile} skipped {decomposerLoader.SkippedTensorCount} tensors.");
            Dictionary<string, Tensor> decomposer = CoLa3DKeyMap.Remap(
                CoLa3DConfig.ForQuerySource(source), decomposerLoader.GetAllTensors());

            SafeTensorsLoader encoderLoader = new();
            loaders.Add(encoderLoader);
            encoderLoader.Load(Locate(modelRoot, ShapeVaeDir, ShapeEncoderFile));

            SafeTensorsLoader decoderLoader = new();
            loaders.Add(decoderLoader);
            decoderLoader.Load(Locate(modelRoot, ShapeVaeDir, ShapeDecoderFile));

            PytorchPickleLoader utoniaLoader = new();
            loaders.Add(utoniaLoader);
            utoniaLoader.Load(Locate(modelRoot, UtoniaDir, UtoniaFile), recursiveFlatten: true);
            UtoniaWeights utonia = new(ToUtoniaNames(utoniaLoader.GetAllTensors()));

            CoLa3DWeights weights = new(decomposer, encoderLoader.GetAllTensors(), decoderLoader.GetAllTensors(), utonia, loaders);
            built = true;
            return weights;
        }
        finally
        {
            if (!built)
                foreach (IDisposable loader in loaders)
                    loader.Dispose();
        }
    }

    /// <summary>Keys the encoder reads, renamed to C# names. Computed buffers and keys the encoder never reads are dropped.</summary>
    internal static Dictionary<string, Tensor> ToUtoniaNames(Dictionary<string, Tensor> stateDict)
    {
        Dictionary<string, Tensor> named = new(StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> entry in stateDict)
        {
            string key = entry.Key.StartsWith(StateDictPrefix, StringComparison.Ordinal) ? entry.Key[StateDictPrefix.Length..] : entry.Key;
            if (UtoniaCheckpointKeys.IsComputedBuffer(key))
                continue;
            string? name = UtoniaCheckpointKeys.ToCSharpName(key);
            if (name is not null)
                named[name] = entry.Value;
        }

        return named;
    }

    // Checks the component folder first, then the root, so both the Models root and a component folder work. No recursive
    // search: the Models root holds tens of gigabytes of unrelated checkpoints.
    private static string Locate(string modelRoot, string subfolder, string fileName)
    {
        string nested = Path.Combine(modelRoot, subfolder, fileName);
        if (File.Exists(nested))
            return nested;
        string direct = Path.Combine(modelRoot, fileName);
        if (File.Exists(direct))
            return direct;
        throw new FileNotFoundException($"CoLa3D checkpoint '{fileName}' not found; expected {nested}.", nested);
    }
}
