using System.Globalization;
using System.Security.Cryptography;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.MiniMaxH3;

/// <summary>Locally rebases the official full-width H3 Fun control AdaLN projections onto a specified pruned base.</summary>
public static unsafe class MiniMaxH3ControlNetPrunedConverter
{
    /// <summary>Writes a self-contained converted branch after an F64 fit with relative residual at most 1e-4.</summary>
    public static MiniMaxH3ControlNetConversionSummary Convert(string controlPath, string fullBasePath,
        string targetPrunedBasePath, string outputPath)
    {
        ValidatePaths(controlPath, fullBasePath, targetPrunedBasePath, outputPath);
        string controlHash = Sha256(controlPath);
        string fullHash = Sha256(fullBasePath);
        string targetHash = Sha256(targetPrunedBasePath);

        using SafeTensorsLoader controlLoader = new SafeTensorsLoader();
        using SafeTensorsLoader fullLoader = new SafeTensorsLoader();
        using SafeTensorsLoader targetLoader = new SafeTensorsLoader();
        controlLoader.Load(controlPath);
        fullLoader.Load(fullBasePath);
        targetLoader.Load(targetPrunedBasePath);
        Dictionary<string, Tensor> control = MiniMaxH3ControlNetCheckpointConverter.Convert(
            controlLoader.GetAllTensors());
        MiniMaxH3CheckpointConverter.ConvertedWeights fullConverted = MiniMaxH3CheckpointConverter.Convert(
            fullLoader.GetAllTensors(), castToF32: false);
        MiniMaxH3CheckpointConverter.ConvertedWeights targetConverted = MiniMaxH3CheckpointConverter.Convert(
            targetLoader.GetAllTensors(), castToF32: false);
        List<Tensor> owned = new List<Tensor>();
        try
        {
            if (fullConverted.Transformer.ContainsKey("adaln_t_table"))
            {
                throw new HartsyInferenceException(
                    "ControlNet conversion fullBase must carry the dense H3 time embedder, not an adaln_t_table.");
            }
            if (!targetConverted.Transformer.ContainsKey("adaln_t_table"))
            {
                throw new HartsyInferenceException(
                    "ControlNet conversion targetPrunedBase must carry adaln_t_table.");
            }

            using MiniMaxH3PddAffineBasis basis = MiniMaxH3PddAffineFitter.Fit(
                fullConverted.Transformer, targetConverted.Transformer, maxResidual: 1e-4);
            int controlBlocks = CountControlBlocks(control);
            int controlTimeDim = checked((int)Require(control,
                "control_blocks.0.adaln_proj.linear.weight").Shape[1]);
            if (controlBlocks != 5)
            {
                throw new HartsyInferenceException(
                    $"The published H3 Fun control branch must carry five blocks; got {controlBlocks}.");
            }
            if (controlTimeDim != basis.Intercept.Shape[0])
            {
                throw new HartsyInferenceException(
                    $"Control AdaLN width {controlTimeDim} does not match dense base curve width "
                    + $"{basis.Intercept.Shape[0]}.");
            }

            for (int index = 0; index < controlBlocks; index++)
            {
                string prefix = $"control_blocks.{index}.adaln_proj.linear";
                Tensor denseWeight = control[prefix + ".weight"];
                Tensor denseBias = control[prefix + ".bias"];
                (Tensor weight, Tensor bias) = RebaseProjection(denseWeight, denseBias, basis);
                owned.Add(weight);
                owned.Add(bias);
                control[prefix + ".weight"] = weight;
                control[prefix + ".bias"] = bias;
            }

            int rebasedTimeDim = checked((int)Require(control,
                "control_blocks.0.adaln_proj.linear.weight").Shape[1]);
            if (rebasedTimeDim != basis.Projection.Shape[1])
            {
                throw new HartsyInferenceException(
                    $"Rebased control AdaLN width {rebasedTimeDim} does not match target curve width "
                    + $"{basis.Projection.Shape[1]}.");
            }

            Dictionary<string, string> metadata = BuildMetadata(controlLoader.Metadata, controlHash, fullHash,
                targetHash, basis.RelativeResidual);
            SafeTensorsWriter.Save(outputPath, control, metadata);
            return new MiniMaxH3ControlNetConversionSummary
            {
                ControlSha256 = controlHash,
                FullBaseSha256 = fullHash,
                TargetBaseSha256 = targetHash,
                RelativeResidual = basis.RelativeResidual,
                RebasedBlocks = controlBlocks,
            };
        }
        finally
        {
            foreach (Tensor tensor in owned)
            {
                tensor.Dispose();
            }
            foreach (Tensor tensor in control.Values.Distinct())
            {
                if (!owned.Contains(tensor))
                {
                    tensor.Dispose();
                }
            }
        }
    }

    /// <summary>Applies a fitted dense-time-to-curve affine basis to one control AdaLN projection. The multiply and
    /// DC-bias accumulation are performed in F64 before the self-contained F32 output tensors are written.</summary>
    public static (Tensor Weight, Tensor Bias) RebaseProjection(Tensor denseWeight, Tensor denseBias,
        MiniMaxH3PddAffineBasis basis)
    {
        int output = checked((int)denseWeight.Shape[0]);
        int dense = checked((int)denseWeight.Shape[1]);
        int curve = checked((int)basis.Projection.Shape[1]);
        if (denseWeight.Shape.Rank != 2 || denseBias.Shape.Rank != 1 || denseBias.Shape[0] != output
            || basis.Intercept.Shape[0] != dense || basis.Projection.Shape[0] != dense)
        {
            throw new HartsyInferenceException(
                $"Cannot rebase control AdaLN shapes weight={denseWeight.Shape}, bias={denseBias.Shape}, "
                + $"intercept={basis.Intercept.Shape}, projection={basis.Projection.Shape}.");
        }

        Tensor weight = new Tensor(new TensorShape(output, curve), DType.F32);
        Tensor bias = new Tensor(new TensorShape(output), DType.F32);
        float* weightPointer = (float*)weight.DataPointer;
        float* biasPointer = (float*)bias.DataPointer;
        try
        {
            Parallel.For(0, output, row =>
            {
                double dc = MiniMaxH3TensorReader.Read(denseBias, row);
                Span<double> projected = stackalloc double[curve];
                long denseBase = (long)row * dense;
                for (int column = 0; column < dense; column++)
                {
                    double value = MiniMaxH3TensorReader.Read(denseWeight, denseBase + column);
                    dc += value * MiniMaxH3TensorReader.Read(basis.Intercept, column);
                    long projectionBase = (long)column * curve;
                    for (int coordinate = 0; coordinate < curve; coordinate++)
                    {
                        projected[coordinate] += value
                            * MiniMaxH3TensorReader.Read(basis.Projection, projectionBase + coordinate);
                    }
                }
                biasPointer[row] = (float)dc;
                for (int coordinate = 0; coordinate < curve; coordinate++)
                {
                    weightPointer[(long)row * curve + coordinate] = (float)projected[coordinate];
                }
            });
            return (weight, bias);
        }
        catch
        {
            weight.Dispose();
            bias.Dispose();
            throw;
        }
    }

    private static Dictionary<string, string> BuildMetadata(IReadOnlyDictionary<string, string>? original,
        string controlHash, string fullHash, string targetHash, double residual)
    {
        Dictionary<string, string> metadata = original is null
            ? new Dictionary<string, string>(StringComparer.Ordinal)
            : new Dictionary<string, string>(original, StringComparer.Ordinal);
        metadata["hartsy.controlnet.format"] = "minimax_h3_fun_pruned_v1";
        metadata["hartsy.controlnet.control_sha256"] = controlHash;
        metadata["hartsy.controlnet.full_base_sha256"] = fullHash;
        metadata["hartsy.controlnet.target_base_sha256"] = targetHash;
        metadata["hartsy.controlnet.affine_residual"] = residual.ToString("E17", CultureInfo.InvariantCulture);
        metadata["hartsy.controlnet.converter"] = "HartsyInference.MiniMaxH3ControlNetPrunedConverter/v1";
        return metadata;
    }

    private static int CountControlBlocks(IReadOnlyDictionary<string, Tensor> tensors)
    {
        int count = 0;
        while (tensors.ContainsKey($"control_blocks.{count}.after_proj.weight"))
        {
            count++;
        }
        return count;
    }

    private static Tensor Require(IReadOnlyDictionary<string, Tensor> tensors, string key)
    {
        return tensors.TryGetValue(key, out Tensor? tensor) ? tensor
            : throw new HartsyInferenceException($"MiniMax-H3 Fun control checkpoint is missing '{key}'.");
    }

    private static void ValidatePaths(string controlPath, string fullBasePath, string targetPath, string outputPath)
    {
        string[] inputs = [controlPath, fullBasePath, targetPath];
        foreach (string input in inputs)
        {
            if (!File.Exists(input))
            {
                throw new FileNotFoundException("MiniMax-H3 ControlNet conversion input not found.", input);
            }
        }
        string output = Path.GetFullPath(outputPath);
        StringComparison comparison = OperatingSystem.IsWindows()
            ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        if (inputs.Any(input => string.Equals(Path.GetFullPath(input), output, comparison)))
        {
            throw new ArgumentException("ControlNet conversion output must not overwrite an input file.", nameof(outputPath));
        }
    }

    private static string Sha256(string path)
    {
        using FileStream stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        return System.Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }
}
