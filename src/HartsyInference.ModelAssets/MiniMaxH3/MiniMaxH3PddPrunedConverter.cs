using System.Globalization;
using System.Security.Cryptography;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.MiniMaxH3;

/// <summary>Local-only converter that rebases an official dense PDD adapter onto an operator-supplied pruned H3 base.</summary>
internal static class MiniMaxH3PddPrunedConverter
{
    /// <summary>Converts and writes one hash-provenanced adapter without downloading or embedding any basis asset.</summary>
    public static MiniMaxH3PddConversionSummary Convert(string adapterPath, string fullBasePath,
        string targetPrunedBasePath, string outputPath, MiniMaxH3PddTask expectedTask)
    {
        if (expectedTask == MiniMaxH3PddTask.Unknown)
            throw new ArgumentException("PDD conversion requires a hash-bound FL2VA or Ref2VA task.", nameof(expectedTask));
        ValidateDistinctPaths(adapterPath, fullBasePath, targetPrunedBasePath, outputPath);
        string adapterHash = Sha256(adapterPath);
        string fullHash = Sha256(fullBasePath);
        string targetHash = Sha256(targetPrunedBasePath);

        using MiniMaxH3PddAdapter adapter = MiniMaxH3PddAdapter.Load(adapterPath,
            MiniMaxH3PddFormatHint.OfficialFullHeads, expectedTask);
        using SafeTensorsLoader fullLoader = new SafeTensorsLoader();
        using SafeTensorsLoader targetLoader = new SafeTensorsLoader();
        fullLoader.Load(fullBasePath);
        targetLoader.Load(targetPrunedBasePath);
        Dictionary<string, Tensor> full = fullLoader.GetAllTensors();
        Dictionary<string, Tensor> target = targetLoader.GetAllTensors();
        if (full.ContainsKey("adaln_t_table"))
            throw new HartsyInferenceException("PDD conversion fullBase must be the dense time-embedder checkpoint, not a pruned build.");
        if (!target.ContainsKey("adaln_t_table"))
            throw new HartsyInferenceException("PDD conversion targetPrunedBase has no adaln_t_table.");

        using MiniMaxH3PddAffineBasis basis = MiniMaxH3PddAffineFitter.Fit(full, target, maxResidual: 1e-4);
        using MiniMaxH3PddRebaseResult rebased = MiniMaxH3PddPrunedRebaser.Rebase(adapter.Trunk.Layers, basis);
        ValidateTargetDiffShapes(target, rebased.FullWeightDiffs);

        Dictionary<string, Tensor> output = new(StringComparer.Ordinal);
        List<Tensor> scalarAlpha = [];
        try
        {
            foreach (LoraLayer layer in rebased.Layers)
            {
                string root = layer.TargetKey[..^".weight".Length];
                output[root + ".lora_A.weight"] = layer.LoraDown;
                output[root + ".lora_B.weight"] = layer.LoraUp;
                Tensor alpha = new Tensor(new TensorShape(1), DType.F32);
                alpha.AsSpan<float>()[0] = layer.Alpha;
                scalarAlpha.Add(alpha);
                output[root + ".alpha"] = alpha;
            }
            foreach (LoraFullWeightDiff diff in rebased.FullWeightDiffs)
            {
                string suffix = diff.IsBias ? ".bias" : ".weight";
                string root = diff.TargetKey[..^suffix.Length];
                output[root + (diff.IsBias ? ".diff_b" : ".diff")] = diff.Diff;
            }
            output["proj_out.weight"] = adapter.VideoHeadWeight;
            output["proj_out.bias"] = adapter.VideoHeadBias;
            output["audio_proj_out.weight"] = adapter.AudioHeadWeight;
            output["audio_proj_out.bias"] = adapter.AudioHeadBias;
            SafeTensorsWriter.Save(outputPath, output, BuildMetadata(adapter, expectedTask, adapterHash,
                fullHash, targetHash, basis.RelativeResidual));
        }
        finally
        {
            foreach (Tensor tensor in scalarAlpha) tensor.Dispose();
        }

        return new MiniMaxH3PddConversionSummary
        {
            AdapterSha256 = adapterHash,
            FullBaseSha256 = fullHash,
            TargetBaseSha256 = targetHash,
            RelativeResidual = basis.RelativeResidual,
            RebasedModules = rebased.FullWeightDiffs.Count / 2,
        };
    }

    private static Dictionary<string, string> BuildMetadata(MiniMaxH3PddAdapter adapter,
        MiniMaxH3PddTask task, string adapterHash, string fullHash, string targetHash, double residual)
    {
        Dictionary<string, string> metadata = adapter.Metadata is null
            ? new Dictionary<string, string>(StringComparer.Ordinal)
            : new Dictionary<string, string>(adapter.Metadata, StringComparer.Ordinal);
        metadata["hartsy.pdd.format"] = "minimax_h3_pdd_hartsy_pruned_v1";
        metadata["hartsy.pdd.head_layout"] = "full_heads_3d";
        metadata["hartsy.pdd.task"] = task == MiniMaxH3PddTask.Fl2Va ? "fl2va" : "ref2va";
        metadata["hartsy.pdd.adapter_sha256"] = adapterHash;
        metadata["hartsy.pdd.full_base_sha256"] = fullHash;
        metadata["hartsy.pdd.target_base_sha256"] = targetHash;
        metadata["hartsy.pdd.affine_residual"] = residual.ToString("E17", CultureInfo.InvariantCulture);
        metadata["hartsy.pdd.converter"] = "HartsyInference.MiniMaxH3PddPrunedConverter/v1";
        metadata["pdd_num_steps"] = adapter.PddNumSteps.ToString(CultureInfo.InvariantCulture);
        metadata["pdd_block_size"] = adapter.PddBlockSize.ToString(CultureInfo.InvariantCulture);
        metadata["lora_rank"] = adapter.Rank.ToString(CultureInfo.InvariantCulture);
        metadata["lora_alpha"] = adapter.Alpha.ToString("R", CultureInfo.InvariantCulture);
        return metadata;
    }

    private static void ValidateTargetDiffShapes(IReadOnlyDictionary<string, Tensor> target,
        IReadOnlyList<LoraFullWeightDiff> diffs)
    {
        foreach (LoraFullWeightDiff diff in diffs)
        {
            if (!target.TryGetValue(diff.TargetKey, out Tensor? targetTensor))
                throw new HartsyInferenceException($"Pruned H3 target is missing rebased AdaLN tensor '{diff.TargetKey}'.");
            if (targetTensor.Shape != diff.Diff.Shape)
            {
                throw new HartsyInferenceException(
                    $"Rebased PDD diff '{diff.TargetKey}' has shape {diff.Diff.Shape}, target has {targetTensor.Shape}.");
            }
        }
    }

    private static void ValidateDistinctPaths(string adapterPath, string fullBasePath,
        string targetPrunedBasePath, string outputPath)
    {
        string[] inputs = [adapterPath, fullBasePath, targetPrunedBasePath];
        foreach (string input in inputs)
        {
            if (!File.Exists(input)) throw new FileNotFoundException("PDD conversion input not found.", input);
        }
        if (inputs.Any(input => FileSystemPathIdentity.SamePath(input, outputPath)))
        {
            throw new ArgumentException("PDD conversion output must not overwrite an input file.", nameof(outputPath));
        }
    }

    private static string Sha256(string path)
    {
        using FileStream stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        return System.Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }
}
