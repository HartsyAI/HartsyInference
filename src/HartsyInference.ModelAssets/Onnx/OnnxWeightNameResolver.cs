using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Onnx;

/// <summary>Recovers the logical weight names that an ONNX export anonymized. When PyTorch's <c>weight_norm</c> (or a similar parametrization) is exported, the fused conv/matmul weight becomes an unnamed initializer (<c>onnx::Conv_8168</c>) feeding a node, while that node's bias keeps its module name (<c>flow.flows.6.enc.in_layers.0.bias</c>). This walks the graph and renames each anonymous weight to <c>&lt;module&gt;.weight</c>, derived from the sibling bias (preferred) or the node's path-encoding name, so a model loader can bind weights by their PyTorch names. Not model-specific — it fixes any weight-norm-fused export.</summary>
public static class OnnxWeightNameResolver
{
    private static readonly HashSet<string> WeightOps = new(StringComparer.Ordinal)
    {
        "Conv", "ConvTranspose", "MatMul", "Gemm",
    };

    /// <summary>Returns a copy of <paramref name="tensors"/> with anonymous weight initializers renamed to their recovered module names. Tensors with no recoverable name keep their original key.</summary>
    public static Dictionary<string, Tensor> Resolve(OnnxModel model, IReadOnlyDictionary<string, Tensor> tensors)
    {
        Dictionary<string, string> rename = new(StringComparer.Ordinal);

        foreach (OnnxNode node in model.Nodes)
        {
            // Embedding tables surface as a Gather whose data input is an (often oddly-named) initializer; recover the
            // module name from the node path (e.g. "/enc_p/emb/Gather" -> "enc_p.emb.weight").
            if (node.OpType == "Gather")
            {
                if (node.Inputs.Length >= 1 && tensors.ContainsKey(node.Inputs[0]) && IsAnonymous(node.Inputs[0])
                    && !rename.ContainsKey(node.Inputs[0]) && node.Name.Length > 0)
                {
                    string embTarget = DeriveFromNodeName(node.Name);
                    if (!tensors.ContainsKey(embTarget) && !rename.ContainsValue(embTarget))
                        rename[node.Inputs[0]] = embTarget;
                }
                continue;
            }

            if (!WeightOps.Contains(node.OpType)) continue;

            string? anonWeight = null;
            string? namedBias = null;
            foreach (string input in node.Inputs)
            {
                if (!tensors.ContainsKey(input)) continue; // only initializer inputs
                if (IsAnonymous(input)) anonWeight = input;
                else if (input.EndsWith(".bias", StringComparison.Ordinal)) namedBias = input;
            }

            if (anonWeight is null || rename.ContainsKey(anonWeight)) continue;

            string? target = null;
            if (namedBias is not null)
                target = string.Concat(namedBias.AsSpan(0, namedBias.Length - ".bias".Length), ".weight");
            else if (node.Name.Length > 0)
                target = DeriveFromNodeName(node.Name);

            if (target is not null && !tensors.ContainsKey(target) && !rename.ContainsValue(target))
                rename[anonWeight] = target;
        }

        Dictionary<string, Tensor> result = new(tensors.Count, StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> kv in tensors)
            result[rename.TryGetValue(kv.Key, out string? renamed) ? renamed : kv.Key] = kv.Value;
        return result;
    }

    private static bool IsAnonymous(string name)
        => name.StartsWith("onnx::", StringComparison.Ordinal) || !name.Contains('.');

    // "/flow/flows.6/enc/in_layers.0/Conv" -> "flow.flows.6.enc.in_layers.0.weight"
    private static string DeriveFromNodeName(string nodeName)
    {
        string[] parts = nodeName.Trim('/').Split('/');
        if (parts.Length == 0) return nodeName;
        // Drop the trailing op-name segment (e.g. "Conv").
        int take = parts.Length > 1 ? parts.Length - 1 : parts.Length;
        return string.Join('.', parts[..take]) + ".weight";
    }
}
