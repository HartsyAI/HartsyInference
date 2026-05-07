namespace SharpInference.ModelHandler.Gguf.KeyMappers;

/// <summary>SD3 / SD3.5 GGUF mapper. SD3 GGUFs use the Stability single-file naming with <c>model.diffusion_model.</c> prefix that <see cref="SharpInference.ModelHandler.CheckpointConverters.Sd3CheckpointConverter.Convert"/> already handles.</summary>
public sealed class Sd3KeyMapper : IGgufKeyMapper
{
    public string Architecture => "sd3";

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        foreach (string name in tensorNames)
        {
            if (name.Contains("joint_blocks.", StringComparison.Ordinal)) return true;
            if (name.Contains("x_block.", StringComparison.Ordinal) && name.Contains("attn.", StringComparison.Ordinal)) return true;
        }
        return false;
    }

    public string? MapKey(string ggufKey) => ggufKey;
}
