namespace HartsyInference.ModelHandler.Gguf.KeyMappers;

/// <summary>Maps Flux GGUF tensor names → BFL single-file naming that <see cref="HartsyInference.ModelHandler.CheckpointConverters.FluxCheckpointConverter.Convert"/> expects.
///
/// <para>city96's `city96/FLUX.1-dev-gguf` ships with the standard ComfyUI BFL prefix <c>model.diffusion_model.</c>. The existing FluxCheckpointConverter strips that prefix; we just need to pass the GGUF tensor names through unchanged. For Flux the GGUF naming and the BFL naming are 1:1.</para>
///
/// <para>If a future GGUF builder uses llama.cpp's preferred <c>blk.{i}.</c> prefix instead, add a remap rule here. As of city96's current dumps (verified 2026-05-06), no such rewrite is needed.</para></summary>
public sealed class FluxKeyMapper : IGgufKeyMapper
{
    public string Architecture => "flux";

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        bool seenDouble = false, seenSingle = false;
        foreach (string name in tensorNames)
        {
            if (name.Contains("double_blocks.", StringComparison.Ordinal)) seenDouble = true;
            if (name.Contains("single_blocks.", StringComparison.Ordinal)) seenSingle = true;
            if (seenDouble && seenSingle) return true;
        }
        return false;
    }

    public string? MapKey(string ggufKey) => ggufKey;
}
