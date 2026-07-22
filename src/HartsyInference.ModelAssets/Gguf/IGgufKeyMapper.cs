using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf;

/// <summary>Per-architecture remap from GGUF tensor names → the format an existing <c>*CheckpointConverter.Convert</c> expects (typically the model's single-file/BFL naming or canonical diffusers naming, depending on the converter).
///
/// <para>The point of splitting this from the existing checkpoint converters is that GGUF tensor naming is its own dialect (llama.cpp / city96 / ComfyUI conventions) — it doesn't match either BFL single-file naming or diffusers naming directly. Each mapper is small (~50-100 lines, mostly a string-replacement table) and hands off to the existing converter for the real work.</para>
///
/// <para><b>Discovery</b>: Mappers self-register at static-ctor time of <see cref="GgufKeyMapperRegistry"/>. To add a new architecture, drop a new file in <c>Gguf/KeyMappers/</c>, implement this interface, and add a <c>Register</c> call in the registry. No other files need changes.</para></summary>
public interface IGgufKeyMapper
{
    /// <summary>The <c>general.architecture</c> metadata value this mapper handles. Lower-case (matching ggml convention). Examples: <c>"flux"</c>, <c>"sd3"</c>, <c>"sdxl"</c>, <c>"sd15"</c>, <c>"flite"</c>, <c>"chroma"</c>, <c>"auraflow"</c>, <c>"zimage"</c>. This is also the mapper's canonical display name.</summary>
    string Architecture { get; }

    /// <summary>Every <c>general.architecture</c> value this mapper handles, when one mapper covers a whole family that shares an identical GGUF tensor-naming dialect (e.g. llama.cpp emits the same <c>blk.N.attn_q.*</c> dialect for <c>llama</c>, <c>qwen2</c> and <c>qwen3</c> dense decoders). Defaults to just <see cref="Architecture"/>. Declaring the family here makes <see cref="GgufKeyMapperRegistry.GetByArchitecture"/> resolve every member directly instead of relying on the key-heuristic fallback.</summary>
    IReadOnlyCollection<string> Architectures => [Architecture];

    /// <summary>Heuristic fallback when GGUF metadata doesn't set <c>general.architecture</c> (some city96 dumps don't). Matches against tensor-name patterns. Return true if this mapper recognizes the file from its keys alone.</summary>
    bool MatchesByKeys(IEnumerable<string> tensorNames);

    /// <summary>Maps one GGUF tensor name to its target name. Return <c>null</c> to drop the tensor (e.g. metadata-only entries that ggml emits but the model doesn't consume). The target naming should match what the corresponding <c>*CheckpointConverter.Convert</c> expects as its input dict.</summary>
    string? MapKey(string ggufKey);
}
