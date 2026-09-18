using HartsyInference.ModelAssets.Gguf.KeyMappers;

namespace HartsyInference.ModelAssets.Gguf;

/// <summary>Static registry of every <see cref="IGgufKeyMapper"/>. Lookup by architecture string (from GGUF metadata <c>general.architecture</c>) or by tensor-name heuristic when metadata is missing.</summary>
public static class GgufKeyMapperRegistry
{
    // Distinct mappers in registration order — drives the key-heuristic fallback (DetectByKeys). A single
    // mapper registered under several architecture aliases appears here exactly once. Declared before
    // _mappers so it is initialized when BuildRegistry() (which appends to it via Register) runs.
    private static readonly List<IGgufKeyMapper> _ordered = [];
    private static readonly Dictionary<string, IGgufKeyMapper> _mappers = BuildRegistry();

    private static Dictionary<string, IGgufKeyMapper> BuildRegistry()
    {
        Dictionary<string, IGgufKeyMapper> r = new(StringComparer.OrdinalIgnoreCase);
        // Diffusion families first, in the order DiffusionGgufFamilies documents: their identity mappers carry only a
        // recognition rule, and several of those rules are supersets of each other (Radiance over Chroma, Zeta over
        // Z-Image, Hunyuan Image's Tencent repack over Flux — that last one silently loaded every HunyuanImage GGUF as
        // garbage Flux weights until the ordering was fixed, 2026-07-21).
        foreach (IGgufKeyMapper family in DiffusionGgufFamilies.InDetectionOrder) Register(r, family);
        // Gemma before Llama: Gemma's heuristic (sandwich norms) is a strict superset of the llama-family keys.
        Register(r, new GemmaKeyMapper());
        Register(r, new Gemma4KeyMapper());
        Register(r, new PhiKeyMapper());
        Register(r, new Gpt2KeyMapper());
        Register(r, new Glm4KeyMapper());
        Register(r, new DeepSeekKeyMapper());
        // mllama before llama: its cross_attn_* keys are a strict superset of the llama-family signature.
        Register(r, new MllamaKeyMapper());
        Register(r, new LlamaKeyMapper());
        Register(r, new PassthroughKeyMapper());
        return r;
    }

    private static void Register(Dictionary<string, IGgufKeyMapper> r, IGgufKeyMapper mapper)
    {
        foreach (string arch in mapper.Architectures)
        {
            if (r.ContainsKey(arch))
                throw new InvalidOperationException($"Duplicate GGUF key-mapper registration for architecture '{arch}'.");
            r[arch] = mapper;
        }
        _ordered.Add(mapper);
    }

    /// <summary>Looks up a mapper by architecture name. Returns null when no mapper matches — caller should fall back to <see cref="DetectByKeys"/>.</summary>
    public static IGgufKeyMapper? GetByArchitecture(string architecture)
    {
        return _mappers.TryGetValue(architecture, out IGgufKeyMapper? m) ? m : null;
    }

    /// <summary>Heuristic detection when the GGUF metadata doesn't set <c>general.architecture</c>. Iterates every registered mapper and returns the first one whose <see cref="IGgufKeyMapper.MatchesByKeys"/> returns true. Returns the passthrough mapper as a final fallback.</summary>
    public static IGgufKeyMapper DetectByKeys(IReadOnlyCollection<string> tensorNames)
    {
        foreach (IGgufKeyMapper mapper in _ordered)
        {
            if (mapper is PassthroughKeyMapper) continue;
            if (mapper.MatchesByKeys(tensorNames)) return mapper;
        }
        return _mappers["passthrough"];
    }

    /// <summary>All registered architecture names.</summary>
    public static IReadOnlyCollection<string> Architectures => _mappers.Keys;
}
