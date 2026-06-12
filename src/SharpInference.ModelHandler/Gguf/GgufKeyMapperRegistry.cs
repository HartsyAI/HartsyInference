using SharpInference.ModelHandler.Gguf.KeyMappers;

namespace SharpInference.ModelHandler.Gguf;

/// <summary>Static registry of every <see cref="IGgufKeyMapper"/>. Lookup by architecture string (from GGUF metadata <c>general.architecture</c>) or by tensor-name heuristic when metadata is missing.</summary>
public static class GgufKeyMapperRegistry
{
    private static readonly Dictionary<string, IGgufKeyMapper> _mappers = BuildRegistry();

    private static Dictionary<string, IGgufKeyMapper> BuildRegistry()
    {
        Dictionary<string, IGgufKeyMapper> r = new(StringComparer.OrdinalIgnoreCase);
        // Pixel-space variants must precede their parent families (and Flux, whose double+single-block
        // signature also matches the whole Chroma family): DetectByKeys probes mappers in registration
        // order and a Radiance/Zeta checkpoint is a strict key-superset of classic Chroma / Z-Image.
        Register(r, new ChromaRadianceKeyMapper());
        Register(r, new ZetaChromaKeyMapper());
        Register(r, new FluxKeyMapper());
        Register(r, new Flux2KeyMapper());
        Register(r, new SdxlKeyMapper());
        Register(r, new Sd3KeyMapper());
        Register(r, new Sd15KeyMapper());
        Register(r, new FLiteKeyMapper());
        Register(r, new ChromaKeyMapper());
        Register(r, new AuraFlowKeyMapper());
        Register(r, new ZImageKeyMapper());
        Register(r, new ErnieImageKeyMapper());
        Register(r, new HunyuanImageKeyMapper());
        Register(r, new QwenImageKeyMapper());
        Register(r, new LlamaKeyMapper());
        Register(r, new PassthroughKeyMapper());
        return r;
    }

    private static void Register(Dictionary<string, IGgufKeyMapper> r, IGgufKeyMapper mapper)
    {
        if (r.ContainsKey(mapper.Architecture))
            throw new InvalidOperationException($"Duplicate GGUF key-mapper registration for architecture '{mapper.Architecture}'.");
        r[mapper.Architecture] = mapper;
    }

    /// <summary>Looks up a mapper by architecture name. Returns null when no mapper matches — caller should fall back to <see cref="DetectByKeys"/>.</summary>
    public static IGgufKeyMapper? GetByArchitecture(string architecture)
    {
        return _mappers.TryGetValue(architecture, out IGgufKeyMapper? m) ? m : null;
    }

    /// <summary>Heuristic detection when the GGUF metadata doesn't set <c>general.architecture</c>. Iterates every registered mapper and returns the first one whose <see cref="IGgufKeyMapper.MatchesByKeys"/> returns true. Returns the passthrough mapper as a final fallback.</summary>
    public static IGgufKeyMapper DetectByKeys(IReadOnlyCollection<string> tensorNames)
    {
        foreach (IGgufKeyMapper mapper in _mappers.Values)
        {
            if (mapper is PassthroughKeyMapper) continue;
            if (mapper.MatchesByKeys(tensorNames)) return mapper;
        }
        return _mappers["passthrough"];
    }

    /// <summary>All registered architecture names.</summary>
    public static IReadOnlyCollection<string> Architectures => _mappers.Keys;
}
