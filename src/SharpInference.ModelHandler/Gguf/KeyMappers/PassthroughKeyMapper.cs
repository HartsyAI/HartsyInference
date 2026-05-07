namespace SharpInference.ModelHandler.Gguf.KeyMappers;

/// <summary>Final-fallback mapper that returns every tensor name unchanged. Selected when neither metadata nor key heuristic finds a match. Useful for novel GGUFs that happen to ship with diffusers naming already.</summary>
public sealed class PassthroughKeyMapper : IGgufKeyMapper
{
    public string Architecture => "passthrough";
    public bool MatchesByKeys(IEnumerable<string> tensorNames) => true;
    public string? MapKey(string ggufKey) => ggufKey;
}
