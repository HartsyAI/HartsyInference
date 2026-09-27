namespace HartsyInference.Engine.Variants;

/// <summary>Filename token sets several families share, so the same guess means the same thing everywhere.</summary>
public static class VariantTokens
{
    /// <summary>Step-distilled builds: <c>turbo</c>, <c>tdm</c>, <c>distill</c>, <c>distilled</c>.</summary>
    public static IReadOnlyList<IReadOnlyList<string>> Distilled { get; } = [["turbo"], ["tdm"], ["distill"], ["distilled"]];

    /// <summary><see cref="Distilled"/> with <paramref name="token"/> also required in each set.</summary>
    public static IReadOnlyList<IReadOnlyList<string>> DistilledWith(string token) =>
        Distilled.Select(set => (IReadOnlyList<string>)[token, .. set]).ToArray();
}
