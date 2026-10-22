namespace HartsyInference.Engine.Recipes;

/// <summary>A model resolved to a family this engine build has no recipe for, so there are no defaults to plan with
/// and nothing to construct. Thrown instead of planning against invented values.</summary>
public sealed class RecipeNotRegisteredException : NotSupportedException
{
    /// <summary>Creates the exception, naming the family and what is drivable instead.</summary>
    public RecipeNotRegisteredException(Modality modality, string familyId, IReadOnlyCollection<string> drivable)
        : base(BuildMessage(modality, familyId, drivable))
    {
        Modality = modality;
        FamilyId = familyId;
        Drivable = [.. drivable];
    }

    /// <summary>The modality whose registry was asked.</summary>
    public Modality Modality { get; }

    /// <summary>The family id that has no recipe.</summary>
    public string FamilyId { get; }

    /// <summary>The families that registry can drive.</summary>
    public IReadOnlyList<string> Drivable { get; }

    private static string BuildMessage(Modality modality, string familyId, IReadOnlyCollection<string> drivable)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(familyId);
        ArgumentNullException.ThrowIfNull(drivable);
        string kind = modality == Modality.Video ? "video" : "image";
        return $"Model family '{familyId}' has no registered {kind} recipe. Currently drivable: {string.Join(", ", drivable)}.";
    }
}
