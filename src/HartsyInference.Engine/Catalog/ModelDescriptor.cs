using System.Text.Json.Serialization;
using HartsyInference.Engine.Planning;

namespace HartsyInference.Engine.Catalog;

/// <summary>What a model's recipe declares about itself, answered without loading weights. The engine serializes this
/// type on the wire: the <c>$type</c> discriminator is <c>image</c> or <c>video</c>.</summary>
[JsonPolymorphic]
[JsonDerivedType(typeof(ImageModelDescriptor), "image")]
[JsonDerivedType(typeof(VideoModelDescriptor), "video")]
public abstract record ModelDescriptor
{
    /// <summary>The family, defaults revision and variant the descriptor resolved to.</summary>
    public required RecipeIdentity Recipe { get; init; }

    /// <summary>The domain the family serves.</summary>
    public required Modality Modality { get; init; }

    /// <summary>The variant ids the family's catalog declares; empty when the family declares none.</summary>
    public required IReadOnlyList<string> Variants { get; init; }

    /// <summary>True when exact-hash checkpoint profiles overlay this family's defaults at plan time.</summary>
    public required bool ProfileBound { get; init; }
}
