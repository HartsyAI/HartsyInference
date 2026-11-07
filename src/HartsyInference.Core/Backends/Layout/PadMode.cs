namespace HartsyInference.Core.Backends.Layout;

/// <summary>How <see cref="IBackend.Pad"/> fills the positions outside the input.</summary>
public enum PadMode
{
    /// <summary>All-zero bits (<c>F.pad(mode="constant", value=0)</c>).</summary>
    Zero,

    /// <summary>The nearest edge element of the input (<c>F.pad(mode="replicate")</c>).</summary>
    Replicate,

    /// <summary>The input mirrored about its edge elements, which are not repeated (<c>F.pad(mode="reflect")</c>).</summary>
    Reflect,
}
