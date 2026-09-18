namespace HartsyInference.ModelAssets.Lora;

/// <summary>Which piece of a LoRA layer a file key carries. The spellings that map onto each role live in <see cref="LoraRoleSuffix"/>; every mapper classifies through it so a format is recognized by its suffix vocabulary rather than by per-mapper string constants.</summary>
public enum LoraRole
{
    /// <summary>Down/A projection, <c>[rank, in]</c> (or <c>[rank, in, kh, kw]</c> for a conv adapter).</summary>
    Down,

    /// <summary>Up/B projection, <c>[out, rank]</c>.</summary>
    Up,

    /// <summary>Scalar alpha; the layer scale is <c>alpha / rank</c>.</summary>
    Alpha,

    /// <summary>LyCORIS LoHa first Hadamard factor's left matrix, <c>[out, rank]</c>.</summary>
    HadaW1A,

    /// <summary>LyCORIS LoHa first Hadamard factor's right matrix, <c>[rank, in]</c>.</summary>
    HadaW1B,

    /// <summary>LyCORIS LoHa second Hadamard factor's left matrix, <c>[out, rank]</c>.</summary>
    HadaW2A,

    /// <summary>LyCORIS LoHa second Hadamard factor's right matrix, <c>[rank, in]</c>.</summary>
    HadaW2B,

    /// <summary>LyCORIS LoHa first factor's CP/Tucker core, <c>[rank, rank, kh, kw]</c>.</summary>
    HadaT1,

    /// <summary>LyCORIS LoHa second factor's CP/Tucker core, <c>[rank, rank, kh, kw]</c>.</summary>
    HadaT2,

    /// <summary>LyCORIS LoKr left Kronecker factor stored whole.</summary>
    LokrW1,

    /// <summary>LyCORIS LoKr left Kronecker factor's low-rank left matrix.</summary>
    LokrW1A,

    /// <summary>LyCORIS LoKr left Kronecker factor's low-rank right matrix.</summary>
    LokrW1B,

    /// <summary>LyCORIS LoKr right Kronecker factor stored whole.</summary>
    LokrW2,

    /// <summary>LyCORIS LoKr right Kronecker factor's low-rank left matrix.</summary>
    LokrW2A,

    /// <summary>LyCORIS LoKr right Kronecker factor's low-rank right matrix.</summary>
    LokrW2B,

    /// <summary>LyCORIS LoKr right factor's CP/Tucker core, <c>[rank, rank, kh, kw]</c>.</summary>
    LokrT2,

    /// <summary>DoRA magnitude vector; orthogonal to the decomposition, applied by <see cref="LoraDoraDecompose"/>.</summary>
    DoraScale,

    /// <summary>Comfy-style full-weight delta targeting the module's <c>.weight</c>. Not low-rank.</summary>
    Diff,

    /// <summary>Comfy-style full-weight delta targeting the module's <c>.bias</c>. Not low-rank.</summary>
    BiasDiff,
}
