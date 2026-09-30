using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>A row-major cache of latent vectors in one <see cref="LatentEncoding"/>; a zero-row source is empty.</summary>
/// <param name="Encoding">Row format.</param>
/// <param name="Codes">F32 <c>[Rows,Dim]</c> for F32, otherwise U8 <c>[Rows, CodeBytesPerRow]</c>; null when empty.</param>
/// <param name="Scales">U8 <c>[Rows, Dim/group]</c> scale bytes; null for F32 or when empty.</param>
/// <param name="Rows">Logical row count.</param>
/// <param name="Dim">Elements per row; a multiple of the encoding's group size.</param>
public readonly record struct LatentSource(LatentEncoding Encoding, Tensor? Codes, Tensor? Scales, int Rows, int Dim)
{
    /// <summary>A source with no rows, usable where an absent cache is allowed.</summary>
    public static LatentSource Empty => new(LatentEncoding.F32, null, null, 0, 0);

    /// <summary>Checks that the tensors match the encoding's geometry; throws with <paramref name="name"/> in the message.</summary>
    public void Validate(string name)
    {
        if (Rows < 0 || Dim < 0) throw new ArgumentException($"{name}: Rows and Dim must not be negative.");
        if (Rows == 0) return;
        int group = LatentEncodings.GroupSize(Encoding);
        if (Dim < 2 || (group != 0 && Dim % group != 0) || Dim % 2 != 0)
            throw new ArgumentException($"{name}: Dim {Dim} must be even and a multiple of the {group}-element group.");
        if (Codes is null) throw new ArgumentException($"{name}: Codes is required when Rows > 0.");
        DType codeType = Encoding == LatentEncoding.F32 ? DType.F32 : DType.U8;
        long codeElements = Encoding == LatentEncoding.F32 ? (long)Rows * Dim : Rows * LatentEncodings.CodeBytesPerRow(Encoding, Dim);
        if (Codes.DType != codeType || Codes.ElementCount != codeElements)
            throw new ArgumentException($"{name}: Codes must be {codeType} with {codeElements} elements; got {Codes.DType} x {Codes.ElementCount}.");
        long scaleElements = (long)Rows * LatentEncodings.ScaleBytesPerRow(Encoding, Dim);
        if (Encoding == LatentEncoding.F32)
        {
            if (Scales is not null) throw new ArgumentException($"{name}: F32 sources carry no scales.");
        }
        else if (Scales is null || Scales.DType != DType.U8 || Scales.ElementCount != scaleElements)
        {
            throw new ArgumentException($"{name}: Scales must be U8 with {scaleElements} elements.");
        }
    }
}
