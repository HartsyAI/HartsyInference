using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.SafeTensors;

/// <summary>Maps safetensors dtype names to engine dtypes, and refuses the ones that parse but must never be read as another format.</summary>
public static class SafeTensorDTypes
{
    /// <summary>Resolves a header dtype string, or returns false for a name this engine does not know.</summary>
    public static bool TryParse(string name, out DType dtype)
    {
        switch (name)
        {
            case "F64": dtype = DType.F64; return true;
            case "F32": dtype = DType.F32; return true;
            case "F16": dtype = DType.F16; return true;
            case "BF16": dtype = DType.BF16; return true;
            case "F8_E4M3": dtype = DType.F8E4M3; return true;
            case "F8_E5M2": dtype = DType.F8E5M2; return true;
            case "F8_E8M0": dtype = DType.F8E8M0; return true;
            case "F8_E4M3FNUZ": dtype = DType.F8E4M3Fnuz; return true;
            case "F8_E5M2FNUZ": dtype = DType.F8E5M2Fnuz; return true;
            // No checkpoint we consume declares this: ComfyUI ships NVFP4 as U8 with companion scales, and the
            // relabel to F4_E2M1 happens in Nvfp4Codec. Mapped anyway so a file that does declare it loads rather
            // than dying here with "Unsupported safetensors dtype" — the packing is identical either way.
            case "F4_E2M1": dtype = DType.F4E2M1; return true;
            case "I64": dtype = DType.I64; return true;
            case "I32": dtype = DType.I32; return true;
            case "I8": dtype = DType.I8; return true;
            case "U8": dtype = DType.U8; return true;
            case "U16": dtype = DType.U16; return true;
            case "U32": dtype = DType.U32; return true;
            case "U64": dtype = DType.U64; return true;
            case "BOOL": dtype = DType.Bool; return true;
            default: dtype = default; return false;
        }
    }

    /// <summary>Resolves a header dtype string or throws naming it.</summary>
    public static DType Parse(string name)
    {
        if (TryParse(name, out DType dtype))
            return dtype;
        throw new HartsyInferenceException($"Unsupported safetensors dtype: {name}");
    }

    /// <summary>Throws for a dtype whose bytes exist in a header but that the engine cannot honestly hand out as a tensor.</summary>
    public static void ThrowIfNotMaterialisable(DType dtype, string tensorName, string filePath)
    {
        if (!dtype.IsFnuz)
            return;
        throw new UnsupportedModelException(
            $"Tensor '{tensorName}' in '{filePath}' is {dtype.Name}. FNUZ float8 uses a different exponent bias from "
            + "the E4M3/E5M2 the engine decodes, so loading it as those would halve every value. Use an OCP "
            + "F8_E4M3 or BF16 build of this model.");
    }
}
