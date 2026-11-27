using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A two-block SANA-WM layout (block 0 GDN, block 1 softmax) at tiny widths, written with the same key names and
/// dtypes as the real checkpoint. Values are deterministic, and every F32 tensor starts with 1 + 3·2⁻⁸, which sits exactly
/// between two BF16 values, so a test can tell round-to-nearest-even from truncation.</summary>
internal static class SanaWmSyntheticLayout
{
    /// <summary>The tie value 1 + 3·2⁻⁸: BF16 0x3F81 truncates, and round-to-nearest-even gives 0x3F82.</summary>
    public const float RoundingTie = 1.01171875f;

    public static SanaWmConfig Config { get; } = new()
    {
        Depth = 2,
        HiddenSize = 224,
        NumHeads = 2,
        HeadDim = 112,
        InChannels = 8,
        OutChannels = 8,
        FfnHidden = 448,
        TextWidth = 16,
        NullCaptionLength = 6,
        TimestepWidth = 16,
        ConvKernelSize = 4,
        PosEmbedTokens = 5,
        ModulationRows = 6,
        SoftmaxBlocks = [1],
        HasCameraBranch = true,
        HasPluckerBranch = true,
        PluckerChannels = 12,
        HasRaymapEmbedder = true,
        RaymapChannels = 3,
    };

    public static float PatternValue(int index) => (index % 251) * 0.004f - 0.5f;

    /// <summary>Writes the layout to <paramref name="path"/>. <paramref name="edit"/> may drop, add or reshape specs first.</summary>
    public static void WriteFile(string path, Action<List<SanaWmTensorSpec>>? edit = null)
    {
        List<SanaWmTensorSpec> specs = [.. SanaWmCheckpointConverter.ExpectedTensors(Config)];
        edit?.Invoke(specs);
        Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal);
        try
        {
            foreach (SanaWmTensorSpec spec in specs)
                tensors.Add(spec.Name, CreateTensor(spec));
            SafeTensorsWriter.Save(path, tensors);
        }
        finally
        {
            foreach (Tensor tensor in tensors.Values)
                tensor.Dispose();
        }
    }

    /// <summary>Stored dtype of a source tensor. The real checkpoint is F32 throughout except <c>y_embedding</c>, so the
    /// converter's F32 to BF16 path is exercised for every weight.</summary>
    private static Tensor CreateTensor(SanaWmTensorSpec spec)
    {
        bool bf16Source = spec.Name == "y_embedder.y_embedding";
        Tensor tensor = new(spec.Shape, bf16Source ? DType.BF16 : DType.F32);
        if (!bf16Source)
        {
            Span<float> values = tensor.AsSpan<float>();
            for (int i = 0; i < values.Length; i++)
                values[i] = PatternValue(i);
            values[0] = RoundingTie;
        }
        else
        {
            Span<ushort> bits = tensor.AsSpan<ushort>();
            for (int i = 0; i < bits.Length; i++)
                bits[i] = SafeTensorsMerger.ToBf16(PatternValue(i));
        }
        return tensor;
    }
}
