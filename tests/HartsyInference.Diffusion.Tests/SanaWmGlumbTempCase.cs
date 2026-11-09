using System.Globalization;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Tests;

/// <summary>How the fixture's weights and input reach the FFN. <c>Exact</c> borrows them as F32. <c>Bf16Rounded</c> keeps F32
/// storage but rounds every value to a BF16-representable number. <c>Bf16Storage</c> stores them as BF16.</summary>
internal enum SanaWmOperandMode
{
    Exact,
    Bf16Rounded,
    Bf16Storage,
}

/// <summary>A <c>sana_wm_s4_*</c> fixture loaded as an FFN plus its input, with the upstream F32 output as <see cref="Expected"/>.
/// Converted copies are owned by this object and released with it.</summary>
internal sealed class SanaWmGlumbTempCase : IDisposable
{
    private static readonly string[] WeightKeys =
    [
        "mlp.inverted_conv.conv.weight",
        "mlp.inverted_conv.conv.bias",
        "mlp.depth_conv.conv.weight",
        "mlp.depth_conv.conv.bias",
        "mlp.point_conv.conv.weight",
        "mlp.t_conv.weight",
    ];

    private readonly SafeTensorsLoader _loader = new();
    private readonly List<Tensor> _owned = [];

    private SanaWmGlumbTempCase(string fileName, SanaWmOperandMode mode)
    {
        _loader.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", fileName));
        IReadOnlyDictionary<string, string> metadata = _loader.Metadata
            ?? throw new InvalidDataException($"{fileName} has no safetensors metadata.");
        Frames = int.Parse(metadata["T"], CultureInfo.InvariantCulture);
        Height = int.Parse(metadata["H"], CultureInfo.InvariantCulture);
        Width = int.Parse(metadata["W"], CultureInfo.InvariantCulture);
        Expected = _loader.GetTensor("expected_output");
        Input = Operand(_loader.GetTensor("input"), mode);

        Tensor[] weights = new Tensor[WeightKeys.Length];
        for (int i = 0; i < WeightKeys.Length; i++)
            weights[i] = Operand(_loader.GetTensor(WeightKeys[i]), mode);
        Ffn = new SanaWmGlumbTempFfn(weights[0], weights[1], weights[2], weights[3], weights[4], weights[5]);
    }

    public SanaWmGlumbTempFfn Ffn { get; }

    public Tensor Input { get; }

    /// <summary>Upstream output, always F32. Borrowed from the loader.</summary>
    public Tensor Expected { get; }

    public int Frames { get; }

    public int Height { get; }

    public int Width { get; }

    public static SanaWmGlumbTempCase Load(string fileName, SanaWmOperandMode mode) => new(fileName, mode);

    /// <summary>Largest magnitude in the upstream output, the scale that BF16 tolerances are measured against.</summary>
    public float ExpectedScale()
    {
        float scale = 0f;
        foreach (float value in Expected.AsReadOnlySpan<float>())
            scale = MathF.Max(scale, MathF.Abs(value));
        return scale;
    }

    public void Dispose()
    {
        foreach (Tensor owned in _owned)
            owned.Dispose();
        _owned.Clear();
        _loader.Dispose();
    }

    private Tensor Operand(Tensor source, SanaWmOperandMode mode)
    {
        if (mode == SanaWmOperandMode.Exact)
            return source;

        DType dtype = mode == SanaWmOperandMode.Bf16Storage ? DType.BF16 : DType.F32;
        Tensor copy = new(source.Shape, dtype);
        _owned.Add(copy);
        ReadOnlySpan<float> values = source.AsReadOnlySpan<float>();
        if (dtype == DType.F32)
        {
            Span<float> rounded = copy.AsSpan<float>();
            for (int i = 0; i < values.Length; i++)
                rounded[i] = BFloat16Bits.ToFloat(BFloat16Bits.FromFloat(values[i]));
        }
        else
        {
            Span<ushort> bits = copy.AsSpan<ushort>();
            for (int i = 0; i < values.Length; i++)
                bits[i] = BFloat16Bits.FromFloat(values[i]);
        }
        return copy;
    }
}
