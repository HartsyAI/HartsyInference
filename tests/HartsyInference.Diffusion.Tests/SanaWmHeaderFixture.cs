using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The real SANA-WM DiT header as names, dtypes and shapes (<c>Fixtures/sana_wm_dit_header.json</c>, 872 tensors, no
/// weights), so layout tests run without the 10.65 GB checkpoint.</summary>
internal static class SanaWmHeaderFixture
{
    public static Dictionary<string, SafeTensorDescriptor> LoadDescriptors()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "sana_wm_dit_header.json");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(path));
        Dictionary<string, SafeTensorDescriptor> tensors = new(StringComparer.Ordinal);
        foreach (JsonProperty entry in document.RootElement.EnumerateObject())
        {
            string dtypeName = entry.Value.GetProperty("dtype").GetString()!;
            if (!SafeTensorDTypes.TryParse(dtypeName, out DType dtype))
                throw new InvalidDataException($"Fixture tensor '{entry.Name}' has unknown dtype '{dtypeName}'.");
            long[] dims = [.. entry.Value.GetProperty("shape").EnumerateArray().Select(dim => dim.GetInt64())];
            TensorShape shape = new(dims);
            tensors[entry.Name] = new SafeTensorDescriptor
            {
                Name = entry.Name,
                DType = dtype,
                Shape = shape,
                DataOffset = 0,
                ByteLength = dtype.ComputeByteCount(shape.ElementCount),
            };
        }
        return tensors;
    }
}
