using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Memory.Vram;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Placement;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>Whether a backend can hold a LongCat DiT. A backend with a streaming cache moves the blocks through the device; one without keeps every
/// block on it, so a DiT larger than the device cannot run there.</summary>
/// <remarks>Only the weights are counted, at the size they are held, so a refusal is a run that could not have loaded. Activations, cast copies and
/// LoRA operands come on top and are not judged here.</remarks>
internal static class LongCatWeightFit
{
    /// <summary>Why <paramref name="backend"/> cannot hold the DiT in <paramref name="directory"/>, or null. Null without reading a file for a
    /// backend that streams blocks or reports no device memory.</summary>
    public static string? Refusal(IBackend backend, string directory) =>
        VideoWeightFit.Refusal(backend, "LongCat", () =>
        {
            (long held, long widened) = HeldBytes(directory, backend.SupportsResidentQuant);
            return (held, widened > 0 ? " (the quantized weights this backend cannot hold packed widen to F32)" : "");
        });

    /// <summary>The bytes the DiT's weights in <paramref name="directory"/> take on a backend, from the file headers: each tensor at its stored
    /// size, and a quantized weight (INT8, or a GGUF block quant) the backend does not hold packed at 4 bytes an element, an INT8 one without
    /// its row scales. <c>Widened</c> counts those weights.</summary>
    public static (long Bytes, long Widened) HeldBytes(string directory, Func<Tensor, bool> supportsResidentQuant)
    {
        if (LongCatCheckpoint.GgufIn(directory) is { } gguf)
            return GgufHeldBytes(gguf, supportsResidentQuant);
        using ShardedSafeTensorSet shards = ShardedSafeTensorSet.OpenIndex(directory, null, LongCatCheckpoint.IndexNameIn(directory));
        // A dense F32 checkpoint is held as BF16 when the loader converts it (numerics.longcatDitBf16), so the device fit counts that size.
        bool heldBf16 = !LongCatInt8Converter.IsInt8Layout(shards.Inventory.Keys) && EngineKnobs.LongCatDitBf16.Value;
        // The question a backend answers about a packed weight is its format, so one element stands for every INT8 weight of the release.
        using Tensor scale = new(new TensorShape(1), DType.F32);
        using Tensor probe = new(new TensorShape(1, 1), DType.I8);
        probe.QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = scale, ConvRotGroupSize = 0 };
        bool packed = supportsResidentQuant(probe);
        // numerics.longcatFp8Ffn holds the selected projections of a dense DiT at one byte an element, on every backend.
        HashSet<string> fp8 = EngineKnobs.LongCatFp8Ffn.Value ? [.. LongCatFp8FfnConverter.Keys()] : [];
        long bytes = 0, widened = 0;
        foreach ((string name, TensorLocation location) in shards.Inventory)
        {
            if (!packed && location.DType == DType.I8 && name.EndsWith(LongCatInt8Converter.QuantizedSuffix, StringComparison.Ordinal))
            {
                bytes += location.Shape.ElementCount * DType.F32.SizeInBytes;
                widened++;
            }
            else if (fp8.Contains(name))
            {
                bytes += location.Shape.ElementCount * DType.F8E4M3.SizeInBytes;
            }
            else if (packed || !name.EndsWith(LongCatInt8Converter.ScaleSuffix, StringComparison.Ordinal))
            {
                bytes += heldBf16 && location.DType == DType.F32
                    ? location.Shape.ElementCount * DType.BF16.SizeInBytes
                    : location.ByteLength;
            }
        }
        return (bytes, widened);
    }

    /// <summary>The same count for a GGUF repack. A projection whose parts have different quant types is held as BF16
    /// (<see cref="LongCatGgufConverter"/>).</summary>
    private static (long Bytes, long Widened) GgufHeldBytes(string path, Func<Tensor, bool> supportsResidentQuant)
    {
        CheckpointHeader header = CheckpointHeader.Read(path);
        Dictionary<string, List<SafeTensorDescriptor>> joined = new(StringComparer.Ordinal);
        foreach (SafeTensorDescriptor descriptor in header.Descriptors.Values)
        {
            string name = LongCatGgufConverter.EngineName(descriptor.Name).Name;
            if (!joined.TryGetValue(name, out List<SafeTensorDescriptor>? parts))
                joined[name] = parts = [];
            parts.Add(descriptor);
        }
        Dictionary<DType, bool> packed = [];
        long bytes = 0, widened = 0;
        foreach (List<SafeTensorDescriptor> parts in joined.Values)
        {
            DType dtype = parts[0].DType;
            long elements = parts.Sum(part => part.Shape.ElementCount);
            bool mixed = parts.Any(part => part.DType != dtype);
            if (!mixed && dtype.IsQuantized)
            {
                if (!packed.TryGetValue(dtype, out bool resident))
                {
                    using Tensor probe = new(new TensorShape(1, dtype.BlockElementCount), dtype);
                    packed[dtype] = resident = supportsResidentQuant(probe);
                }
                bytes += resident ? parts.Sum(part => part.ByteLength) : elements * DType.F32.SizeInBytes;
                widened += resident ? 0 : 1;
            }
            else if (mixed || (dtype == DType.F32 && EngineKnobs.LongCatDitBf16.Value))
            {
                bytes += elements * DType.BF16.SizeInBytes;
            }
            else
            {
                bytes += parts.Sum(part => part.ByteLength);
            }
        }
        return (bytes, widened);
    }
}
