using HartsyInference.Core.Backends;
using HartsyInference.Core.Memory.Vram;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Placement;
using HartsyInference.ModelAssets.CheckpointConverters;
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
            return (held, widened > 0 ? " (its INT8 weights widen to F32 on this backend)" : "");
        });

    /// <summary>The bytes the DiT's weights in <paramref name="directory"/> take on a backend, from the shard headers: each tensor at its stored
    /// size, and an INT8 weight the backend does not hold packed at 4 bytes an element without its row scales. <c>Widened</c> counts those
    /// weights.</summary>
    public static (long Bytes, long Widened) HeldBytes(string directory, Func<Tensor, bool> supportsResidentQuant)
    {
        using ShardedSafeTensorSet shards = ShardedSafeTensorSet.OpenIndex(directory, null, LongCatCheckpoint.IndexNameIn(directory));
        // The question a backend answers about a packed weight is its format, so one element stands for every INT8 weight of the release.
        using Tensor scale = new(new TensorShape(1), DType.F32);
        using Tensor probe = new(new TensorShape(1, 1), DType.I8);
        probe.QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = scale, ConvRotGroupSize = 0 };
        bool packed = supportsResidentQuant(probe);
        long bytes = 0, widened = 0;
        foreach ((string name, TensorLocation location) in shards.Inventory)
        {
            if (!packed && location.DType == DType.I8 && name.EndsWith(LongCatInt8Converter.QuantizedSuffix, StringComparison.Ordinal))
            {
                bytes += location.Shape.ElementCount * DType.F32.SizeInBytes;
                widened++;
            }
            else if (packed || !name.EndsWith(LongCatInt8Converter.ScaleSuffix, StringComparison.Ordinal))
            {
                bytes += location.ByteLength;
            }
        }
        return (bytes, widened);
    }
}
