using System.Globalization;
using System.Text.RegularExpressions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Planning.Memory;

/// <summary>A checkpoint's weight bytes grouped by component and dtype, built once from its header so that sizing it for
/// any device afterwards is a handful of additions.</summary>
/// <remarks><para>Grouped by dtype because the resident size is a property of the device, not the file: a quantized
/// matrix the backend has a kernel for stays packed, one it has not is widened to F16 at load
/// (<c>QuantizedWeightPolicy</c>). Walking every descriptor per question would cost a millisecond per device per
/// request; the buckets make it constant.</para>
/// <para>Components are separated by the key prefixes all-in-one checkpoints use for their bundled VAE and text
/// encoders, so a single-file SD/SDXL checkpoint is not charged as one giant denoiser. Anything unprefixed is the
/// denoiser.</para></remarks>
internal sealed class CheckpointWeightInventory
{
    /// <summary>What a widened quantized weight becomes, matching <c>QuantizedWeightPolicy</c>'s default.</summary>
    private const int WidenedBytesPerElement = 2;

    /// <summary>ComfyUI's all-in-one prefix for the denoiser, checked first so nothing under it is misattributed.</summary>
    private const string DiffusionPrefix = "model.diffusion_model.";

    private static readonly string[] VaePrefixes = ["first_stage_model.", "vae."];

    private static readonly string[] TextEncoderPrefixes =
        ["cond_stage_model.", "conditioner.", "text_encoders.", "text_encoder.", "text_encoder_2.", "text_model."];

    /// <summary>A repeated transformer block: the group name and its index. Group + index, not index alone, because
    /// Flux-style models number their double and single blocks independently.</summary>
    private static readonly Regex BlockKey = new(
        @"(?:^|\.)(blocks|transformer_blocks|double_blocks|single_blocks|single_transformer_blocks|joint_blocks|layers)\.(\d+)\.",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    private readonly Dictionary<MemoryComponent, Dictionary<DType, DTypeBucket>> _buckets = [];

    private CheckpointWeightInventory()
    {
    }

    /// <summary>Distinct denoiser blocks found by key pattern; zero when the model names its blocks some other way.</summary>
    public int DenoiserBlockCount { get; private set; }

    /// <summary>On-disk bytes of the denoiser's repeated blocks.</summary>
    public long DenoiserBlockBytes { get; private set; }

    /// <summary>Whether any weight was attributed to <paramref name="component"/>.</summary>
    public bool Has(MemoryComponent component) => _buckets.ContainsKey(component);

    /// <summary>Builds an inventory from one file's descriptors.</summary>
    /// <param name="defaultComponent">The component unprefixed keys belong to: the denoiser for a checkpoint, or the
    /// component a side-model file is being read as.</param>
    public static CheckpointWeightInventory FromDescriptors(IEnumerable<SafeTensorDescriptor> descriptors,
        MemoryComponent defaultComponent = MemoryComponent.Denoiser) =>
        FromFiles([(descriptors, defaultComponent)]);

    /// <summary>Builds one inventory from several files of one model (shards, or a diffusers folder's components),
    /// each with the component its unprefixed keys belong to.</summary>
    public static CheckpointWeightInventory FromFiles(
        IEnumerable<(IEnumerable<SafeTensorDescriptor> Descriptors, MemoryComponent DefaultComponent)> files)
    {
        ArgumentNullException.ThrowIfNull(files);
        CheckpointWeightInventory inventory = new();
        HashSet<(string Group, int Index)> blocks = [];
        foreach ((IEnumerable<SafeTensorDescriptor> descriptors, MemoryComponent defaultComponent) in files)
        {
            foreach (SafeTensorDescriptor descriptor in descriptors)
            {
                MemoryComponent component = Classify(descriptor.Name, defaultComponent);
                inventory.Add(component, descriptor);
                if (component != MemoryComponent.Denoiser)
                {
                    continue;
                }
                Match block = BlockKey.Match(descriptor.Name);
                if (block.Success)
                {
                    blocks.Add((block.Groups[1].Value, int.Parse(block.Groups[2].ValueSpan, CultureInfo.InvariantCulture)));
                    inventory.DenoiserBlockBytes += descriptor.ByteLength;
                }
            }
        }
        inventory.DenoiserBlockCount = blocks.Count;
        return inventory;
    }

    /// <summary>Bytes <paramref name="component"/> occupies on the checkpoint's own storage.</summary>
    public long StoredBytes(MemoryComponent component) =>
        _buckets.TryGetValue(component, out Dictionary<DType, DTypeBucket>? byDtype)
            ? byDtype.Values.Sum(bucket => bucket.StoredBytes)
            : 0;

    /// <summary>Bytes <paramref name="component"/> occupies once loaded onto a device: quantized weights the device
    /// cannot hold packed — and every non-matrix quantized tensor, which no GEMM reads — are widened to F16.</summary>
    /// <param name="supportsResidentQuant">The target backend's <c>SupportsResidentQuant</c>; always-true sizes the
    /// weights as stored.</param>
    public long ResidentBytes(MemoryComponent component, Func<DType, bool> supportsResidentQuant)
    {
        ArgumentNullException.ThrowIfNull(supportsResidentQuant);
        if (!_buckets.TryGetValue(component, out Dictionary<DType, DTypeBucket>? byDtype))
        {
            return 0;
        }
        long total = 0;
        foreach ((DType dtype, DTypeBucket bucket) in byDtype)
        {
            if (!dtype.IsQuantized)
            {
                total += bucket.StoredBytes;
                continue;
            }
            total += supportsResidentQuant(dtype)
                ? bucket.MatrixBytes + bucket.NonMatrixElements * WidenedBytesPerElement
                : (bucket.MatrixElements + bucket.NonMatrixElements) * WidenedBytesPerElement;
        }
        return total;
    }

    /// <summary>Denoiser bytes that must stay resident while its blocks stream: the shared (non-block) weights plus
    /// <paramref name="windowBlocks"/> blocks, scaled by the same stored-to-resident ratio as the whole denoiser.
    /// Equal to the full resident size when no repeated blocks were recognised.</summary>
    public long DenoiserStreamFloorBytes(Func<DType, bool> supportsResidentQuant, int windowBlocks)
    {
        long resident = ResidentBytes(MemoryComponent.Denoiser, supportsResidentQuant);
        long stored = StoredBytes(MemoryComponent.Denoiser);
        if (DenoiserBlockCount == 0 || stored == 0)
        {
            return resident;
        }
        long sharedStored = stored - DenoiserBlockBytes;
        long windowStored = DenoiserBlockBytes / DenoiserBlockCount * Math.Min(windowBlocks, DenoiserBlockCount);
        // Int128 keeps the scale exact: a double loses the low bits of a 20 GB byte count.
        return (long)((Int128)(sharedStored + windowStored) * resident / stored);
    }

    private static MemoryComponent Classify(string key, MemoryComponent defaultComponent)
    {
        if (key.StartsWith(DiffusionPrefix, StringComparison.Ordinal))
        {
            return MemoryComponent.Denoiser;
        }
        foreach (string prefix in VaePrefixes)
        {
            if (key.StartsWith(prefix, StringComparison.Ordinal))
            {
                return MemoryComponent.Vae;
            }
        }
        foreach (string prefix in TextEncoderPrefixes)
        {
            if (key.StartsWith(prefix, StringComparison.Ordinal))
            {
                return MemoryComponent.TextEncoder;
            }
        }
        return defaultComponent;
    }

    private void Add(MemoryComponent component, SafeTensorDescriptor descriptor)
    {
        if (!_buckets.TryGetValue(component, out Dictionary<DType, DTypeBucket>? byDtype))
        {
            byDtype = [];
            _buckets[component] = byDtype;
        }
        if (!byDtype.TryGetValue(descriptor.DType, out DTypeBucket? bucket))
        {
            bucket = new DTypeBucket();
            byDtype[descriptor.DType] = bucket;
        }
        bucket.StoredBytes += descriptor.ByteLength;
        if (descriptor.Shape.Rank == 2)
        {
            bucket.MatrixBytes += descriptor.ByteLength;
            bucket.MatrixElements += descriptor.Shape.ElementCount;
        }
        else
        {
            bucket.NonMatrixElements += descriptor.Shape.ElementCount;
        }
    }

    /// <summary>Running totals for one dtype within one component.</summary>
    private sealed class DTypeBucket
    {
        public long StoredBytes;
        public long MatrixBytes;
        public long MatrixElements;
        public long NonMatrixElements;
    }
}
