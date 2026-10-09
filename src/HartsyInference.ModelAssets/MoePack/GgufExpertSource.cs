using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.ModelAssets.MoePack;

/// <summary>
/// Reads per-expert F32 weights from a GGUF checkpoint's stacked expert tensors. Two layouts are supported: separate
/// <c>blk.N.ffn_gate_exps</c> and <c>ffn_up_exps</c>, or fused <c>ffn_gate_up_exps</c> (each expert's gate rows then its up rows).
/// The loader reports GGUF's ggml order, fastest dimension first: gate and up <c>[H, I, E]</c>, down <c>[I, H, E]</c>, fused
/// <c>[H, 2I, E]</c>. Memory is expert-major either way, so an expert's rows are contiguous.
/// </summary>
/// <remarks>By default the fingerprint is provisional: it identifies the GGUF's expert geometry, not the runtime topology. A caller
/// that has the model's topology passes its SparseModelTopology.Fingerprint to <see cref="Open(string, string?)"/> to bind the
/// pack to that topology. A pack is refused by any reader expecting a different fingerprint.</remarks>
public sealed partial class GgufExpertSource : IDisposable
{
    private readonly GgufLoader _loader;
    private readonly Dictionary<int, LayerTensors> _layers;

    private sealed record LayerTensors(Tensor Gate, Tensor? Up, Tensor Down);

    private GgufExpertSource(GgufLoader loader, Dictionary<int, LayerTensors> layers, int experts, int hidden, int intermediate,
        string architecture, string? topologyFingerprint)
    {
        _loader = loader;
        _layers = layers;
        ExpertCount = experts;
        Hidden = hidden;
        Intermediate = intermediate;
        Layers = layers.Keys.Order().ToArray();
        TopologyFingerprint = topologyFingerprint ?? "gguf-moe:" + Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(
            $"v1;arch={architecture};layers={string.Join(',', Layers)};experts={experts};hidden={hidden};intermediate={intermediate}")))
            .ToLowerInvariant();
    }

    /// <summary>Routed experts per layer.</summary>
    public int ExpertCount { get; }

    /// <summary>Model width H.</summary>
    public int Hidden { get; }

    /// <summary>Expert inner width I.</summary>
    public int Intermediate { get; }

    /// <summary>GGUF block indices of the MoE layers, ascending.</summary>
    public IReadOnlyList<int> Layers { get; }

    /// <summary>Identity recorded in the pack manifest; see the type remarks.</summary>
    public string TopologyFingerprint { get; }

    [GeneratedRegex(@"^blk\.(\d+)\.ffn_down_exps\.weight$")]
    private static partial Regex DownExpertsName();

    /// <summary>Opens the checkpoint with the provisional GGUF-geometry fingerprint; see <see cref="Open(string, string?)"/>.</summary>
    /// <exception cref="InvalidDataException">The file has no MoE layers, or a layer's expert tensors are missing or mis-shaped.</exception>
    public static GgufExpertSource Open(string path) => Open(path, topologyFingerprint: null);

    /// <summary>
    /// Opens the checkpoint and checks that every MoE layer has consistent expert shapes. When
    /// <paramref name="topologyFingerprint"/> is given, the pack identity is that runtime topology fingerprint
    /// (SparseModelTopology.Fingerprint), supplied by the caller that built the model's topology.
    /// </summary>
    /// <param name="path">GGUF checkpoint path.</param>
    /// <param name="topologyFingerprint">Runtime topology fingerprint, or null for the provisional GGUF-geometry identity.</param>
    /// <exception cref="InvalidDataException">The file has no MoE layers, or a layer's expert tensors are missing or mis-shaped.</exception>
    /// <exception cref="ArgumentException">The supplied fingerprint is empty.</exception>
    public static GgufExpertSource Open(string path, string? topologyFingerprint)
    {
        if (topologyFingerprint is not null && topologyFingerprint.Length == 0)
            throw new ArgumentException("The topology fingerprint must not be empty.", nameof(topologyFingerprint));
        GgufLoader loader = new();
        try
        {
            loader.Load(path);
            Dictionary<int, LayerTensors> layers = [];
            int experts = 0, hidden = 0, intermediate = 0;
            foreach (string name in loader.Descriptors.Keys.Where(static n => DownExpertsName().IsMatch(n)).Order())
            {
                int layer = int.Parse(DownExpertsName().Match(name).Groups[1].ValueSpan, CultureInfo.InvariantCulture);
                Tensor down = loader.GetTensor(name);
                if (down.Shape.Rank != 3) throw new InvalidDataException($"{name} must be rank 3 [intermediate, hidden, experts].");
                int i = (int)down.Shape[0], h = (int)down.Shape[1], e = (int)down.Shape[2];
                if (layers.Count == 0) (experts, hidden, intermediate) = (e, h, i);
                else if (e != experts || h != hidden || i != intermediate)
                    throw new InvalidDataException(
                        $"Layer {layer} has expert shape [{e}, {h}, {i}]; earlier layers have [{experts}, {hidden}, {intermediate}].");

                string fused = $"blk.{layer}.ffn_gate_up_exps.weight";
                LayerTensors tensors;
                if (loader.Descriptors.ContainsKey(fused))
                {
                    Tensor gateUp = loader.GetTensor(fused);
                    if (gateUp.Shape.Rank != 3 || gateUp.Shape[0] != h || gateUp.Shape[1] != 2 * i || gateUp.Shape[2] != e)
                        throw new InvalidDataException($"{fused} must be [{h}, {2 * i}, {e}].");
                    tensors = new LayerTensors(gateUp, null, down);
                }
                else
                {
                    Tensor gate = loader.GetTensor($"blk.{layer}.ffn_gate_exps.weight");
                    Tensor up = loader.GetTensor($"blk.{layer}.ffn_up_exps.weight");
                    if (gate.Shape.Rank != 3 || gate.Shape[0] != h || gate.Shape[1] != i || gate.Shape[2] != e)
                        throw new InvalidDataException($"ffn_gate_exps of layer {layer} must be [{h}, {i}, {e}].");
                    if (up.Shape.Rank != 3 || up.Shape[0] != h || up.Shape[1] != i || up.Shape[2] != e)
                        throw new InvalidDataException($"ffn_up_exps of layer {layer} must be [{h}, {i}, {e}].");
                    tensors = new LayerTensors(gate, up, down);
                }
                layers[layer] = tensors;
            }
            if (layers.Count == 0) throw new InvalidDataException($"'{path}' has no stacked expert tensors (blk.N.ffn_down_exps.weight).");

            string architecture = loader.Metadata.GetString("general.architecture") ?? "unknown";
            return new GgufExpertSource(loader, layers, experts, hidden, intermediate, architecture, topologyFingerprint);
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    /// <summary>Every expert in the checkpoint, layer by layer, in ascending expert order. Bank is always 0.</summary>
    public IEnumerable<ExpertKey> Keys()
    {
        foreach (int layer in Layers)
            for (int expert = 0; expert < ExpertCount; expert++)
                yield return new ExpertKey(layer, expert, 0);
    }

    /// <summary>Reads one expert as F32, dequantizing if the checkpoint stores it quantized. Gate and up are <c>[I, H]</c>; down is
    /// <c>[H, I]</c>, all row-major.</summary>
    /// <exception cref="KeyNotFoundException">The layer is not a MoE layer of this checkpoint.</exception>
    public (float[] Gate, float[] Up, float[] Down) Read(int layer, int expert)
    {
        if (!_layers.TryGetValue(layer, out LayerTensors? tensors)) throw new KeyNotFoundException($"Layer {layer} has no experts.");
        ArgumentOutOfRangeException.ThrowIfNegative(expert);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(expert, ExpertCount);
        if (tensors.Up is null)
        {
            long gateUpStart = (long)expert * 2 * Intermediate;
            return (Matrix(tensors.Gate, gateUpStart, Intermediate),
                Matrix(tensors.Gate, gateUpStart + Intermediate, Intermediate),
                Matrix(tensors.Down, (long)expert * Hidden, Hidden));
        }
        return (Matrix(tensors.Gate, (long)expert * Intermediate, Intermediate),
            Matrix(tensors.Up, (long)expert * Intermediate, Intermediate),
            Matrix(tensors.Down, (long)expert * Hidden, Hidden));
    }

    /// <summary>Copies rows <c>[rowStart, rowStart + rows)</c> of a stacked tensor, viewed as <c>[E·rowsPerExpert, cols]</c>, as F32.</summary>
    private static unsafe float[] Matrix(Tensor stacked, long rowStart, int rows)
    {
        int cols = (int)stacked.Shape[0];
        long rowBytes = stacked.DType.ComputeByteCount(cols);
        byte* start = (byte*)stacked.DataPointer + rowStart * rowBytes;
        using Tensor view = new(start, new TensorShape(rows, cols), stacked.DType);
        Tensor values = stacked.DType == DType.F32 ? view : GgufDequantizer.Dequantize(view, DType.F32);
        try
        {
            float[] result = new float[(long)rows * cols];
            new ReadOnlySpan<float>((float*)values.DataPointer, result.Length).CopyTo(result);
            return result;
        }
        finally
        {
            if (!ReferenceEquals(values, view)) values.Dispose();
        }
    }

    /// <inheritdoc/>
    public void Dispose() => _loader.Dispose();
}
