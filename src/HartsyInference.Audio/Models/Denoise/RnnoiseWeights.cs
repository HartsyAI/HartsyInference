using HartsyInference.Audio.Models.Wake;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Audio.Models.Denoise;

/// <summary>The 20 trained tensors of an RNNoise checkpoint, loaded once and shared by every stream.
///
/// <para>Split out from <see cref="RnnoiseModel"/> because the weights are stateless and 11.5 MB, while the
/// model around them is pure per-stream state (conv history, three GRU hidden vectors, scratch). A listener
/// serving dozens of satellites would otherwise hold dozens of identical copies. Same division the wake models
/// already use, where <c>WakeModelSet</c> shares one mel front-end and embedding across all sessions.</para>
///
/// <para>Borrowed by models, never owned: dispose this only after every <see cref="RnnoiseModel"/> built on it
/// is gone.</para>
///
/// <para>The precision is fixed at load (<see cref="Precision"/>). <see cref="RnnoisePrecision.Int8"/> adds the int8
/// tables of upstream's default C build (<see cref="LoadInt8Tables"/>) on top of the F32 weights, which conv1 and the two
/// heads still use. <see cref="LoadFile"/> is where a caller picks one: the wake stack loads F32, the voice front end
/// int8.</para></summary>
public sealed class RnnoiseWeights : IDisposable
{
    /// <summary>Tensors in a complete checkpoint: two convs, three GRUs of four, and two dense heads.</summary>
    public const int TensorCount = 20;

    private const string Source = "rnnoise.safetensors converted from xiph's rnnoise10Ga_12.pth (RnnoiseCheckpoint)";
    private const int Gates = 3 * RnnoiseModel.GruSize;
    private const int CatSize = 4 * RnnoiseModel.GruSize;

    private int _disposed;

    public Tensor Conv1Weight { get; private set; } = null!;
    public Tensor Conv1Bias { get; private set; } = null!;
    public Tensor Conv2Weight { get; private set; } = null!;
    public Tensor Conv2Bias { get; private set; } = null!;
    public Tensor DenseOutWeight { get; private set; } = null!;
    public Tensor DenseOutBias { get; private set; } = null!;
    public Tensor VadWeight { get; private set; } = null!;
    public Tensor VadBias { get; private set; } = null!;

    /// <summary>Per-GRU input-side weights, indexed 0-2 for gru1-gru3.</summary>
    public Tensor[] GruWeightIh { get; } = new Tensor[3];
    public Tensor[] GruWeightHh { get; } = new Tensor[3];
    public Tensor[] GruBiasIh { get; } = new Tensor[3];
    public Tensor[] GruBiasHh { get; } = new Tensor[3];

    /// <summary>True once <see cref="Load"/> has bound every tensor.</summary>
    public bool IsLoaded { get; private set; }

    /// <summary>What the network runs on: F32 after <see cref="Load"/>, int8 once <see cref="LoadInt8Tables"/> has
    /// added the tables.</summary>
    public RnnoisePrecision Precision { get; private set; } = RnnoisePrecision.Float;

    /// <summary>conv2's int8 weight, <c>[48, 96, 8, 4]</c> in <see cref="Int8Tiles"/> order, its inputs channel-major like
    /// <see cref="Conv2Weight"/>; null at F32. <see cref="Conv2Scale"/> and <see cref="Conv2Subias"/> go with it.</summary>
    public Tensor? Conv2Int8 { get; private set; }
    public Tensor? Conv2Scale { get; private set; }
    public Tensor? Conv2Subias { get; private set; }

    /// <summary>Per-GRU int8 input weights, <c>[144, 96, 8, 4]</c>, rows in PyTorch's (r, z, n) gate order like
    /// <see cref="GruWeightIh"/>; null at F32. The scales, SU biases and recurrent diagonals follow the same order.</summary>
    public Tensor?[] GruInputInt8 { get; } = new Tensor?[3];
    public Tensor?[] GruInputScale { get; } = new Tensor?[3];
    public Tensor?[] GruInputSubias { get; } = new Tensor?[3];
    public Tensor?[] GruRecurrentInt8 { get; } = new Tensor?[3];
    public Tensor?[] GruRecurrentScale { get; } = new Tensor?[3];
    public Tensor?[] GruRecurrentSubias { get; } = new Tensor?[3];
    public Tensor?[] GruRecurrentDiag { get; } = new Tensor?[3];

    /// <summary>Loads RNNoise from <paramref name="weightsPath"/> at <paramref name="precision"/>. This is where a
    /// caller chooses: the wake stack passes <see cref="RnnoisePrecision.Float"/>, the voice front end
    /// <see cref="RnnoisePrecision.Int8"/>, which also reads <paramref name="int8TablesPath"/>, by default
    /// <see cref="RnnoiseInt8Tables.FileName"/> beside the weights.</summary>
    public static RnnoiseWeights LoadFile(string weightsPath, RnnoisePrecision precision = RnnoisePrecision.Float,
        string? int8TablesPath = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(weightsPath);
        RnnoiseWeights weights = new();
        try
        {
            ReadInto(weightsPath, weights.Load);
            if (precision == RnnoisePrecision.Int8)
            {
                string tables = int8TablesPath
                    ?? Path.Combine(Path.GetDirectoryName(Path.GetFullPath(weightsPath))!, RnnoiseInt8Tables.FileName);
                ReadInto(tables, weights.LoadInt8Tables);
            }
            return weights;
        }
        catch
        {
            weights.Dispose();
            throw;
        }
    }

    private static void ReadInto(string path, Action<IReadOnlyDictionary<string, Tensor>> load)
    {
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> tensors = loader.GetAllTensors();
        try
        {
            load(tensors);
        }
        finally
        {
            foreach (Tensor tensor in tensors.Values) tensor.Dispose();
        }
    }

    /// <summary>Takes owned F32 copies via <see cref="WakeWeights"/>, so the loader that supplied them can be
    /// disposed immediately afterwards. Throws naming the tensor when one is missing or shaped for another
    /// architecture, which would otherwise surface as a shape error on the first speech frame. Call once: a second
    /// load would orphan the first set, so it throws instead.</summary>
    public void Load(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (IsLoaded || Conv1Weight is not null)
            throw new InvalidOperationException(
                "RnnoiseWeights are already loaded; build a new instance to load another set.");
        Conv1Weight = Require(weights, "conv1.weight",
            new TensorShape(RnnoiseModel.CondSize, RnnoiseModel.InputDim, RnnoiseModel.KernelSize));
        Conv1Bias = Require(weights, "conv1.bias", new TensorShape(RnnoiseModel.CondSize));
        Conv2Weight = Require(weights, "conv2.weight",
            new TensorShape(RnnoiseModel.GruSize, RnnoiseModel.CondSize, RnnoiseModel.KernelSize));
        Conv2Bias = Require(weights, "conv2.bias", new TensorShape(RnnoiseModel.GruSize));
        for (int i = 0; i < 3; i++)
        {
            string gru = $"gru{i + 1}";
            GruWeightIh[i] = Require(weights, $"{gru}.weight_ih_l0", new TensorShape(Gates, RnnoiseModel.GruSize));
            GruWeightHh[i] = Require(weights, $"{gru}.weight_hh_l0", new TensorShape(Gates, RnnoiseModel.GruSize));
            GruBiasIh[i] = Require(weights, $"{gru}.bias_ih_l0", new TensorShape(Gates));
            GruBiasHh[i] = Require(weights, $"{gru}.bias_hh_l0", new TensorShape(Gates));
        }
        DenseOutWeight = Require(weights, "dense_out.weight", new TensorShape(RnnoiseModel.OutputDim, CatSize));
        DenseOutBias = Require(weights, "dense_out.bias", new TensorShape(RnnoiseModel.OutputDim));
        VadWeight = Require(weights, "vad_dense.weight", new TensorShape(1, CatSize));
        VadBias = Require(weights, "vad_dense.bias", new TensorShape(1));
        IsLoaded = true;
    }

    /// <summary>Adds upstream's int8 tables (<see cref="RnnoiseInt8Tables"/>) and switches to
    /// <see cref="RnnoisePrecision.Int8"/>. The F32 weights must be loaded first.</summary>
    /// <remarks><para>The tables come in upstream's arrangement and are copied into the model's. Two things differ:</para>
    /// <list type="bullet">
    /// <item>Upstream's GRU rows run (z, r, n); PyTorch's, which <see cref="RnnoiseModel"/> and
    /// <c>GruOps.GateAndUpdate</c> use, run (r, z, n). The gate blocks of each table, scale, SU bias and diagonal swap.
    /// Each gate is 48 whole tiles, so this moves tiles and changes no sum.</item>
    /// <item>Upstream feeds conv2 its three-frame window time-major, <c>t·128 + c</c>, where the model's window is
    /// channel-major, <c>c·3 + t</c>, as <see cref="Conv2Weight"/> flattens. conv2's columns are permuted to match.
    /// Each output's int8 sum is exact, so the order of its terms cannot change it.</item>
    /// </list></remarks>
    public void LoadInt8Tables(IReadOnlyDictionary<string, Tensor> tables)
    {
        ArgumentNullException.ThrowIfNull(tables);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!IsLoaded)
            throw new InvalidOperationException(
                "Load the F32 weights first: conv1 and the two heads run in F32 at either precision.");
        if (Precision == RnnoisePrecision.Int8)
            throw new InvalidOperationException("The int8 tables are already loaded.");
        RnnoiseInt8Tables.Require(tables);
        try
        {
            Conv2Int8 = Conv2Tiles(tables["conv2_weights_int8"]);
            Conv2Scale = Copy(tables["conv2_scale"]);
            Conv2Subias = Copy(tables["conv2_subias"]);
            for (int i = 0; i < 3; i++)
            {
                string input = $"gru{i + 1}_input", recurrent = $"gru{i + 1}_recurrent";
                GruInputInt8[i] = GateOrderTiles(tables[$"{input}_weights_int8"]);
                GruInputScale[i] = GateOrder(tables[$"{input}_scale"]);
                GruInputSubias[i] = GateOrder(tables[$"{input}_subias"]);
                GruRecurrentInt8[i] = GateOrderTiles(tables[$"{recurrent}_weights_int8"]);
                GruRecurrentScale[i] = GateOrder(tables[$"{recurrent}_scale"]);
                GruRecurrentSubias[i] = GateOrder(tables[$"{recurrent}_subias"]);
                GruRecurrentDiag[i] = GateOrder(tables[$"{recurrent}_weights_diag"]);
            }
            Precision = RnnoisePrecision.Int8;
        }
        catch
        {
            DisposeInt8();
            throw;
        }
    }

    /// <summary>conv2's table with its columns moved from upstream's time-major window order to the model's
    /// channel-major one.</summary>
    private static Tensor Conv2Tiles(Tensor source)
    {
        const int outputs = RnnoiseModel.GruSize, channels = RnnoiseModel.CondSize, taps = RnnoiseModel.KernelSize;
        const int inputs = channels * taps;
        sbyte[] upstream = new sbyte[outputs * inputs];
        Int8Tiles.Unpack(source.AsSpan<sbyte>(), outputs, inputs, upstream);
        sbyte[] model = new sbyte[outputs * inputs];
        for (int row = 0; row < outputs; row++)
        {
            for (int t = 0; t < taps; t++)
            {
                for (int c = 0; c < channels; c++)
                    model[row * inputs + c * taps + t] = upstream[row * inputs + t * channels + c];
            }
        }
        Tensor tiles = new(new TensorShape(outputs / Int8Tiles.Rows, inputs / Int8Tiles.Cols, Int8Tiles.Rows,
            Int8Tiles.Cols), DType.I8);
        Int8Tiles.Pack(model, outputs, inputs, tiles.AsSpan<sbyte>());
        return tiles;
    }

    /// <summary>A GRU table with its (z, r, n) gate blocks put in (r, z, n) order: whole 8-row blocks of tiles move.</summary>
    private static Tensor GateOrderTiles(Tensor source)
    {
        const int inputs = RnnoiseModel.GruSize;
        Tensor tiles = new(new TensorShape(Gates / Int8Tiles.Rows, inputs / Int8Tiles.Cols, Int8Tiles.Rows,
            Int8Tiles.Cols), DType.I8);
        SwapFirstTwoGates(source.AsSpan<sbyte>(), tiles.AsSpan<sbyte>(), RnnoiseModel.GruSize * inputs);
        return tiles;
    }

    /// <summary>A per-row vector of a GRU table in (r, z, n) order.</summary>
    private static Tensor GateOrder(Tensor source)
    {
        Tensor vector = new(new TensorShape(Gates), DType.F32);
        SwapFirstTwoGates(source.AsSpan<float>(), vector.AsSpan<float>(), RnnoiseModel.GruSize);
        return vector;
    }

    private static void SwapFirstTwoGates<T>(ReadOnlySpan<T> source, Span<T> target, int gateLength)
    {
        source.Slice(gateLength, gateLength).CopyTo(target[..gateLength]);
        source[..gateLength].CopyTo(target.Slice(gateLength, gateLength));
        source.Slice(2 * gateLength, gateLength).CopyTo(target.Slice(2 * gateLength, gateLength));
    }

    private static Tensor Copy(Tensor source)
    {
        Tensor copy = new(source.Shape, source.DType);
        source.AsSpan<byte>().CopyTo(copy.AsSpan<byte>());
        return copy;
    }

    private void DisposeInt8()
    {
        Conv2Int8?.Dispose(); Conv2Scale?.Dispose(); Conv2Subias?.Dispose();
        Conv2Int8 = Conv2Scale = Conv2Subias = null;
        foreach (Tensor?[] set in new[] { GruInputInt8, GruInputScale, GruInputSubias, GruRecurrentInt8,
            GruRecurrentScale, GruRecurrentSubias, GruRecurrentDiag })
        {
            for (int i = 0; i < set.Length; i++)
            {
                set[i]?.Dispose();
                set[i] = null;
            }
        }
    }

    private static Tensor Require(IReadOnlyDictionary<string, Tensor> weights, string name, TensorShape expected)
    {
        if (weights.TryGetValue(name, out Tensor? tensor) && tensor.Shape != expected)
            throw new HartsyInferenceException(
                $"Weight '{name}' has shape {tensor.Shape}, expected {expected}. Expected {Source}.");
        return WakeWeights.Require(weights, name, Source);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        // Null-conditional throughout: a Load that threw part-way has bound only some of these.
        Conv1Weight?.Dispose(); Conv1Bias?.Dispose(); Conv2Weight?.Dispose(); Conv2Bias?.Dispose();
        DenseOutWeight?.Dispose(); DenseOutBias?.Dispose(); VadWeight?.Dispose(); VadBias?.Dispose();
        for (int i = 0; i < 3; i++)
        {
            GruWeightIh[i]?.Dispose(); GruWeightHh[i]?.Dispose();
            GruBiasIh[i]?.Dispose(); GruBiasHh[i]?.Dispose();
        }
        DisposeInt8();
    }
}
