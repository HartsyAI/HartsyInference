using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Omni;

/// <summary>The control signals of Hunyuan3D-Omni and the id of each (<c>cond_signal_embedding</c>).</summary>
public enum OmniControlKind
{
    /// <summary>Skeleton: one token per bone, <c>[x1 y1 z1 x2 y2 z2]</c>.</summary>
    Pose = 0,

    /// <summary>Bounding box: one token from the three extents.</summary>
    BoundingBox = 1,

    /// <summary>Voxels: one token per occupied voxel of a 16 x 16 x 16 grid over <c>[-1, 1]</c>.</summary>
    Voxel = 2,

    /// <summary>Point cloud: one token per point.</summary>
    Point = 3,
}

/// <summary>The control encoder of Hunyuan3D-Omni (<c>OmniEncoder</c> of <c>hy3dshape/models/conditioners/omni_encoder.py</c>, <c>cond_encoder/pytorch_model.bin</c>, 6 tensors): the control
/// input becomes rows of 6 values, each row is Fourier-embedded (8 frequencies, no pi, with the input: 102 channels), projected to 1024 channels by <c>Linear</c>, <c>RMSNorm</c> and GELU, and
/// followed by 10 copies of the signal token (<c>Embedding(4, 8)</c> then <c>Linear(8, 1024)</c>). The tokens go behind the DINO tokens in the DiT's context.</summary>
public sealed class OmniControlEncoder : IDisposable
{
    /// <summary>Copies of the signal token behind the control tokens.</summary>
    public const int SignalTokens = 10;

    /// <summary>Voxels per axis of the voxel control.</summary>
    public const int VoxelResolution = 16;

    private const int Width = 1024, FourierBands = 8, RowWidth = 6;
    // nn.RMSNorm(width) without eps uses the float32 machine epsilon.
    private const float RmsEps = 1.1920929e-7f;

    private readonly IBackend _backend;
    private readonly FourierEmbedSpec _fourier = new(InputDims: RowWidth, FourierBands, IncludeInput: true);
    private readonly List<Tensor> _owned = [];
    private readonly Tensor _linearW, _linearB, _normW, _signalTable, _signalW, _signalB;
    private bool _disposed;

    /// <summary>Builds the encoder from the tensors of <c>cond_encoder/pytorch_model.bin</c> (<c>linear.0</c> and <c>linear.1</c>, <c>cond_signal_embedding</c>, <c>cond_signal_linear</c>). The caller keeps ownership
    /// of <paramref name="weights"/> and must keep them alive for as long as the encoder is used: tensors that are already F32 are borrowed, not copied.</summary>
    public OmniControlEncoder(IBackend backend, IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(weights);
        _backend = backend;
        _linearW = Get(weights, "linear.0.weight", Width, _fourier.Width);
        _linearB = Get(weights, "linear.0.bias", Width);
        _normW = Get(weights, "linear.1.weight", Width);
        _signalTable = Get(weights, "cond_signal_embedding.weight", 4, 8);
        _signalW = Get(weights, "cond_signal_linear.weight", Width, 8);
        _signalB = Get(weights, "cond_signal_linear.bias", Width);
    }

    /// <summary>Encodes <paramref name="rows"/> (<c>M</c> rows of 6 values) as control tokens followed by the signal tokens: <c>[1, M + 10, 1024]</c>. The caller disposes it.</summary>
    public Tensor Encode(OmniControlKind kind, ReadOnlySpan<float> rows)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!Enum.IsDefined(kind))
            throw new ArgumentOutOfRangeException(nameof(kind), kind, "The control kind must be Pose, BoundingBox, Voxel or Point.");
        if (rows.Length == 0 || rows.Length % RowWidth != 0)
            throw new ArgumentException($"The control needs rows of {RowWidth} values and at least one row; got {rows.Length} values.", nameof(rows));
        int count = rows.Length / RowWidth;
        using Tensor coords = new(new TensorShape(count, RowWidth), DType.F32);
        rows.CopyTo(coords.AsSpan<float>());
        using Tensor fourier = new(new TensorShape(1, count, _fourier.Width), DType.F32);
        _backend.FourierEmbed(fourier, coords, _fourier);
        using Tensor projected = new(new TensorShape(1, count, Width), DType.F32);
        _backend.Linear(projected, fourier, _linearW, _linearB);
        using Tensor normed = new(projected.Shape, DType.F32);
        _backend.RmsNorm(normed, projected, _normW, RmsEps);
        using Tensor activated = new(projected.Shape, DType.F32);
        _backend.GeluErf(activated, normed);

        // The signal token: its 8 channels from the table, then Linear(8, 1024).
        using Tensor signalIn = new(new TensorShape(1, 8), DType.F32);
        _signalTable.AsSpan<float>().Slice((int)kind * 8, 8).CopyTo(signalIn.AsSpan<float>());
        using Tensor signal = new(new TensorShape(1, Width), DType.F32);
        _backend.Linear(signal, signalIn, _signalW, _signalB);

        Tensor tokens = new(new TensorShape(1, count + SignalTokens, Width), DType.F32);
        Span<float> target = tokens.AsSpan<float>();
        activated.AsSpan<float>().CopyTo(target[..(count * Width)]);
        for (int i = 0; i < SignalTokens; i++)
            signal.AsSpan<float>().CopyTo(target.Slice((count + i) * Width, Width));
        return tokens;
    }

    /// <summary>Pose rows: the bones as given, each <c>[x1 y1 z1 x2 y2 z2]</c>.</summary>
    public static float[] PoseRows(ReadOnlySpan<float> bones) => Rows(bones, 6, "bone");

    /// <summary>The bounding-box row: the three extents (each 0 to 1) repeated to 6 values (<c>bbox.repeat(1, 1, 2)</c>).</summary>
    public static float[] BoundingBoxRow(float length, float height, float width) => [length, height, width, length, height, width];

    /// <summary>Point rows: each point <c>[x y z]</c> repeated to 6 values.</summary>
    public static float[] PointRows(ReadOnlySpan<float> points) => Rows(points, 3, "point", repeat: true);

    /// <summary>Voxel rows: the points (in <c>[-1, 1]</c>) are quantised to the 16 x 16 x 16 grid and the occupied voxels' centres, in ascending <c>x + 16 y + 256 z</c> order, are repeated to 6 values
    /// (upstream <c>generate_voxel</c>).</summary>
    public static float[] VoxelRows(ReadOnlySpan<float> points)
    {
        if (points.Length == 0 || points.Length % 3 != 0)
            throw new ArgumentException("The voxel control needs points of 3 values.", nameof(points));
        SortedSet<int> occupied = [];
        for (int i = 0; i < points.Length; i += 3)
        {
            if (!float.IsFinite(points[i]) || !float.IsFinite(points[i + 1]) || !float.IsFinite(points[i + 2]))
                throw new ArgumentException($"Point {i / 3} is not finite.", nameof(points));
            int x = Quantise(points[i]), y = Quantise(points[i + 1]), z = Quantise(points[i + 2]);
            occupied.Add(x + y * VoxelResolution + z * VoxelResolution * VoxelResolution);
        }

        float size = 2f / VoxelResolution;
        float[] rows = new float[occupied.Count * RowWidth];
        int row = 0;
        foreach (int index in occupied)
        {
            int z = index / (VoxelResolution * VoxelResolution), y = index % (VoxelResolution * VoxelResolution) / VoxelResolution, x = index % VoxelResolution;
            float cx = x * size + size / 2 - 1, cy = y * size + size / 2 - 1, cz = z * size + size / 2 - 1;
            rows[row * RowWidth] = rows[row * RowWidth + 3] = cx;
            rows[row * RowWidth + 1] = rows[row * RowWidth + 4] = cy;
            rows[row * RowWidth + 2] = rows[row * RowWidth + 5] = cz;

            row++;
        }

        return rows;
    }

    /// <summary>Releases the encoder.</summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        foreach (Tensor t in _owned)
            t.Dispose();
    }

    private static int Quantise(float v) => Math.Clamp((int)MathF.Floor((v + 1) / 2 * VoxelResolution), 0, VoxelResolution - 1);

    private static float[] Rows(ReadOnlySpan<float> values, int width, string name, bool repeat = false)
    {
        if (values.Length == 0 || values.Length % width != 0)
            throw new ArgumentException($"The control needs {name}s of {width} values and at least one; got {values.Length} values.", nameof(values));
        if (!repeat)
            return values.ToArray();
        int count = values.Length / width;
        float[] rows = new float[count * RowWidth];
        for (int i = 0; i < count; i++)
        {
            for (int c = 0; c < width; c++)
            {
                rows[i * RowWidth + c] = values[i * width + c];
                rows[i * RowWidth + width + c] = values[i * width + c];
            }
        }

        return rows;
    }

    private Tensor Get(IReadOnlyDictionary<string, Tensor> weights, string key, params long[] shape)
    {
        if (!weights.TryGetValue(key, out Tensor? source))
            throw new KeyNotFoundException($"The Omni control tensor '{key}' is missing.");
        if (!source.Shape.Equals(new TensorShape(shape)))
            throw new InvalidDataException($"The Omni control tensor '{key}' has shape {source.Shape}, expected {new TensorShape(shape)}.");
        Tensor widened = Hunyuan3DShapeVae.F32(source);
        if (ReferenceEquals(widened, source))
            return source;
        _owned.Add(widened);
        return widened;
    }
}
