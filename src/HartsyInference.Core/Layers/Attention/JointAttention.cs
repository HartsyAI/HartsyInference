using System.Buffers;
using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Layers.Attention;

/// <summary>When each stream's QK norm runs relative to the other streams' projections.</summary>
public enum JointStageOrder
{
    /// <summary>Every stream projects, then every stream norms its Q and K.</summary>
    ProjectAllThenNorm,

    /// <summary>Each stream projects and norms before the next stream projects.</summary>
    ProjectNormPerStream,
}

/// <summary>The fixed geometry and policy of an attention over several token streams, decided at load.</summary>
public sealed record JointAttentionConfig
{
    /// <summary>Heads of the joint attention; every stream projects to this many.</summary>
    public required int Heads { get; init; }

    /// <summary>Dims per head.</summary>
    public required int HeadDim { get; init; }

    /// <summary>Stream indices in joint row order: <c>[1, 0]</c> puts stream 1's rows first.</summary>
    public required int[] ConcatOrder { get; init; }

    /// <summary>The legacy <c>allowF16</c> assertion the joint attention call carries.</summary>
    public bool AllowF16 { get; init; }

    /// <summary>The assertion above one batch item when it differs; null follows <see cref="AllowF16"/>.</summary>
    public bool? AllowF16Batched { get; init; }

    /// <summary>The joint call's <see cref="AttentionLayerConfig.AllowBf16"/> grant; exclusive with either F16 assertion.</summary>
    public bool AllowBf16 { get; init; }

    /// <summary>The softmax scale; 0 means <c>1/√headDim</c>.</summary>
    public float Scale { get; init; }

    /// <summary>The order of the projection and norm stages across streams.</summary>
    public JointStageOrder StageOrder { get; init; }

    /// <summary>The joint call's <see cref="AttentionLayerConfig.TokenMajorRoute"/>; the stream layers never attend, so
    /// they stay Dense.</summary>
    public AttentionRoute TokenMajorRoute { get; init; } = AttentionRoute.Dense;

    /// <summary>The joint call's <see cref="AttentionLayerConfig.HeadMajorForSage"/>.</summary>
    public bool HeadMajorForSage { get; init; }
}

/// <summary>The per-call inputs of a joint attention. Every member is borrowed for the call.</summary>
public readonly record struct JointCallContext
{
    /// <summary>The joint call's rotation, bias, probe and K/V hooks, over the joint rows.</summary>
    public AttentionCallContext Joint { get; init; }

    /// <summary>The one stream whose Q and K rotate alone, after the norms and before the concatenation (a model that rotates
    /// its image rows only); null for none.</summary>
    public int? RotatedStream { get; init; }

    /// <summary>The rotation of <see cref="RotatedStream"/>: its tables or model rotation.</summary>
    public AttentionCallContext StreamRotation { get; init; }

    /// <summary>Each stream input is disposed once its projections have read it.</summary>
    public bool ReleaseInputs { get; init; }
}

/// <summary>Attention over several token streams (MMDiT): each stream projects and norms through its own
/// <see cref="AttentionLayer"/>, the streams' rows are concatenated in <see cref="JointAttentionConfig.ConcatOrder"/>, one
/// attention call runs over the joint sequence, and its output is split back per stream.</summary>
public sealed class JointAttention
{
    /// <summary>The most streams one joint attention takes.</summary>
    public const int MaxStreams = 4;

    private readonly JointAttentionConfig _config;
    private readonly AttentionLayer[] _streams;
    private readonly AttentionLayer _joint;
    private readonly AttentionLayer _jointBatched;

    /// <summary>Creates a joint attention over <paramref name="streams"/>, given in stream index order.</summary>
    public JointAttention(JointAttentionConfig config, ReadOnlySpan<AttentionLayer> streams)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (streams.Length is < 2 or > MaxStreams)
            throw new ArgumentException($"A joint attention takes 2 to {MaxStreams} streams; got {streams.Length}.", nameof(streams));
        int count = streams.Length;
        if (config.ConcatOrder is null || config.ConcatOrder.Length != count
            || config.ConcatOrder.Distinct().Count() != count || config.ConcatOrder.Any(i => i < 0 || i >= count))
            throw new ArgumentException("ConcatOrder must name every stream exactly once.", nameof(config));
        foreach (AttentionLayer stream in streams)
        {
            AttentionLayerConfig c = stream.Config;
            if (c.Heads != config.Heads || c.KvHeads != config.Heads || c.HeadDim != config.HeadDim)
                throw new ArgumentException($"Every stream projects {config.Heads} heads of {config.HeadDim}; one has {c.Heads}/{c.KvHeads} of {c.HeadDim}.",
                    nameof(streams));
            if (c.Gate != AttentionGate.None || c.QkNorm == AttentionQkNorm.FullWidth || c.TokenMajorRoute != AttentionRoute.Dense)
                throw new ArgumentException("A joint stream has no gate, no fused full-width norm and attends head-major.", nameof(streams));
        }
        _config = config;
        _streams = streams.ToArray();
        _joint = JointLayer(config, config.AllowF16);
        _jointBatched = config.AllowF16Batched is { } batched && batched != config.AllowF16 ? JointLayer(config, batched) : _joint;
    }

    /// <summary>The joint attention's configuration.</summary>
    public JointAttentionConfig Config => _config;

    /// <summary>Number of streams.</summary>
    public int StreamCount => _streams.Length;

    /// <summary>Runs the projections, norms, concatenation, attention and split. <paramref name="inputs"/> are the streams'
    /// <c>[B, S_i, C_i]</c> inputs in stream index order; <paramref name="attended"/>[i] receives stream i's rows of the
    /// attention output, <c>[B, S_i, heads·D]</c>, unprojected and owned by the caller.</summary>
    public void Attend(IBackend backend, ReadOnlySpan<Tensor> inputs, Span<Tensor?> attended, in JointCallContext context)
    {
        int n = _streams.Length;
        if (inputs.Length != n || attended.Length != n)
            throw new ArgumentException($"A joint attention over {n} streams takes {n} inputs and {n} outputs.");
        if (context.RotatedStream is { } rotatedIndex && (uint)rotatedIndex >= (uint)n)
            throw new ArgumentOutOfRangeException(nameof(context), $"Rotated stream {rotatedIndex} is not one of the {n} streams.");
        if (context.RotatedStream is not null && HasRotation(context.Joint))
            throw new ArgumentException("A rotated stream and a joint rotation would rotate that stream twice.", nameof(context));
        int batch = (int)inputs[0].Shape[0];
        for (int i = 0; i < n; i++)
            if (inputs[i].Shape.Rank != 3 || inputs[i].Shape[0] != batch)
                throw new ArgumentException($"Joint stream inputs are [B, S, C] with one batch size; got {inputs[i].Shape}.", nameof(inputs));
        DType act = _streams[0].Config.ComputeDType ?? inputs[0].DType;
        if (batch > 1 && act != DType.F32)
            throw new NotSupportedException($"A batched joint attention splits F32 rows; got {act} at batch {batch}.");

        StreamTensors q = default, k = default, v = default;
        Span<int> rows = stackalloc int[MaxStreams];
        try
        {
            for (int i = 0; i < n; i++)
            {
                rows[i] = (int)inputs[i].Shape[1];
                (q[i], k[i], v[i]) = _streams[i].ProjectSelf(backend, inputs[i], context.ReleaseInputs);
                if (_config.StageOrder == JointStageOrder.ProjectNormPerStream)
                    _streams[i].NormalizeQk(backend, ref q[i], ref k[i]);
            }
            if (_config.StageOrder == JointStageOrder.ProjectAllThenNorm)
                for (int i = 0; i < n; i++)
                    _streams[i].NormalizeQk(backend, ref q[i], ref k[i]);
            if (context.RotatedStream is { } rotated)
            {
                AttentionCallContext rotation = context.StreamRotation;
                _streams[rotated].RotateQk(backend, q[rotated], k[rotated], in rotation);
            }

            AttentionCallContext joint = context.Joint;
            AttendJoint(backend, ref q, ref k, ref v, rows, batch, attended, in joint);
        }
        catch
        {
            for (int i = 0; i < n; i++)
            {
                q[i]?.Dispose();
                k[i]?.Dispose();
                v[i]?.Dispose();
                attended[i]?.Dispose();
                attended[i] = null;
            }
            throw;
        }
    }

    /// <summary>The concatenation, attention and split over streams already projected, normed and rotated by the model:
    /// token-major <paramref name="q"/>, <paramref name="k"/> and <paramref name="v"/> of each stream, <c>[B, S_i, heads, D]</c>, in
    /// stream index order, all taken over and disposed, also when the call throws (a refused argument takes nothing over). <paramref name="attended"/>[i] receives stream i's rows
    /// as in <see cref="Attend"/>.</summary>
    public void AttendProjected(IBackend backend, ReadOnlySpan<Tensor> q, ReadOnlySpan<Tensor> k, ReadOnlySpan<Tensor> v,
        Span<Tensor?> attended, in AttentionCallContext joint)
    {
        int n = _streams.Length;
        if (q.Length != n || k.Length != n || v.Length != n || attended.Length != n)
            throw new ArgumentException($"A joint attention over {n} streams takes {n} each of q, k, v and outputs.");
        int batch = (int)q[0].Shape[0];
        Span<int> rows = stackalloc int[MaxStreams];
        for (int i = 0; i < n; i++)
        {
            if (q[i].Shape.Rank < 3 || q[i].Shape[0] != batch || k[i].Shape[0] != batch || v[i].Shape[0] != batch
                || k[i].Shape[1] != q[i].Shape[1] || v[i].Shape[1] != q[i].Shape[1])
                throw new ArgumentException($"Joint stream operands are [B, S, ..] with one batch size and length per stream; got {q[i].Shape}, {k[i].Shape}, {v[i].Shape}.");
            rows[i] = (int)q[i].Shape[1];
        }
        if (batch > 1 && (_streams[0].Config.ComputeDType ?? q[0].DType) != DType.F32)
            throw new NotSupportedException($"A batched joint attention splits F32 rows; got {q[0].DType} at batch {batch}.");
        StreamTensors qs = default, ks = default, vs = default;
        for (int i = 0; i < n; i++) (qs[i], ks[i], vs[i]) = (q[i], k[i], v[i]);
        try
        {
            AttendJoint(backend, ref qs, ref ks, ref vs, rows, batch, attended, in joint);
        }
        catch
        {
            for (int i = 0; i < n; i++)
            {
                qs[i]?.Dispose();
                ks[i]?.Dispose();
                vs[i]?.Dispose();
                attended[i]?.Dispose();
                attended[i] = null;
            }
            throw;
        }
    }

    private void AttendJoint(IBackend backend, ref StreamTensors q, ref StreamTensors k, ref StreamTensors v, ReadOnlySpan<int> rows,
        int batch, Span<Tensor?> attended, in AttentionCallContext joint)
    {
        int n = _streams.Length;
        Tensor? jq = null, jk = null, jv = null, flat = null;
        try
        {
            int total = 0;
            for (int i = 0; i < n; i++) total += rows[i];
            int inner = _config.Heads * _config.HeadDim;
            TensorShape jointShape = new(batch, total, inner);
            jq = ConcatInOrder(backend, ref q, n, jointShape);
            jk = ConcatInOrder(backend, ref k, n, jointShape);
            jv = ConcatInOrder(backend, ref v, n, jointShape);

            flat = (batch == 1 ? _joint : _jointBatched).AttendFlat(backend, jq, jk, jv, jointShape, in joint);
            int offset = 0;
            foreach (int stream in _config.ConcatOrder)
            {
                Tensor part = new(new TensorShape(batch, rows[stream], inner), flat.DType);
                attended[stream] = part;
                SplitRows(backend, part, flat, batch, total, offset);
                offset += rows[stream];
            }
        }
        catch
        {
            jq?.Dispose();
            jk?.Dispose();
            jv?.Dispose();
            flat?.Dispose();
            throw;
        }
        flat.Dispose();
    }

    /// <summary>Stream <paramref name="stream"/>'s output projection of its attended rows, which are disposed.</summary>
    public Tensor ProjectOutput(IBackend backend, int stream, Tensor attended) => _streams[stream].ProjectOutput(backend, attended);

    /// <summary>Rows <c>[offset, offset + output rows)</c> of each batch item of a <c>[B, seq, W]</c> tensor into
    /// <paramref name="output"/> <c>[B, rows, W]</c>, on the device: the split of a joint sequence. At B = 1 it is one
    /// contiguous <see cref="IBackend.SliceRows"/>; above, each batch item's rows are gathered (F32, as every batched joint
    /// path runs), since one contiguous block would hand batch 0's rows to every item.</summary>
    public static void SplitRows(IBackend backend, Tensor output, Tensor input, int batch, int seq, int offset)
    {
        if (batch == 1)
        {
            backend.SliceRows(output, input, offset);
            return;
        }
        if (input.DType != DType.F32 || output.DType != DType.F32)
            throw new NotSupportedException($"A batched joint split gathers F32 rows; got {input.DType} at batch {batch}.");
        long width = input.ElementCount / ((long)batch * seq);
        int count = (int)(output.ElementCount / ((long)batch * width));
        int[] rented = ArrayPool<int>.Shared.Rent(batch * count);
        try
        {
            Span<int> index = rented.AsSpan(0, batch * count);
            for (int b = 0; b < batch; b++)
                for (int r = 0; r < count; r++) index[b * count + r] = b * seq + offset + r;
            backend.GatherRows(output, input, index);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(rented);
        }
    }

    private static bool HasRotation(in AttentionCallContext context) =>
        context.Rotation is not null || context.HeadMajorRotation is not null
        || context.QueryRotation.Kind != QkRotationKind.None || context.KeyRotation.Kind != QkRotationKind.None;

    private Tensor ConcatInOrder(IBackend backend, ref StreamTensors parts, int n, TensorShape shape)
    {
        StreamTensors ordered = default;
        for (int j = 0; j < n; j++) ordered[j] = parts[_config.ConcatOrder[j]];
        ReadOnlySpan<Tensor> sources = ((ReadOnlySpan<Tensor>)ordered)[..n];
        Tensor joint = new(shape, sources[0].DType);
        backend.Concat(joint, sources, 1);
        foreach (Tensor source in sources) source.Dispose();
        return joint;
    }

    private static AttentionLayer JointLayer(JointAttentionConfig config, bool allowF16) => new(new AttentionLayerConfig
    {
        Heads = config.Heads, KvHeads = config.Heads, HeadDim = config.HeadDim, AllowF16 = allowF16, AllowBf16 = config.AllowBf16,
        Scale = config.Scale, TokenMajorRoute = config.TokenMajorRoute, HeadMajorForSage = config.HeadMajorForSage,
    });

    /// <summary>One tensor per stream, held on the stack.</summary>
    [InlineArray(MaxStreams)]
    private struct StreamTensors
    {
        private Tensor _element;
    }
}
