using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Idol;

/// <summary>Idol's learned rotary position (upstream <c>RePo3DRotaryEmbedding</c>), one per block. Each token's features predict a 3D offset
/// per head: <c>final_map(silu(gate_map(h)) · content_map(h))</c> on the affine-LayerNorm of the features (eps 1e-5). The offset times the
/// learned per-axis frequencies (10, 10 and 12 of them for a 64-wide head) times π is the rotation angle of each channel pair, so a head
/// rotates 32 pairs, interleaved <c>(x[2i], x[2i+1])</c>. The upstream <c>clamp_mul</c> equals the plain product in the forward pass.
/// Everything stays F32: the angles reach 16π and a half-precision angle moves the output by several percent.
/// <para>The per-axis product is one 3 → 32 matrix: row <c>j</c> holds the π-scaled frequency of the axis that owns angle <c>j</c> and zeros
/// elsewhere, so <c>Linear</c> forms all angles at once. Borrows the tensors it is given and owns the matrix.</para></summary>
internal sealed class IdolRePo : IDisposable
{
    private const float NormEps = 1e-5f;

    private readonly int _heads, _pairs;
    private readonly Tensor _normW, _normB, _gateW, _contentW, _finalW;
    private readonly Tensor _angleMatrix;
    private int _disposed;

    internal IdolRePo(IdolWeights weights, string prefix, IdolConfig config)
    {
        _heads = config.Heads;
        _normW = weights[$"{prefix}.norm.weight"];
        _normB = weights[$"{prefix}.norm.bias"];
        _gateW = weights[$"{prefix}.gate_map.weight"];
        _contentW = weights[$"{prefix}.content_map.weight"];
        _finalW = weights[$"{prefix}.final_map.weight"];
        Tensor[] freqs = [weights[$"{prefix}.freqs_0"], weights[$"{prefix}.freqs_1"], weights[$"{prefix}.freqs_2"]];
        _pairs = (int)(freqs[0].Shape[0] + freqs[1].Shape[0] + freqs[2].Shape[0]);
        _angleMatrix = new Tensor(new TensorShape(_pairs, 3), DType.F32);
        Span<float> m = _angleMatrix.AsSpan<float>();
        int row = 0;
        float pi = (float)Math.PI;
        for (int axis = 0; axis < 3; axis++)
        {
            ReadOnlySpan<float> f = freqs[axis].AsReadOnlySpan<float>();
            for (int j = 0; j < f.Length; j++)
                m[(row++) * 3 + axis] = f[j] * pi;
        }
    }

    /// <summary>Rotation angles pairs per head.</summary>
    internal int Pairs => _pairs;

    /// <summary>The cosines and sines of the rotation of every token and head: two <c>[L, heads, pairs]</c> tensors computed from the block
    /// input <paramref name="x"/> <c>[1, L, C]</c>. The caller owns both.</summary>
    internal (Tensor Cos, Tensor Sin) Angles(IBackend backend, Tensor x)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        int tokens = (int)x.Shape[1];
        long rows = (long)tokens * _heads;

        using Tensor normed = new(x.Shape, DType.F32);
        backend.LayerNorm(normed, x, _normW, _normB, NormEps);
        int hidden = (int)_gateW.Shape[0];
        using Tensor gate = new(new TensorShape(1, tokens, hidden), DType.F32);
        backend.Linear(gate, normed, _gateW, null);
        using Tensor gateAct = new(gate.Shape, DType.F32);
        backend.Silu(gateAct, gate);
        using Tensor content = new(gate.Shape, DType.F32);
        backend.Linear(content, normed, _contentW, null);
        using Tensor gated = new(gate.Shape, DType.F32);
        backend.Mul(gated, gateAct, content);
        using Tensor offsets = new(new TensorShape(1, tokens, 3L * _heads), DType.F32);
        backend.Linear(offsets, gated, _finalW, null);

        using Tensor offsetRows = offsets.Reshape(new TensorShape(rows, 3));
        using Tensor angles = new(new TensorShape(rows, _pairs), DType.F32);
        backend.Linear(angles, offsetRows, _angleMatrix, null);
        using Tensor embedded = new(new TensorShape(rows, 2L * _pairs), DType.F32);
        backend.FourierEmbed(embedded, angles, new FourierEmbedSpec(_pairs, 1, false));

        Tensor cos = new(new TensorShape(tokens, _heads, _pairs), DType.F32);
        Tensor sin = new(cos.Shape, DType.F32);
        backend.SliceLastDim(sin, embedded, 0);
        backend.SliceLastDim(cos, embedded, _pairs);
        return (cos, sin);
    }

    /// <summary>Rotates token-major <paramref name="x"/> <c>[1, L, heads, headDim]</c> in place by the tables of <see cref="Angles"/>.</summary>
    internal void Apply(IBackend backend, Tensor x, Tensor cos, Tensor sin)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        int tokens = (int)x.Shape[1];
        StridedView vectors = RotaryViews.TokenMajor(x.Shape);
        // The table [L, H, P] read as [1, H, L, 1, 1, P]: one row of angles per token and head.
        StridedView table = RotaryViews.Explicit([1, _heads, tokens, 1, 1, _pairs], [0, _pairs, (long)_heads * _pairs, 0, 0, 1]);
        backend.Rotary(new RotaryCall(x, cos, sin, RotaryViews.Create(vectors, table, RotaryPairing.Interleaved, RopeTableLayout.InterleavedPrefix)));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _angleMatrix.Dispose();
    }
}
