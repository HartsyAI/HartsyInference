using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ThreeD.Models.HairDeck;

namespace HartsyInference.ThreeD.Models.Dora;

/// <summary>
/// Forward pass of the Dora-VAE 1.1 shape encoder: the point-cloud condition of HairDeck's point-cloud model.
/// Input: a point cloud <c>[N, 6]</c> of positions (scaled to about [-1, 1]) and unit normals. Output: the pre-KL tokens
/// <c>[2t, 768]</c>; the first <c>t</c> tokens come from the farthest-point samples of the coarse points and the next
/// <c>t</c> from the samples of the sharp points. HairDeck's released condition uses <c>t = 256</c> and feeds the same
/// points to both branches.
/// </summary>
/// <remarks>
/// Follows <c>hairdeck/encoders/dora.py</c> (Apache-2.0, HairDeck) for the encoder forward pass. Farthest-point sampling
/// starts at index 0 and takes the first maximum on ties, as the reference does. The attention and linear layers run on
/// the backend; the point sampling and the Fourier embedding run on the host.
/// </remarks>
public sealed unsafe class DoraShapeEncoder : IDisposable
{
    private const float LayerNormEps = 1e-5f;
    private const int FourierBands = 8;
    private const int Width = DoraShapeEncoderCheckpointConverter.Width;
    private const int Heads = DoraShapeEncoderCheckpointConverter.Heads;
    private const int HeadDim = Width / Heads;

    private readonly DoraShapeEncoderCheckpointConverter.Weights _w;
    private readonly List<Tensor> _reordered = new();
    private bool _disposed;

    /// <summary>Builds the encoder. The fused qkv and kv projections are reordered once from the checkpoint's per-head
    /// layout to all-q / all-k / all-v, so the shared head splitting reads them.</summary>
    public DoraShapeEncoder(DoraShapeEncoderCheckpointConverter.Weights weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        _w = new DoraShapeEncoderCheckpointConverter.Weights(
            weights.InputProj,
            weights.InputProjSharp,
            ReorderCross(weights.Cross),
            ReorderCross(weights.CrossSharp),
            weights.Blocks.Select(ReorderSelfBlock).ToList(),
            weights.LnPostWeight,
            weights.LnPostBias);
    }

    /// <summary>Encodes one point cloud into <c>2 * tokensPerBranch</c> pre-KL tokens; both branches use the same points.</summary>
    /// <param name="backend">The backend the attention and linear layers run on.</param>
    /// <param name="points">The cloud <c>[N, 6]</c>, row-major (x y z nx ny nz).</param>
    /// <param name="tokensPerBranch">Query points per branch (256 for the released condition).</param>
    /// <returns>A new <c>[2 * tokensPerBranch, 768]</c> tensor; the caller disposes it.</returns>
    public Tensor Encode(IBackend backend, ReadOnlySpan<float> points, int tokensPerBranch)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (points.Length == 0 || points.Length % 6 != 0)
        {
            throw new ArgumentException("A point cloud is [N, 6] with N > 0.", nameof(points));
        }
        int n = points.Length / 6;
        if (tokensPerBranch <= 0 || tokensPerBranch > n)
        {
            throw new ArgumentOutOfRangeException(nameof(tokensPerBranch), tokensPerBranch,
                $"Between 1 and the point count ({n}) query points per branch.");
        }

        float[] cloud = points.ToArray();
        return EncodeBranches(backend, cloud, cloud, n, tokensPerBranch);
    }

    /// <summary>Encodes with separate coarse and sharp point sets, both <c>[N, 6]</c> with the same N.</summary>
    public Tensor Encode(IBackend backend, float[] coarse, float[] sharp, int n, int tokensPerBranch)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(coarse);
        ArgumentNullException.ThrowIfNull(sharp);
        if (coarse.Length != n * 6 || sharp.Length != n * 6)
        {
            throw new ArgumentException($"Both point sets must be [{n}, 6].");
        }
        if (tokensPerBranch <= 0 || tokensPerBranch > n)
        {
            throw new ArgumentOutOfRangeException(nameof(tokensPerBranch), tokensPerBranch,
                $"Between 1 and the point count ({n}) query points per branch.");
        }
        return EncodeBranches(backend, coarse, sharp, n, tokensPerBranch);
    }

    private Tensor EncodeBranches(IBackend backend, float[] coarse, float[] sharp, int n, int t)
    {
        int[] coarseIdx = FarthestPointSample(coarse, n, t);
        int[] sharpIdx = FarthestPointSample(sharp, n, t);

        Tensor dc = EmbedPoints(backend, coarse, n, _w.InputProj);
        Tensor ds = EmbedPoints(backend, sharp, n, _w.InputProjSharp);
        Tensor query = GatherQueries(dc, coarseIdx, ds, sharpIdx, t);

        // Both cross blocks read the same queries; the block outputs are summed.
        Tensor crossCoarse = CrossBlock(backend, query, dc, _w.Cross);
        Tensor crossSharp = CrossBlock(backend, query, ds, _w.CrossSharp);
        dc.Dispose();
        ds.Dispose();
        query.Dispose();

        Tensor x = new(crossCoarse.Shape, DType.F32);
        backend.Add(x, crossCoarse, crossSharp);
        crossCoarse.Dispose();
        crossSharp.Dispose();

        foreach (DoraShapeEncoderCheckpointConverter.SelfBlock block in _w.Blocks)
        {
            Tensor next = SelfBlock(backend, x, block);
            x.Dispose();
            x = next;
        }

        Tensor tokens = HairDeckNumerics.LayerNorm(backend, x, _w.LnPostWeight, _w.LnPostBias, LayerNormEps);
        x.Dispose();
        return tokens;
    }

    /// <summary>The 54-dim point features of every point, projected by the branch's input layer to <c>[N, 768]</c>.</summary>
    private static Tensor EmbedPoints(IBackend backend, float[] points, int n, HairDeckSetDitCheckpointConverter.Linear proj)
    {
        const int embed = DoraShapeEncoderCheckpointConverter.PointEmbeddingWidth;
        float[] features = new float[(long)n * embed];
        Span<float> row = stackalloc float[embed];
        for (int i = 0; i < n; i++)
        {
            FourierFeatures(points.AsSpan(i * 6, 6), row);
            row.CopyTo(features.AsSpan(i * embed, embed));
        }
        Tensor input = HairDeckNumerics.FromArray(features, n, embed);
        Tensor output = HairDeckNumerics.Project(backend, input, proj);
        input.Dispose();
        return output;
    }

    /// <summary>The point features: the coordinates, then sin and cos of the coordinates times 2^k (k = 0..7),
    /// coordinate-major, then the normal. Matches <c>fourier_xyz</c> of the reference, with the normal appended.</summary>
    internal static void FourierFeatures(ReadOnlySpan<float> point, Span<float> row)
    {
        int o = 0;
        for (int c = 0; c < 3; c++) row[o++] = point[c];
        for (int c = 0; c < 3; c++)
        {
            for (int k = 0; k < FourierBands; k++) row[o++] = MathF.Sin(point[c] * (1 << k));
        }
        for (int c = 0; c < 3; c++)
        {
            for (int k = 0; k < FourierBands; k++) row[o++] = MathF.Cos(point[c] * (1 << k));
        }
        for (int c = 3; c < 6; c++) row[o++] = point[c];
    }

    /// <summary>Greedy farthest-point sampling starting at index 0. Each next index maximises the squared distance to the
    /// chosen set, the first maximum on ties (<c>torch.argmax</c>), in float32 in the reference's order.</summary>
    internal static int[] FarthestPointSample(float[] points, int n, int k)
    {
        int[] chosen = new int[k];
        float[] dist = new float[n];
        Array.Fill(dist, float.PositiveInfinity);
        int cur = 0;
        for (int i = 0; i < k; i++)
        {
            chosen[i] = cur;
            float cx = points[cur * 6], cy = points[cur * 6 + 1], cz = points[cur * 6 + 2];
            int next = 0;
            float best = float.NegativeInfinity;
            for (int j = 0; j < n; j++)
            {
                float dx = points[j * 6] - cx, dy = points[j * 6 + 1] - cy, dz = points[j * 6 + 2] - cz;
                float d = dx * dx + dy * dy + dz * dz;
                if (d < dist[j])
                {
                    dist[j] = d;
                }
                if (dist[j] > best)
                {
                    best = dist[j];
                    next = j;
                }
            }
            cur = next;
        }
        return chosen;
    }

    /// <summary>The query tokens: the coarse-branch rows of the coarse samples, then the sharp-branch rows of the sharp samples.</summary>
    private static Tensor GatherQueries(Tensor coarseRows, int[] coarseIdx, Tensor sharpRows, int[] sharpIdx, int t)
    {
        Tensor q = new(new TensorShape(2 * t, Width), DType.F32);
        float* dst = (float*)q.DataPointer;
        float* a = (float*)coarseRows.DataPointer;
        float* b = (float*)sharpRows.DataPointer;
        for (int i = 0; i < t; i++)
        {
            new ReadOnlySpan<float>(a + (long)coarseIdx[i] * Width, Width).CopyTo(new Span<float>(dst + (long)i * Width, Width));
            new ReadOnlySpan<float>(b + (long)sharpIdx[i] * Width, Width).CopyTo(new Span<float>(dst + (long)(t + i) * Width, Width));
        }
        return q;
    }

    /// <summary>x + attn(ln_1 x -> q, ln_2 data -> k, v), then + mlp(ln_3 x). Returns a new tensor.</summary>
    private static Tensor CrossBlock(IBackend backend, Tensor x, Tensor data, DoraShapeEncoderCheckpointConverter.CrossBlock b)
    {
        Tensor xn = HairDeckNumerics.LayerNorm(backend, x, b.Ln1Weight, b.Ln1Bias, LayerNormEps);
        Tensor q = HairDeckNumerics.Project(backend, xn, b.Query);
        xn.Dispose();
        Tensor dn = HairDeckNumerics.LayerNorm(backend, data, b.Ln2Weight, b.Ln2Bias, LayerNormEps);
        Tensor kv = HairDeckNumerics.Project(backend, dn, b.KeyValue);
        dn.Dispose();
        Tensor attended = Attend(backend, q, kv, data.Shape[0], x.Shape[0], keyValueOffsetV: Width);
        q.Dispose();
        kv.Dispose();
        Tensor projected = HairDeckNumerics.Project(backend, attended, b.Output);
        attended.Dispose();
        Tensor afterAttn = new(x.Shape, DType.F32);
        backend.Add(afterAttn, x, projected);
        projected.Dispose();
        return MlpResidual(backend, afterAttn, b.Ln3Weight, b.Ln3Bias, b.MlpIn, b.MlpOut);
    }

    /// <summary>x + attn(ln_1 x) over the fused qkv projection, then + mlp(ln_2 x). Returns a new tensor.</summary>
    private static Tensor SelfBlock(IBackend backend, Tensor x, DoraShapeEncoderCheckpointConverter.SelfBlock b)
    {
        Tensor xn = HairDeckNumerics.LayerNorm(backend, x, b.Ln1Weight, b.Ln1Bias, LayerNormEps);
        Tensor qkv = HairDeckNumerics.Project(backend, xn, b.Qkv);
        xn.Dispose();
        Tensor attended = AttendFused(backend, qkv, x.Shape[0]);
        qkv.Dispose();
        Tensor projected = HairDeckNumerics.Project(backend, attended, b.Output);
        attended.Dispose();
        Tensor afterAttn = new(x.Shape, DType.F32);
        backend.Add(afterAttn, x, projected);
        projected.Dispose();
        return MlpResidual(backend, afterAttn, b.Ln2Weight, b.Ln2Bias, b.MlpIn, b.MlpOut);
    }

    /// <summary>Attention of token-major queries <c>[n, 768]</c> over keys (columns 0..767) and values (columns from
    /// <paramref name="keyValueOffsetV"/>) of a <c>[m, 1536]</c> projection; returns <c>[n, 768]</c>.</summary>
    private static Tensor Attend(IBackend backend, Tensor q, Tensor kv, long m, long n, int keyValueOffsetV)
    {
        using Tensor qh = HairDeckNumerics.Create(1, Heads, n, HeadDim);
        using Tensor kh = HairDeckNumerics.Create(1, Heads, m, HeadDim);
        using Tensor vh = HairDeckNumerics.Create(1, Heads, m, HeadDim);
        HairDeckNumerics.SplitHeads(q, 0, qh, Heads, HeadDim);
        HairDeckNumerics.SplitHeads(kv, 0, kh, Heads, HeadDim);
        HairDeckNumerics.SplitHeads(kv, keyValueOffsetV, vh, Heads, HeadDim);
        return HeadsToTokens(backend, qh, kh, vh, n);
    }

    /// <summary>Self-attention over the fused <c>[n, 3 * 768]</c> projection (q, then k, then v, head-major in each).</summary>
    private static Tensor AttendFused(IBackend backend, Tensor qkv, long n)
    {
        using Tensor qh = HairDeckNumerics.Create(1, Heads, n, HeadDim);
        using Tensor kh = HairDeckNumerics.Create(1, Heads, n, HeadDim);
        using Tensor vh = HairDeckNumerics.Create(1, Heads, n, HeadDim);
        HairDeckNumerics.SplitHeads(qkv, 0, qh, Heads, HeadDim);
        HairDeckNumerics.SplitHeads(qkv, Width, kh, Heads, HeadDim);
        HairDeckNumerics.SplitHeads(qkv, 2 * Width, vh, Heads, HeadDim);
        return HeadsToTokens(backend, qh, kh, vh, n);
    }

    private static Tensor HeadsToTokens(IBackend backend, Tensor qh, Tensor kh, Tensor vh, long n)
    {
        using Tensor oh = HairDeckNumerics.Create(1, Heads, n, HeadDim);
        backend.ScaledDotProductAttention(oh, qh, kh, vh, null, 1f / MathF.Sqrt(HeadDim));
        Tensor merged = HairDeckNumerics.Create(n, Width);
        HairDeckNumerics.MergeHeads(oh, merged, Heads, HeadDim);
        return merged;
    }

    /// <summary>x + mlp(ln x) with the exact GELU; consumes <paramref name="x"/> and returns a new tensor.</summary>
    private static Tensor MlpResidual(IBackend backend, Tensor x, Tensor lnWeight, Tensor lnBias,
        HairDeckSetDitCheckpointConverter.Linear fc, HairDeckSetDitCheckpointConverter.Linear proj)
    {
        Tensor xn = HairDeckNumerics.LayerNorm(backend, x, lnWeight, lnBias, LayerNormEps);
        Tensor hidden = HairDeckNumerics.Project(backend, xn, fc);
        xn.Dispose();
        Tensor act = HairDeckNumerics.GeluExact(backend, hidden);
        hidden.Dispose();
        Tensor mlp = HairDeckNumerics.Project(backend, act, proj);
        act.Dispose();
        Tensor result = new(x.Shape, DType.F32);
        backend.Add(result, x, mlp);
        mlp.Dispose();
        x.Dispose();
        return result;
    }

    private DoraShapeEncoderCheckpointConverter.CrossBlock ReorderCross(DoraShapeEncoderCheckpointConverter.CrossBlock b) => b with
    {
        KeyValue = new HairDeckSetDitCheckpointConverter.Linear(ReorderRows(b.KeyValue.Weight, parts: 2), null),
    };

    private DoraShapeEncoderCheckpointConverter.SelfBlock ReorderSelfBlock(DoraShapeEncoderCheckpointConverter.SelfBlock b) => b with
    {
        Qkv = new HairDeckSetDitCheckpointConverter.Linear(ReorderRows(b.Qkv.Weight, parts: 3), null),
    };

    /// <summary>Permutes the rows of a per-head projection <c>[heads * parts * headDim, in]</c> (each head holds its
    /// parts back to back) to <c>[parts * width, in]</c> with each part's heads contiguous.</summary>
    private Tensor ReorderRows(Tensor weight, int parts)
    {
        int rows = (int)weight.Shape[0];
        int cols = (int)weight.Shape[1];
        float[] src = HairDeckNumerics.ToArray(weight);
        float[] dst = new float[src.Length];
        int perHead = parts * HeadDim;
        for (int h = 0; h < Heads; h++)
        {
            for (int p = 0; p < parts; p++)
            {
                for (int d = 0; d < HeadDim; d++)
                {
                    int from = (h * perHead + p * HeadDim + d) * cols;
                    int to = (p * Width + h * HeadDim + d) * cols;
                    Array.Copy(src, from, dst, to, cols);
                }
            }
        }
        Tensor result = HairDeckNumerics.FromArray(dst, rows, cols);
        _reordered.Add(result);
        return result;
    }

    /// <summary>Releases the reordered weights. Tensors from <see cref="Encode(IBackend, ReadOnlySpan{float}, int)"/> belong to the caller.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        foreach (Tensor t in _reordered)
        {
            t.Dispose();
        }
        _reordered.Clear();
    }
}
