namespace HartsyInference.LLM.DeepSeekV41.Engram;

/// <summary>Host reference for upstream <c>Engram.forward</c>: an n-gram table lookup written into the hyper-connection residual, gated by how well it matches it.</summary>
/// <remarks>Rows are looked up by hash id, projected by <c>wkv</c> into one key per residual copy plus a value shared by all copies, and each copy
/// takes <c>gate * value</c> where the gate is a sigmoid of the signed square root of a normalized dot product with its key. The table's FP8
/// decoding lives behind <see cref="EngramRowGather"/>, and <c>wkv</c> arrives dequantized.</remarks>
public sealed class DeepSeekV41EngramModule
{
    private const float GateClamp = 1e-6f;

    private readonly EngramRowGather _gather;
    private readonly float _eps;
    private readonly DeepSeekV41Weight _wkv;
    private readonly float[] _product;

    /// <summary>Residual width.</summary>
    public int Dim { get; }

    /// <summary>Hyper-connection copies of the residual.</summary>
    public int HcMult { get; }

    /// <summary>Hash ids per position, <c>(max_ngram_size - 1) * n_heads</c>.</summary>
    public int Columns { get; }

    /// <summary>Width of one table row.</summary>
    public int HeadDim { get; }

    /// <param name="dim">Residual width.</param>
    /// <param name="hcMult">Hyper-connection copies.</param>
    /// <param name="columns">Hash ids per position.</param>
    /// <param name="headDim">Table row width.</param>
    /// <param name="eps">Norm epsilon (config <c>rms_norm_eps</c>).</param>
    /// <param name="wkv">Dequantized projection, <c>[dim * (hcMult + 1), columns * headDim]</c>.</param>
    /// <param name="qWeight"><c>[hcMult, dim]</c>; only ever used multiplied by <paramref name="kWeight"/>.</param>
    /// <param name="kWeight"><c>[hcMult, dim]</c>.</param>
    /// <param name="gather">Row lookup for this layer's table.</param>
    public DeepSeekV41EngramModule(int dim, int hcMult, int columns, int headDim, float eps, DeepSeekV41Weight wkv, float[] qWeight, float[] kWeight,
        EngramRowGather gather)
    {
        ArgumentNullException.ThrowIfNull(wkv);
        ArgumentNullException.ThrowIfNull(qWeight);
        ArgumentNullException.ThrowIfNull(kWeight);
        ArgumentNullException.ThrowIfNull(gather);
        ArgumentOutOfRangeException.ThrowIfLessThan(dim, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(hcMult, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(columns, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(headDim, 1);
        if (wkv.Elements != (long)dim * (hcMult + 1) * columns * headDim) throw new ArgumentException("wkv must be [dim * (hcMult + 1), columns * headDim].", nameof(wkv));
        if (qWeight.Length != hcMult * dim || kWeight.Length != qWeight.Length) throw new ArgumentException("q and k weights must each be [hcMult, dim].");
        Dim = dim;
        HcMult = hcMult;
        Columns = columns;
        HeadDim = headDim;
        _eps = eps;
        _wkv = wkv;
        _gather = gather;
        _product = new float[qWeight.Length];
        for (int i = 0; i < _product.Length; i++) _product[i] = qWeight[i] * kWeight[i];
    }

    /// <summary>Adds the gated lookup to <paramref name="x"/> in place.</summary>
    /// <param name="x">Residual, <c>[tokens, hcMult, dim]</c>.</param>
    /// <param name="tokens">Position count.</param>
    /// <param name="hashIds">This layer's hash ids, <c>[tokens, Columns]</c>.</param>
    /// <param name="tokenMask">False shuts the gate for that position; empty means every position is live.</param>
    public void Apply(Span<float> x, int tokens, ReadOnlySpan<long> hashIds, ReadOnlySpan<bool> tokenMask)
    {
        if (x.Length != (long)tokens * HcMult * Dim) throw new ArgumentException("x must hold tokens x hcMult x dim values.", nameof(x));
        if (hashIds.Length != (long)tokens * Columns) throw new ArgumentException("hashIds must hold tokens x columns values.", nameof(hashIds));
        if (!tokenMask.IsEmpty && tokenMask.Length != tokens) throw new ArgumentException("tokenMask must hold one flag per token.", nameof(tokenMask));

        int rowWidth = Columns * HeadDim, outWidth = Dim * (HcMult + 1);
        ushort[] bits = new ushort[rowWidth];
        float[] rows = new float[rowWidth], kv = new float[outWidth];
        for (int t = 0; t < tokens; t++)
        {
            if (!tokenMask.IsEmpty && !tokenMask[t]) continue;
            _gather(hashIds.Slice(t * Columns, Columns), bits);
            for (int i = 0; i < rowWidth; i++) rows[i] = BitConverter.UInt32BitsToSingle((uint)bits[i] << 16);
            _wkv.Linear(rows, 1, rowWidth, outWidth).CopyTo(kv, 0);

            ReadOnlySpan<float> value = kv.AsSpan(HcMult * Dim, Dim);
            for (int c = 0; c < HcMult; c++)
            {
                Span<float> h = x.Slice((t * HcMult + c) * Dim, Dim);
                ReadOnlySpan<float> key = kv.AsSpan(c * Dim, Dim), weight = _product.AsSpan(c * Dim, Dim);
                float hh = 0f, kk = 0f, dotRaw = 0f;
                for (int d = 0; d < Dim; d++)
                {
                    hh += h[d] * h[d];
                    kk += key[d] * key[d];
                    dotRaw += h[d] * weight[d] * key[d];
                }
                float rstd = 1f / MathF.Sqrt(hh / Dim + _eps) * (1f / MathF.Sqrt(kk / Dim + _eps));
                float dot = dotRaw * rstd * (1f / MathF.Sqrt(Dim));
                float gate = 1f / (1f + MathF.Exp(-MathF.CopySign(MathF.Sqrt(MathF.Max(MathF.Abs(dot), GateClamp)), dot)));
                for (int d = 0; d < Dim; d++) h[d] += gate * value[d];
            }
        }
    }
}
