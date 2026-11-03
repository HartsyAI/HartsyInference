using System.Buffers;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>docs/ATTENTION_CONTRACT.md computed on the host from strided operands of any storage dtype: the default
/// <see cref="IBackend.Attention"/> for a backend without kernel rows. It reads tensors through
/// <see cref="Tensor.DataPointer"/> and never calls a legacy entry point, so it cannot recurse into a backend's wrappers.</summary>
public static class AttentionHostReference
{
    /// <summary>Runs <paramref name="call"/>. Device positions need a backend to read them and are refused.</summary>
    /// <exception cref="ArgumentException">The operands disagree on geometry or address memory outside their tensors.</exception>
    public static void Run(in AttentionCall call) => Run(in call, default);

    /// <summary>Runs <paramref name="call"/> and, when <paramref name="lse"/> is not empty, writes each row's natural-log LSE
    /// (§3; −∞ for a fully excluded row) into it as <c>[B, Hq, Sq]</c>.</summary>
    /// <exception cref="ArgumentException">The operands disagree on geometry or address memory outside their tensors.</exception>
    public static unsafe void Run(in AttentionCall call, Span<float> lse)
    {
        if (call.Positions.OnDevice)
            throw new AttentionUnsupportedException("The host reference cannot read device positions.");
        Validate(in call);
        if (!lse.IsEmpty && lse.Length != (long)call.Batch * call.Heads * call.QueryTokens)
            throw new ArgumentException($"LSE holds {lse.Length} values; [B, Hq, Sq] needs {(long)call.Batch * call.Heads * call.QueryTokens}.",
                nameof(lse));
        AttentionOperand q = call.Query, k = call.Key, v = call.Value, o = call.Output;
        int dqk = call.QkHeadDim, dv = call.ValueHeadDim, group = call.KvGroup;
        int kvLen = call.ValidKvLength;
        AttentionMask mask = call.Mask;
        BiasView bias = BiasView.Of(mask.Bias, call);
        float* sinks = call.Logits.Sinks is null ? null : (float*)call.Logits.Sinks.DataPointer;
        float* slopes = call.Logits.AlibiSlopes is null ? null : (float*)call.Logits.AlibiSlopes.DataPointer;
        float softcap = call.Logits.Softcap;
        double[] scores = ArrayPool<double>.Shared.Rent(Math.Max(1, kvLen));
        double[] acc = ArrayPool<double>.Shared.Rent(dv);
        try
        {
            Compute(in call, q, k, v, o, dqk, dv, group, kvLen, in mask, bias, sinks, slopes, softcap, scores, acc, lse);
        }
        finally
        {
            ArrayPool<double>.Shared.Return(scores);
            ArrayPool<double>.Shared.Return(acc);
        }
    }

    private static unsafe void Compute(in AttentionCall call, AttentionOperand q, AttentionOperand k, AttentionOperand v,
        AttentionOperand o, int dqk, int dv, int group, int kvLen, in AttentionMask mask, BiasView bias, float* sinks, float* slopes,
        float softcap, double[] scores, double[] acc, Span<float> lse)
    {
        for (int b = 0; b < call.Batch; b++)
            for (int h = 0; h < call.Heads; h++)
            {
                int hk = h / group;
                for (int r = 0; r < call.QueryTokens; r++)
                {
                    int qpos = call.Positions.QueryOffset + r;
                    double max = double.NegativeInfinity;
                    for (int j = 0; j < kvLen; j++)
                    {
                        if (!Visible(in mask, qpos, j)) { scores[j] = double.NegativeInfinity; continue; }
                        double dot = 0;
                        for (int c = 0; c < dqk; c++) dot += Read(in q, b, h, r, c) * Read(in k, b, hk, j, c);
                        double s = call.Scale * dot;
                        if (softcap > 0f) s = softcap * Math.Tanh(s / softcap);
                        s += bias.At(b, h, r, j);
                        if (slopes != null) s += slopes[h] * (mask.Causal ? j - qpos : -Math.Abs(j - qpos));
                        scores[j] = s;
                        if (s > max || double.IsNaN(s)) max = s;
                    }
                    double sink = sinks == null ? double.NegativeInfinity : sinks[h];
                    double m = Math.Max(max, sink);
                    Array.Clear(acc, 0, dv);
                    double sum = double.IsNegativeInfinity(sink) ? 0 : Math.Exp(sink - m);
                    if (!double.IsNegativeInfinity(max))
                        for (int j = 0; j < kvLen; j++)
                        {
                            if (double.IsNegativeInfinity(scores[j])) continue;
                            double p = Math.Exp(scores[j] - m);
                            sum += p;
                            for (int c = 0; c < dv; c++) acc[c] += p * Read(in v, b, hk, j, c);
                        }
                    // A fully excluded row with no sink writes zeros; a NaN anywhere in the row stays NaN (contract §3).
                    double inv = sum > 0 ? 1.0 / sum : double.IsNaN(sum) ? double.NaN : 0.0;
                    for (int c = 0; c < dv; c++) Write(in o, b, h, r, c, (float)(acc[c] * inv));
                    if (!lse.IsEmpty)
                        lse[(b * call.Heads + h) * call.QueryTokens + r] =
                            sum > 0 ? (float)(m + Math.Log(sum)) : double.IsNaN(sum) ? float.NaN : float.NegativeInfinity;
                }
            }
    }

    /// <summary>The geometry the computation indexes by: shared heads that divide, matching extents, every operand inside
    /// its tensor, a bias broadcastable on every axis.</summary>
    /// <exception cref="ArgumentException">The operands disagree on geometry or address memory outside their tensors.</exception>
    internal static void Validate(in AttentionCall call)
    {
        AttentionFormats refused = call.Numerics.Refused & AttentionNumerics.OperandFormats(in call);
        if (refused != AttentionFormats.None)
            throw new AttentionUnsupportedException($"The call refuses {refused}, and an operand or the output is stored in it.");
        AttentionOperand q = call.Query, k = call.Key, v = call.Value, o = call.Output;
        if (q.Batch <= 0 || q.Heads <= 0 || q.Tokens <= 0 || q.HeadDim <= 0 || k.Heads <= 0 || k.Tokens <= 0 || v.HeadDim <= 0)
            throw new ArgumentException($"Attention operands need positive extents; got Q={Describe(q)}, K={Describe(k)}, V={Describe(v)}.");
        if (q.Heads % k.Heads != 0)
            throw new ArgumentException($"Query heads ({q.Heads}) must be a multiple of key/value heads ({k.Heads}).");
        if (k.HeadDim != q.HeadDim || v.Heads != k.Heads || v.Tokens != k.Tokens || o.HeadDim != v.HeadDim
            || o.Heads != q.Heads || o.Tokens != q.Tokens || k.Batch != q.Batch || v.Batch != q.Batch || o.Batch != q.Batch)
            throw new ArgumentException(
                $"Attention operands disagree: Q={Describe(q)}, K={Describe(k)}, V={Describe(v)}, O={Describe(o)}.");
        if (call.ValidKvLength < 0 || call.ValidKvLength > k.Tokens)
            throw new ArgumentException($"Valid key length {call.ValidKvLength} is outside the key operand's {k.Tokens} rows.");
        foreach (AttentionOperand x in (ReadOnlySpan<AttentionOperand>)[q, k, v, o])
            if (x.Offset < 0 || x.Extent > x.Storage.ElementCount)
                throw new ArgumentException($"Operand {Describe(x)} addresses past its {x.Storage.ElementCount}-element tensor.");
        if (call.Mask.Bias is { } bias)
        {
            TensorShape s = bias.Shape;
            int rank = s.Rank;
            bool fits = rank is >= 2 and <= 4 && (s[rank - 1] == 1 || s[rank - 1] == k.Tokens) && (s[rank - 2] == 1 || s[rank - 2] == q.Tokens)
                && (rank < 3 || s[rank - 3] == 1 || s[rank - 3] == q.Heads) && (rank < 4 || s[0] == 1 || s[0] == q.Batch);
            if (!fits)
                throw new ArgumentException($"Bias {s} does not broadcast to [{q.Batch},{q.Heads},{q.Tokens},{k.Tokens}].");
        }
        if (call.Logits.Sinks is { } sinks && sinks.ElementCount < q.Heads)
            throw new ArgumentException($"Sinks hold {sinks.ElementCount} values for {q.Heads} heads.");
        if (call.Logits.AlibiSlopes is { } slopes && slopes.ElementCount < q.Heads)
            throw new ArgumentException($"ALiBi slopes hold {slopes.ElementCount} values for {q.Heads} heads.");
    }

    private static string Describe(in AttentionOperand x) =>
        $"{x.Layout}[b{x.Batch} h{x.Heads} s{x.Tokens} d{x.HeadDim}] of {x.Storage.Shape}";

    private static bool Visible(in AttentionMask m, int qpos, int kpos)
    {
        int qi = m.BlockSize > 0 ? qpos / m.BlockSize : qpos, ki = m.BlockSize > 0 ? kpos / m.BlockSize : kpos;
        if (m.Causal && ki > qi) return false;
        if (m.WindowLeft is { } left && ki < qi - left) return false;
        if (m.WindowRight is { } right && ki > qi + right) return false;
        return true;
    }

    private static unsafe double Read(in AttentionOperand x, int b, int h, int t, int c)
    {
        long i = x.Offset + b * x.BatchStride + h * x.HeadStride + t * x.TokenStride + c;
        void* p = x.Storage.DataPointer;
        DType dt = x.Storage.DType;
        if (dt == DType.F32) return ((float*)p)[i];
        if (dt == DType.F16) return (double)BitConverter.UInt16BitsToHalf(((ushort*)p)[i]);
        if (dt == DType.BF16) return BitConverter.UInt32BitsToSingle((uint)((ushort*)p)[i] << 16);
        throw new AttentionUnsupportedException($"The host reference reads F32, F16 and BF16 operands, not {dt}.");
    }

    internal static unsafe void Write(in AttentionOperand x, int b, int h, int t, int c, float value)
    {
        long i = x.Offset + b * x.BatchStride + h * x.HeadStride + t * x.TokenStride + c;
        void* p = x.Storage.DataPointer;
        DType dt = x.Storage.DType;
        if (dt == DType.F32) ((float*)p)[i] = value;
        else if (dt == DType.F16) ((ushort*)p)[i] = BitConverter.HalfToUInt16Bits((Half)value);
        else if (dt == DType.BF16)
        {
            uint u = BitConverter.SingleToUInt32Bits(value);
            ((ushort*)p)[i] = (ushort)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
        }
        else throw new AttentionUnsupportedException($"The host reference writes F32, F16 and BF16 outputs, not {dt}.");
    }

    /// <summary>An F32 bias of rank 2–4 read as <c>[B, H, Sq, Skv]</c>, axes of extent 1 broadcast.</summary>
    private readonly unsafe struct BiasView
    {
        private readonly float* _p;
        private readonly long _sb, _sh, _sq, _sk;

        private BiasView(float* p, long sb, long sh, long sq, long sk) { _p = p; _sb = sb; _sh = sh; _sq = sq; _sk = sk; }

        public static BiasView Of(Tensor? bias, in AttentionCall call)
        {
            if (bias is null) return default;
            if (bias.DType != DType.F32) throw new AttentionUnsupportedException("The host reference reads an F32 bias only.");
            TensorShape s = bias.Shape;
            int rank = s.Rank;
            long skv = s[rank - 1];
            long dq = s[rank - 2];
            long dh = rank >= 3 ? s[rank - 3] : 1;
            long db = rank >= 4 ? s[rank - 4] : 1;
            long sk = skv == 1 ? 0 : 1;
            long sq = dq == 1 ? 0 : skv;
            long sh = dh == 1 ? 0 : dq * skv;
            long sb = db == 1 ? 0 : dh * dq * skv;
            return new BiasView((float*)bias.DataPointer, sb, sh, sq, sk);
        }

        public double At(int b, int h, int r, int j) => _p == null ? 0.0 : _p[b * _sb + h * _sh + r * _sq + j * _sk];
    }
}
