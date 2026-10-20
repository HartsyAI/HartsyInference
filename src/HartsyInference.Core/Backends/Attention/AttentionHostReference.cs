using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>docs/ATTENTION_CONTRACT.md computed on the host from strided operands of any storage dtype: the default
/// <see cref="IBackend.Attention"/> for a backend without kernel rows. It reads tensors through
/// <see cref="Tensor.DataPointer"/> and never calls a legacy entry point, so it cannot recurse into a backend's wrappers.</summary>
public static class AttentionHostReference
{
    /// <summary>Runs <paramref name="call"/>. Device positions need a backend to read them and are refused.</summary>
    public static unsafe void Run(in AttentionCall call)
    {
        if (call.Positions.OnDevice)
            throw new AttentionUnsupportedException("The host reference cannot read device positions.");
        AttentionOperand q = call.Query, k = call.Key, v = call.Value, o = call.Output;
        int dqk = call.QkHeadDim, dv = call.ValueHeadDim, group = Math.Max(1, call.KvGroup);
        int kvLen = call.ValidKvLength;
        AttentionMask mask = call.Mask;
        BiasView bias = BiasView.Of(mask.Bias, call);
        float* sinks = call.Logits.Sinks is null ? null : (float*)call.Logits.Sinks.DataPointer;
        float* slopes = call.Logits.AlibiSlopes is null ? null : (float*)call.Logits.AlibiSlopes.DataPointer;
        float softcap = call.Logits.Softcap;
        double[] scores = new double[kvLen];
        double[] acc = new double[dv];
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
                        if (softcap != 0f) s = softcap * Math.Tanh(s / softcap);
                        s += bias.At(b, h, r, j);
                        if (slopes != null) s += slopes[h] * (mask.Causal ? j - qpos : -Math.Abs(j - qpos));
                        scores[j] = s;
                        if (s > max) max = s;
                    }
                    double sink = sinks == null ? double.NegativeInfinity : sinks[h];
                    double m = Math.Max(max, sink);
                    Array.Clear(acc);
                    double sum = double.IsNegativeInfinity(sink) ? 0 : Math.Exp(sink - m);
                    if (!double.IsNegativeInfinity(max))
                        for (int j = 0; j < kvLen; j++)
                        {
                            if (double.IsNegativeInfinity(scores[j])) continue;
                            double p = Math.Exp(scores[j] - m);
                            sum += p;
                            for (int c = 0; c < dv; c++) acc[c] += p * Read(in v, b, hk, j, c);
                        }
                    // A fully excluded row with no sink writes zeros (contract §3).
                    double inv = sum > 0 ? 1.0 / sum : 0.0;
                    for (int c = 0; c < dv; c++) Write(in o, b, h, r, c, (float)(acc[c] * inv));
                }
            }
    }

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

    private static unsafe void Write(in AttentionOperand x, int b, int h, int t, int c, float value)
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
        private readonly long _sb, _sh, _sq;

        private BiasView(float* p, long sb, long sh, long sq) { _p = p; _sb = sb; _sh = sh; _sq = sq; }

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
            long sq = dq == 1 ? 0 : skv;
            long sh = dh == 1 ? 0 : dq * skv;
            long sb = db == 1 ? 0 : dh * dq * skv;
            return new BiasView((float*)bias.DataPointer, sb, sh, sq);
        }

        public double At(int b, int h, int r, int j) => _p == null ? 0.0 : _p[b * _sb + h * _sh + r * _sq + j];
    }
}
