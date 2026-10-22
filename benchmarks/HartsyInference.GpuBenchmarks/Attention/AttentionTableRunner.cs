using System.Diagnostics;
using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.GpuBenchmarks.Attention;

/// <summary>Times one (shape, arm) pair per process and appends TSV rows: the bare kernel call on head-major MHA
/// operands, and for token-major or grouped-query shapes the call sequence the model actually issues.</summary>
/// <remarks>One pair per process keeps constructor-captured knobs, cuDNN failure state and the allocator pool from
/// leaking between arms. Peak VRAM and throttling are sampled by the driver script around the process.</remarks>
public static unsafe class AttentionTableRunner
{
    /// <summary>Column header of the rows this runner writes.</summary>
    public const string Header = "utc\tdevice\tshape\tarm\tcall\tkernel\tstatus\tdtype\tkv_dtype\tlayout\tmask\tcausal\tb\thq\thkv\tsq\tskv\td"
        + "\twarmup\ttrials\tmedian_ms\tp10_ms\tp90_ms\tmin_ms\ttflops\trel_l2\tcos\tmax_abs\toracle_rows\tknobs";

    /// <summary>Entry point: <c>attention-table --shapes F --shape ID --arm NAME --out F [--warmup 3] [--trials 20]
    /// [--oracle-rows 64] [--outlier] [--seed 1234]</c>.</summary>
    public static int Run(string[] args)
    {
        Dictionary<string, string> o = ParseArgs(args);
        AttentionBenchShape s = AttentionBenchShape.Load(o["shapes"]).SingleOrDefault(x => x.Id == o["shape"])
            ?? throw new ArgumentException($"shape '{o["shape"]}' not in {o["shapes"]}");
        AttentionArm arm = AttentionArms.Get(o["arm"]);
        int warmup = int.Parse(o.GetValueOrDefault("warmup", "3"), CultureInfo.InvariantCulture);
        int trials = int.Parse(o.GetValueOrDefault("trials", "20"), CultureInfo.InvariantCulture);
        int oracleRows = int.Parse(o.GetValueOrDefault("oracle-rows", "64"), CultureInfo.InvariantCulture);
        ulong seed = ulong.Parse(o.GetValueOrDefault("seed", "1234"), CultureInfo.InvariantCulture);
        bool outlier = o.ContainsKey("outlier");
        string outPath = o["out"];
        bool newFile = !File.Exists(outPath);
        using StreamWriter w = new(outPath, append: true);
        if (newFile) w.WriteLine(Header);

        Result Na(string call, string status) => new(call, "-", status, warmup, trials, [], double.NaN, double.NaN, double.NaN, double.NaN, 0);
        if (arm.Entry != s.Entry)
        {
            w.WriteLine(Format("-", s, arm, Na("kernel", "n/a-entry"), "-"));
            return 0;
        }

        AttentionArms.ClearAll();
        arm.Apply();
        string knobs = AttentionArms.Provenance();
        string device = "-";
        try
        {
            using BenchmarkFixture fx = new();
            IBackend be = fx.Backend;
            device = string.IsNullOrEmpty(be.Capabilities.DeviceName) ? be.Capabilities.Name : be.Capabilities.DeviceName;
            Inputs inp = Inputs.Create(s, arm, seed, outlier);
            float scale = 1f / MathF.Sqrt(s.D);
            AttentionOracle oracle = new(s, inp.Q, inp.K, inp.V, inp.Mask, scale);
            foreach (Result r in Measure(be, s, arm, inp, scale, oracle, warmup, trials, oracleRows))
                w.WriteLine(Format(device, s, arm, r, knobs));
            return 0;
        }
        catch (Exception ex)
        {
            string msg = (ex.GetType().Name + ":" + ex.Message).Replace('\t', ' ').Replace('\n', ' ');
            w.WriteLine(Format(device, s, arm, Na("kernel", "error:" + msg[..Math.Min(160, msg.Length)]), knobs));
            return 1;
        }
        finally
        {
            AttentionArms.ClearAll();
        }
    }

    private sealed record Result(string Call, string Kernel, string Status, int Warmup, int Trials, double[] Ms,
        double RelL2, double Cos, double MaxAbs, double Tflops, int OracleRows);

    private static IEnumerable<Result> Measure(IBackend be, AttentionBenchShape s, AttentionArm arm, Inputs inp, float scale,
        AttentionOracle oracle, int warmup, int trials, int oracleRows)
    {
        if (s.Entry == "flash")
        {
            using Tensor q = inp.Build(Operand.Q, tokenMajor: false, widen: false, arm.Dtype);
            using Tensor k = inp.Build(Operand.K, tokenMajor: false, widen: false, arm.KvDtype);
            using Tensor v = inp.Build(Operand.V, tokenMajor: false, widen: false, arm.KvDtype);
            using Tensor outT = new(new TensorShape(s.B, s.Hq, s.Sq, s.D), DType.F32);
            Func<Tensor> call = () =>
            {
                be.FlashAttention(outT, q, k, v, kvLen: s.Skv, kvGroup: s.Hq / s.Hkv, causal: s.Causal, qOffset: s.QOffset, scale: scale);
                return outT;
            };
            yield return Time(be, s, arm, "kernel", call, warmup, trials, oracle, oracleRows, tokenMajorOut: false, ownsOutput: false);
            yield break;
        }

        using Tensor? mask = inp.BuildMask();
        // The bare kernel: head-major MHA operands, K/V widened ahead of time for grouped-query shapes.
        using (Tensor qh = inp.Build(Operand.Q, false, false, arm.Dtype))
        using (Tensor kh = inp.Build(Operand.K, false, widen: true, arm.Dtype))
        using (Tensor vh = inp.Build(Operand.V, false, widen: true, arm.Dtype))
        using (Tensor oh = new(new TensorShape(s.B, s.Hq, s.Sq, s.D), arm.Dtype))
        {
            Func<Tensor> kernel = () => { HeadMajor(be, oh, qh, kh, vh, mask, scale, arm); return oh; };
            yield return Time(be, s, arm, "kernel", kernel, warmup, trials, oracle, oracleRows, tokenMajorOut: false, ownsOutput: false);
        }
        if (s.Layout != "tm" && !s.Gqa) yield break;

        // The model path: what a block hands attention, including the permutes and head repeats it pays today.
        bool tm = s.Layout == "tm";
        using Tensor qp = inp.Build(Operand.Q, tm, false, arm.Dtype);
        using Tensor kp = inp.Build(Operand.K, tm, false, arm.Dtype);
        using Tensor vp = inp.Build(Operand.V, tm, false, arm.Dtype);
        Func<Tensor> path = ModelPath(be, s, arm, qp, kp, vp, mask, scale);
        yield return Time(be, s, arm, "path", path, warmup, trials, oracle, oracleRows, tokenMajorOut: tm, ownsOutput: true);
    }

    /// <summary>The sequence a model issues today for this layout and head grouping on this backend.</summary>
    private static Func<Tensor> ModelPath(IBackend be, AttentionBenchShape s, AttentionArm arm, Tensor q, Tensor k, Tensor v,
        Tensor? mask, float scale)
    {
        int g = s.Hq / s.Hkv;
        TensorShape qMh = new(s.B, s.Hq, s.Sq, s.D), kvMh = new(s.B, s.Hkv, s.Skv, s.D), kvRep = new(s.B, s.Hq, s.Skv, s.D);
        TensorShape tmOut = new(s.B, s.Sq, s.Hq, s.D);
        if (s.Layout == "tm" && arm.Grant is { } grant
            && (s.Gqa ? be.SupportsTokenMajorGqaAttention : be.SupportsTokenMajorAttention))
            return () =>
            {
                Tensor o = new(tmOut, arm.Dtype);
                be.Attention(AttentionCall.FromTokenMajor(o, q, k, v, mask, s.Hq, s.Hkv, s.D, scale, arm.AllowF16) with { Numerics = grant });
                return o;
            };
        if (s.Layout == "tm" && s.Gqa && be.SupportsTokenMajorGqaAttention)
            return () => { Tensor o = new(tmOut, arm.Dtype); be.ScaledDotProductAttentionTokenMajor(o, q, k, v, mask, s.Hq, s.Hkv, s.D, scale, arm.AllowF16); return o; };
        if (s.Layout == "tm" && !s.Gqa && be.SupportsTokenMajorAttention)
            return () => { Tensor o = new(tmOut, arm.Dtype); be.ScaledDotProductAttentionTokenMajor(o, q, k, v, mask, s.Hq, s.D, scale, arm.AllowF16); return o; };
        if (s.Layout == "tm")
        {
            // Krea2's AttendHeadMajor: three permutes in, two head repeats, SDPA, one permute out.
            return () =>
            {
                Tensor qh = new(qMh, arm.Dtype); be.Permute0213(qh, q, s.Sq, s.Hq, s.D);
                Tensor kh = new(kvMh, arm.Dtype); be.Permute0213(kh, k, s.Skv, s.Hkv, s.D);
                Tensor vh = new(kvMh, arm.Dtype); be.Permute0213(vh, v, s.Skv, s.Hkv, s.D);
                Tensor kr = kh, vr = vh;
                if (g > 1)
                {
                    kr = new(kvRep, arm.Dtype); be.RepeatKvHeads(kr, kh, s.Hkv, g); kh.Dispose();
                    vr = new(kvRep, arm.Dtype); be.RepeatKvHeads(vr, vh, s.Hkv, g); vh.Dispose();
                }
                Tensor oh = new(qMh, arm.Dtype);
                HeadMajor(be, oh, qh, kr, vr, mask, scale, arm);
                qh.Dispose(); kr.Dispose(); vr.Dispose();
                Tensor o = new(tmOut, arm.Dtype);
                be.Permute0213(o, oh, s.Hq, s.Sq, s.D);
                oh.Dispose();
                return o;
            };
        }
        // Head-major grouped-query: repeat K/V heads on the device, then SDPA.
        return () =>
        {
            Tensor kr = new(kvRep, arm.Dtype); be.RepeatKvHeads(kr, k, s.Hkv, g);
            Tensor vr = new(kvRep, arm.Dtype); be.RepeatKvHeads(vr, v, s.Hkv, g);
            Tensor o = new(qMh, arm.Dtype);
            HeadMajor(be, o, q, kr, vr, mask, scale, arm);
            kr.Dispose(); vr.Dispose();
            return o;
        };
    }

    /// <summary>The head-major call: the legacy entry, or the descriptor when the arm carries a grant the legacy flag cannot.</summary>
    private static void HeadMajor(IBackend be, Tensor o, Tensor q, Tensor k, Tensor v, Tensor? mask, float scale, AttentionArm arm)
    {
        if (arm.Grant is { } grant)
            be.Attention(AttentionCall.FromScaledDotProductAttention(o, q, k, v, mask, scale, arm.AllowF16) with { Numerics = grant });
        else
            be.ScaledDotProductAttention(o, q, k, v, mask, scale, arm.AllowF16);
    }

    private static Result Time(IBackend be, AttentionBenchShape s, AttentionArm arm, string callName, Func<Tensor> call,
        int warmup, int trials, AttentionOracle oracle, int oracleRows, bool tokenMajorOut, bool ownsOutput)
    {
        Tensor? last = null;
        void Keep(Tensor t)
        {
            if (ownsOutput && last is not null && !ReferenceEquals(last, t)) last.Dispose();
            last = t;
        }
        string kernel = AttentionArms.Identify(be, s.Entry, () => Keep(call()));
        if (callName == "kernel" && arm.Expect is not null && kernel != arm.Expect)
        {
            if (ownsOutput) last?.Dispose();
            return new(callName, kernel, "n/a-kernel", warmup, trials, [], double.NaN, double.NaN, double.NaN, double.NaN, 0);
        }
        for (int i = 0; i < warmup; i++) { Keep(call()); be.Sync(); }
        double[] ms = new double[trials];
        Stopwatch sw = new();
        for (int i = 0; i < trials; i++)
        {
            sw.Restart();
            Tensor t = call();
            be.Sync();
            ms[i] = sw.Elapsed.TotalMilliseconds;
            Keep(t);
        }
        Tensor outT = last!;
        Func<int, int, int, int, double> read = Reader(outT, s, tokenMajorOut);
        (double rel, double cos, double maxAbs, int rows) = oracle.Compare(read, oracleRows);
        if (ownsOutput) outT.Dispose();
        double median = Percentile(ms, 50);
        return new(callName, kernel, "ok", warmup, trials, ms, rel, cos, maxAbs, s.Flops / (median * 1e-3) / 1e12, rows);
    }

    /// <summary>Reads one output element as double from a head-major <c>[B,H,S,D]</c> or token-major <c>[B,S,H,D]</c> tensor.</summary>
    private static Func<int, int, int, int, double> Reader(Tensor t, AttentionBenchShape s, bool tokenMajor)
    {
        void* p = t.DataPointer;
        DType dt = t.DType;
        return (b, h, row, c) =>
        {
            long idx = tokenMajor
                ? (((long)b * s.Sq + row) * s.Hq + h) * s.D + c
                : (((long)b * s.Hq + h) * s.Sq + row) * s.D + c;
            if (dt == DType.F32) return ((float*)p)[idx];
            if (dt == DType.F16) return (double)BitConverter.UInt16BitsToHalf(((ushort*)p)[idx]);
            if (dt == DType.BF16) return BitConverter.UInt32BitsToSingle((uint)((ushort*)p)[idx] << 16);
            throw new NotSupportedException(dt.ToString());
        };
    }

    private static double Percentile(double[] xs, double pct)
    {
        if (xs.Length == 0) return double.NaN;
        double[] sorted = xs.OrderBy(x => x).ToArray();
        double rank = pct / 100.0 * (sorted.Length - 1);
        int lo = (int)Math.Floor(rank), hi = (int)Math.Ceiling(rank);
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo);
    }

    private static string Format(string device, AttentionBenchShape s, AttentionArm arm, Result r, string knobs)
    {
        static string F(double x, string fmt) => double.IsNaN(x) ? "NA" : x.ToString(fmt, CultureInfo.InvariantCulture);
        string[] cols =
        [
            DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ", CultureInfo.InvariantCulture), device, s.Id, arm.Name, r.Call, r.Kernel, r.Status,
            arm.Dtype.ToString(), arm.KvDtype.ToString(), s.Layout, s.Mask, s.Causal ? "1" : "0",
            s.B.ToString(CultureInfo.InvariantCulture), s.Hq.ToString(CultureInfo.InvariantCulture), s.Hkv.ToString(CultureInfo.InvariantCulture),
            s.Sq.ToString(CultureInfo.InvariantCulture), s.Skv.ToString(CultureInfo.InvariantCulture), s.D.ToString(CultureInfo.InvariantCulture),
            r.Warmup.ToString(CultureInfo.InvariantCulture), r.Trials.ToString(CultureInfo.InvariantCulture),
            F(Percentile(r.Ms, 50), "F3"), F(Percentile(r.Ms, 10), "F3"), F(Percentile(r.Ms, 90), "F3"), F(r.Ms.Length > 0 ? r.Ms.Min() : double.NaN, "F3"),
            F(r.Tflops, "F2"), F(r.RelL2, "E3"), F(r.Cos, "F7"), F(r.MaxAbs, "E3"), r.OracleRows.ToString(CultureInfo.InvariantCulture), knobs,
        ];
        return string.Join('\t', cols);
    }

    private static Dictionary<string, string> ParseArgs(string[] args)
    {
        Dictionary<string, string> o = new(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i++)
        {
            if (!args[i].StartsWith("--", StringComparison.Ordinal)) throw new ArgumentException($"unexpected argument '{args[i]}'");
            string key = args[i][2..];
            bool flag = i + 1 >= args.Length || args[i + 1].StartsWith("--", StringComparison.Ordinal);
            o[key] = flag ? "1" : args[++i];
        }
        foreach (string req in new[] { "shapes", "shape", "arm", "out" })
            if (!o.ContainsKey(req)) throw new ArgumentException($"--{req} is required");
        return o;
    }

    private enum Operand { Q, K, V }

    /// <summary>Canonical head-major host inputs, already rounded to the arm's storage precision so the oracle sees the
    /// exact values the kernel sees. F32 arms reuse the F16-rounded values, so F16 and F32 arms time identical data.</summary>
    private sealed class Inputs
    {
        public required AttentionBenchShape S { get; init; }
        public required float[] Q { get; init; }
        public required float[] K { get; init; }
        public required float[] V { get; init; }
        public float[]? Mask { get; init; }

        public static Inputs Create(AttentionBenchShape s, AttentionArm arm, ulong seed, bool outlier)
        {
            Func<float, float> round = arm.Dtype == DType.BF16 || arm.KvDtype == DType.BF16 ? RoundBf16 : RoundF16;
            float[] q = Uniform((long)s.B * s.Hq * s.Sq * s.D, seed * 7 + 1, round);
            float[] k = Uniform((long)s.B * s.Hkv * s.Skv * s.D, seed * 7 + 2, round);
            float[] v = Uniform((long)s.B * s.Hkv * s.Skv * s.D, seed * 7 + 3, round);
            if (outlier)
            {
                // Four K channels offset by +8: the channel-outlier case K smoothing exists for.
                int[] channels = [0, s.D / 4, s.D / 2, 3 * s.D / 4];
                for (long row = 0; row < (long)s.B * s.Hkv * s.Skv; row++)
                    foreach (int c in channels) k[row * s.D + c] = round(k[row * s.D + c] + 8f);
            }
            float[]? mask = s.Mask switch
            {
                "bias-head" => Uniform((long)s.Hq * s.Sq * s.Skv, seed * 7 + 4, x => 4f * x),
                "bias-key" => KeyPadding(s.Skv),
                _ => null,
            };
            return new() { S = s, Q = q, K = k, V = v, Mask = mask };
        }

        /// <summary>Key padding: the last tenth of the keys masked with the large finite bias the padded models use.</summary>
        private static float[] KeyPadding(int skv)
        {
            float[] m = new float[skv];
            for (int j = skv - Math.Max(1, skv / 10); j < skv; j++) m[j] = -1e30f;
            return m;
        }

        public Tensor? BuildMask()
        {
            if (Mask is null) return null;
            TensorShape shape = S.Mask == "bias-head" ? new(1, S.Hq, S.Sq, S.Skv) : new(1, 1, 1, S.Skv);
            Tensor t = new(shape, DType.F32);
            Mask.AsSpan().CopyTo(new Span<float>(t.DataPointer, Mask.Length));
            return t;
        }

        /// <summary>Builds a device-uploadable tensor of one operand: head-major or token-major, K/V widened to Hq heads when asked.</summary>
        public Tensor Build(Operand which, bool tokenMajor, bool widen, DType dt)
        {
            (float[] src, int heads, int seq) = which switch
            {
                Operand.Q => (Q, S.Hq, S.Sq),
                _ => (which == Operand.K ? K : V, S.Hkv, S.Skv),
            };
            int outHeads = which != Operand.Q && widen ? S.Hq : heads;
            int group = outHeads / heads;
            int b = S.B, d = S.D;
            Tensor t = new(tokenMajor ? new TensorShape(b, seq, outHeads, d) : new TensorShape(b, outHeads, seq, d), dt);
            void* dst = t.DataPointer;
            Parallel.For(0, b * outHeads, bh =>
            {
                int bi = bh / outHeads, h = bh % outHeads, srcH = h / group;
                for (int sIdx = 0; sIdx < seq; sIdx++)
                {
                    long from = (((long)bi * heads + srcH) * seq + sIdx) * d;
                    long to = tokenMajor ? (((long)bi * seq + sIdx) * outHeads + h) * d : (((long)bi * outHeads + h) * seq + sIdx) * d;
                    if (dt == DType.F32)
                        for (int c = 0; c < d; c++) ((float*)dst)[to + c] = src[from + c];
                    else if (dt == DType.F16)
                        for (int c = 0; c < d; c++) ((ushort*)dst)[to + c] = BitConverter.HalfToUInt16Bits((Half)src[from + c]);
                    else if (dt == DType.BF16)
                        for (int c = 0; c < d; c++) ((ushort*)dst)[to + c] = (ushort)(BitConverter.SingleToUInt32Bits(src[from + c]) >> 16);
                    else
                        throw new NotSupportedException(dt.ToString());
                }
            });
            return t;
        }

        private static float RoundF16(float x) => (float)(Half)x;

        private static float RoundBf16(float x)
        {
            uint u = BitConverter.SingleToUInt32Bits(x);
            u += 0x7FFFu + ((u >> 16) & 1u);
            return BitConverter.UInt32BitsToSingle(u & 0xFFFF0000u);
        }

        /// <summary>Uniform [-1,1) from SplitMix64, generated in parallel chunks with per-chunk seeds (deterministic).</summary>
        private static float[] Uniform(long n, ulong seed, Func<float, float> map)
        {
            float[] a = new float[n];
            const long Chunk = 1 << 20;
            long chunks = (n + Chunk - 1) / Chunk;
            Parallel.For(0, chunks, ci =>
            {
                ulong x = seed ^ ((ulong)ci * 0xD1B54A32D192ED03UL);
                long end = Math.Min(n, (ci + 1) * Chunk);
                for (long i = ci * Chunk; i < end; i++)
                {
                    x += 0x9E3779B97F4A7C15UL;
                    ulong z = x;
                    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
                    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
                    z ^= z >> 31;
                    a[i] = map((float)((z >> 40) * (1.0 / (1UL << 24))) * 2f - 1f);
                }
            });
            return a;
        }
    }
}
