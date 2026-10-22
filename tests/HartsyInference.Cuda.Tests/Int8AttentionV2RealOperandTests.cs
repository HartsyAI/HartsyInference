using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common;
using HartsyInference.Tests.Common.Attention;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>INT8 attention v2 on real attention operands captured from model runs (<c>diagnostics.attentionCaptureDir</c>): each
/// captured call's four heads and 512 query rows against the double-precision oracle for both V scale modes, the F16-PV row,
/// the call's production row and (F16) Sage v1, printed per call. A record, not a gate: the gate is the model's SSIM.</summary>
[Collection("CudaSerial")]
public sealed class Int8AttentionV2RealOperandTests(ITestOutputHelper output)
{
    private const string CaptureRoot = "/home/hartsy/Desktop/attention-overhaul-research-2026-10-20/qkv-captures";

    [Trait("Category", "Integration")]
    [Fact]
    public void CapturedCalls_AgainstTheOracle()
    {
        if (!RealWeightGate.Require(output.WriteLine, CaptureRoot)) return;
        if (!BackendGate.TryOpen("cuda", output.WriteLine, out IBackend? backend)) return;
        using CudaBackend cuda = (CudaBackend)backend!;
        output.WriteLine("model\tcall\tdtype\tkeys\tpath\trelL2\tcos\tmax|Δ|");
        foreach (string meta in Directory.GetFiles(CaptureRoot, "meta.json", SearchOption.AllDirectories).Order())
        {
            string dir = Path.GetDirectoryName(meta)!;
            using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(meta));
            JsonElement m = doc.RootElement;
            int d = m.GetProperty("headDim").GetInt32(), keys = m.GetProperty("keys").GetInt32();
            if (d != 128) continue;
            DType dt = m.GetProperty("dtype").GetString() == "F32" ? DType.F32 : DType.F16;
            int heads = m.GetProperty("capturedHeads").GetArrayLength(), rows = m.GetProperty("queryRows").GetArrayLength();
            float scale = m.GetProperty("scale").GetSingle();
            bool allowF16 = m.TryGetProperty("allowF16", out JsonElement a) && a.GetBoolean();
            float[] q = Read(Path.Combine(dir, "q.bin"), dt), k = Read(Path.Combine(dir, "k.bin"), dt), v = Read(Path.Combine(dir, "v.bin"), dt);
            Int8V2Spec spec = new()
            {
                Id = Path.GetFileName(dir), Heads = heads, KvHeads = heads, Sq = rows, Skv = keys, HeadDim = d, DType = dt, Scale = scale,
                SampledRows = 0,
            };
            using Int8V2Case c = Int8V2Case.FromArrays(spec, q, k, v);
            float[] oracle = CachedOracle(dir, q, k, v, heads, rows, keys, d, scale);
            string model = Path.GetFileName(Path.GetDirectoryName(dir)) ?? "-";
            void Row(string path, float[] actual)
            {
                (double rel, double cos, double max) = Int8V2Measure.Of(actual, oracle);
                output.WriteLine($"{model}\t{Path.GetFileName(dir)}\t{dt}\t{keys}\t{path}\t{rel:E3}\t{cos:F7}\t{max:E2}");
            }
            foreach ((string name, Int8AttentionV2Pv pv) in new[] { ("int8v2-seq", Int8AttentionV2Pv.U8SequenceScales),
                ("int8v2-tile", Int8AttentionV2Pv.U8TileScales), ("int8v2-f16pv", Int8AttentionV2Pv.F16) })
            {
                using Tensor o = new(new TensorShape(1, heads, rows, d), dt);
                AttentionCall call = c.Call with { Output = AttentionOperand.HeadMajor(o) };
                Row(name, Widen(cuda.RunInt8AttentionV2ForTest(call, pv, cuda.Int8V2BlockRows).Output, dt));
            }
            using Tensor qt = Make(q, new TensorShape(1, heads, rows, d), dt), kt = Make(k, new TensorShape(1, heads, keys, d), dt);
            using Tensor vt = Make(v, new TensorShape(1, heads, keys, d), dt), ot = new(qt.Shape, dt);
            cuda.ScaledDotProductAttention(ot, qt, kt, vt, null, scale, allowF16);
            using Tensor wide = ot.CastTo(DType.F32);
            Row($"production({CudaBackend.LastAttentionRoute})", wide.AsReadOnlySpan<float>().ToArray());
            if (dt != DType.F16) continue;
            // Sage v1 needs both sequences at its engage length, which a 512-row slice is not.
            KnobStore.Set(EngineKnobs.SageAttn, true);
            KnobStore.Set(EngineKnobs.SageF16MinSkv, 1);
            try
            {
                using Tensor o1 = new(qt.Shape, dt);
                cuda.ScaledDotProductAttention(o1, qt, kt, vt, null, scale, allowF16);
                using Tensor w1 = o1.CastTo(DType.F32);
                Row($"sage-v1({CudaBackend.LastAttentionRoute})", w1.AsReadOnlySpan<float>().ToArray());
            }
            finally
            {
                KnobStore.Clear(EngineKnobs.SageAttn);
                KnobStore.Clear(EngineKnobs.SageF16MinSkv);
            }
        }
    }

    /// <summary>The emulator's variants on the same calls (64 sampled rows), host only: K scales per lane class or per key,
    /// 128- or 64-key tiles, u8 or F16 PV.</summary>
    [Trait("Category", "Integration")]
    [Fact]
    public void CapturedCalls_EmulatorVariants()
    {
        if (!RealWeightGate.Require(output.WriteLine, CaptureRoot)) return;
        (string Name, Int8AttentionV2Options Options)[] variants =
        [
            ("class-128-u8tile", new() { ValueScalesPerTile = true }),
            ("perkey-128-u8tile", new() { KeyScales = Int8KeyScaleGrouping.PerKey, ValueScalesPerTile = true }),
            ("class-64-u8tile", new() { BlockCols = 64, ValueScalesPerTile = true }),
            ("class-128-f16pv", new() { Probabilities = Int8ProbabilityFormat.F16 }),
            ("perkey-128-f16pv", new() { KeyScales = Int8KeyScaleGrouping.PerKey, Probabilities = Int8ProbabilityFormat.F16 }),
        ];
        string[] metas = [.. Directory.GetFiles(CaptureRoot, "meta.json", SearchOption.AllDirectories).Order()];
        string[] lines = new string[metas.Length];
        Parallel.For(0, metas.Length, i =>
        {
            string dir = Path.GetDirectoryName(metas[i])!;
            using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(metas[i]));
            JsonElement m = doc.RootElement;
            int d = m.GetProperty("headDim").GetInt32(), keys = m.GetProperty("keys").GetInt32();
            if (d != 128) return;
            DType dt = m.GetProperty("dtype").GetString() == "F32" ? DType.F32 : DType.F16;
            int heads = m.GetProperty("capturedHeads").GetArrayLength(), rows = m.GetProperty("queryRows").GetArrayLength();
            Int8V2Spec spec = new()
            {
                Id = Path.GetFileName(dir), Heads = heads, KvHeads = heads, Sq = rows, Skv = keys, HeadDim = d, DType = dt,
                Scale = m.GetProperty("scale").GetSingle(), SampledRows = 64,
            };
            using Int8V2Case c = Int8V2Case.FromArrays(spec, Read(Path.Combine(dir, "q.bin"), dt), Read(Path.Combine(dir, "k.bin"), dt),
                Read(Path.Combine(dir, "v.bin"), dt));
            float[] oracle = c.Oracle();
            string model = Path.GetFileName(Path.GetDirectoryName(dir)) ?? "-";
            lines[i] = string.Join("\n", variants.Select(v =>
            {
                (double rel, double cos, _) = Int8V2Measure.Of(c.Emulate(v.Options), oracle);
                return $"{model}\t{Path.GetFileName(dir)}\t{v.Name}\t{rel:E3}\t{cos:F7}";
            }));
        });
        output.WriteLine("model\tcall\tvariant\trelL2\tcos");
        foreach (string l in lines) if (l is not null) output.WriteLine(l);
    }

    /// <summary>Fills the oracle cache on the host alone, so the GPU fact holds its card only for the kernels.</summary>
    [Trait("Category", "Integration")]
    [Fact]
    public void CapturedCalls_OracleCache()
    {
        if (!RealWeightGate.Require(output.WriteLine, CaptureRoot)) return;
        foreach (string meta in Directory.GetFiles(CaptureRoot, "meta.json", SearchOption.AllDirectories).Order())
        {
            string dir = Path.GetDirectoryName(meta)!;
            using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(meta));
            JsonElement m = doc.RootElement;
            int d = m.GetProperty("headDim").GetInt32(), keys = m.GetProperty("keys").GetInt32();
            if (d != 128) continue;
            DType dt = m.GetProperty("dtype").GetString() == "F32" ? DType.F32 : DType.F16;
            CachedOracle(dir, Read(Path.Combine(dir, "q.bin"), dt), Read(Path.Combine(dir, "k.bin"), dt), Read(Path.Combine(dir, "v.bin"), dt),
                m.GetProperty("capturedHeads").GetArrayLength(), m.GetProperty("queryRows").GetArrayLength(), keys, d,
                m.GetProperty("scale").GetSingle());
            output.WriteLine($"{dir}: oracle cached");
        }
    }

    /// <summary>Softmax(scale·QKᵀ)V in double per (head, row), kept beside the capture as <c>oracle.f32</c>.</summary>
    private static float[] CachedOracle(string dir, float[] q, float[] k, float[] v, int heads, int rows, int keys, int d, float scale)
    {
        string path = Path.Combine(dir, "oracle.f32");
        if (File.Exists(path) && new FileInfo(path).Length == (long)heads * rows * d * sizeof(float)) return Widen(File.ReadAllBytes(path), DType.F32);
        float[] o = new float[heads * rows * d];
        Parallel.For(0, heads * rows, hr =>
        {
            int h = hr / rows;
            ReadOnlySpan<float> qr = q.AsSpan(hr * d, d);
            double[] s = new double[keys], acc = new double[d];
            double max = double.NegativeInfinity, sum = 0;
            for (int j = 0; j < keys; j++)
            {
                ReadOnlySpan<float> kj = k.AsSpan((h * keys + j) * d, d);
                double dot = 0;
                for (int c = 0; c < d; c++) dot += (double)qr[c] * kj[c];
                s[j] = dot * scale;
                max = Math.Max(max, s[j]);
            }
            for (int j = 0; j < keys; j++)
            {
                double p = Math.Exp(s[j] - max);
                sum += p;
                ReadOnlySpan<float> vj = v.AsSpan((h * keys + j) * d, d);
                for (int c = 0; c < d; c++) acc[c] += p * vj[c];
            }
            for (int c = 0; c < d; c++) o[hr * d + c] = (float)(acc[c] / sum);
        });
        byte[] bytes = new byte[o.Length * sizeof(float)];
        Buffer.BlockCopy(o, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(path, bytes);
        return o;
    }

    private static float[] Read(string path, DType dt)
    {
        byte[] bytes = File.ReadAllBytes(path);
        return Widen(bytes, dt);
    }

    private static float[] Widen(byte[] bytes, DType dt)
    {
        int n = bytes.Length / (dt == DType.F32 ? 4 : 2);
        float[] values = new float[n];
        for (int i = 0; i < n; i++)
            values[i] = dt == DType.F32 ? BitConverter.ToSingle(bytes, 4 * i) : (float)BitConverter.UInt16BitsToHalf(BitConverter.ToUInt16(bytes, 2 * i));
        return values;
    }

    private static Tensor Make(float[] values, TensorShape shape, DType dt)
    {
        Tensor f32 = new(shape, DType.F32);
        values.CopyTo(f32.AsSpan<float>());
        if (dt == DType.F32) return f32;
        using (f32) return f32.CastTo(dt);
    }
}
