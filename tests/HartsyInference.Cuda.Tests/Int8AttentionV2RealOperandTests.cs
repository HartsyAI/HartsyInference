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
/// captured call's four heads and 512 query rows against the double-precision oracle for both V scale modes, the F16-PV row
/// and the call's production row, printed per call. A record, not a gate: the gate is the model's SSIM.</summary>
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
            float[] oracle = c.Oracle();
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
        }
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
