using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Hunyuan3D;
using HartsyInference.ThreeD.Models.Trellis;
using HartsyInference.ThreeD.Models.TripoSr;
using HartsyInference.Tests.Common.Attention;
using Xunit;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.ThreeD.Tests.Attention;

/// <summary>The op sequence of each 3D attention site on the recording stub (no compute): every backend op in order, with the attention call
/// pinned to its scale, F16 assertion, operand shapes, output dtype and mask. The sequences were recorded from the code before the sites moved onto
/// the shared layer and must not change. Ops the stub leaves to the interface's host defaults are not recorded.</summary>
public sealed class AttentionWaveE3TraceTests
{
    private static void Pin(string site, string[] actual)
    {
        if (!AttentionTraceOps.TryRecord(site, actual)) Assert.Equal(AttentionWaveE3TracePins.All[site], actual);
    }

    private static Tensor T(params long[] dims) => new(new TensorShape(dims), DType.F32);

    private static void LayerNormW(Weights w, string name, int n) { w.AddNorm($"{name}.weight", n); w.AddBias($"{name}.bias", n); }

    [Fact]
    public void TripoSrBlock_AttendsSelfThenCrossWithAnF16Assertion()
    {
        const int width = 16, heads = 2, hd = 8, cross = 12, n = 9, s = 5;
        TripoSrConfig cfg = new()
        {
            TriplaneChannels = 4, PlaneSize = 2, NumChannels = width, Width = width, Depth = 1, NumHeads = heads, HeadDim = hd,
            CrossAttentionDim = cross, NerfHidden = 8, NerfMidLayers = 1,
        };
        Weights w = new(300);
        LayerNormW(w, "b.norm1", width); LayerNormW(w, "b.norm2", width); LayerNormW(w, "b.norm3", width);
        foreach (string n1 in new[] { "to_q", "to_k", "to_v" }) w.Add($"b.attn1.{n1}.weight", width, width);
        w.Linear("b.attn1.to_out.0", width, width, true);
        w.Add("b.attn2.to_q.weight", width, width); w.Add("b.attn2.to_k.weight", width, cross); w.Add("b.attn2.to_v.weight", width, cross);
        w.Linear("b.attn2.to_out.0", width, width, true);
        w.Linear("b.ff.net.0.proj", 2 * 24, width, true); w.Linear("b.ff.net.2", width, 24, true);
        TripoSrBlock block = new(cfg);
        block.LoadWeights(w, "b");
        Pin("triposr", AttentionTraceOps.Record(b => block.Forward(b, T(1, n, width), T(1, s, cross))));
    }

    [Fact]
    public void VaeResBlock_ProjectsNormsPermutesAndAttends()
    {
        const int width = 16, heads = 2, hd = 8, n = 9;
        Weights w = new(301);
        LayerNormW(w, "r.ln_1", width); LayerNormW(w, "r.ln_2", width);
        w.Add("r.attn.c_qkv.weight", 3 * width, width); w.Linear("r.attn.c_proj", width, width, true);
        LayerNormW(w, "r.attn.attention.q_norm", hd); LayerNormW(w, "r.attn.attention.k_norm", hd);
        w.Linear("r.mlp.c_fc", 4 * width, width, true); w.Linear("r.mlp.c_proj", width, 4 * width, true);
        Hunyuan3DVaeResBlock block = new(width, heads);
        block.LoadWeights(w, "r");
        Pin("hunyuan3d_vae_resblock", AttentionTraceOps.Record(b => block.Forward(b, T(1, n, width))));
    }

    [Fact]
    public void GeoDecoderQuery_AttendsOverThePrecomputedKeyValueAndKeepsIt()
    {
        const int width = 16, heads = 2, hd = 8, bands = 2, fourier = 3 + 3 * 2 * bands, nl = 10;
        Weights w = new(302);
        w.Linear("g.query_proj", width, fourier, true);
        const string c = "g.cross_attn_decoder";
        foreach (string n in new[] { "ln_1", "ln_2", "ln_3" }) LayerNormW(w, $"{c}.{n}", width);
        w.Add($"{c}.attn.c_q.weight", width, width); w.Add($"{c}.attn.c_kv.weight", 2 * width, width);
        w.Linear($"{c}.attn.c_proj", width, width, true);
        LayerNormW(w, $"{c}.attn.attention.q_norm", hd); LayerNormW(w, $"{c}.attn.attention.k_norm", hd);
        w.Linear($"{c}.mlp.c_fc", 4 * width, width, true); w.Linear($"{c}.mlp.c_proj", width, 4 * width, true);
        LayerNormW(w, "g.ln_post", width); w.Linear("g.output_proj", 1, width, true);
        Hunyuan3DGeoDecoder dec = new(width, heads, bands, fourier);
        dec.LoadWeights(w, "g");
        Tensor kP = T(1, heads, nl, hd), vP = T(1, heads, nl, hd), coords = T(6, 3);
        Pin("hunyuan3d_geo_query", AttentionTraceOps.Record(b =>
        {
            dec.Query(b, (kP, vP), coords, 6);
            dec.Query(b, (kP, vP), coords, 6);   // the second chunk reads the same K/V, so the first must not have freed them
        }));
    }

    [Fact]
    public void DoubleBlock_ProjectsEachStreamThenAttendsJointTextFirst()
    {
        const int width = 16, heads = 2, hd = 8, mlp = 32, nImg = 8, nTxt = 5;
        Weights w = new(303);
        foreach (string s in new[] { "img", "txt" })
        {
            w.Linear($"d.{s}_mod.lin", 6 * width, width, true);
            w.Linear($"d.{s}_attn.qkv", 3 * width, width, true); w.Linear($"d.{s}_attn.proj", width, width, true);
            w.AddNorm($"d.{s}_attn.norm.query_norm.scale", hd); w.AddNorm($"d.{s}_attn.norm.key_norm.scale", hd);
            w.Linear($"d.{s}_mlp.0", mlp, width, true); w.Linear($"d.{s}_mlp.2", width, mlp, true);
        }
        Hunyuan3DDoubleBlock block = new(width, heads, mlp);
        block.LoadWeights(w, "d");
        Pin("hunyuan3d_dit_double", AttentionTraceOps.Record(b => block.Forward(b, T(1, nImg, width), T(1, nTxt, width), T(1, width))));
    }

    [Fact]
    public void SingleBlock_SplitsNormsPermutesAndAttends()
    {
        const int width = 16, heads = 2, hd = 8, mlp = 32, n = 11;
        Weights w = new(304);
        w.Linear("s.modulation.lin", 3 * width, width, true);
        w.Linear("s.linear1", 3 * width + mlp, width, true); w.Linear("s.linear2", width, width + mlp, true);
        w.AddNorm("s.norm.query_norm.scale", hd); w.AddNorm("s.norm.key_norm.scale", hd);
        Hunyuan3DSingleBlock block = new(width, heads, mlp);
        block.LoadWeights(w, "s");
        Pin("hunyuan3d_dit_single", AttentionTraceOps.Record(b => block.Forward(b, T(1, n, width), T(1, width))));
    }

    [Fact]
    public void SsFlowBlock_AttendsSelfThenCrossOverHeadMajorOperands()
    {
        const int w = 1024, hd = 64, mlp = 4096, n = 9, lc = 6;
        Weights wt = new(305);
        wt.AddTensor("b.adaLN_modulation.1.weight", 0f, 6 * w, w); wt.AddBias("b.adaLN_modulation.1.bias", 6 * w);
        LayerNormW(wt, "b.norm2", w);
        wt.AddTensor("b.self_attn.to_qkv.weight", 0f, 3 * w, w); wt.AddBias("b.self_attn.to_qkv.bias", 3 * w);
        wt.AddTensor("b.self_attn.to_out.weight", 0f, w, w); wt.AddBias("b.self_attn.to_out.bias", w);
        wt.AddTensor("b.self_attn.q_rms_norm.gamma", 0f, 16, hd); wt.AddTensor("b.self_attn.k_rms_norm.gamma", 0f, 16, hd);
        wt.AddTensor("b.cross_attn.to_q.weight", 0f, w, w); wt.AddBias("b.cross_attn.to_q.bias", w);
        wt.AddTensor("b.cross_attn.to_kv.weight", 0f, 2 * w, w); wt.AddBias("b.cross_attn.to_kv.bias", 2 * w);
        wt.AddTensor("b.cross_attn.to_out.weight", 0f, w, w); wt.AddBias("b.cross_attn.to_out.bias", w);
        wt.AddTensor("b.mlp.mlp.0.weight", 0f, mlp, w); wt.AddBias("b.mlp.mlp.0.bias", mlp);
        wt.AddTensor("b.mlp.mlp.2.weight", 0f, w, mlp); wt.AddBias("b.mlp.mlp.2.bias", w);
        SsFlowBlock block = SsFlowBlock.Load(wt, "b");
        Tensor ones = T(hd);
        Pin("trellis_ssflow", AttentionTraceOps.Record(b => block.Forward(b, T(1, n, w), T(1, w), T(1, lc, w), ones)));
    }

    [Fact]
    public void GsBlock_ProjectsThroughTheFusedSliceAndAttendsWithTheWindowMask()
    {
        const int w = 768, mlp = 3072, n = 12;
        Weights wt = new(306);
        wt.AddTensor("b.attn.to_qkv.weight", 0f, 3 * w, w); wt.AddBias("b.attn.to_qkv.bias", 3 * w);
        wt.AddTensor("b.attn.to_out.weight", 0f, w, w); wt.AddBias("b.attn.to_out.bias", w);
        wt.AddTensor("b.mlp.mlp.0.weight", 0f, mlp, w); wt.AddBias("b.mlp.mlp.0.bias", mlp);
        wt.AddTensor("b.mlp.mlp.2.weight", 0f, w, mlp); wt.AddBias("b.mlp.mlp.2.bias", w);
        GsBlock block = GsBlock.Load(wt, "b");
        Tensor mask = T(1, 1, n, n);
        Pin("trellis_gs", AttentionTraceOps.Record(b => block.Forward(b, T(1, n, w), mask, n)));
    }
}
