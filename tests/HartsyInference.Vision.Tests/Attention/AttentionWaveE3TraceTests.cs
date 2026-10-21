using System.Reflection;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.TextEncoders.Bert;
using HartsyInference.Vision.Detection;
using HartsyInference.Vision.Detection.GroundingDino;
using HartsyInference.Vision.Dinov2;
using HartsyInference.Vision.DinoVit;
using HartsyInference.Vision.Segmentation;
using HartsyInference.Vision.Segmentation.Sam2;
using HartsyInference.Vision.Siglip;
using HartsyInference.Tests.Common.Attention;
using Xunit;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.Vision.Tests.Attention;

/// <summary>The op sequence of each vision attention site on the recording stub (no compute): every backend op in order, with the attention call
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

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void Dinov2Layer_ProjectsPermutesAttendsAndProjects(int batch)
    {
        const int h = 16, inter = 24;
        Weights w = new(200u + (uint)batch);
        LayerNormW(w, "l.norm1", h); LayerNormW(w, "l.norm2", h);
        foreach (string n in new[] { "query", "key", "value" }) w.Linear($"l.attention.attention.{n}", h, h, true);
        w.Linear("l.attention.output.dense", h, h, true);
        w.AddTensor("l.layer_scale1.lambda1", 0.5f, h); w.AddTensor("l.layer_scale2.lambda1", 0.5f, h);
        w.Linear("l.mlp.fc1", inter, h, true); w.Linear("l.mlp.fc2", h, inter, true);
        Dinov2Layer layer = new(h, 2, inter, 1e-6f);
        layer.LoadWeights(w, "l");
        Pin($"dinov2_b{batch}", AttentionTraceOps.Record(b => layer.Forward(b, T(batch, 7, h))));
    }

    [Fact]
    public void DinoVitLayer_ProjectsPermutesAttendsAndProjects()
    {
        const int h = 16, inter = 24;
        Weights w = new(203);
        LayerNormW(w, "l.layernorm_before", h); LayerNormW(w, "l.layernorm_after", h);
        foreach (string n in new[] { "query", "key", "value" }) w.Linear($"l.attention.attention.{n}", h, h, true);
        w.Linear("l.attention.output.dense", h, h, true);
        w.Linear("l.intermediate.dense", inter, h, true); w.Linear("l.output.dense", h, inter, true);
        DinoViTLayer layer = new(h, 2, inter, 1e-6f);
        layer.LoadWeights(w, "l");
        Pin("dinovit", AttentionTraceOps.Record(b => layer.Forward(b, T(2, 7, h))));
    }

    [Fact]
    public void SiglipLayer_ProjectsPermutesAttendsAndProjects()
    {
        const int h = 16, inter = 24;
        Weights w = new(204);
        LayerNormW(w, "l.layer_norm1", h); LayerNormW(w, "l.layer_norm2", h);
        foreach (string n in new[] { "q_proj", "k_proj", "v_proj", "out_proj" }) w.Linear($"l.self_attn.{n}", h, h, true);
        w.Linear("l.mlp.fc1", inter, h, true); w.Linear("l.mlp.fc2", h, inter, true);
        SiglipTransformerLayer layer = new(h, 2, inter, 1e-6f);
        layer.LoadWeights(w, "l");
        Pin("siglip", AttentionTraceOps.Record(b => layer.Forward(b, T(2, 7, h))));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void SiglipAttentionPool_ProjectsTheProbeAndPatchTokensThroughTheInProjSlices(int batch)
    {
        const int h = 16, patches = 4;
        SiglipVisionEncoder enc = new(new SiglipPreset
        {
            Name = "tiny", HiddenSize = h, NumLayers = 1, NumHeads = 2, IntermediateSize = 24, ImageSize = 8, PatchSize = 4, EmbeddingDim = h,
        });
        void Set(string field, Tensor value) =>
            typeof(SiglipVisionEncoder).GetField(field, BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(enc, value);
        Set("_headProbe", T(1, 1, h));
        Set("_headAttnInProjWeight", T(3 * h, h)); Set("_headAttnInProjBias", T(3 * h));
        Set("_headAttnOutProjWeight", T(h, h)); Set("_headAttnOutProjBias", T(h));
        MethodInfo pool = typeof(SiglipVisionEncoder).GetMethod("AttentionPool", BindingFlags.Instance | BindingFlags.NonPublic)!;
        Pin($"siglip_pool_b{batch}", AttentionTraceOps.Record(b => pool.Invoke(enc, [b, T(batch, patches, h), batch, h])));
    }

    [Theory]
    [InlineData(16, false)]
    [InlineData(8, true)]
    public void Sam2DecoderAttention_ProjectsEachInputThenAttendsFlat(int internalDim, bool cross)
    {
        const int embed = 16, nq = 5, nk = 9;
        Weights w = new(cross ? 205u : 206u);
        w.Linear("a.q_proj", internalDim, embed, true); w.Linear("a.k_proj", internalDim, embed, true);
        w.Linear("a.v_proj", internalDim, embed, true); w.Linear("a.out_proj", embed, internalDim, true);
        DecoderAttention attn = new(2);
        attn.LoadWeights(w, "a.");
        Tensor q = T(2, nq, embed), k = cross ? T(2, nk, embed) : T(2, nq, embed), v = cross ? T(2, nk, embed) : T(2, nq, embed);
        Pin(cross ? "sam2_cross" : "sam2_self", AttentionTraceOps.Record(b => attn.Forward(b, q, k, v)));
    }

    [Theory]
    [InlineData("hiera_windowed", 8, 8, 2, false, 5, 5)]
    [InlineData("hiera_global", 8, 8, 0, false, 4, 4)]
    [InlineData("hiera_pooled", 8, 16, 4, true, 6, 6)]
    public void HieraBlock_AttendsOverHeadMajorOperands(string site, int dimIn, int dimOut, int ws, bool qPool, int h, int w)
    {
        Weights wt = new(207u + (uint)dimOut + (uint)ws);
        LayerNormW(wt, "h.norm1", dimIn); LayerNormW(wt, "h.norm2", dimOut);
        wt.Linear("h.attn.qkv", 3 * dimOut, dimIn, true); wt.Linear("h.attn.proj", dimOut, dimOut, true);
        wt.Linear("h.mlp.layers.0", 4 * dimOut, dimOut, true); wt.Linear("h.mlp.layers.1", dimOut, 4 * dimOut, true);
        if (dimIn != dimOut) wt.Linear("h.proj", dimOut, dimIn, true);
        HieraBlock block = new(dimIn, dimOut, 2, ws, qPool, 4, 1e-6f);
        block.LoadWeights(wt, "h.");
        Pin(site, AttentionTraceOps.Record(b => block.Forward(b, T(1, h, w, dimIn), h, w)));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    public void SwinWindowAttention_AttendsWithTheRelativePositionMask(int shift)
    {
        const int dim = 8, heads = 2, ws = 4, hp = 8, wp = 8;
        using SwinBackbone backbone = new(new GroundingDinoConfig());
        MethodInfo method = typeof(SwinBackbone).GetMethod("WindowAttention", BindingFlags.Instance | BindingFlags.NonPublic)!;
        Tensor q = T(1, hp * wp, dim), k = T(1, hp * wp, dim), v = T(1, hp * wp, dim), rel = T((2 * ws - 1) * (2 * ws - 1), heads);
        Pin($"swin_shift{shift}", AttentionTraceOps.Record(b => method.Invoke(backbone, [b, q, k, v, hp, wp, dim, heads, ws, shift, rel])));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RtDetrMultiHead_PermutesAttendsAndMerges(bool masked)
    {
        const int heads = 2, hd = 8, sq = 6, skv = 6;
        Tensor q = T(1, sq, heads * hd), k = T(1, skv, heads * hd), v = T(1, skv, heads * hd);
        Tensor? mask = masked ? T(1, 1, sq, skv) : null;
        Pin(masked ? "rtdetr_masked" : "rtdetr", AttentionTraceOps.Record(b => RtDetrAttention.MultiHead(b, q, k, v, heads, hd, mask)));
    }

    [Fact]
    public void GroundingDinoMultiheadAttention_ProjectsAttendsAndProjectsInBothRoles()
    {
        const int hidden = 16, heads = 2, nImg = 7, nTxt = 5;
        Weights w = new(210);
        foreach (string n in new[] { "query", "key", "value", "out_proj" }) w.Linear($"m.{n}", hidden, hidden, true);
        using GroundingDinoMultiheadAttention mha = new(hidden, heads);
        mha.LoadWeights(w, "m");
        Tensor img = T(1, nImg, hidden), txt = T(1, nTxt, hidden), pad = T(1, heads, nImg, nTxt);
        Pin("groundingdino_mha", AttentionTraceOps.Record(b =>
        {
            mha.Forward(b, img, txt, txt, pad);
            mha.Forward(b, txt, img, img, null);
        }));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void BertLayer_AttendsOverTheHeadInvariantMask(bool masked)
    {
        const int h = 16, inter = 24, heads = 2, vocab = 20, t = 7;
        BertConfig cfg = new() { Hidden = h, NumLayers = 1, NumHeads = heads, Intermediate = inter, VocabSize = vocab, MaxPositions = 16 };
        Weights w = new(masked ? 211u : 212u);
        w.Add("bert.embeddings.word_embeddings.weight", vocab, h); w.Add("bert.embeddings.position_embeddings.weight", 16, h);
        w.Add("bert.embeddings.token_type_embeddings.weight", 2, h);
        LayerNormW(w, "bert.embeddings.LayerNorm", h);
        const string p = "bert.encoder.layer.0";
        foreach (string n in new[] { "query", "key", "value" }) w.Linear($"{p}.attention.self.{n}", h, h, true);
        w.Linear($"{p}.attention.output.dense", h, h, true); LayerNormW(w, $"{p}.attention.output.LayerNorm", h);
        w.Linear($"{p}.intermediate.dense", inter, h, true); w.Linear($"{p}.output.dense", h, inter, true);
        LayerNormW(w, $"{p}.output.LayerNorm", h);
        using BertModel bert = new(cfg);
        bert.LoadWeights(w);
        int[] ids = [3, 7, 1, 12, 5, 9, 2];
        Tensor? mask = masked ? T(1, heads, t, t) : null;
        Pin(masked ? "bert_masked" : "bert_null", AttentionTraceOps.Record(b => bert.Forward(b, ids, default, default, mask)));
    }

    [Fact]
    public void ClipSegDecoderLayer_ProjectsPermutesAttendsAndProjects()
    {
        const int dim = 64, seq = 5;
        Type layerType = typeof(ClipSegPipeline).GetNestedType("DecoderLayerWeights", BindingFlags.NonPublic)!;
        object layer = Activator.CreateInstance(layerType, nonPublic: true)!;
        void Set(string field, Tensor value) => layerType.GetField(field)!.SetValue(layer, value);
        foreach (string n in new[] { "Q", "K", "V", "O" }) { Set($"{n}W", T(dim, dim)); Set($"{n}B", T(dim)); }
        foreach (string n in new[] { "Ln1", "Ln2" }) { Set($"{n}W", T(dim)); Set($"{n}B", T(dim)); }
        Set("Fc1W", T(2048, dim)); Set("Fc1B", T(2048)); Set("Fc2W", T(dim, 2048)); Set("Fc2B", T(dim));
        ClipSegPipeline pipeline = (ClipSegPipeline)System.Runtime.CompilerServices.RuntimeHelpers.GetUninitializedObject(typeof(ClipSegPipeline));
        MethodInfo method = typeof(ClipSegPipeline).GetMethod("DecoderLayer", BindingFlags.Instance | BindingFlags.NonPublic)!;
        Pin("clipseg_decoder", AttentionTraceOps.Record(b => method.Invoke(pipeline, [b, T(1, seq, dim), layer, seq])));
    }
}
