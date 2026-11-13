using System.Text;
using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Imaging;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Kimi;
using HartsyInference.LLM.Kimi.Vision;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Tests.Kimi.Vision;
using Xunit;

namespace HartsyInference.LLM.Tests.Kimi;

/// <summary>The Kimi K3.1 vision tower loads on the first image request. A text load reads no vision tensor, the image logits match an eagerly
/// loaded tower bit for bit, disposing the model releases the tower, and a broken or missing tower refuses images without failing the
/// text load.</summary>
public sealed class KimiVisionLazyLoadTests : IDisposable
{
    // The media pad sits above the fixture's named specials, as the real tokenizer's <|media_pad|> does.
    private const int MediaPadId = 267, MaxSequence = 32, PublishedGrid = 4;
    private const string TowerShard = "vision.safetensors";
    private const string PreprocessorConfig = "{\"media_proc_cfg\": {\"in_patch_limit\": 65536, \"patch_size\": 2, \"image_mean\": [0.5, 0.5, 0.5], "
        + "\"image_std\": [0.5, 0.5, 0.5], \"merge_kernel_size\": 2, \"fixed_output_tokens\": null, \"patch_limit_on_one_side\": 512}}";

    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-kimi-lazy-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    [Fact]
    public void Text_only_load_reads_no_vision_tensor_and_builds_no_tower()
    {
        string dir = WriteCheckpoint("text-only");
        List<string> reads = [];
        using CpuBackend backend = new();

        using KimiK31TextModel model = KimiK31TextModel.Load(backend, dir, MaxSequence, Recording(reads));

        Assert.NotEmpty(reads);
        Assert.DoesNotContain(reads, IsVisionTensor);
        Assert.False(model.Vision.IsResolved);
        Assert.Equal(KimiSyntheticCheckpoint.Vocab, DecodeLogits(model).Length);
    }

    [Fact]
    public void First_image_request_loads_the_tower_once_and_later_requests_reuse_it_bit_for_bit()
    {
        string dir = WriteCheckpoint("reuse");
        List<string> reads = [];
        using CpuBackend backend = new();
        using KimiK31TextModel model = KimiK31TextModel.Load(backend, dir, MaxSequence, Recording(reads));
        Assert.DoesNotContain(reads, IsVisionTensor);

        float[] first = ImageLogits(model);
        List<string> towerReads = reads.Where(IsVisionTensor).Order(StringComparer.Ordinal).ToList();
        float[] second = ImageLogits(model);

        Assert.Equal(ExpectedTowerNames(), towerReads);
        Assert.Equal(towerReads.Count, reads.Count(IsVisionTensor));

        // The eager reference: the same checkpoint with the tower loaded before any image arrives.
        using KimiK31TextModel eager = KimiK31TextModel.Load(backend, dir, MaxSequence, Recording([]));
        eager.Vision.Route();
        float[] reference = ImageLogits(eager);
        Assert.Equal(reference, first);
        Assert.Equal(reference, second);
    }

    [Fact]
    public void Disposing_the_model_releases_the_tower_and_refuses_later_image_requests()
    {
        string dir = WriteCheckpoint("dispose");
        using CpuBackend backend = new();
        KimiK31TextModel model = KimiK31TextModel.Load(backend, dir, MaxSequence, KimiK31CheckpointWeights.Open);
        _ = ImageLogits(model);
        List<Tensor> owned = [.. ((MoonViTEncoder)model.Vision.Route().Encoder).Weights.OwnedTensors];
        Assert.NotEmpty(owned);

        model.Dispose();

        Assert.All(owned, t => Assert.Throws<ObjectDisposedException>(() => { _ = t.AsReadOnlySpan<float>(); }));
        Assert.Throws<ObjectDisposedException>(() => model.Vision.Route());
    }

    [Fact]
    public void Corrupted_tower_tensors_leave_the_text_load_working_and_refuse_images_by_name()
    {
        // The shard holds the position table at 3x3 while config.json names a 4x4 grid: the shapes disagree, the headers still parse.
        string dir = WriteCheckpoint("corrupt", writtenGrid: PublishedGrid - 1);
        List<string> reads = [];
        using CpuBackend backend = new();
        using KimiK31TextModel model = KimiK31TextModel.Load(backend, dir, MaxSequence, Recording(reads));
        Assert.Equal(KimiSyntheticCheckpoint.Vocab, DecodeLogits(model).Length);
        Assert.DoesNotContain(reads, IsVisionTensor);

        NotSupportedException refused = Assert.Throws<NotSupportedException>(() => ImageLogits(model));
        Assert.Contains("vision tower cannot take images", refused.Message, StringComparison.Ordinal);
        Assert.Contains("vision_tower.patch_embed.pos_emb.weight", refused.Message, StringComparison.Ordinal);
        int readsAfterRefusal = reads.Count(IsVisionTensor);

        Assert.Throws<NotSupportedException>(() => model.Vision.Route());
        Assert.Equal(readsAfterRefusal, reads.Count(IsVisionTensor));
    }

    [Fact]
    public void Missing_tower_tensors_leave_the_text_load_working_and_are_named_on_an_image_request()
    {
        string dir = WriteCheckpoint("missing", withTower: false);
        using CpuBackend backend = new();
        using KimiK31TextModel model = KimiK31TextModel.Load(backend, dir, MaxSequence, KimiK31CheckpointWeights.Open);
        Assert.Equal(KimiSyntheticCheckpoint.Vocab, DecodeLogits(model).Length);

        NotSupportedException refused = Assert.Throws<NotSupportedException>(() => ImageLogits(model));
        Assert.Contains("lacks", refused.Message, StringComparison.Ordinal);
        Assert.Contains("vision_tower.patch_embed.proj.weight", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void A_cancelled_request_stops_before_the_prefill_runs()
    {
        string dir = WriteCheckpoint("cancelled");
        using CpuBackend backend = new();
        using KimiK31TextModel model = KimiK31TextModel.Load(backend, dir, MaxSequence, KimiK31CheckpointWeights.Open);
        KimiK31VisionGenerator generator = new(model.Generation, model.Tokenizer, model.Vision);
        using CancellationTokenSource cancel = new();
        cancel.Cancel();

        Assert.Throws<OperationCanceledException>(() =>
            generator.Generate([1, 2, MediaPadId, 3], [Frame(seed: 3)], new SamplingOptions { Temperature = 0f }, maxTokens: 4, onDelta: null, cancel.Token));
    }

    [Fact]
    public void The_reply_reports_a_token_limit_only_when_it_used_its_whole_budget()
    {
        string dir = WriteCheckpoint("limit");
        using CpuBackend backend = new();
        using KimiK31TextModel model = KimiK31TextModel.Load(backend, dir, MaxSequence, KimiK31CheckpointWeights.Open);
        KimiK31VisionGenerator generator = new(model.Generation, model.Tokenizer, model.Vision);

        // A stop id ends the reply before its budget; a reply that samples a token for every step of a one-token budget hit the limit.
        KimiK31VisionReply reply = generator.Generate([1, 2, MediaPadId, 3], [Frame(seed: 3)], new SamplingOptions { Temperature = 0f }, maxTokens: 1,
            onDelta: null, CancellationToken.None);

        Assert.Equal(reply.Tokens == 1, reply.HitTokenLimit);
    }

    // The engine's route: a generator is built for each image request, and the tower loads inside it on the first one.
    private static float[] ImageLogits(KimiK31TextModel model) =>
        new KimiK31VisionGenerator(model.Generation, model.Tokenizer, model.Vision).PrefillLogits([1, 2, MediaPadId, 3], [Frame(seed: 3)]);

    private static float[] DecodeLogits(KimiK31TextModel model)
    {
        using KimiK31State state = new(model.Generation.DecoderConfig, capacity: 8);
        return model.Generation.Decoder.Forward([1, 2, 3], state);
    }

    private static bool IsVisionTensor(string name) =>
        name.StartsWith("vision_tower.", StringComparison.Ordinal) || name.StartsWith("mm_projector.", StringComparison.Ordinal);

    private static List<string> ExpectedTowerNames() =>
        KimiVisionSyntheticCheckpoint.TensorNames(Tower()).Select(t => t.Name).Order(StringComparer.Ordinal).ToList();

    // The tower of the tiny config, widened to the synthetic decoder's hidden width.
    private static KimiVisionConfig Tower() => KimiVisionSyntheticCheckpoint.Tiny(layers: 1) with { TextHidden = KimiSyntheticCheckpoint.Hidden };

    private static Func<string, IKimiK31WeightStore> Recording(List<string> reads) =>
        directory => new RecordingWeights(KimiK31CheckpointWeights.Open(directory), reads);

    // A combined checkpoint: the text decoder's shard and the tower's shard under one index, with a config that names both. The tower's shard holds
    // a position grid of writtenGrid while the config names the published one.
    private string WriteCheckpoint(string name, bool withTower = true, int writtenGrid = PublishedGrid)
    {
        string dir = Path.Combine(_root, name);
        KimiSyntheticCheckpoint.Write(dir);
        File.WriteAllText(Path.Combine(dir, "config.json"), CombinedConfigJson());
        KimiTiktokenFixture.Write(dir);
        File.WriteAllText(Path.Combine(dir, "tokenizer_config.json"), TokenizerConfig());
        File.WriteAllText(Path.Combine(dir, "preprocessor_config.json"), PreprocessorConfig);
        if (withTower) AddTower(dir, writtenGrid);
        return dir;
    }

    private static void AddTower(string dir, int writtenGrid)
    {
        KimiVisionConfig written = Tower() with { InitPosGrid = writtenGrid };
        string scratch = dir + "-tower";
        KimiVisionSyntheticCheckpoint.Write(scratch, written, KimiVisionSyntheticCheckpoint.Values(written, seed: 31));
        File.Move(Path.Combine(scratch, "model.safetensors"), Path.Combine(dir, TowerShard));
        Directory.Delete(scratch, recursive: true);

        Dictionary<string, string> weightMap = KimiSyntheticCheckpoint.Specs(KimiSyntheticCheckpoint.DefaultConfig, packedExperts: false, bf16Dense: false)
            .ToDictionary(t => t.Name, _ => "model-00001-of-00001.safetensors", StringComparer.Ordinal);
        foreach ((string tensor, _) in KimiVisionSyntheticCheckpoint.TensorNames(written)) weightMap[tensor] = TowerShard;
        File.WriteAllText(Path.Combine(dir, "model.safetensors.index.json"),
            JsonSerializer.Serialize(new Dictionary<string, object> { ["weight_map"] = weightMap }));
    }

    // The text config with the tower's vision_config and the media placeholder id spliced in at the top level.
    private static string CombinedConfigJson()
    {
        KimiVisionConfig t = Tower();
        string vision = $"\"media_placeholder_token_id\": {MediaPadId}, \"vision_config\": {{"
            + $"\"vt_hidden_size\": {t.Hidden}, \"vt_num_attention_heads\": {t.Heads}, \"qkv_hidden_size\": {t.QkvHidden}, "
            + $"\"vt_intermediate_size\": {t.MlpDim}, \"vt_num_hidden_layers\": {t.Layers}, \"patch_size\": {t.PatchSize}, "
            + $"\"merge_kernel_size\": [{t.MergeSize}, {t.MergeSize}], \"init_pos_emb_height\": {PublishedGrid}, "
            + $"\"init_pos_emb_width\": {PublishedGrid}, \"text_hidden_size\": {t.TextHidden}, \"projector_ln_eps\": 1e-05}}, ";
        return KimiSyntheticCheckpoint.ConfigJson(KimiSyntheticCheckpoint.DefaultConfig).Insert(1, vision);
    }

    // The fixture's named specials, plus the four media literals with the pad at its published id.
    private static string TokenizerConfig()
    {
        StringBuilder sb = new("{\"added_tokens_decoder\": {");
        sb.Append("\"259\": {\"content\": \"[BOS]\", \"special\": true}, \"260\": {\"content\": \"[EOS]\", \"special\": true}, ");
        sb.Append("\"261\": {\"content\": \"<|end_of_msg|>\", \"special\": true}, \"262\": {\"content\": \"<|open|>\", \"special\": true}, ");
        sb.Append("\"263\": {\"content\": \"<|close|>\", \"special\": true}, \"264\": {\"content\": \"<|sep|>\", \"special\": true}, ");
        sb.Append("\"265\": {\"content\": \"<|media_begin|>\", \"special\": true}, ");
        sb.Append("\"266\": {\"content\": \"<|media_content|>\", \"special\": true}, ");
        sb.Append($"\"{MediaPadId}\": {{\"content\": \"<|media_pad|>\", \"special\": true}}, ");
        sb.Append("\"268\": {\"content\": \"<|media_end|>\", \"special\": true}}}");
        return sb.ToString();
    }

    // A deterministic 8x6 RGB frame, patterned by seed.
    private static RawRgbFrame Frame(int seed)
    {
        byte[] rgb = new byte[8 * 6 * 3];
        for (int i = 0; i < rgb.Length; i++) rgb[i] = (byte)((i * 31 + seed * 17) % 256);
        return new RawRgbFrame(rgb, 8, 6);
    }

    // A checkpoint store that records the name of every tensor it reads.
    private sealed class RecordingWeights(KimiK31CheckpointWeights inner, List<string> reads) : IKimiK31WeightStore
    {
        public bool Has(string name) => inner.Has(name);

        public Tensor Read(string name)
        {
            reads.Add(name);
            return inner.Read(name);
        }

        public Tensor ReadSlice(string name, int dim, long start, long count) => inner.ReadSlice(name, dim, start, count);

        public void Dispose() => inner.Dispose();
    }
}
