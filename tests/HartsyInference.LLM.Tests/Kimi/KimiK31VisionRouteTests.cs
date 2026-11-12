using System.Linq;
using System.Text;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Imaging;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Kimi;
using HartsyInference.LLM.Kimi.Vision;
using HartsyInference.LLM.Tests.Kimi.Vision;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests.Kimi;

/// <summary>The Kimi K3.1 image route on a synthetic decoder and a synthetic MoonViT-V2 tower: the embedding path against the token path, the splice,
/// the prefill of a spliced image, and the refusals by name.</summary>
public sealed class KimiK31VisionRouteTests : IDisposable
{
    // The media pad sits above the fixture's named specials, inside the reserved range, as the real tokenizer's <|media_pad|> does.
    private const int MediaPadId = 267;

    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-kimi-route-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    [Fact]
    public void Template_encodes_the_image_to_the_media_ids_with_one_pad()
    {
        ChatMessage user = new("user", "hi")
        {
            Blocks = [new ImageBlock(0) { Width = 8, Height = 6 }, new TextBlock("hi")],
        };

        int[] ids = new KimiK31ChatTemplate().Encode(Tokenizer(), [user], addGenerationPrompt: false, enableThinking: false);

        Assert.Equal(1, ids.Count(id => id == MediaPadId));
        int begin = Array.IndexOf(ids, 265), content = Array.IndexOf(ids, 266), pad = Array.IndexOf(ids, MediaPadId), end = Array.IndexOf(ids, 268);
        Assert.True(begin >= 0 && begin < content && content < pad && pad < end, "the media markers are out of order");
    }

    [Fact]
    public void Embedding_path_reproduces_the_token_path_bit_for_bit()
    {
        using CpuBackend backend = new();
        using KimiK31GenerationModel generation = LoadDecoder(backend);
        KimiK31Model decoder = generation.Decoder;
        int hidden = generation.DecoderConfig.Hidden;

        using KimiK31State tokenState = new(generation.DecoderConfig, capacity: 8);
        float[] byToken = decoder.Forward([1, 2, 3], tokenState);

        using KimiK31State embeddingState = new(generation.DecoderConfig, capacity: 8);
        float[] row = new float[hidden];
        float[] byEmbedding = [];
        foreach (int token in new[] { 1, 2, 3 })
        {
            decoder.EmbedTokens([token], row);
            byEmbedding = decoder.ForwardEmbedding(row, embeddingState);
        }

        Assert.Equal(byToken, byEmbedding);
    }

    [Fact]
    public void Splice_writes_each_feature_row_at_its_placeholder_and_leaves_the_text_rows_alone()
    {
        using CpuBackend backend = new();
        using KimiK31GenerationModel generation = LoadDecoder(backend);
        KimiK31Model decoder = generation.Decoder;
        int hidden = generation.DecoderConfig.Hidden, pad = generation.DecoderConfig.Vocab - 1;
        int[] ids = [4, pad, 9, pad, 2];

        using Tensor embeds = new(new TensorShape(1, ids.Length, hidden), DType.F32);
        decoder.EmbedTokens(ids, embeds.AsSpan<float>());
        using Tensor first = new(new TensorShape(1, 1, hidden), DType.F32);
        using Tensor second = new(new TensorShape(1, 1, hidden), DType.F32);
        first.AsSpan<float>().Fill(1f);
        second.AsSpan<float>().Fill(2f);
        Assert.Equal(2, VisionTokenSplice.Apply(embeds, ids, pad, [first, second]));

        ReadOnlySpan<float> rows = embeds.AsReadOnlySpan<float>();
        Assert.All(rows.Slice(1 * hidden, hidden).ToArray(), v => Assert.Equal(1f, v));
        Assert.All(rows.Slice(3 * hidden, hidden).ToArray(), v => Assert.Equal(2f, v));
        int[] text = [4, 9, 2];
        int[] textRows = [0, 2, 4];
        using KimiK31State state = new(generation.DecoderConfig, capacity: 8);
        for (int i = 0; i < text.Length; i++)
        {
            float[] looked = new float[hidden];
            decoder.EmbedTokens([text[i]], looked);
            Assert.Equal(looked, rows.Slice(textRows[i] * hidden, hidden).ToArray());
        }
    }

    [Fact]
    public void Image_prefill_is_finite_and_has_the_token_only_shape()
    {
        using CpuBackend backend = new();
        using KimiK31GenerationModel generation = LoadDecoder(backend);
        KimiVisionInput vision = LoadVision(textHidden: generation.DecoderConfig.Hidden);
        using KimiK31State tokenState = new(generation.DecoderConfig, capacity: 8);
        float[] tokenOnly = generation.Decoder.Forward([1, 2, 3], tokenState);

        KimiK31VisionGenerator generator = new(generation, Tokenizer(), vision);
        float[] withImage = generator.PrefillLogits([1, 2, MediaPadId, 3], [Frame(seed: 3)]);

        Assert.Equal(tokenOnly.Length, withImage.Length);
        Assert.Equal(generation.DecoderConfig.Vocab, withImage.Length);
        foreach (float v in withImage) Assert.True(float.IsFinite(v));
        vision.Dispose();
    }

    [Fact]
    public void Different_images_give_different_logits_so_the_splice_reaches_the_decoder()
    {
        using CpuBackend backend = new();
        using KimiK31GenerationModel generation = LoadDecoder(backend);
        KimiVisionInput vision = LoadVision(textHidden: generation.DecoderConfig.Hidden);
        KimiK31VisionGenerator generator = new(generation, Tokenizer(), vision);

        float[] dark = generator.PrefillLogits([1, MediaPadId, 3], [Frame(seed: 0, bright: false)]);
        float[] light = generator.PrefillLogits([1, MediaPadId, 3], [Frame(seed: 0, bright: true)]);

        bool differs = false;
        for (int i = 0; i < dark.Length; i++) differs |= MathF.Abs(dark[i] - light[i]) > 1e-5f;
        Assert.True(differs, "the image did not change the logits");
        vision.Dispose();
    }

    [Fact]
    public void Prompt_pads_must_match_the_images()
    {
        using CpuBackend backend = new();
        using KimiK31GenerationModel generation = LoadDecoder(backend);
        KimiVisionInput vision = LoadVision(textHidden: generation.DecoderConfig.Hidden);
        KimiK31VisionGenerator generator = new(generation, Tokenizer(), vision);

        Assert.Throws<ArgumentException>(() => generator.PrefillLogits([1, MediaPadId, 3], []));
        Assert.Throws<ArgumentException>(() => generator.PrefillLogits([1, 3], [Frame(seed: 1)]));
        vision.Dispose();
    }

    [Fact]
    public void Vision_input_accepts_a_tower_whose_pad_and_geometry_load()
    {
        string dir = Path.Combine(_root, "vision");
        WriteVisionDirectory(dir, textHidden: 16, withMediaIds: true);

        using KimiVisionInput vision = KimiVisionInput.Load(dir, Tokenizer(dir), textHidden: 16, xtmlTemplate: true);

        Assert.True(vision.Accepts, vision.Refusal);
        Assert.NotNull(vision.Encoder);
        Assert.Equal(MediaPadId, vision.PadId);
        Assert.Equal(2, vision.Image!.PatchSize);
    }

    [Fact]
    public void Vision_input_refuses_by_name_when_the_checkpoint_has_no_media_ids()
    {
        string dir = Path.Combine(_root, "no-media");
        WriteVisionDirectory(dir, textHidden: 16, withMediaIds: false);

        using KimiVisionInput vision = KimiVisionInput.Load(dir, Tokenizer(dir), textHidden: 16, xtmlTemplate: true);

        Assert.False(vision.Accepts);
        Assert.Null(vision.Encoder);
        Assert.Contains("media_placeholder_token_id", vision.Refusal, StringComparison.Ordinal);
        Assert.Contains("vision tower cannot take images", vision.Refusal, StringComparison.Ordinal);
    }

    [Fact]
    public void Vision_input_refuses_when_the_projector_width_differs_from_the_decoder()
    {
        string dir = Path.Combine(_root, "width");
        WriteVisionDirectory(dir, textHidden: 6, withMediaIds: true);

        using KimiVisionInput vision = KimiVisionInput.Load(dir, Tokenizer(dir), textHidden: 16, xtmlTemplate: true);

        Assert.False(vision.Accepts);
        Assert.Contains("projector emits 6", vision.Refusal, StringComparison.Ordinal);
    }

    [Fact]
    public void Generator_refuses_a_vision_input_that_is_not_accepted()
    {
        using CpuBackend backend = new();
        using KimiK31GenerationModel generation = LoadDecoder(backend);
        string dir = Path.Combine(_root, "refused");
        WriteVisionDirectory(dir, textHidden: 16, withMediaIds: false);
        using KimiVisionInput vision = KimiVisionInput.Load(dir, Tokenizer(dir), textHidden: 16, xtmlTemplate: true);

        NotSupportedException refused = Assert.Throws<NotSupportedException>(() => new KimiK31VisionGenerator(generation, Tokenizer(dir), vision));

        Assert.Contains("Kimi K3.1 image input is not supported", refused.Message, StringComparison.Ordinal);
    }

    // The text decoder over a synthetic checkpoint, loaded through the checkpoint reader the engine uses.
    private KimiK31GenerationModel LoadDecoder(CpuBackend backend)
    {
        string dir = Path.Combine(_root, "text");
        if (!Directory.Exists(dir)) KimiSyntheticCheckpoint.Write(dir);
        KimiK31Checkpoint checkpoint = KimiK31Checkpoint.Load(dir, backend, maxPositions: 32);
        return new KimiK31GenerationModel(checkpoint, backend, maxTokens: 32);
    }

    // The vision tower of the tiny config, widened to the decoder's hidden width, written under the names a checkpoint index carries.
    private KimiVisionInput LoadVision(int textHidden)
    {
        string dir = Path.Combine(_root, $"vision-{textHidden}");
        if (!Directory.Exists(dir)) WriteVisionDirectory(dir, textHidden, withMediaIds: true);
        return KimiVisionInput.Load(dir, Tokenizer(dir), textHidden, xtmlTemplate: true);
    }

    private ILlmTokenizer Tokenizer(string? dir = null)
    {
        string path = dir ?? Path.Combine(_root, "tokenizer");
        if (!File.Exists(Path.Combine(path, "tiktoken.model")))
        {
            KimiTiktokenFixture.Write(path);
            File.WriteAllText(Path.Combine(path, "tokenizer_config.json"), TokenizerConfig(withMediaIds: true));
        }
        return KimiK31Tokenizer.Load(path, stopIds: [], bosId: KimiTiktokenFixture.BosId, eosId: KimiTiktokenFixture.EosId);
    }

    // A checkpoint directory's tokenizer, vision config and processor config, with the tower's tensors in its index.
    private static void WriteVisionDirectory(string dir, int textHidden, bool withMediaIds)
    {
        Directory.CreateDirectory(dir);
        KimiTiktokenFixture.Write(dir);
        File.WriteAllText(Path.Combine(dir, "tokenizer_config.json"), TokenizerConfig(withMediaIds));
        KimiVisionConfig tower = KimiVisionSyntheticCheckpoint.Tiny(layers: 1) with { TextHidden = textHidden };
        KimiVisionSyntheticCheckpoint.Write(dir, tower, KimiVisionSyntheticCheckpoint.Values(tower, seed: 31));
        File.WriteAllText(Path.Combine(dir, "config.json"), ConfigJson(tower, withMediaIds));
        File.WriteAllText(Path.Combine(dir, "preprocessor_config.json"),
            "{\"media_proc_cfg\": {\"in_patch_limit\": 65536, \"patch_size\": 2, \"image_mean\": [0.5, 0.5, 0.5], \"image_std\": [0.5, 0.5, 0.5], "
            + "\"merge_kernel_size\": 2, \"fixed_output_tokens\": null, \"patch_limit_on_one_side\": 512}}");
    }

    private static string ConfigJson(KimiVisionConfig c, bool withMediaIds)
    {
        string media = withMediaIds ? $"\"media_placeholder_token_id\": {MediaPadId}, " : string.Empty;
        return "{" + media + "\"model_type\": \"kimi_k31\", \"vision_config\": {"
            + $"\"vt_hidden_size\": {c.Hidden}, \"vt_num_attention_heads\": {c.Heads}, \"qkv_hidden_size\": {c.QkvHidden}, "
            + $"\"vt_intermediate_size\": {c.MlpDim}, \"vt_num_hidden_layers\": {c.Layers}, \"patch_size\": {c.PatchSize}, "
            + $"\"merge_kernel_size\": [{c.MergeSize}, {c.MergeSize}], \"init_pos_emb_height\": {c.InitPosGrid}, "
            + $"\"init_pos_emb_width\": {c.InitPosGrid}, \"text_hidden_size\": {c.TextHidden}, \"projector_ln_eps\": 1e-05}}}}";
    }

    // The fixture's named specials, plus the four media literals when the checkpoint names them.
    private static string TokenizerConfig(bool withMediaIds)
    {
        StringBuilder sb = new("{\"added_tokens_decoder\": {");
        sb.Append("\"259\": {\"content\": \"[BOS]\", \"special\": true}, \"260\": {\"content\": \"[EOS]\", \"special\": true}, ");
        sb.Append("\"261\": {\"content\": \"<|end_of_msg|>\", \"special\": true}, \"262\": {\"content\": \"<|open|>\", \"special\": true}, ");
        sb.Append("\"263\": {\"content\": \"<|close|>\", \"special\": true}, \"264\": {\"content\": \"<|sep|>\", \"special\": true}");
        if (withMediaIds)
        {
            sb.Append(", \"265\": {\"content\": \"<|media_begin|>\", \"special\": true}, \"266\": {\"content\": \"<|media_content|>\", \"special\": true}");
            sb.Append($", \"{MediaPadId}\": {{\"content\": \"<|media_pad|>\", \"special\": true}}");
            sb.Append(", \"268\": {\"content\": \"<|media_end|>\", \"special\": true}");
        }
        sb.Append("}}");
        return sb.ToString();
    }

    // A deterministic RGB frame, 8x6, either dark or light, or patterned by seed.
    private static RawRgbFrame Frame(int seed, bool? bright = null)
    {
        byte[] rgb = new byte[8 * 6 * 3];
        for (int i = 0; i < rgb.Length; i++)
        {
            rgb[i] = bright switch
            {
                true => (byte)255,
                false => (byte)0,
                null => (byte)((i * 31 + seed * 17) % 256),
            };
        }
        return new RawRgbFrame(rgb, 8, 6);
    }
}
