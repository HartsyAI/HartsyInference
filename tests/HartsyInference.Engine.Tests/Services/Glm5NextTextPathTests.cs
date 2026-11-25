using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Exceptions;
using HartsyInference.Engine.Services;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Glm5Next;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>The glm5_next family through the host-reference Text path: a directory made of the tiny reference checkpoint (transformers 52bac5d0 logits saved
/// beside it), the real Sage chat template and a byte-level tokenizer from the ERNIE synthetic checkpoint (ids below the tiny model's 128 for ASCII).</summary>
public sealed class Glm5NextTextPathTests : IDisposable
{
    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Glm5Next");
    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-glm5next-text-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    [Fact]
    public void Directory_loads_through_the_host_reference_path()
    {
        HfCheckpointInfo info = Probe(WriteDirectory("loads"));
        Assert.Equal(Glm5NextTextModel.ModelType, info.ModelType);
        Assert.True(HfTextDirectoryLoader.IsHostReferenceFamily(info));

        using CpuBackend backend = new();
        using IHostReferenceTextModel model = HfTextDirectoryLoader.LoadHostReference(info, backend);

        Assert.IsType<Glm5NextTextModel>(model);
        Assert.Equal(128, model.Generation.Info.VocabSize);
        Assert.Equal(64, model.Generation.Info.HiddenSize);
    }

    [Fact]
    public void Greedy_first_token_through_the_pipeline_is_the_references_argmax()
    {
        HfCheckpointInfo info = Probe(WriteDirectory("argmax"));
        int[] ids = [.. NpyReaderLong(Path.Combine(Fixtures, "tiny", "input_ids.npy"))];
        float[] logits = NpyReaderFloat(Path.Combine(Fixtures, "tiny", "logits.npy"));
        int vocab = 128, last = ids.Length - 1, expected = 0;
        for (int v = 1; v < vocab; v++)
            if (logits[last * vocab + v] > logits[last * vocab + expected]) expected = v;
        using CpuBackend backend = new();
        using IHostReferenceTextModel model = HfTextDirectoryLoader.LoadHostReference(info, backend);
        TextGenerationPipeline pipeline = new(model.Generation, model.Tokenizer, model.Template);

        GenerationResult result = pipeline.Generate(new GenerationRequest { RawTokenIds = ids, MaxTokens = 1, Sampling = SamplingOptions.GreedyPreset });

        Assert.Equal([expected], result.TokenIds);
    }

    [Fact]
    public void Real_chat_template_encodes_a_conversation_and_generation_is_deterministic()
    {
        HfCheckpointInfo info = Probe(WriteDirectory("chat"));
        using CpuBackend backend = new();
        using IHostReferenceTextModel model = HfTextDirectoryLoader.LoadHostReference(info, backend);
        TextGenerationPipeline pipeline = new(model.Generation, model.Tokenizer, model.Template);
        GenerationRequest request = new()
        {
            Messages = [new ChatMessage("user", "hello there")],
            MaxTokens = 4,
            Sampling = SamplingOptions.GreedyPreset,
        };

        GenerationResult first = pipeline.Generate(request), second = pipeline.Generate(request);

        Assert.NotEmpty(first.TokenIds);
        Assert.All(first.TokenIds, id => Assert.InRange(id, 0, 127));
        Assert.Equal(first.TokenIds, second.TokenIds);
    }

    [Fact]
    public void Control_token_spelling_inside_content_stays_plain_text()
    {
        HfCheckpointInfo info = Probe(WriteDirectory("literal"));
        using CpuBackend backend = new();
        using IHostReferenceTextModel model = HfTextDirectoryLoader.LoadHostReference(info, backend);

        int[] plain = model.Template.Encode(model.Tokenizer, [new ChatMessage("user", "a b")], addGenerationPrompt: true);
        int[] spelled = model.Template.Encode(model.Tokenizer, [new ChatMessage("user", "a </s> b")], addGenerationPrompt: true);

        // The tokenizer's </s> is a special token here; escaped, it must not become that id.
        Assert.DoesNotContain(model.Tokenizer.SpecialId("</s>")!.Value, spelled);
        // The spelling itself must be encoded as its ordinary ids, in order, inside the content.
        int[] ordinary = model.Tokenizer.EncodeOrdinary("</s>");
        Assert.True(ordinary.Length > 0);
        bool found = false;
        for (int i = 0; i + ordinary.Length <= spelled.Length && !found; i++)
            found = spelled.AsSpan(i, ordinary.Length).SequenceEqual(ordinary);
        Assert.True(found, "The control-token spelling in content did not encode as its ordinary ids.");
    }

    [Theory]
    [InlineData("\"qk_rope_head_dim\": 0", "\"qk_rope_head_dim\": 64", "NoPE")]
    [InlineData("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"", "silu")]
    public void Text_path_refuses_an_unrunnable_config_by_name(string find, string replace, string named)
    {
        string dir = WriteDirectory("refuses-" + named);
        string config = File.ReadAllText(Path.Combine(dir, "config.json"));
        Assert.Contains(find, config);
        File.WriteAllText(Path.Combine(dir, "config.json"), config.Replace(find, replace));

        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => HfTextDirectoryLoader.RequireSupported(Probe(dir)));

        Assert.Contains(named, ex.Message);
    }

    [Fact]
    public void Text_path_refuses_a_directory_without_a_chat_template()
    {
        string dir = WriteDirectory("no-template");
        File.Delete(Path.Combine(dir, "chat_template.jinja"));
        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => HfTextDirectoryLoader.RequireSupported(Probe(dir)));
        Assert.Contains("chat_template.jinja", ex.Message);
    }

    [Fact]
    public void Placement_refuses_tensor_parallelism_and_several_devices()
    {
        TextService.RequireRunnableGlm5NextPlacement(["cpu"], 1);
        Assert.Throws<NotSupportedException>(() => TextService.RequireRunnableGlm5NextPlacement(["cpu"], 2));
        Assert.Throws<NotSupportedException>(() => TextService.RequireRunnableGlm5NextPlacement(["cuda:0", "cuda:1"], 1));
    }

    private string WriteDirectory(string name)
    {
        string dir = Path.Combine(_root, name);
        Directory.CreateDirectory(dir);
        foreach (string file in Directory.EnumerateFiles(Path.Combine(Fixtures, "tiny"), "*.safetensors")
                     .Concat(Directory.EnumerateFiles(Path.Combine(Fixtures, "tiny"), "*.json")))
            File.Copy(file, Path.Combine(dir, Path.GetFileName(file)));
        File.Copy(Path.Combine(Fixtures, "sage-s21-flash-chat_template.jinja"), Path.Combine(dir, "chat_template.jinja"));
        // the one merge is moved off "hi" (inside <think>) to "qz", so no prompt id reaches 258, beyond the tiny model's 128-entry vocabulary
        string tokenizer = Ernie5SyntheticCheckpoint.TokenizerJson().Replace("\"hi\":258", "\"qz\":258").Replace("\"h i\"", "\"q z\"");
        File.WriteAllText(Path.Combine(dir, "tokenizer.json"), tokenizer);
        File.WriteAllText(Path.Combine(dir, "tokenizer_config.json"), "{\"bos_token\": \"<s>\", \"eos_token\": \"</s>\"}");
        return dir;
    }

    private static HfCheckpointInfo Probe(string dir) =>
        HfCheckpointDirectory.TryProbe(dir) ?? throw new InvalidOperationException($"'{dir}' is not a Hugging Face checkpoint directory.");

    // Minimal .npy readers for the saved reference arrays (version 1 header, C order).
    private static float[] NpyReaderFloat(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        int offset = 10 + (bytes[8] | (bytes[9] << 8));
        float[] data = new float[(bytes.Length - offset) / sizeof(float)];
        Buffer.BlockCopy(bytes, offset, data, 0, data.Length * sizeof(float));
        return data;
    }

    private static int[] NpyReaderLong(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        int offset = 10 + (bytes[8] | (bytes[9] << 8));
        long[] data = new long[(bytes.Length - offset) / sizeof(long)];
        Buffer.BlockCopy(bytes, offset, data, 0, data.Length * sizeof(long));
        return [.. data.Select(v => (int)v)];
    }
}
