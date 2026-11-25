using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Hy4Flash;
using HartsyInference.LLM.Transformer;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Hy4Flash;

/// <summary>The Hy4 Flash Text path: the text model loaded from a directory, its generation adapter against the Hugging Face dump, and the chat template against jinja2 renders.
/// The directory is a synthetic checkpoint: the tiny-config weights and config of the dump, a byte-level tokenizer whose special tokens are the template's own literals, and the
/// checkpoint's chat template.</summary>
public sealed class Hy4FlashTextModelTests(ITestOutputHelper output)
{
    private static string RepoDir
    {
        get
        {
            string? dir = AppContext.BaseDirectory;
            while (dir is not null && !Directory.Exists(Path.Combine(dir, "tests", "python-reference", "hy4_flash", "data"))) dir = Path.GetDirectoryName(dir);
            return dir ?? throw new DirectoryNotFoundException("tests/python-reference/hy4_flash/data not found.");
        }
    }

    private static string Reference(string name) => Path.Combine(RepoDir, "tests", "python-reference", "hy4_flash", "data", name);

    // Byte-level alphabet for printable ASCII, space, tab and newline; every literal the chat template names gets its own id after them.
    private static string Synthetic()
    {
        string template = File.ReadAllText(Path.Combine(RepoDir, "tests", "python-reference", "hy4_flash", "chat_template.jinja"));
        // The template builds its control tokens as '<...{}...>'.format(HYTK) with HYTK = ':opensource'; expand them the same way.
        List<string> literals = [.. Regex.Matches(template, @"'(<[^']*\{\}[^']*>)'").Select(static m => m.Groups[1].Value.Replace("{}", ":opensource")).Distinct()];
        Dictionary<string, int> vocab = new(StringComparer.Ordinal);
        for (int c = 33; c <= 126; c++) vocab[((char)c).ToString()] = vocab.Count + 3;
        vocab["Ġ"] = vocab.Count + 3;
        vocab["Ċ"] = vocab.Count + 3;
        vocab["ĉ"] = vocab.Count + 3;
        // one merge the templates never produce ("zq"), so the loader accepts a BPE model that still encodes text character by character
        vocab["zq"] = vocab.Count + 3;
        int next = vocab.Count + 3;
        List<object> added = [new { id = 1, content = "<｜hy_start:opensource｜>", special = true }, new { id = 2, content = "<｜hy_end:opensource｜>", special = true }];
        foreach (string literal in literals)
        {
            if (literal is "<｜hy_start:opensource｜>" or "<｜hy_end:opensource｜>") continue;
            added.Add(new { id = next, content = literal, special = true });
            next++;
        }
        if (next > 256) throw new InvalidOperationException($"The synthetic vocabulary needs {next} ids; the tiny config has 256.");
        return JsonSerializer.Serialize(new
        {
            version = "1.0",
            added_tokens = added,
            pre_tokenizer = new { type = "ByteLevel", add_prefix_space = false, trim_offsets = true, use_regex = true },
            post_processor = (object?)null,
            decoder = new { type = "ByteLevel" },
            model = new { type = "BPE", vocab, merges = new[] { "z q" }, ignore_merges = false },
        });
    }

    // A checkpoint directory the Text path loads: the dump's config and weights, the synthetic tokenizer and the template.
    private static string SyntheticCheckpoint()
    {
        string dir = Directory.CreateTempSubdirectory("hy4-text").FullName;
        File.Copy(Reference("config.json"), Path.Combine(dir, "config.json"));
        File.Copy(Reference("weights.safetensors"), Path.Combine(dir, "weights.safetensors"));
        File.Copy(Path.Combine(RepoDir, "tests", "python-reference", "hy4_flash", "chat_template.jinja"), Path.Combine(dir, "chat_template.jinja"));
        File.WriteAllText(Path.Combine(dir, "tokenizer.json"), Synthetic());
        File.WriteAllText(Path.Combine(dir, "tokenizer_config.json"), JsonSerializer.Serialize(new { bos_token = "<｜hy_start:opensource｜>", eos_token = "<｜hy_end:opensource｜>" }));
        File.WriteAllText(Path.Combine(dir, "generation_config.json"), "{\"eos_token_id\": 2}");
        return dir;
    }

    private static float MaxRelative(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        float scale = 1e-30f, worst = 0f;
        foreach (float v in expected) scale = MathF.Max(scale, MathF.Abs(v));
        for (int i = 0; i < expected.Length; i++) worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]) / scale);
        return worst;
    }

    private static float[] Npy(string name) => NpyReader.ReadFloat32(Reference(name)).Data;

    [Fact]
    public void Load_ReadsTheSyntheticCheckpointAndItsSpecialIds()
    {
        string dir = SyntheticCheckpoint();
        try
        {
            using Hy4FlashTextModel model = Hy4FlashTextModel.Load(new CpuBackend(), dir, 64);
            Assert.Equal(1, model.Tokenizer.BosId);
            Assert.Equal(2, model.Tokenizer.EosId);
            Assert.IsType<Hy4FlashChatTemplate>(model.Template);
            Assert.Equal(256, model.Generation.Info.VocabSize);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Prefill_ThroughTheGenerationAdapter_MatchesTheHfDumpLogits()
    {
        string dir = SyntheticCheckpoint();
        try
        {
            using Hy4FlashTextModel model = Hy4FlashTextModel.Load(new CpuBackend(), dir, 64);
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Reference("inputs_text.json")));
            int[] ids = [.. document.RootElement.GetProperty("input_ids").EnumerateArray().Select(static e => e.GetInt32())];
            using ISequenceState state = model.Generation.CreateSequenceState(new SequenceStateOptions { MaxSequenceTokens = 64 });
            using Tensor hidden = model.Generation.Prefill(new PrefillChunk(ids.AsMemory(), 0), state);
            using Tensor logits = model.Generation.ProjectLogits(hidden, ids.Length);
            float worst = MaxRelative(Npy("text__logits.npy"), logits.AsReadOnlySpan<float>());
            output.WriteLine($"generation adapter vs HF dump logits: max relative difference {worst:E2}");
            Assert.True(worst < 1e-5f, $"{worst:E2}");
            Assert.Equal(ids.Length, state.Length);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void RenderedChat_MatchesJinja2ForEveryCase()
    {
        string dir = SyntheticCheckpoint();
        try
        {
            using Hy4FlashTextModel model = Hy4FlashTextModel.Load(new CpuBackend(), dir, 64);
            using JsonDocument cases = JsonDocument.Parse(File.ReadAllText(Reference("chat_renders.json")));
            Regex literal = new("<｜[^｜<>]+｜>|</?[a-z_]+:opensource>");
            foreach (JsonElement c in cases.RootElement.EnumerateArray())
            {
                List<ChatMessage> messages = [.. c.GetProperty("messages").EnumerateArray()
                    .Select(static m => new ChatMessage(m.GetProperty("role").GetString()!, m.GetProperty("content").GetString()!))];
                bool add = c.GetProperty("add_generation_prompt").GetBoolean();
                string? effort = c.GetProperty("reasoning_effort").ValueKind == JsonValueKind.String ? c.GetProperty("reasoning_effort").GetString() : null;
                int? reasoning = effort == "no_think" ? 1 : null;
                int[] ids = model.Template.Encode(model.Tokenizer, messages, add, null, null, reasoning);
                string decoded = model.Tokenizer.Decode(ids);
                string expected = literal.Replace(c.GetProperty("text").GetString()!, string.Empty);
                Assert.Equal(expected, decoded);
            }
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void NoThinkEffort_SelectsTheNoThinkVariant_AndTheDefaultMatchesHigh()
    {
        string dir = SyntheticCheckpoint();
        try
        {
            using Hy4FlashTextModel model = Hy4FlashTextModel.Load(new CpuBackend(), dir, 64);
            List<ChatMessage> messages = [new("user", "hello")];
            int[] defaults = model.Template.Encode(model.Tokenizer, messages, true, null, null, null);
            int[] noThink = model.Template.Encode(model.Tokenizer, messages, true, null, null, 1);
            int[] high = model.Template.Encode(model.Tokenizer, messages, true, null, null, 50);
            Assert.NotEqual(defaults, noThink);
            Assert.Equal(defaults, high);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void WeightsOverTheLimit_AreRefusedByName()
    {
        string dir = SyntheticCheckpoint();
        try
        {
            HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => Hy4FlashTextModel.Load(new CpuBackend(), dir, 64, maxWeightBytes: 1000));
            Assert.Contains("host reference", ex.Message, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void EmbeddingsPrefill_IsRefused()
    {
        string dir = SyntheticCheckpoint();
        try
        {
            using Hy4FlashTextModel model = Hy4FlashTextModel.Load(new CpuBackend(), dir, 64);
            using ISequenceState state = model.Generation.CreateSequenceState(new SequenceStateOptions { MaxSequenceTokens = 64 });
            using Tensor embeds = new(new TensorShape(1, 1, 48), DType.F32);
            Assert.Throws<NotSupportedException>(() =>
            {
                model.Generation.Prefill(new PrefillChunk(new[] { 3 }.AsMemory(), 0, Embeds: embeds), state);
            });
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
