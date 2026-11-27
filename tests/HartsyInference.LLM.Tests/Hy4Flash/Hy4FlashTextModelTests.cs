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
            // The image literals take the tiny config's image ids (250, 251, 252): the model reads image runs by those ids, and the loader checks them.
            int id = literal switch
            {
                Hy4FlashChatTemplate.ImagePadLiteral => 250,
                Hy4FlashChatTemplate.ImageStartLiteral => 251,
                Hy4FlashChatTemplate.ImageEndLiteral => 252,
                _ => next++,
            };
            added.Add(new { id, content = literal, special = true });
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
    public void NoThinkVariant_IsSelectedByThinkingOffOrALowEffort_AndTheDefaultIsHigh()
    {
        string dir = SyntheticCheckpoint();
        try
        {
            using Hy4FlashTextModel model = Hy4FlashTextModel.Load(new CpuBackend(), dir, 64);
            List<ChatMessage> messages = [new("user", "hello")];
            int[] defaults = model.Template.Encode(model.Tokenizer, messages, true, null, null, null);
            int[] high = model.Template.Encode(model.Tokenizer, messages, true, null, null, 75);
            int[] thinkingOn = model.Template.Encode(model.Tokenizer, messages, true, true, null, 75);
            int[] noThinkLow = model.Template.Encode(model.Tokenizer, messages, true, null, null, 50);
            int[] noThinkMin = model.Template.Encode(model.Tokenizer, messages, true, null, null, 1);
            int[] thinkingOff = model.Template.Encode(model.Tokenizer, messages, true, false, null, null);
            int[] thinkingOffHighEffort = model.Template.Encode(model.Tokenizer, messages, true, false, null, 100);

            Assert.NotEqual(defaults, noThinkLow);
            Assert.Equal(defaults, high);
            Assert.Equal(defaults, thinkingOn);
            Assert.Equal(noThinkLow, noThinkMin);
            Assert.Equal(noThinkLow, thinkingOff);
            Assert.Equal(noThinkLow, thinkingOffHighEffort);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void EncodeWithImages_ExpandsEachPadToItsFeatureRows_BetweenTheStartAndEnd()
    {
        string dir = SyntheticCheckpoint();
        try
        {
            using Hy4FlashTextModel model = Hy4FlashTextModel.Load(new CpuBackend(), dir, 64);
            Hy4FlashChatTemplate template = Assert.IsType<Hy4FlashChatTemplate>(model.Template);
            ChatMessage user = new("user", "look") { Blocks = [new TextBlock("look"), new ImageBlock(0)] };
            int pad = model.Tokenizer.SpecialId(Hy4FlashChatTemplate.ImagePadLiteral) ?? throw new InvalidOperationException("no pad id");
            int start = model.Tokenizer.SpecialId(Hy4FlashChatTemplate.ImageStartLiteral) ?? throw new InvalidOperationException("no start id");
            int end = model.Tokenizer.SpecialId(Hy4FlashChatTemplate.ImageEndLiteral) ?? throw new InvalidOperationException("no end id");
            Hy4FlashConfig config = Hy4FlashConfig.Load(Path.Combine(dir, "config.json"));
            Assert.Equal(config.ImageTokenId, pad);
            Assert.Equal(config.ImageStartTokenId, start);
            Assert.Equal(config.ImageEndTokenId, end);

            // A 4 x 6 patch grid with merge 2: 2 merged rows of 3 columns each plus a newline, so 2 * 4 = 8 feature tokens.
            int[] ids = template.EncodeWithImages(model.Tokenizer, [user], true, null, null, null, [(4, 6)], 2);
            int[] padRun = [.. ids.Select((id, i) => (id, i)).Where(p => p.id == pad).Select(p => p.i)];
            Assert.Equal(8, padRun.Length);
            Assert.Equal(padRun[0], Array.IndexOf(ids, pad));
            Assert.Equal(padRun[^1] - padRun[0] + 1, padRun.Length);
            Assert.Equal(start, ids[padRun[0] - 1]);
            Assert.Equal(end, ids[padRun[^1] + 1]);

            // Text-only turns encode exactly as Encode does.
            List<ChatMessage> text = [new("user", "hello")];
            Assert.Equal(template.Encode(model.Tokenizer, text, true, null, null, null), template.EncodeWithImages(model.Tokenizer, text, true, null, null, null, [], 2));

            // A grid that does not match the pads is refused.
            Assert.Throws<ArgumentException>(() => template.EncodeWithImages(model.Tokenizer, [user], true, null, null, null, [], 2));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void ReasoningState_OpensOnThinkingAndCloses_OnTheHy4Markers()
    {
        string dir = SyntheticCheckpoint();
        try
        {
            using Hy4FlashTextModel model = Hy4FlashTextModel.Load(new CpuBackend(), dir, 64);
            List<ChatMessage> messages = [new("user", "hello")];
            Assert.NotNull(model.Tokenizer.SpecialId(Hy4FlashChatTemplate.ReasoningMarkers[0]));
            Assert.NotNull(model.Tokenizer.SpecialId(Hy4FlashChatTemplate.ReasoningMarkers[1]));

            // The default prompt prefills the think block the template does not close, so the completion starts inside it.
            Hy4FlashChatTemplate template = Assert.IsType<Hy4FlashChatTemplate>(model.Template);
            Assert.True(template.ResolveParserState(model.Tokenizer, messages, null, null, null).ReasoningOpen);
            Assert.True(template.ResolveParserState(model.Tokenizer, messages, true, null, 75).ReasoningOpen);
            // The no_think variant writes the empty pair, so nothing is left open.
            Assert.False(template.ResolveParserState(model.Tokenizer, messages, false, null, null).ReasoningOpen);
            Assert.False(template.ResolveParserState(model.Tokenizer, messages, null, null, 50).ReasoningOpen);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Reset_ClearsTheImageRunFlagAndTheCounters()
    {
        using Hy4FlashSequenceState state = new([], new Hy4FlashMropeRows(2, 8, 10000.0, [2, 0, 0]), new Hy4FlashMropeRows(2, 8, 10000.0, [2, 0, 0])) { EndsInImageRun = true, NextPosition = 5, ImagesUsed = 1 };
        state.Reset();
        Assert.False(state.EndsInImageRun);
        Assert.Equal(0, state.NextPosition);
        Assert.Equal(0, state.ImagesUsed);
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
