using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Hy4Flash;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>The Hy4 Flash Text path at the loader: a directory is probed as a Hugging Face checkpoint, recognised as a host reference family, and loaded as a
/// <see cref="Hy4FlashTextModel"/>; the refusals name the file or the size. The directory is a tiny synthetic checkpoint built from the reference dump.</summary>
public sealed class Hy4FlashTextPathTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-hy4-text-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private static string RepoDir
    {
        get
        {
            string? dir = AppContext.BaseDirectory;
            while (dir is not null && !Directory.Exists(Path.Combine(dir, "tests", "python-reference", "hy4_flash", "data"))) dir = Path.GetDirectoryName(dir);
            return dir ?? throw new DirectoryNotFoundException("tests/python-reference/hy4_flash/data not found.");
        }
    }

    private static string Hy4Dir(string name) => Path.Combine(RepoDir, "tests", "python-reference", "hy4_flash", name);

    // The loader needs config, weights, tokenizer and template; the tokenizer is the byte-level one the LLM tests build, with the template's own control tokens.
    private string Synthetic()
    {
        string dir = Path.Combine(_root, "hy4");
        Directory.CreateDirectory(dir);
        File.Copy(Hy4Dir("data/config.json"), Path.Combine(dir, "config.json"));
        File.Copy(Hy4Dir("data/weights.safetensors"), Path.Combine(dir, "weights.safetensors"));
        File.Copy(Hy4Dir("chat_template.jinja"), Path.Combine(dir, "chat_template.jinja"));
        string template = File.ReadAllText(Hy4Dir("chat_template.jinja"));
        List<string> literals = [.. System.Text.RegularExpressions.Regex.Matches(template, @"'(<[^']*\{\}[^']*>)'")
            .Select(static m => m.Groups[1].Value.Replace("{}", ":opensource")).Distinct()];
        Dictionary<string, int> vocab = new(StringComparer.Ordinal);
        for (int c = 33; c <= 126; c++) vocab[((char)c).ToString()] = vocab.Count + 3;
        vocab["Ġ"] = vocab.Count + 3;
        vocab["Ċ"] = vocab.Count + 3;
        vocab["ĉ"] = vocab.Count + 3;
        vocab["zq"] = vocab.Count + 3;
        int next = vocab.Count + 3;
        List<object> added = [new { id = 1, content = "<｜hy_start:opensource｜>", special = true }, new { id = 2, content = "<｜hy_end:opensource｜>", special = true }];
        foreach (string literal in literals)
        {
            if (literal is "<｜hy_start:opensource｜>" or "<｜hy_end:opensource｜>") continue;
            // The image literals take the tiny config's image ids (250 to 252): the loader checks them against config.json.
            int id = literal switch
            {
                Hy4FlashChatTemplate.ImagePadLiteral => 250,
                Hy4FlashChatTemplate.ImageStartLiteral => 251,
                Hy4FlashChatTemplate.ImageEndLiteral => 252,
                _ => next++,
            };
            added.Add(new { id, content = literal, special = true });
        }
        File.WriteAllText(Path.Combine(dir, "tokenizer.json"), JsonSerializer.Serialize(new
        {
            version = "1.0",
            added_tokens = added,
            pre_tokenizer = new { type = "ByteLevel", add_prefix_space = false, trim_offsets = true, use_regex = true },
            post_processor = (object?)null,
            decoder = new { type = "ByteLevel" },
            model = new { type = "BPE", vocab, merges = new[] { "z q" }, ignore_merges = false },
        }));
        File.WriteAllText(Path.Combine(dir, "tokenizer_config.json"), JsonSerializer.Serialize(new { bos_token = "<｜hy_start:opensource｜>", eos_token = "<｜hy_end:opensource｜>" }));
        File.WriteAllText(Path.Combine(dir, "generation_config.json"), "{\"eos_token_id\": 2}");
        return dir;
    }

    [Fact]
    public void Probe_RecognisesTheDirectoryAsAHostReferenceFamily_AndLoadsIt()
    {
        string dir = Synthetic();
        HfCheckpointInfo info = HfCheckpointDirectory.TryProbe(dir) ?? throw new InvalidOperationException("not probed as a Hugging Face checkpoint");
        Assert.Equal("hy_v4_flash", info.ModelType);
        Assert.True(HfTextDirectoryLoader.IsHy4Flash(info));
        Assert.True(HfTextDirectoryLoader.IsHostReferenceFamily(info));
        HfTextDirectoryLoader.RequireSupported(info);
        using CpuBackend backend = new();
        using IHostReferenceTextModel model = HfTextDirectoryLoader.LoadHostReference(info, backend);
        Assert.IsType<Hy4FlashTextModel>(model);
        Assert.Equal(1, model.Tokenizer.BosId);
    }

    [Fact]
    public void OpenReasoningCloseId_IsTheHy4CloseMarker_OnlyWhenThinkingIsOpen()
    {
        string dir = Synthetic();
        HfCheckpointInfo info = HfCheckpointDirectory.TryProbe(dir) ?? throw new InvalidOperationException("not probed as a Hugging Face checkpoint");
        using CpuBackend backend = new();
        using IHostReferenceTextModel model = HfTextDirectoryLoader.LoadHostReference(info, backend);
        Hy4FlashTextModel hy4 = Assert.IsType<Hy4FlashTextModel>(model);
        int close = hy4.Tokenizer.SpecialId(Hy4FlashChatTemplate.ReasoningMarkers[1]) ?? throw new InvalidOperationException("no close marker");
        List<TextMessage> messages = [new() { Role = TextRole.User, Content = "hello" }];
        GenerationRequest genRequest = new() { Messages = [new ChatMessage("user", "hello")] };
        TextRequest request = new() { Messages = messages };

        // The default prompt prefills the think block, so the close marker is the reasoning close.
        Assert.Equal(close, TextService.OpenReasoningCloseId(hy4.Template, hy4.Tokenizer, genRequest, request, rawCompletion: false));
        // Thinking switched off renders the empty pair: nothing is open, so no reasoning close.
        TextRequest thinkingOff = request with { EnableThinking = false };
        Assert.Null(TextService.OpenReasoningCloseId(hy4.Template, hy4.Tokenizer, genRequest, thinkingOff, rawCompletion: false));
        // A raw completion never splits.
        Assert.Null(TextService.OpenReasoningCloseId(hy4.Template, hy4.Tokenizer, genRequest, request, rawCompletion: true));
    }

    // The engine route end to end: the synthetic checkpoint is loaded as the request's model on the CPU and the image request runs through RunHy4Vision.
    [Fact]
    public async Task ImageRequest_RunsThroughTheEngineRoute_AndIsDeterministic()
    {
        string dir = Synthetic();
        using InferenceEngine engine = new("cpu");
        ModelSpec spec = new() { Requested = dir, Modality = Modality.Text, LocalPath = dir };
        byte[] rgb = new byte[8 * 8 * 3];
        for (int i = 0; i < rgb.Length; i++) rgb[i] = (byte)((i * 31) % 256);
        TextRequest request = new()
        {
            Messages = [new TextMessage { Role = TextRole.User, Content = "What is this?", Images = [new ImageData { Rgb = rgb, Width = 8, Height = 8 }] }],
            Device = "cpu",
            Temperature = 0,
            MaxTokens = 3,
        };

        TextResult first = await engine.Text.GenerateAsync(spec, request);
        TextResult second = await engine.Text.GenerateAsync(spec, request);
        Assert.Equal(first.Text, second.Text);
        Assert.Equal(first.Reasoning, second.Reasoning);
    }

    [Fact]
    public void FileWithAMalformedConfig_IsRefusedWithTheFieldName()
    {
        string dir = Synthetic();
        File.WriteAllText(Path.Combine(dir, "config.json"), File.ReadAllText(Path.Combine(dir, "config.json")).Replace("\"num_hidden_layers\": 3", "\"num_hidden_layers\": \"three\"", StringComparison.Ordinal));
        HfCheckpointInfo info = HfCheckpointDirectory.TryProbe(dir) ?? throw new InvalidOperationException("not probed");
        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => HfTextDirectoryLoader.RequireSupported(info));
        Assert.Contains("num_hidden_layers", ex.Message, StringComparison.Ordinal);
    }

    // A sparse-looking weights file just over the host limit: the size is what is refused, so its contents need not be a model.
    private string Oversize()
    {
        string dir = Path.Combine(_root, "oversize");
        Directory.CreateDirectory(dir);
        File.Copy(Hy4Dir("data/config.json"), Path.Combine(dir, "config.json"));
        File.WriteAllText(Path.Combine(dir, "tokenizer.json"), "{}");
        byte[] header = [2, 0, 0, 0, 0, 0, 0, 0, (byte)'{', (byte)'}'];
        using (FileStream stream = File.Create(Path.Combine(dir, "weights.safetensors")))
        {
            stream.Write(header);
            stream.SetLength(Hy4FlashTextModel.MaxHostWeightBytes + (1L << 20));
        }
        return dir;
    }

    // The loaded model must survive a refused load: the slot keeps its backend and the path it was loaded from.
    private async Task RefusalKeepsTheLoadedModel(string dir, string expected)
    {
        ModelSpec spec = new() { Requested = dir, Modality = Modality.Text, LocalPath = dir };
        TextRequest request = new()
        {
            Messages = [new TextMessage { Role = TextRole.User, Content = "Hello" }],
            Device = "vulkan:0",
            Temperature = 0,
            MaxTokens = 1,
        };
        using InferenceEngine engine = new("cpu");
        TextService text = (TextService)engine.Text;
        TextDeviceSlot slot = text.EnsureSlot("vulkan:0");
        CpuBackend previous = new();
        slot.Backend = previous;
        slot.LoadedPath = "previous-model-directory";

        HartsyInferenceException error = await Assert.ThrowsAsync<HartsyInferenceException>(() => engine.Text.GenerateAsync(spec, request));

        Assert.Contains(expected, error.Message, StringComparison.Ordinal);
        Assert.Same(previous, slot.Backend);
        Assert.Equal("previous-model-directory", slot.LoadedPath);
    }

    [Fact]
    public async Task OversizeHy4_IsRefusedBeforeTheLoadedModelIsUnloaded()
    {
        await RefusalKeepsTheLoadedModel(Oversize(), "MiB of weights");
    }

    [Fact]
    public async Task Hy4WithoutTokenizer_IsRefusedBeforeTheLoadedModelIsUnloaded()
    {
        string dir = Synthetic();
        File.Delete(Path.Combine(dir, "tokenizer.json"));
        await RefusalKeepsTheLoadedModel(dir, "tokenizer.json");
    }
}
