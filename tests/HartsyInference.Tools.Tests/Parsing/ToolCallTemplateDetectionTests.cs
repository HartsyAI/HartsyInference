using System.Text;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.Tests.Common;
using HartsyInference.Tools.Parsing;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Tools.Tests.Parsing;

/// <summary>Tests <see cref="ToolCallFormats.TryDetectFromTemplate"/> against the real <c>tokenizer.chat_template</c>
/// of eight local GGUFs (see <c>Fixtures/ChatTemplates/README.md</c> for exactly what each one proves) plus
/// synthetic templates for the two markers no local checkpoint carries (Llama-3's <c>&lt;|python_tag|&gt;</c>,
/// Mistral's <c>[TOOL_CALLS]</c>) and the failure modes the review that prompted this API named by name.</summary>
public sealed class ToolCallTemplateDetectionTests
{
    private readonly ITestOutputHelper _output;
    public ToolCallTemplateDetectionTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void ToolsReferencedButNoSupportedMarker_IsFalse()
        => Assert.False(ToolCallFormats.TryDetectFromTemplate(
            "{%- if tools %}You have tools available.{%- endif %}", out _));

    [Fact]
    public void HermesMarkerWithoutToolsReference_IsFalse()
        // The envelope alone isn't enough — nothing here ever loops over the caller-supplied list, so there's
        // nothing for the model to have actually been instructed about.
        => Assert.False(ToolCallFormats.TryDetectFromTemplate(
            "<tool_call>\n{\"name\": \"x\", \"arguments\": {}}\n</tool_call>", out _));

    // ── Real fixtures: Hermes (true) ────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Qwen3_4B_RealTemplate_DetectsHermes()
    {
        bool detected = ToolCallFormats.TryDetectFromTemplate(ChatTemplateFixtures.Qwen3_4B, out ToolCallFormat format);
        Assert.True(detected);
        Assert.Equal(ToolCallFormat.Hermes, format);
    }

    // ── Real fixtures: Gemma (true) ─────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Gemma4E2BIt_RealTemplate_DetectsGemma()
    {
        bool detected = ToolCallFormats.TryDetectFromTemplate(ChatTemplateFixtures.Gemma4_E2B_It, out ToolCallFormat format);
        Assert.True(detected);
        Assert.Equal(ToolCallFormat.Gemma, format);
    }

    // ── Real fixtures: false, for family-specific reasons ───────────────────────────────────────────────────

    [Fact]
    public void Qwen35_0_8B_RealTemplate_XmlFunctionArgsDoNotMatchHermes()
        // Qwen3.5 / Qwen3-Coder style: opens with the identical "<tool_call>" tag as the Hermes pair above, but
        // instructs "<function=name><parameter=...>" — neither "name" nor "arguments" appears anywhere in it.
        => Assert.False(ToolCallFormats.TryDetectFromTemplate(ChatTemplateFixtures.Qwen35_0_8B, out _));

    [Fact]
    public void Llama32_1B_RealTemplate_BareJsonConventionIsFalse()
        // A real Llama-3.2 template that references "tools" (and "tools_in_user_message") but never renders
        // "<|python_tag|>" for custom tools — it instructs a bare {"name":..,"parameters":..} object instead,
        // which isn't one of the four supported envelopes. Proves the detector doesn't assume family == format.
        => Assert.False(ToolCallFormats.TryDetectFromTemplate(ChatTemplateFixtures.Llama32_1B_Instruct, out _));

    // ── Synthetic: the two markers no local GGUF happens to carry ──────────────────────────────────────────

    [Fact]
    public void SyntheticLlama3Template_PythonTagWithTools_DetectsLlama3()
    {
        // The trailing {"name": x} is filler, not a Hermes decoy: Hermes needs a literal "<tool_call>" marker,
        // which this template never has, so only the "<|python_tag|>" branch can match it.
        const string template = "{%- if tools %}Environment: ipython{%- endif %}<|python_tag|>{\"name\": x}";
        bool detected = ToolCallFormats.TryDetectFromTemplate(template, out ToolCallFormat format);
        Assert.True(detected);
        Assert.Equal(ToolCallFormat.Llama3, format);
    }

    [Fact]
    public void SyntheticMistralTemplate_ToolCallsWithTools_DetectsMistral()
    {
        const string template = "{%- if tools %}[AVAILABLE_TOOLS]{{ tools }}[/AVAILABLE_TOOLS]{%- endif %}[TOOL_CALLS][{\"name\": x}]";
        bool detected = ToolCallFormats.TryDetectFromTemplate(template, out ToolCallFormat format);
        Assert.True(detected);
        Assert.Equal(ToolCallFormat.Mistral, format);
    }

    // ── Synthetic: the exact failure modes the review named ───────────────────────────────────────────────

    [Fact]
    public void SyntheticGlm45StyleXmlArgKey_IsFalse()
    {
        // GLM-4.5's actual convention per the review: "<tool_call>name\n<arg_key>...". Same opening tag as
        // Hermes, XML arguments instead of JSON — must not be confused with the Hermes envelope.
        const string template = "{% if tools %}Tools: {{ tools | tojson }}{% endif %}"
            + "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>";
        Assert.False(ToolCallFormats.TryDetectFromTemplate(template, out _));
    }

    [Fact]
    public void EscapedQuotesInJinjaStringLiteral_StillDetectHermes()
    {
        // Exactly the Qwen shape: the JSON example is embedded inside a Jinja string literal, so the raw GGUF
        // metadata string carries \" rather than ". TryDetectFromTemplate must normalize before matching.
        const string template = "{%- if tools %}"
            + "{{- \"Call with <tool_call>\\n{\\\"name\\\": <fn>, \\\"arguments\\\": <args>}\\n</tool_call>\" }}"
            + "{%- endif %}";
        bool detected = ToolCallFormats.TryDetectFromTemplate(template, out ToolCallFormat format);
        Assert.True(detected);
        Assert.Equal(ToolCallFormat.Hermes, format);
    }

    // ── Live re-extraction: the committed fixtures must still match the real GGUFs ────────────────────────

    public static TheoryData<string> FixtureFiles()
    {
        TheoryData<string> data = new();
        foreach (string name in ChatTemplateFixtures.SourceGgufPaths.Keys) data.Add(name);
        return data;
    }

    /// <summary>Re-reads <c>tokenizer.chat_template</c> straight from each source GGUF (metadata only — the
    /// loader mmaps the file but tensor bytes are never touched) and asserts it still matches the committed
    /// fixture byte-for-byte, so the two can't silently drift apart. Guarded: skips cleanly (or, under
    /// <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c>, fails loudly) when a path isn't mounted on this machine.</summary>
    [Trait("Category", "Integration")]
    [Theory]
    [MemberData(nameof(FixtureFiles))]
    public void CommittedFixture_MatchesLiveGgufMetadata(string fixtureName)
    {
        string path = ChatTemplateFixtures.SourceGgufPaths[fixtureName];
        if (!RealWeightGate.Require(_output.WriteLine, path)) return;

        using GgufLoader loader = new();
        loader.Load(path);
        string? liveTemplate = loader.Metadata.GetString("tokenizer.chat_template");

        Assert.NotNull(liveTemplate);
        // Bytes, not File.ReadAllText: ReadAllText detects and strips a BOM, which would silently pass even if
        // a future re-extraction introduced one — the fixtures and the README's claim are byte-exact.
        byte[] committed = File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "Fixtures", "ChatTemplates", fixtureName));
        Assert.Equal(committed, Encoding.UTF8.GetBytes(liveTemplate));
    }
}
