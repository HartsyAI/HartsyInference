using System.Text;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>The special ids a <c>tokenizer.json</c> post-processor wraps a single raw prompt in, as Hugging Face adds them with
/// <c>add_special_tokens=True</c>.</summary>
public sealed class HfPostProcessorSpecialsTests
{
    private const string BosEos = "\"special_tokens\":{\"<s>\":{\"id\":\"<s>\",\"ids\":[1],\"tokens\":[\"<s>\"]},\"</s>\":{\"id\":\"</s>\",\"ids\":[2],\"tokens\":[\"</s>\"]}}";

    [Fact]
    public void No_post_processor_and_a_byte_level_one_add_nothing()
    {
        Assert.Equal((0, 0), Lengths(Read("{}")));
        Assert.Equal((0, 0), Lengths(Read("{\"post_processor\":null}")));
        Assert.Equal((0, 0), Lengths(Read("{\"post_processor\":{\"type\":\"ByteLevel\",\"trim_offsets\":false}}")));
    }

    [Fact]
    public void Template_puts_its_specials_before_and_after_the_text()
    {
        // The ERNIE-5.5 base layout is [end token, $A]; Llama 3 is [begin, $A]; this one wraps both sides.
        (int[] prefix, int[] suffix) = Read(Template("{\"SpecialToken\":{\"id\":\"<s>\",\"type_id\":0}},{\"Sequence\":{\"id\":\"A\",\"type_id\":0}},{\"SpecialToken\":{\"id\":\"</s>\",\"type_id\":0}}"));

        Assert.Equal([1], prefix);
        Assert.Equal([2], suffix);
    }

    [Fact]
    public void Sequence_applies_each_processor_around_the_last_ones_output()
    {
        string inner = TemplateObject("{\"SpecialToken\":{\"id\":\"<s>\",\"type_id\":0}},{\"Sequence\":{\"id\":\"A\",\"type_id\":0}}");
        string outer = TemplateObject("{\"SpecialToken\":{\"id\":\"</s>\",\"type_id\":0}},{\"Sequence\":{\"id\":\"A\",\"type_id\":0}}");
        string json = "{\"post_processor\":{\"type\":\"Sequence\",\"processors\":[{\"type\":\"ByteLevel\"}," + inner + "," + outer + "]}}";

        (int[] prefix, int[] suffix) = Read(json);

        Assert.Equal([2, 1], prefix);
        Assert.Empty(suffix);
    }

    [Fact]
    public void Unsupported_shapes_are_refused_by_name()
    {
        NotSupportedException bert = Assert.Throws<NotSupportedException>(() => Read("{\"post_processor\":{\"type\":\"BertProcessing\",\"sep\":[\"</s>\",2],\"cls\":[\"<s>\",1]}}"));
        Assert.Contains("BertProcessing", bert.Message);
        Assert.Throws<NotSupportedException>(() => Read(Template("{\"SpecialToken\":{\"id\":\"<s>\",\"type_id\":0}}")));
        Assert.Throws<NotSupportedException>(() => Read(Template("{\"Sequence\":{\"id\":\"B\",\"type_id\":0}}")));
        InvalidOperationException missing = Assert.Throws<InvalidOperationException>(() => Read(Template("{\"SpecialToken\":{\"id\":\"<x>\",\"type_id\":0}},{\"Sequence\":{\"id\":\"A\",\"type_id\":0}}")));
        Assert.Contains("<x>", missing.Message);
    }

    [Fact]
    public void A_post_processor_missing_a_required_key_is_refused_by_key_name()
    {
        // InvalidOperationException, not KeyNotFoundException, so a model load's error filter can wrap it with the file name.
        string noSpecials = "{\"post_processor\":{\"type\":\"TemplateProcessing\",\"single\":[{\"Sequence\":{\"id\":\"A\",\"type_id\":0}}],\"pair\":[]}}";
        Assert.Contains("special_tokens", Assert.Throws<InvalidOperationException>(() => Read(noSpecials)).Message);
        Assert.Contains("processors", Assert.Throws<InvalidOperationException>(() => Read("{\"post_processor\":{\"type\":\"Sequence\"}}")).Message);
        string noIds = "{\"post_processor\":{\"type\":\"TemplateProcessing\",\"single\":[{\"SpecialToken\":{\"id\":\"<s>\",\"type_id\":0}},"
            + "{\"Sequence\":{\"id\":\"A\",\"type_id\":0}}],\"pair\":[],\"special_tokens\":{\"<s>\":{\"id\":\"<s>\"}}}}";
        Assert.Contains("ids", Assert.Throws<InvalidOperationException>(() => Read(noIds)).Message);
    }

    private static string TemplateObject(string single) => "{\"type\":\"TemplateProcessing\",\"single\":[" + single + "],\"pair\":[]," + BosEos + "}";

    private static string Template(string single) => "{\"post_processor\":" + TemplateObject(single) + "}";

    private static (int[] Prefix, int[] Suffix) Read(string json)
    {
        using MemoryStream stream = new(Encoding.UTF8.GetBytes(json));
        return HfTokenizerJson.ReadSingleSequenceSpecials(stream);
    }

    private static (int, int) Lengths((int[] Prefix, int[] Suffix) specials) => (specials.Prefix.Length, specials.Suffix.Length);
}
