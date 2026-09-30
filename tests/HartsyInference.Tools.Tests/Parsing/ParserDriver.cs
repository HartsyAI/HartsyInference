using System.Text;
using HartsyInference.Engine.Requests;
using HartsyInference.Tools.Parsing;

namespace HartsyInference.Tools.Tests.Parsing;

/// <summary>Feeds a completion to a <see cref="ToolCallParser"/> in pieces (random 1..<c>maxPiece</c> chars from a seed, or fixed split points) and collects the forwarded text and completed calls, ending with <see cref="ToolCallParser.Flush"/>.</summary>
internal static class ParserDriver
{
    public static (string Forwarded, List<NativeToolCall> Calls) Drive(ToolCallFormat format, string completion, int seed, int maxPiece = 4,
        int maxSpanChars = ToolCallParser.DefaultMaxSpanChars, IEnumerable<string>? knownTools = null)
        => Drive(new ToolCallParser(format, maxSpanChars, knownTools), Pieces(completion, new Random(seed), maxPiece));

    public static (string Forwarded, List<NativeToolCall> Calls) Drive(ToolCallFormat format, params string[] pieces)
        => Drive(new ToolCallParser(format), pieces);

    public static (string Forwarded, List<NativeToolCall> Calls) Drive(ToolCallParser parser, IEnumerable<string> pieces)
    {
        StringBuilder forwarded = new();
        List<NativeToolCall> calls = [];
        foreach (string piece in pieces) Collect(parser.Push(piece), forwarded, calls);
        Collect(parser.Flush(), forwarded, calls);
        return (forwarded.ToString(), calls);
    }

    public static IEnumerable<string> Pieces(string text, Random random, int maxPiece)
    {
        int pos = 0;
        while (pos < text.Length)
        {
            int len = Math.Min(text.Length - pos, random.Next(1, maxPiece + 1));
            yield return text.Substring(pos, len);
            pos += len;
        }
    }

    private static void Collect(ToolCallParseResult result, StringBuilder forwarded, List<NativeToolCall> calls)
    {
        forwarded.Append(result.ForwardText);
        if (result.Calls is { } completed) calls.AddRange(completed);
    }
}
