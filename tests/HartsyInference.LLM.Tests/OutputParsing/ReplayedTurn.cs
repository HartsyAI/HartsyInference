using System.Text;
using HartsyInference.LLM.OutputParsing;

namespace HartsyInference.LLM.Tests.OutputParsing;

/// <summary>What a consumer of the event stream reconstructs; must always equal the parser's own <see cref="ParsedAssistant"/>.</summary>
internal sealed class ReplayedTurn
{
    public StringBuilder Reasoning { get; } = new();
    public StringBuilder Content { get; } = new();
    public List<(string Name, string? Namespace, StringBuilder Args, bool Ended)> Calls { get; } = [];
    public HashSet<int> Aborted { get; } = [];
    public int Malformed { get; private set; }
    public int Stops { get; private set; }

    public void Handle(ParsedEvent e)
    {
        switch (e.Kind)
        {
            case ParsedEventKind.ReasoningDelta: Reasoning.Append(e.Text); break;
            case ParsedEventKind.ContentDelta: Content.Append(e.Text); break;
            case ParsedEventKind.ToolCallBegin:
                Assert(e.ToolCallIndex == Calls.Count, "call index out of order");
                Assert(Calls.Count == 0 || Calls[^1].Ended || Aborted.Contains(Calls.Count - 1), "call opened before the previous one ended");
                Calls.Add((e.Text!, e.Namespace, new StringBuilder(), false));
                break;
            case ParsedEventKind.ToolCallArgsDelta:
                Assert(e.ToolCallIndex == Calls.Count - 1, "args delta for a call that is not open");
                Calls[e.ToolCallIndex].Args.Append(e.Text);
                break;
            case ParsedEventKind.ToolCallEnd:
                Assert(e.ToolCallIndex == Calls.Count - 1, "end for a call that is not open");
                Calls[e.ToolCallIndex] = Calls[e.ToolCallIndex] with { Ended = true };
                break;
            case ParsedEventKind.ToolCallAbort:
                Assert(e.ToolCallIndex == Calls.Count - 1 && !Calls[e.ToolCallIndex].Ended, "abort for a call that is not open");
                Assert(Aborted.Add(e.ToolCallIndex), "call aborted twice");
                break;
            case ParsedEventKind.Malformed: Malformed++; break;
            case ParsedEventKind.Stop: Stops++; break;
        }
    }

    public static ParsedAssistant Run(IOutputParser parser, IEnumerable<int> ids, out ReplayedTurn replay)
    {
        ReplayedTurn turn = new();
        foreach (int id in ids) parser.Push(id, turn.Handle);
        parser.Finish(turn.Handle);
        replay = turn;
        return parser.Result;
    }

    private static void Assert(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }
}
