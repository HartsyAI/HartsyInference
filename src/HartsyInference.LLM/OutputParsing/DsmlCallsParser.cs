using System.Text;
using System.Text.RegularExpressions;
using HartsyInference.LLM.ChatTemplates;

namespace HartsyInference.LLM.OutputParsing;

/// <summary>Incremental parser for the DSML calls block that follows the calls marker: validates the same grammar as the reference <c>parse_tool_calls</c> while streaming each call's JSON arguments as they arrive.</summary>
internal sealed class DsmlCallsParser
{
    private const string Dsml = DeepSeekV41Tools.DsmlToken;
    private const int MaxHeaderChars = 4096;
    private const string GapText = ">\n";

    private static readonly string InvokeStart = "<" + Dsml + DeepSeekV41Tools.ToolCallTagName;
    private static readonly string InvokeEnd = "</" + Dsml + DeepSeekV41Tools.ToolCallTagName;
    private static readonly string CallsEnd = "</" + Dsml + DeepSeekV41Tools.ToolCallsBlockName + ">";
    private static readonly string ParamStart = "<" + Dsml + DeepSeekV41Tools.ToolParameterTagName;
    private static readonly string ParamEnd = "</" + Dsml + DeepSeekV41Tools.ToolParameterTagName;
    private static readonly string[] GapAfterCallsStops = [InvokeStart, CallsEnd];
    private static readonly string[] HeadStops = [ParamStart, InvokeEnd];
    private static readonly string[] ParamValueStops = [ParamEnd];
    private static readonly Regex NameRegex = new("^\\s*name=\"(.*?)\">\n$", RegexOptions.Singleline | RegexOptions.CultureInvariant);
    private static readonly Regex ParamHeaderRegex =
        new("^ name=\"(.*?)\" string=\"(true|false)\">", RegexOptions.Singleline | RegexOptions.CultureInvariant);

    private enum Phase { GapBeforeInvoke, InvokeHead, ParamHeader, ParamValue, GapAfterParam, AfterCallsEnd, Faulted }

    private readonly List<ChatToolCall> _calls;
    private readonly HashSet<string> _seen = new(StringComparer.Ordinal);
    private readonly StringBuilder _args = new();
    private Phase _phase = Phase.GapBeforeInvoke;
    private string _rest = "";
    private string _name = "";
    private string? _namespace;
    private bool _stringValue;
    private bool _callOpen;

    /// <summary>Creates a parser that appends completed calls to <paramref name="calls"/> (its current count is the first call index).</summary>
    public DsmlCallsParser(List<ChatToolCall> calls) => _calls = calls;

    /// <summary>True once the block was closed and nothing but the end of the turn may follow.</summary>
    public bool Closed => _phase == Phase.AfterCallsEnd && _rest.Length == 0;

    /// <summary>True after a format violation; all further input is swallowed.</summary>
    public bool Faulted => _phase == Phase.Faulted;

    /// <summary>Consumes block text (never the end-of-turn marker).</summary>
    public void Feed(string text, Action<ParsedEvent> sink)
    {
        if (_phase == Phase.Faulted || text.Length == 0) return;
        _rest += text;
        while (_phase != Phase.Faulted && Step(sink)) { }
    }

    /// <summary>The turn ended; a block that is not closed is truncated and its open call is dropped.</summary>
    public void End(Action<ParsedEvent> sink)
    {
        if (_phase == Phase.Faulted || Closed) return;
        Fault(sink, _phase == Phase.AfterCallsEnd ? "Unexpected content after tool calls" : "Tool calls truncated before the calls block closed");
    }

    private bool Step(Action<ParsedEvent> sink)
    {
        switch (_phase)
        {
            case Phase.GapBeforeInvoke: return StepGap(GapAfterCallsStops, sink);
            case Phase.GapAfterParam: return StepGap(HeadStops, sink);
            case Phase.InvokeHead: return StepInvokeHead(sink);
            case Phase.ParamHeader: return StepParamHeader(sink);
            case Phase.ParamValue: return StepParamValue(sink);
            case Phase.AfterCallsEnd:
                if (_rest.Length == 0) return false;
                Fault(sink, "Unexpected content after tool calls");
                return false;
            default: return false;
        }
    }

    private bool StepGap(string[] stops, Action<ParsedEvent> sink)
    {
        MarkerScanner.Scan(_rest, stops, out int at, out int which, out _);
        if (at < 0)
        {
            bool consistent = _rest.Length <= GapText.Length ? GapText.StartsWith(_rest, StringComparison.Ordinal)
                : _rest.StartsWith(GapText, StringComparison.Ordinal);
            if (!consistent || _rest.Length > MaxHeaderChars)
                Fault(sink, $"Tool call format error: expected '>\\n' but got '{_rest}'");
            return false;
        }
        string segment = _rest[..at];
        if (segment != GapText)
        {
            Fault(sink, $"Tool call format error: expected '>\\n' but got '{segment}'");
            return false;
        }
        _rest = _rest[(at + stops[which].Length)..];
        if (stops[which] == CallsEnd) _phase = Phase.AfterCallsEnd;
        else if (stops[which] == InvokeStart) _phase = Phase.InvokeHead;
        else if (stops[which] == ParamStart) _phase = Phase.ParamHeader;
        else CompleteCall(sink);
        return true;
    }

    private bool StepInvokeHead(Action<ParsedEvent> sink)
    {
        MarkerScanner.Scan(_rest, HeadStops, out int at, out int which, out _);
        if (at < 0)
        {
            if (_rest.Length > MaxHeaderChars) Fault(sink, "Tool name header too long");
            return false;
        }
        string segment = _rest[..at];
        Match match = NameRegex.Match(segment);
        if (!match.Success)
        {
            Fault(sink, $"Tool name format error: '{segment}'");
            return false;
        }
        string qualified = match.Groups[1].Value;
        int sep = qualified.IndexOf("::", StringComparison.Ordinal);
        _namespace = sep >= 0 ? qualified[..sep] : null;
        _name = sep >= 0 ? qualified[(sep + 2)..] : qualified;
        if (_name.Contains("::", StringComparison.Ordinal))
        {
            Fault(sink, $"Tool name must not contain '::': {_name}");
            return false;
        }
        _seen.Clear();
        _args.Clear();
        _callOpen = true;
        sink(new ParsedEvent(ParsedEventKind.ToolCallBegin, _name, _calls.Count, _namespace));
        EmitArgs("{", sink);
        string stop = HeadStops[which];
        _rest = _rest[(at + stop.Length)..];
        if (stop == ParamStart) _phase = Phase.ParamHeader;
        else CompleteCall(sink);
        return true;
    }

    private bool StepParamHeader(Action<ParsedEvent> sink)
    {
        Match match = ParamHeaderRegex.Match(_rest);
        if (!match.Success)
        {
            const string prefix = " name=\"";
            bool consistent = _rest.Length <= prefix.Length ? prefix.StartsWith(_rest, StringComparison.Ordinal)
                : _rest.StartsWith(prefix, StringComparison.Ordinal);
            if (!consistent || _rest.Length > MaxHeaderChars) Fault(sink, $"Parameter format error: '{_rest}'");
            return false;
        }
        string key = match.Groups[1].Value;
        if (!_seen.Add(key))
        {
            Fault(sink, $"Duplicate parameter name: '{key}'");
            return false;
        }
        _stringValue = match.Groups[2].Value == "true";
        StringBuilder head = new();
        if (_seen.Count > 1) head.Append(", ");
        head.Append('"');
        PyJson.AppendEscaped(head, key);
        head.Append("\": ");
        if (_stringValue) head.Append('"');
        EmitArgs(head.ToString(), sink);
        _rest = _rest[match.Length..];
        _phase = Phase.ParamValue;
        return true;
    }

    private bool StepParamValue(Action<ParsedEvent> sink)
    {
        MarkerScanner.Scan(_rest, ParamValueStops, out int at, out _, out int hold);
        if (at >= 0)
        {
            EmitValue(_rest[..at], sink);
            if (_stringValue) EmitArgs("\"", sink);
            _rest = _rest[(at + ParamEnd.Length)..];
            _phase = Phase.GapAfterParam;
            return true;
        }
        int cut = hold < 0 ? _rest.Length : hold;
        EmitValue(_rest[..cut], sink);
        _rest = _rest[cut..];
        return false;
    }

    private void EmitValue(string value, Action<ParsedEvent> sink)
    {
        if (value.Length == 0) return;
        if (!_stringValue)
        {
            EmitArgs(value, sink);
            return;
        }
        StringBuilder escaped = new(value.Length + 8);
        PyJson.AppendEscaped(escaped, value);
        EmitArgs(escaped.ToString(), sink);
    }

    private void EmitArgs(string fragment, Action<ParsedEvent> sink)
    {
        _args.Append(fragment);
        sink(new ParsedEvent(ParsedEventKind.ToolCallArgsDelta, fragment, _calls.Count));
    }

    private void CompleteCall(Action<ParsedEvent> sink)
    {
        EmitArgs("}", sink);
        int index = _calls.Count;
        _calls.Add(new ChatToolCall($"call_{index}", _name, _args.ToString()) { Namespace = _namespace });
        _callOpen = false;
        sink(new ParsedEvent(ParsedEventKind.ToolCallEnd, null, index));
        _phase = Phase.GapBeforeInvoke;
    }

    private void Fault(Action<ParsedEvent> sink, string reason)
    {
        _phase = Phase.Faulted;
        _rest = "";
        if (_callOpen) sink(new ParsedEvent(ParsedEventKind.ToolCallAbort, null, _calls.Count));
        _callOpen = false;
        sink(new ParsedEvent(ParsedEventKind.Malformed, reason));
    }
}
