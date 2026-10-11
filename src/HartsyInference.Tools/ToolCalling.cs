using System.Globalization;
using HartsyInference.Engine;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Tools.Parsing;

namespace HartsyInference.Tools;

/// <summary>Installs tool-call parsing into an engine: sets <see cref="EngineOptions.TextStreamFilterFactory"/> to create a <see cref="ToolCallStreamFilter"/> for every request that offers <see cref="TextRequest.Tools"/>, and nothing for any other request, so plain generation keeps its untouched code path.</summary>
public static class ToolCalling
{
    /// <summary>Sets the filter factory on <paramref name="options"/> (replacing any earlier factory). <paramref name="format"/> applies to every request; null means <see cref="ToolCallFormat.Hermes"/>, and <see cref="ToolCallFormats.Detect"/> picks one from a model id when the host knows which family it loads.</summary>
    public static void Install(EngineOptions options, ToolCallFormat? format = null, bool stopAfterFirstCall = true)
    {
        ArgumentNullException.ThrowIfNull(options);
        ToolCallFormat resolved = format ?? ToolCallFormat.Hermes;
        options.TextStreamFilterFactory = context => CreateFilter(context, resolved, stopAfterFirstCall);
    }

    /// <summary>The filter for <paramref name="context"/>'s request: a <see cref="ToolCallStreamFilter"/> restricted to the offered tool names when it offers tools, else null. A template with its own structured parser needs none, and its call ids are <c>call_{RequestId}_{n}</c> like the parser's.</summary>
    public static ITextStreamFilter? CreateFilter(TextStreamFilterContext context, ToolCallFormat format = ToolCallFormat.Hermes, bool stopAfterFirstCall = true)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (context.HasStructuredParser) return null;
        if (context.Request.Tools is not { Count: > 0 } tools) return null;
        string[] names = new string[tools.Count];
        for (int i = 0; i < names.Length; i++) names[i] = tools[i].Name;
        string idPrefix = "call_" + context.RequestId.ToString(CultureInfo.InvariantCulture) + "_";
        return new ToolCallStreamFilter(format, stopAfterFirstCall, names, idPrefix);
    }
}
