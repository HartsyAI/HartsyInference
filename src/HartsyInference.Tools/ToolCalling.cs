using System.Globalization;
using HartsyInference.Engine;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Tools.Parsing;

namespace HartsyInference.Tools;

/// <summary>Installs tool-call parsing into an engine: sets <see cref="EngineOptions.TextStreamFilterFactory"/> to create a <see cref="ToolCallStreamFilter"/> for every request that offers <see cref="TextRequest.Tools"/>, and nothing for any other request, so plain generation keeps its untouched code path.</summary>
public static class ToolCalling
{
    /// <summary>Sets the filter factory on <paramref name="options"/> (replacing any earlier factory). A null <paramref name="format"/> resolves per request from the model's own template and identity (<see cref="ResolveFormat"/>); a non-null one applies to every request.</summary>
    public static void Install(EngineOptions options, ToolCallFormat? format = null, bool stopAfterFirstCall = true)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.TextStreamFilterFactory = context => CreateFilter(context, format, stopAfterFirstCall);
    }

    /// <summary>The format a request's model speaks: a structured parser or a tool-less prompt decides it first; otherwise the chat template's own markers, then the family name in the architecture or model path, then Hermes. Null when the template has its own parser and no filter applies.</summary>
    public static ToolCallFormat? ResolveFormat(TextStreamFilterContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (context.HasStructuredParser) return null;
        if (context.ToolPromptInjected) return ToolCallFormat.Hermes;
        if (string.Equals(context.TemplateName, "chatml", StringComparison.OrdinalIgnoreCase)) return ToolCallFormat.Hermes;
        if (context.ChatTemplateSource is { } source && ToolCallFormats.TryDetectFromTemplate(source, out ToolCallFormat fromTemplate)) return fromTemplate;
        return ToolCallFormats.Detect(string.Join(' ', context.Architecture, context.ModelPath));
    }

    /// <summary>The filter for <paramref name="context"/>'s request: a <see cref="ToolCallStreamFilter"/> restricted to the offered tool names when it offers tools, else null. A forced tool restricts the bare forms to that one name and stops after its call. A template with its own structured parser needs none, and its call ids are <c>call_{RequestId}_{n}</c> like the parser's.</summary>
    public static ITextStreamFilter? CreateFilter(TextStreamFilterContext context, ToolCallFormat? format = null, bool stopAfterFirstCall = true)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (context.HasStructuredParser) return null;
        if (context.Request.Tools is not { Count: > 0 } tools) return null;
        ToolCallFormat? resolved = format ?? ResolveFormat(context);
        if (resolved is not { } chosen) return null;
        IReadOnlyList<ToolDefinition> offered = tools;
        bool stop = stopAfterFirstCall;
        if (context.ForcedToolName is { Length: > 0 } forced)
        {
            offered = tools.Where(t => t.Name == forced).ToList();
            stop = true;
        }
        string idPrefix = "call_" + context.RequestId.ToString(CultureInfo.InvariantCulture) + "_";
        return new ToolCallStreamFilter(ToolCallParser.ForTools(chosen, offered, idPrefix), stop);
    }
}
