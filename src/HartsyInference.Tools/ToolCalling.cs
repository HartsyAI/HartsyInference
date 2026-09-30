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
        options.TextStreamFilterFactory = request => CreateFilter(request, resolved, stopAfterFirstCall);
    }

    /// <summary>The filter for <paramref name="request"/>: a <see cref="ToolCallStreamFilter"/> when it offers tools, else null.</summary>
    public static ITextStreamFilter? CreateFilter(TextRequest request, ToolCallFormat format = ToolCallFormat.Hermes, bool stopAfterFirstCall = true)
        => request.Tools is { Count: > 0 } ? new ToolCallStreamFilter(format, stopAfterFirstCall) : null;
}
