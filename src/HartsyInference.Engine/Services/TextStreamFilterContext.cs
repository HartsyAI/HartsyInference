using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>What <see cref="EngineOptions.TextStreamFilterFactory"/> sees about one text request: the request itself, the request id that also numbers its tool calls, and the model's template and identity, so a host can pick a parser per model.</summary>
public sealed record TextStreamFilterContext
{
    /// <summary>The native request being run.</summary>
    public required TextRequest Request { get; init; }

    /// <summary>The engine's request number; tool-call ids are <c>call_{RequestId}_{index}</c>, unique across rounds.</summary>
    public required long RequestId { get; init; }

    /// <summary>The chat template's registry name (<c>jinja</c>, <c>chatml</c>, or an encoder name).</summary>
    public required string TemplateName { get; init; }

    /// <summary>The Jinja source of the model's chat template, when it has one.</summary>
    public string? ChatTemplateSource { get; init; }

    /// <summary>The model's architecture string from its GGUF metadata (for example <c>llama</c> or <c>qwen2</c>), when known.</summary>
    public string? Architecture { get; init; }

    /// <summary>The loaded model file's path, when known.</summary>
    public string? ModelPath { get; init; }

    /// <summary>True when the template has its own structured output parser, which already reports tool calls and needs no filter.</summary>
    public bool HasStructuredParser { get; init; }

    /// <summary>True when the engine injected a tool prompt into a template that does not offer tools itself.</summary>
    public bool ToolPromptInjected { get; init; }

    /// <summary>The one tool the request forces, or null.</summary>
    public string? ForcedToolName { get; init; }
}
