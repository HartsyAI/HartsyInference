using System;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Thrown when a chat template invokes its own <c>raise_exception(...)</c> — i.e. the template
/// rejected the message structure (e.g. Mistral's "roles must alternate"). Distinct from a real template
/// bug so callers can normalize the messages and retry. Derives from <see cref="InvalidOperationException"/>
/// for backward compatibility with code that caught the prior exception type.</summary>
public sealed class ChatTemplateRaiseException : InvalidOperationException
{
    public ChatTemplateRaiseException(string message) : base(message) { }
}
