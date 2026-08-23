using System;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Thrown when a chat template invokes its own <c>raise_exception(...)</c> to reject the message structure (e.g. Mistral's "roles must alternate"), distinct from a real template bug; derives from <see cref="InvalidOperationException"/> for backward compatibility with code catching the prior exception type.</summary>
public sealed class ChatTemplateRaiseException : InvalidOperationException
{
    public ChatTemplateRaiseException(string message) : base(message) { }
}
