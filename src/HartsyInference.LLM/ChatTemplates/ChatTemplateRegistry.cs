using System;
using System.Collections.Generic;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Name-keyed lookup of <see cref="IChatTemplate"/> implementations.</summary>
public sealed class ChatTemplateRegistry
{
    /// <summary>Shared registry pre-populated with <see cref="ChatMlTemplate"/> under the keys "chatml" and "qwen".</summary>
    public static ChatTemplateRegistry Default { get; } = CreateDefault();

    private readonly Dictionary<string, IChatTemplate> _templates =
        new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Registers (or replaces) a template under its <see cref="IChatTemplate.Name"/>.</summary>
    public void Register(IChatTemplate template)
    {
        ArgumentNullException.ThrowIfNull(template);
        _templates[template.Name] = template;
    }

    /// <summary>Adds an additional lookup alias pointing at an already-registered template.</summary>
    public void RegisterAlias(string alias, IChatTemplate template)
    {
        ArgumentException.ThrowIfNullOrEmpty(alias);
        ArgumentNullException.ThrowIfNull(template);
        _templates[alias] = template;
    }

    /// <summary>Returns the template registered under <paramref name="name"/>, throwing if none exists.</summary>
    public IChatTemplate Get(string name)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        if (_templates.TryGetValue(name, out IChatTemplate? template))
            return template;
        throw new KeyNotFoundException($"No chat template registered under '{name}'.");
    }

    /// <summary>Attempts to resolve the template registered under <paramref name="name"/>.</summary>
    public bool TryGet(string name, out IChatTemplate template)
    {
        if (!string.IsNullOrEmpty(name) && _templates.TryGetValue(name, out IChatTemplate? found))
        {
            template = found;
            return true;
        }
        template = null!;
        return false;
    }

    private static ChatTemplateRegistry CreateDefault()
    {
        ChatTemplateRegistry registry = new();
        ChatMlTemplate chatMl = new();
        registry.Register(chatMl);
        registry.RegisterAlias("qwen", chatMl);
        return registry;
    }
}
