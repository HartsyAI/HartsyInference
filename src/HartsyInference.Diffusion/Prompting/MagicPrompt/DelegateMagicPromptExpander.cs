using HartsyInference.Core.Exceptions;
using HartsyInference.Diffusion.Prompting.Dialects;

namespace HartsyInference.Diffusion.Prompting.MagicPrompt;

/// <summary>A concrete <see cref="IMagicPromptExpander"/> that delegates the actual LLM call to a caller-supplied
/// function and parses the returned Ideogram-4 JSON into a <see cref="StructuredPrompt"/> via
/// <see cref="Ideogram4Dialect.Parse"/>. This keeps the prompt package free of any LLM dependency: wire it to a
/// Claude API client, a local dotLLM instance, or any service by passing a delegate that turns
/// <c>(systemPrompt, userPrompt)</c> into the model's JSON response.
///
/// <para>Example:
/// <code>
/// var expander = new DelegateMagicPromptExpander(MyIdeogramSystemPrompt, (sys, user, ct) => myLlm.CompleteAsync(sys, user, ct));
/// StructuredPrompt sp = await expander.ExpandAsync("a neon ramen shop at night");
/// string json = new Ideogram4Dialect().Serialize(sp);   // validated, ready for the pipeline
/// </code></para></summary>
public sealed class DelegateMagicPromptExpander : IMagicPromptExpander
{
    private readonly string _systemPrompt;
    private readonly Func<string, string, CancellationToken, Task<string>> _complete;
    private readonly Ideogram4Dialect _dialect = new();
    private readonly bool _validate;

    /// <summary>Creates an expander.</summary>
    /// <param name="systemPrompt">The magic-prompt system instruction handed to the LLM (defines the JSON schema it must emit). Supply the Ideogram-4 magic-prompt text, or your own.</param>
    /// <param name="complete">The LLM call: <c>(systemPrompt, userPrompt, ct) =&gt; Task&lt;jsonResponse&gt;</c>. Must return the model's raw Ideogram-4 JSON.</param>
    /// <param name="validate">When true (default), the parsed prompt is validated against the Ideogram schema before being returned, so a malformed LLM response fails fast.</param>
    public DelegateMagicPromptExpander(
        string systemPrompt,
        Func<string, string, CancellationToken, Task<string>> complete,
        bool validate = true)
    {
        _systemPrompt = systemPrompt ?? throw new ArgumentNullException(nameof(systemPrompt));
        _complete = complete ?? throw new ArgumentNullException(nameof(complete));
        _validate = validate;
    }

    /// <inheritdoc/>
    public async Task<StructuredPrompt> ExpandAsync(string plainPrompt, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(plainPrompt))
            throw new ArgumentException("Plain prompt must be non-empty.", nameof(plainPrompt));

        string json = await _complete(_systemPrompt, plainPrompt, cancellationToken).ConfigureAwait(false);
        if (string.IsNullOrWhiteSpace(json))
            throw new HartsyInferenceException("The magic-prompt LLM returned an empty response.");

        StructuredPrompt prompt = _dialect.Parse(ExtractJsonObject(json));
        if (_validate)
            Ideogram4Dialect.Validate(prompt);
        return prompt;
    }

    /// <summary>LLMs often wrap JSON in prose or a <c>```json</c> fence. Extract the outermost <c>{…}</c> object so the
    /// raw response is tolerated. Throws if no object is present.</summary>
    private static string ExtractJsonObject(string response)
    {
        int start = response.IndexOf('{');
        int end = response.LastIndexOf('}');
        if (start < 0 || end <= start)
            throw new HartsyInferenceException("The magic-prompt LLM response contained no JSON object.");
        return response.Substring(start, end - start + 1);
    }
}
