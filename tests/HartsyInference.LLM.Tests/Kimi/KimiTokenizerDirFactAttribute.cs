using Xunit;

namespace HartsyInference.LLM.Tests.Kimi;

/// <summary>A fact reported as skipped, never passed, unless <c>HARTSY_KIMI_TOKENIZER_DIR</c> names a directory holding the checkpoint's <c>tiktoken.model</c>.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
internal sealed class KimiTokenizerDirFactAttribute : FactAttribute
{
    public const string EnvironmentVariable = "HARTSY_KIMI_TOKENIZER_DIR";

    public KimiTokenizerDirFactAttribute()
    {
        string? dir = Environment.GetEnvironmentVariable(EnvironmentVariable);
        if (string.IsNullOrWhiteSpace(dir) || !File.Exists(Path.Combine(dir, "tiktoken.model")))
            Skip = $"SKIPPED: {EnvironmentVariable} is unset or holds no tiktoken.model; the real-vocabulary check did not run.";
    }
}
