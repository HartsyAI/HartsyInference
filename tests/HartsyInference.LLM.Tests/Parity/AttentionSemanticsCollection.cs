using Xunit;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>Route observations remain isolated from other GPU checks.</summary>
[CollectionDefinition("AttentionSemantics", DisableParallelization = true)]
public sealed class AttentionSemanticsCollection;
