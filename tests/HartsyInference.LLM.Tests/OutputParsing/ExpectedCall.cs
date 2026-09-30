namespace HartsyInference.LLM.Tests.OutputParsing;

internal readonly record struct ExpectedCall(string Name, string? Namespace, string Arguments);
