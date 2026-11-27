using Xunit;

namespace HartsyInference.Audio.Tests.Attention;

/// <summary>The attention-evidence capture is process-wide, so the tests that open one run with nothing beside them.</summary>
[CollectionDefinition("AttentionEvidence", DisableParallelization = true)]
public sealed class AttentionEvidenceCollection;
