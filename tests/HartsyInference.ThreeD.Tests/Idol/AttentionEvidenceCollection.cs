using Xunit;

namespace HartsyInference.ThreeD.Tests.Idol;

/// <summary>Attention evidence is one process-wide session, so a test that records it must not run beside another test that
/// computes attention. A collection that disables parallelization keeps it alone.</summary>
[CollectionDefinition("AttentionEvidence", DisableParallelization = true)]
public sealed class AttentionEvidenceCollection;
