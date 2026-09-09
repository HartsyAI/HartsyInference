using HartsyInference.BenchmarkRunner.Contracts;

namespace HartsyInference.BenchmarkRunner.Publication;

/// <summary>A reviewed campaign paired with one complete workload.</summary>
internal sealed record AcceptedCase(SubmissionRecord Submission, CampaignRecord Campaign, CaseDefinition Case);
