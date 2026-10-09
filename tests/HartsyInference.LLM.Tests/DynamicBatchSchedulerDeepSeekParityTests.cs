using HartsyInference.Cpu;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Tests.DeepSeekV41;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Concurrent V4.1 requests on the scheduled route produce the same token ids as each request run alone through the pipeline. The V4.1 host model
/// decodes one sequence at a time, so the batch is interleaved per sequence: this gate covers admission, per-sequence state and the prefill hook, not a fused
/// batched kernel, and it runs on the synthetic fixture (real weights are not needed for the route's correctness).</summary>
public sealed class DynamicBatchSchedulerDeepSeekParityTests
{
    [Fact]
    public async Task Concurrent_V41_Requests_Match_Their_Solo_Pipeline_Runs()
    {
        using CpuBackend cpu = new();
        DeepSeekV41HostModel host = DeepSeekV41HostModelTests.BuildModel(cpu);
        DeepSeekV41GenerationModel model = new(host, maxTokens: 64, outputBackend: cpu);
        DynamicBatchSchedulerTests.StubTokenizer tokenizer = new();
        int[][] prompts = [[1, 2, 3], [4], [5, 6, 7, 8, 9]];
        const int maxTokens = 6;

        // Oracle: each prompt alone through the pipeline, which is the route every V4.1 request took before this change.
        string[] reference = new string[prompts.Length];
        for (int i = 0; i < prompts.Length; i++)
        {
            GenerationResult solo = new TextGenerationPipeline(model, tokenizer)
                .Generate(DynamicBatchSchedulerTests.Req(prompts[i], maxTokens, seed: 0));
            reference[i] = string.Join(",", solo.TokenIds);
        }

        using DynamicBatchScheduler scheduler = new(model, tokenizer, pool: null);
        int prefilled = -1;
        Task<GenerationResult>[] tasks = new Task<GenerationResult>[prompts.Length];
        for (int i = 0; i < prompts.Length; i++)
        {
            GenerationRequest request = DynamicBatchSchedulerTests.Req(prompts[i], maxTokens, seed: 0);
            if (i == 0) request = request with { OnPrefillCompleted = count => prefilled = count };
            tasks[i] = scheduler.SubmitAsync(request, onToken: null, CancellationToken.None);
        }
        GenerationResult[] results = await Task.WhenAll(tasks);

        for (int i = 0; i < prompts.Length; i++)
            Assert.Equal(reference[i], string.Join(",", results[i].TokenIds));
        // The prefill hook fires once the prompt is in the cache, as it does on the pipeline; the caller reads the prompt length from it.
        Assert.Equal(prompts[0].Length, prefilled);
    }

    [Fact]
    public async Task A_Pool_Less_Model_Serves_Every_Request_When_Only_A_Few_Decode_At_Once()
    {
        using CpuBackend cpu = new();
        DeepSeekV41HostModel host = DeepSeekV41HostModelTests.BuildModel(cpu);
        DeepSeekV41GenerationModel model = new(host, maxTokens: 64, outputBackend: cpu);
        DynamicBatchSchedulerTests.StubTokenizer tokenizer = new();
        int[][] prompts = [[1, 2, 3], [4], [5, 6, 7, 8, 9], [2, 3]];
        const int maxTokens = 6;
        string[] reference = [.. prompts.Select(p => string.Join(",",
            new TextGenerationPipeline(model, tokenizer).Generate(DynamicBatchSchedulerTests.Req(p, maxTokens, seed: 0)).TokenIds))];

        // No pool bounds this model, so maxActiveSequences is the only limit: two decode at once and the rest wait their turn in order.
        using DynamicBatchScheduler scheduler = new(model, tokenizer, pool: null, maxActiveSequences: 2);
        Task<GenerationResult>[] tasks = [.. prompts.Select(p => scheduler.SubmitAsync(DynamicBatchSchedulerTests.Req(p, maxTokens, seed: 0), null, CancellationToken.None))];
        GenerationResult[] results = await Task.WhenAll(tasks).WaitAsync(TimeSpan.FromSeconds(120));

        for (int i = 0; i < prompts.Length; i++)
            Assert.Equal(reference[i], string.Join(",", results[i].TokenIds));
    }
}
