using Xunit;

// Every test in this assembly ultimately touches one real, shared physical GPU (VRAM, device queues,
// pipeline cache). xUnit's default cross-class test parallelism assumes tests are independent; here it
// means multiple VulkanBackend instances from DIFFERENT test classes can end up allocating concurrently
// against the same ~21-24 GB device budget, causing spurious ErrorOutOfDeviceMemory failures in tests that
// pass cleanly alone (confirmed directly: Backend_AffineBroadcastLastDim_MatchesCpu and
// Compare_Naive_Vs_Blocked_Coopmat_GpuOnlyTime — both pre-existing, unrelated to any single test's own
// footprint — failed only when several VRAM-heavy benchmark classes ran at the same time). There is no
// legitimate reason for GPU tests sharing one physical device to run concurrently, so disable parallelism
// for the whole assembly rather than annotate every class individually.
[assembly: CollectionBehavior(DisableTestParallelization = true)]
