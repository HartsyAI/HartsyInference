using Xunit;

// Timing tests (cadence, jitter harness with burner threads, loopback call) cannot share the box with each other:
// one class at a time, in declaration order within a class.
[assembly: CollectionBehavior(DisableTestParallelization = true)]
