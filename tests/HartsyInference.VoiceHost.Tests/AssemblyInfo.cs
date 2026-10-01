using System.Runtime.Versioning;
using Xunit;

// The GC latency mode, the pool floor and Logs.SetLogger are process-wide, and the cadence and loopback tests time a
// paced thread: one class at a time.
[assembly: CollectionBehavior(DisableTestParallelization = true)]

// The host serves a Unix socket and reads owner-only secret files; these tests exercise exactly that.
[assembly: SupportedOSPlatform("linux")]
