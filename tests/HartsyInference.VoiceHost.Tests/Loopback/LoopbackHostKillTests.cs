using HartsyInference.Core.Runtime;
using HartsyInference.PhoneGateway.Sip;
using HartsyInference.Tests.Common;
using HartsyInference.VoiceHost.Tests.Support;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.VoiceHost.Tests.Loopback;

/// <summary>The host is killed with SIGKILL in the middle of a call. The gateway must end the call cleanly on its own,
/// within 2 s after its outage period (it plays "one moment", then "goodbye", then sends the BYE), and must accept the next
/// call once a new host is up on the same socket path, the dead host's socket file still in place. The host runs as its
/// own process, real models on the RTX 3060, scripted language model. Run alone, after the quiet window:
/// <c>CUDA_VISIBLE_DEVICES=1 HARTSY_REQUIRE_REAL_WEIGHTS=1 HARTSYINFERENCE_MODELS_DIR=/mnt/model-storage/Models dotnet test tests/HartsyInference.VoiceHost.Tests -c Release --filter "FullyQualifiedName~LoopbackHostKillTests"</c>.</summary>
[Trait("Category", "Slow")]
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
public sealed class LoopbackHostKillTests
{
    private const int OutageHangupMs = 3_000;
    private const int AfterOutageGateMs = 2_000;
    private const int HostStartMs = 180_000;
    private const string Token = "kill-test-link-token";

    private readonly ITestOutputHelper _output;

    public LoopbackHostKillTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task AKilledHostEndsTheCallCleanlyAndTheNextHostTakesTheNextCall()
    {
        if (!RealWeightGate.Require(_output.WriteLine, LoopbackAssets.All()))
        {
            return;
        }
        string? device = LoopbackAssets.AudioDevice(_output.WriteLine);
        if (device is null)
        {
            return;
        }
        string directory = HostRig.NewSocketDirectory();
        try
        {
            string socket = Path.Combine(directory, "phone.sock");
            string config = WriteConfig(directory, socket, device);
            using LoopbackGateway gateway = LoopbackGateway.Start(socket, Token, OutageHangupMs);
            using LoopbackPhone phone = new();

            using (HostProcess first = HostProcess.Start(config, "Okay."))
            {
                try
                {
                    Assert.True(gateway.WaitUntil(() => gateway.Link.IsConnected, HostStartMs), "the first host never came up.");
                    long start = MonotonicClock.NowNs();
                    Assert.True(await phone.CallAsync(gateway.Port), $"call failed: {phone.LastFailure}");
                    Assert.True(gateway.WaitUntil(() => phone.AudibleSince(start, 1) >= 25, 30_000), "the greeting never reached the phone.");

                    long killed = MonotonicClock.NowNs();
                    first.Kill();
                    Assert.True(phone.HungUp.Wait(OutageHangupMs + 10_000), "the gateway never ended the call after the host died.");
                    double endedMs = (phone.HungUpNs - killed) / 1e6;
                    _output.WriteLine($"host killed; the phone got the BYE {endedMs:F0} ms later ({endedMs - OutageHangupMs:F0} ms after the "
                        + $"{OutageHangupMs} ms outage period); outages={gateway.Metrics.Outages} outage hang-ups={gateway.Metrics.OutageHangups}");
                    Assert.True(endedMs <= OutageHangupMs + AfterOutageGateMs, $"the call ended {endedMs:F0} ms after the kill.");
                    Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Idle, 5_000));
                    Assert.True(File.Exists(socket), "the killed host's socket file is gone; the restart path is not exercised.");
                }
                finally
                {
                    Dump("first host", first);
                }
            }

            using HostProcess second = HostProcess.Start(config, "Okay.");
            try
            {
                Assert.True(gateway.WaitUntil(() => gateway.Link.IsConnected, HostStartMs), "the gateway never reconnected to the new host.");
                long again = MonotonicClock.NowNs();
                Assert.True(await phone.CallAsync(gateway.Port), $"the next call was refused: {phone.LastFailureStatus} {phone.LastFailure}");
                Assert.True(gateway.WaitUntil(() => phone.AudibleSince(again, 1) >= 25, 30_000), "the new host's greeting never reached the phone.");
                phone.Agent.Hangup();
                Assert.True(gateway.WaitUntil(() => gateway.Controller.State == CallState.Idle, 10_000));
                Assert.True(second.Terminate(30_000), "the host did not stop on SIGTERM.");
                Assert.Equal(0, second.ExitCode);
                Assert.False(File.Exists(socket), "a cleanly stopped host left its socket file behind.");
            }
            finally
            {
                Dump("second host", second);
            }
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static string WriteConfig(string directory, string socket, string device)
    {
        string token = Path.Combine(directory, "link-token");
        File.WriteAllText(token, Token + "\n");
        File.SetUnixFileMode(token, UnixFileMode.UserRead | UnixFileMode.UserWrite);
        string path = Path.Combine(directory, "voice.json");
        File.WriteAllText(path, $$"""
            {
              "link": { "socketPath": "{{socket}}", "tokenFile": "{{token}}" },
              "models": { "audioDevice": "{{device}}", "llmDevice": "cpu", "wakeModelRoot": "{{LoopbackAssets.WakeRoot}}" },
              "agent": { "greeting": "Hello, how can I help you?" }
            }
            """);
        return path;
    }

    private void Dump(string name, HostProcess host)
    {
        _output.WriteLine($"--- {name} (pid {host.Id}) ---");
        foreach (string line in host.Output.TakeLast(60))
        {
            _output.WriteLine(line);
        }
    }
}
