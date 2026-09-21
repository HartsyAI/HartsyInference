using HartsyInference.BenchmarkRunner.Contracts;
using HartsyInference.BenchmarkRunner.Execution;
using Xunit;

namespace HartsyInference.BenchmarkRunner.Tests;
/// <summary>Device attestation binds by UUID and degrades to unattested; it never guesses a device.</summary>
public sealed class AttestationTests
{
    [Fact]
    public void BothUuidSpellingsHashToOneDeviceIdentity()
    {
        // cuDeviceGetUuid yields bare hex and nvidia-smi prints GPU-8-4-4-4-12; the match is on the hash of the
        // normalized form, which is what keeps the raw UUID out of the exported record.
        const string Hex = "03eac12476d822a21ff12be8e2fa098d";
        Assert.Equal(Hex, NvidiaSmi.NormalizeUuid("GPU-03eac124-76d8-22a2-1ff1-2be8e2fa098d"));
        Assert.Equal(Hardware.IdentityHash(Hex), Hardware.IdentityHash(NvidiaSmi.NormalizeUuid(
            "  GPU-03EAC124-76D8-22A2-1FF1-2BE8E2FA098D  ")));
    }

    [Fact]
    public void AnUnknownDeviceRefusesInsteadOfFallingBackToAnOrdinal()
    {
        // CUDA enumerates fastest-first, so "the first card nvidia-smi lists" is a different GPU on this host.
        Assert.Null(DeviceAttestation.ResolveUuid(Device("cuda:0")));
        Assert.Null(DeviceAttestation.ResolveUuid(Device("cpu")));
        Assert.Null(DeviceAttestation.ResolveUuid(Device("vulkan:0")));
    }

    [Fact]
    public void ThePowerProfileSeparatesConfigurationsAndMarksTheUnattested()
    {
        Assert.Equal("unattested", DeviceAttestation.Profile(new AttestationRecord { Source = AttestationRecord.Unavailable }));
        AttestationRecord stock = new()
        {
            Source = AttestationRecord.Smi,
            Fields = new(StringComparer.Ordinal) { ["power.limit"] = "450.00 W", ["clocks.max.sm"] = "3150 MHz" }
        };
        AttestationRecord limited = stock with
        {
            Fields = new(StringComparer.Ordinal) { ["power.limit"] = "300.00 W", ["clocks.max.sm"] = "3150 MHz" }
        };
        Assert.NotEqual("unattested", DeviceAttestation.Profile(stock));
        Assert.NotEqual(DeviceAttestation.Profile(stock), DeviceAttestation.Profile(limited));
    }

    [Fact]
    public void AnInertSamplerReportsUnavailableRatherThanAnEmptyAggregate()
    {
        using DeviceSampler sampler = DeviceSampler.Start(null, 100);
        DeviceTelemetry telemetry = sampler.Aggregate(1000, 2000);
        Assert.Equal(DeviceTelemetry.Unavailable, telemetry.Source);
        Assert.Equal(0, telemetry.SampleCount);
        Assert.Null(telemetry.PeakUsedDeviceBytes);
        Assert.NotNull(sampler.Note);
    }

    private static DeviceRecord Device(string selector) => new()
    {
        Selector = selector,
        Name = "device that is not on this host",
        Identity = new string('9', 64),
        Driver = "test"
    };
}
