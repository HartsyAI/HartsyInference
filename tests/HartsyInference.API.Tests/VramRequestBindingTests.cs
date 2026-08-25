using System.Text.Json;
using System.Text.Json.Serialization;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>Pins the wire shape of the per-request VRAM override, which is the only way an HTTP caller can ask for a different memory posture without touching the server's environment.</summary>
/// <remarks>Uses the same options the host configures (<c>JsonSerializerDefaults.Web</c> for camelCase +
/// case-insensitive matching, plus the string-enum converter registered in
/// <c>HartsyInferenceServiceExtensions</c>), so a change to either would fail here rather than silently
/// turning a caller's override into a no-op — which is exactly how this field was inert before.</remarks>
public sealed class VramRequestBindingTests
{
    private static readonly JsonSerializerOptions Options = Build();

    private static JsonSerializerOptions Build()
    {
        JsonSerializerOptions o = new JsonSerializerOptions(JsonSerializerDefaults.Web);
        o.Converters.Add(new JsonStringEnumConverter());
        return o;
    }

    [Fact]
    public void TierOnlyOverrideBinds()
    {
        ImageRequest request = JsonSerializer.Deserialize<ImageRequest>(
            """{"prompt":"a red fox","vram":{"tier":"Aggressive"}}""", Options)!;

        Assert.Equal(VramTier.Aggressive, request.Vram!.Tier);
        Assert.Null(request.Vram.WeightStreaming);
        Assert.False(request.Vram.IsEmpty);
    }

    /// <summary>Individual levers must bind too — the whole point is pinning one knob without restating a tier.</summary>
    [Fact]
    public void IndividualLeversBind()
    {
        ImageRequest request = JsonSerializer.Deserialize<ImageRequest>(
            """
            {"prompt":"a red fox","vram":{"weightStreaming":"On","keepResident":"Off","caches":"Half","chunkScale":0.5}}
            """, Options)!;

        VramOverrides v = request.Vram!;
        Assert.Equal(LeverState.On, v.WeightStreaming);
        Assert.Equal(LeverState.Off, v.KeepResident);
        Assert.Equal(CachePrecision.Half, v.Caches);
        Assert.Equal(0.5f, v.ChunkScale);
        Assert.Null(v.Tier);
    }

    /// <summary>Web defaults are case-insensitive, so a caller sending lowercase enum names is not silently ignored.</summary>
    [Fact]
    public void EnumNamesAreCaseInsensitive()
    {
        ImageRequest request = JsonSerializer.Deserialize<ImageRequest>(
            """{"prompt":"x","vram":{"tier":"maximum","weightStreaming":"off"}}""", Options)!;

        Assert.Equal(VramTier.Maximum, request.Vram!.Tier);
        Assert.Equal(LeverState.Off, request.Vram.WeightStreaming);
    }

    /// <summary>Omitting the field entirely must mean "follow the backend", not an empty override that decides something.</summary>
    [Fact]
    public void OmittedFieldStaysNull()
    {
        ImageRequest request = JsonSerializer.Deserialize<ImageRequest>("""{"prompt":"a red fox"}""", Options)!;
        Assert.Null(request.Vram);
    }

    /// <summary>Video and music carry the same field, so one documented shape covers every modality.</summary>
    [Fact]
    public void OtherModalitiesCarryTheSameShape()
    {
        VideoRequest video = JsonSerializer.Deserialize<VideoRequest>(
            """{"prompt":"x","vram":{"tier":"Balanced"}}""", Options)!;
        Assert.Equal(VramTier.Balanced, video.Vram!.Tier);

        MusicRequest music = JsonSerializer.Deserialize<MusicRequest>(
            """{"prompt":"x","vram":{"tier":"Balanced"}}""", Options)!;
        Assert.Equal(VramTier.Balanced, music.Vram!.Tier);
    }
}
