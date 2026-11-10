using System.Buffers.Binary;
using System.Net.Http.Headers;
using System.Text.Json;
using HartsyInference.ModelAssets.Registry;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Reads only the safetensors header of the published int8 image transformer and runs detection on its keys.
/// Network=Real: excluded from the CPU lane; run with <c>--filter "Network=Real"</c>.</summary>
[Trait("Network", "Real")]
public sealed class MiniMaxH3ImageRealHeaderTests
{
    private const string Url =
        "https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors";

    [Fact]
    public async Task PublishedImageHeaderDetectsAsMiniMaxH3Image()
    {
        using HttpClient http = new();
        byte[] lengthPrefix = await RangeAsync(http, 0, 7);
        long headerLength = BinaryPrimitives.ReadInt64LittleEndian(lengthPrefix);
        byte[] headerJson = await RangeAsync(http, 8, 8 + headerLength - 1);

        using JsonDocument header = JsonDocument.Parse(headerJson);
        List<string> keys = [];
        foreach (JsonProperty property in header.RootElement.EnumerateObject())
        {
            if (property.Name != "__metadata__")
            {
                keys.Add(property.Name);
            }
        }

        Assert.Equal(ModelArchitecture.MiniMaxH3Image, ModelArchitectureDetector.Detect(keys));
    }

    private static async Task<byte[]> RangeAsync(HttpClient http, long start, long end)
    {
        using HttpRequestMessage request = new(HttpMethod.Get, Url);
        request.Headers.Range = new RangeHeaderValue(start, end);
        using HttpResponseMessage response = await http.SendAsync(request);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadAsByteArrayAsync();
    }
}
