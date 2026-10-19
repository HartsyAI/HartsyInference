using System.Text.Json;
using HartsyInference.API.Endpoints;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>A non-streaming completion carries the model's reasoning in <c>reasoning_content</c>, beside the answer alone
/// in <c>content</c>.</summary>
public sealed class CompatReasoningMappingTests
{
    [Fact]
    public void ReasoningGoesToReasoningContentAndTheAnswerStaysInContent()
    {
        TextResult result = new() { Text = "The sky is blue.", Reasoning = "Rayleigh scattering favours short wavelengths." };

        using JsonDocument json = JsonDocument.Parse(JsonSerializer.Serialize(CompatEndpoints.ToAssistantMessage(result)));

        Assert.Equal("The sky is blue.", json.RootElement.GetProperty("content").GetString());
        Assert.Equal("Rayleigh scattering favours short wavelengths.", json.RootElement.GetProperty("reasoning_content").GetString());
    }

    [Fact]
    public void NoReasoningOmitsReasoningContent()
    {
        TextResult result = new() { Text = "Hello." };

        using JsonDocument json = JsonDocument.Parse(JsonSerializer.Serialize(CompatEndpoints.ToAssistantMessage(result)));

        Assert.False(json.RootElement.TryGetProperty("reasoning_content", out _));
    }
}
