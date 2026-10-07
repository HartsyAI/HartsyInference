using System.Text;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Services;

namespace HartsyInference.API.Endpoints;

/// <summary>Typed-decision route in the Jev / SystemOne shape: the body names the model, the state and the questions.</summary>
public static class DecisionEndpoints
{
    /// <summary>Maps <c>/v1/systemone</c>.</summary>
    public static void MapDecisionEndpoints(this WebApplication app)
    {
        app.MapPost("/v1/systemone", async (HttpRequest http, IInferenceEngine engine, InferenceQueue queue, CancellationToken ct) =>
        {
            using StreamReader reader = new(http.Body, Encoding.UTF8);
            string body = await reader.ReadToEndAsync(ct);
            string model;
            try
            {
                using System.Text.Json.JsonDocument doc = System.Text.Json.JsonDocument.Parse(body);
                model = doc.RootElement.TryGetProperty("model", out System.Text.Json.JsonElement m) && m.ValueKind == System.Text.Json.JsonValueKind.String
                    ? m.GetString()! : string.Empty;
            }
            catch (System.Text.Json.JsonException)
            {
                return Results.BadRequest(new { error = "request body must be JSON" });
            }
            if (model.Length == 0)
            {
                return Results.BadRequest(new { error = "model and state are required" });
            }
            ModelSpec spec = ModelResolver.Resolve(model, null, Modality.Decision);
            try
            {
                string json = await queue.EnqueueAsync(() => engine.Decisions.DecideAsync(spec, body, ct), ct);
                return Results.Content(json, "application/json");
            }
            catch (ArgumentException ex)
            {
                return Results.BadRequest(new { error = ex.Message });
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });
    }
}
