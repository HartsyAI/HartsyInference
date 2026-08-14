using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.API.Endpoints;

/// <summary>Native audio routes: speech synthesis, music generation, transcription, voice conversion, and fx (stem separation /
/// enhancement). All one-shot (no step-preview progress, unlike image/video), so none of these have a <c>/stream</c>
/// variant. <see cref="AudioResult"/>/<see cref="TranscriptResult"/>/<see cref="StemsResult"/> already carry encoded
/// bytes (or plain text/timestamps) — no PNG-style re-encoding step needed, just pass the native result straight
/// through as JSON.</summary>
public static class AudioEndpoints
{
    /// <summary>Maps <c>/v1/native/speech</c>, <c>/music</c>, <c>/transcribe</c>, <c>/voice-convert</c>, and <c>/fx/*</c>.</summary>
    public static void MapAudioEndpoints(this WebApplication app)
    {
        app.MapPost("/v1/native/speech", async (NativeSpeechRequest req, IInferenceEngine engine, InferenceQueue queue, CancellationToken ct) =>
        {
            ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Speech);
            try
            {
                AudioResult result = await queue.EnqueueAsync(() => engine.Speech.SynthesizeAsync(spec, req.Request, ct), ct);
                return Results.Ok(WithSavedPath(result, ArtifactPersistence.Save(req, result.Data, req.Request.Text, result.Format)));
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });

        // Music had no native route until MiniMax Music 3 landed: every music model was reachable from the CLI
        // and the SwarmUI extension but not over HTTP.
        app.MapPost("/v1/native/music", async (NativeMusicRequest req, IInferenceEngine engine, InferenceQueue queue, CancellationToken ct) =>
        {
            ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Music);
            try
            {
                AudioResult result = await queue.EnqueueAsync(() => engine.Music.GenerateAsync(spec, req.Request, null, ct), ct);
                return Results.Ok(WithSavedPath(result, ArtifactPersistence.Save(req, result.Data, req.Request.Prompt, result.Format)));
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });

        app.MapPost("/v1/native/transcribe", async (NativeTranscribeRequest req, IInferenceEngine engine, InferenceQueue queue, CancellationToken ct) =>
        {
            ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Transcribe);
            try
            {
                TranscriptResult result = await queue.EnqueueAsync(() => engine.Transcribe.RunAsync(spec, req.Request, ct), ct);
                return Results.Ok(result);
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });

        app.MapPost("/v1/native/voice-convert", async (NativeVoiceConvertRequest req, IInferenceEngine engine, InferenceQueue queue, CancellationToken ct) =>
        {
            ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.VoiceConvert);
            try
            {
                AudioResult result = await queue.EnqueueAsync(() => engine.VoiceConversion.ConvertAsync(spec, req.Request, ct), ct);
                return Results.Ok(WithSavedPath(result, ArtifactPersistence.Save(req, result.Data, "voice-convert", result.Format)));
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });

        app.MapPost("/v1/native/fx/separate", async (NativeFxSeparateRequest req, IInferenceEngine engine, InferenceQueue queue, CancellationToken ct) =>
        {
            ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Fx);
            try
            {
                StemsResult result = await queue.EnqueueAsync(() => engine.Fx.SeparateAsync(spec, req.Request, ct), ct);
                return Results.Ok(new { result.Stems, result.Format, result.SampleRate,
                    savedPath = ArtifactPersistence.SaveGroup(req, result.Stems, "separate", result.Format) });
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });

        app.MapPost("/v1/native/fx/enhance", async (NativeFxEnhanceRequest req, IInferenceEngine engine, InferenceQueue queue, CancellationToken ct) =>
        {
            ModelSpec spec = ModelResolver.Resolve(req.Model, req.ModelPath, Modality.Fx);
            try
            {
                AudioResult result = await queue.EnqueueAsync(() => engine.Fx.EnhanceAsync(spec, req.Request, ct), ct);
                return Results.Ok(WithSavedPath(result, ArtifactPersistence.Save(req, result.Data, "enhance", result.Format)));
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });
    }

    /// <summary>Re-emits the native audio fields plus where the file landed. Property names match
    /// <see cref="AudioResult"/> exactly, so this is additive for existing clients.</summary>
    private static object WithSavedPath(AudioResult result, string? savedPath) => new
    {
        result.Data,
        result.Format,
        result.DurationSeconds,
        result.SampleRate,
        savedPath,
    };
}
