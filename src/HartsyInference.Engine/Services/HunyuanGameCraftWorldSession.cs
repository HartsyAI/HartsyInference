using System.Runtime.CompilerServices;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.World.Pipelines;

namespace HartsyInference.Engine.Services;

/// <summary>A world session over Hunyuan-GameCraft. Every <see cref="WorldRequest"/> carries a mandatory
/// <see cref="WorldRequest.InitImage"/> (<see cref="WorldService.Open"/> enforces this for all world models), and
/// turning that seed frame into the chunk-0 history latent <see cref="HunyuanGameCraftPipeline.DenoiseChunk"/>
/// requires a <c>HunyuanVideoVaeEncoder</c>, which <see cref="HunyuanGameCraftPipeline.LoadFromPath"/> does not
/// build — see <see cref="HunyuanGameCraftPipeline.CanEncodeReferenceFrame"/>'s remarks for exactly what's missing
/// (a Conv3d→Linear reshape on four mid-attention tensors; the structural ldm→diffusers key remap already exists
/// via <c>CheckpointConvertUtils.ConvertVaeEncoderKey</c>). This constructor fails fast on that specific, named gap
/// (before spending any Llava/CLIP encode compute, since no session can proceed past it regardless) rather than
/// guessing at an unvalidated encode path — and <see cref="WorldService.LoadHunyuanGameCraft"/> checks the same
/// gap even earlier, before loading the checkpoint set at all.
/// <para>Chunked generation itself — driving <c>GameCraftFrameStepper</c> from queued WASD-style actions — is not
/// implemented in this pass either (this task is the checkpoint <b>loader</b>; wiring a live session is real
/// follow-up work once a VAE encoder lands). <see cref="SendAction"/>/<see cref="StreamAsync"/> are stubbed to the
/// same effect for the same reason, in case a future caller ever reaches this class with a
/// <see cref="HunyuanGameCraftPipeline"/> built by some other path that does supply a VAE encoder.</para></summary>
public sealed class HunyuanGameCraftWorldSession : IWorldSession
{
    private const string NotImplementedMessage =
        "Hunyuan-GameCraft interactive generation is not implemented yet — this pass only built the checkpoint " +
        "loader (HunyuanGameCraftPipeline.LoadFromPath). Driving a live session (text conditioning, WASD action " +
        "parsing, chunked GameCraftFrameStepper stepping) is real follow-up work.";

    internal const string MissingVaeEncoderMessage =
        "Hunyuan-GameCraft world sessions need a VAE encoder to turn WorldRequest.InitImage into chunk-0 history, " +
        "and HunyuanGameCraftPipeline.LoadFromPath does not build one yet (the structural ldm->diffusers key remap " +
        "exists via CheckpointConvertUtils.ConvertVaeEncoderKey, but the mid-attention Conv3d->Linear reshape " +
        "HunyuanVideoCheckpointConverter.ConvertVaeDecoder's AttnProj applies for the decoder has no encoder-side " +
        "counterpart wired up yet). See HunyuanGameCraftPipeline.CanEncodeReferenceFrame.";

    /// <summary>Validates the request shape, then fails fast with <see cref="MissingVaeEncoderMessage"/> unless
    /// <paramref name="pipeline"/> was built with a VAE encoder (never true for a <see cref="WorldService"/>-loaded
    /// pipeline today). Parameters are accepted (rather than an argumentless constructor) so the shape matches
    /// every other loaded world model's session — <see cref="WorldService.LoadHunyuanGameCraft"/> passes its real,
    /// loaded components — and so a future VAE-encoder follow-up has everything it needs already threaded through.</summary>
    internal HunyuanGameCraftWorldSession(HunyuanGameCraftPipeline pipeline, LlamaStyleEncoder llava,
        ClipTextEncoder clipL, ClipTokenizer clipTokenizer, WorldRequest request)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentNullException.ThrowIfNull(llava);
        ArgumentNullException.ThrowIfNull(clipL);
        ArgumentNullException.ThrowIfNull(clipTokenizer);
        if (request.InitImage is null)
        {
            throw new ArgumentException("Hunyuan-GameCraft rolls out from a first frame.", nameof(request));
        }
        if (!pipeline.CanEncodeReferenceFrame)
        {
            throw new NotSupportedException(MissingVaeEncoderMessage);
        }
        throw new NotSupportedException(NotImplementedMessage);
    }

    /// <inheritdoc/>
    public void SendAction(string action) => throw new NotSupportedException(NotImplementedMessage);

    /// <inheritdoc/>
    public async IAsyncEnumerable<VideoFrame> StreamAsync([EnumeratorCancellation] CancellationToken cancel)
    {
        await Task.CompletedTask;
        throw new NotSupportedException(NotImplementedMessage);
#pragma warning disable CS0162 // unreachable — keeps the method a valid iterator (IAsyncEnumerable needs a yield).
        yield break;
#pragma warning restore CS0162
    }

    /// <inheritdoc/>
    public void Dispose() { }
}
