using HartsyInference.Audio.Models.LongCat;
using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>The multi-speaker side of the LongCat-Video-Avatar recipe: what planning refuses, the two speakers' audio as the encoder hears it,
/// and their regions on the token grid.</summary>
internal static class LongCatAvatarSpeakers
{
    /// <summary>The most silent people a request can mark: the DiT's attention head is 128 wide and holds one region per element, three of them taken
    /// by the two speakers and the background.</summary>
    internal const int MaxSilentBoxes = 128 - LongCatSpeakers.MinRegions;

    /// <summary>Why a multi-speaker request cannot run, or null. Nothing is read but the request.</summary>
    internal static string? Refusal(VideoRequest request, VideoSpeakers speakers)
    {
        if (request.VideoAudioReference is not null)
            return "A multi-speaker request gives each speaker's audio in VideoRequest.Speakers; leave VideoAudioReference unset.";
        if (request.InitImage is not { } image)
            return "Multi-speaker LongCat-Video-Avatar needs an init image: it runs only from a picture, and the speakers' boxes are in its pixels.";
        if (speakers.First is null || speakers.Second is null)
            return "A multi-speaker request names both speakers; leave a silent speaker's audio unset.";
        if (speakers.First.Audio is null && speakers.Second.Audio is null)
            return "A multi-speaker request needs driving speech from at least one speaker.";
        if (!Enum.IsDefined(speakers.Timing))
            return $"Unknown speaker timing '{speakers.Timing}'.";
        if (speakers.First.Box is null != speakers.Second.Box is null)
            return "Give both speakers' boxes, or neither (the left and right halves of the picture).";
        if (speakers.SilentBoxes is { Count: > 0 } && speakers.First.Box is null)
            return "Silent people's boxes need both speakers' boxes: the reference cannot combine them with the default halves.";
        if (speakers.SilentBoxes is { Count: > MaxSilentBoxes })
            return $"A multi-speaker request marks at most {MaxSilentBoxes} silent people.";
        if (speakers.SilentBoxes?.Any(static box => box is null) == true)
            return "A silent person's box is missing.";
        return BoxRefusal(speakers.First.Box, "first speaker's", image)
            ?? BoxRefusal(speakers.Second.Box, "second speaker's", image)
            ?? speakers.SilentBoxes?.Select(box => BoxRefusal(box, "silent person's", image)).FirstOrDefault(static refusal => refusal is not null);
    }

    private static string? BoxRefusal(VideoSpeakerBox? box, string whose, ImageData image) =>
        box is null || (Box(box).IsValid && box.YMin < image.Height && box.XMin < image.Width)
            ? null
            : $"The {whose} box (rows {box.YMin} to {box.YMax}, columns {box.XMin} to {box.XMax}) must have non-negative corners, a positive size "
                + $"and start inside the {image.Width}x{image.Height} init image.";

    /// <summary>The two speakers' encoder inputs and the soundtrack for a clip of <paramref name="clipFrames"/> frames.</summary>
    internal static LongCatSpeakerAudio PrepareAudio(VideoSpeakers speakers, int clipFrames, int fps, bool normalizeLoudness)
    {
        LongCatSpeakerSpeech? first = Speech(speakers.First), second = Speech(speakers.Second);
        LongCatSpeakerTiming timing = speakers.Timing == VideoSpeakerTiming.Sequential
            ? LongCatSpeakerTiming.Sequential
            : LongCatSpeakerTiming.Parallel;
        double clipSeconds = clipFrames / (double)fps;
        if (first is { } one && second is { } two)
        {
            double oneSeconds = one.Recording.Length / (double)LongCatDrivingAudio.SampleRate;
            double twoSeconds = two.Recording.Length / (double)LongCatDrivingAudio.SampleRate;
            if (timing == LongCatSpeakerTiming.Parallel && Math.Min(oneSeconds, twoSeconds) + 1e-6 < clipSeconds)
                Logs.Warning($"[LongCatAvatarSpeakers] Parallel speech ends with the shorter recording ({Math.Min(oneSeconds, twoSeconds):F2} s of a "
                    + $"{clipSeconds:F2} s clip); the rest of the clip is silence.");
            if (timing == LongCatSpeakerTiming.Sequential && oneSeconds >= clipSeconds)
                Logs.Warning($"[LongCatAvatarSpeakers] The first speaker's recording ({oneSeconds:F2} s) fills the {clipSeconds:F2} s "
                    + "clip, so the second speaker is not heard.");
        }
        return LongCatDrivingAudio.PrepareSpeakers(first, second, timing, clipFrames, fps, normalizeLoudness,
            silentTrack: speakers.SilentBoxes is { Count: > 0 });
    }

    /// <summary>The conditioning of the DiT: the regions of <paramref name="image"/> on the token grid of a <paramref name="width"/> by
    /// <paramref name="height"/> generation, for <paramref name="tracks"/> audio tracks.</summary>
    internal static LongCatSpeakers Conditioning(VideoSpeakers speakers, ImageData image, int tracks, int width, int height, int gridHeight,
        int gridWidth)
    {
        LongCatSpeakerBox[] silent = [.. (speakers.SilentBoxes ?? []).Select(Box)];
        (int regions, float[] masks) = LongCatSpeakerMasks.TokenMasks(image.Height, image.Width,
            speakers.First.Box is { } first ? Box(first) : null, speakers.Second.Box is { } second ? Box(second) : null,
            silent, height, width, gridHeight, gridWidth);
        int tokens = gridHeight * gridWidth;
        for (int region = 0; region < 2; region++)
            if (!masks.AsSpan(region * tokens, tokens).Contains(1f))
                Logs.Warning($"[LongCatAvatarSpeakers] The {(region == 0 ? "first" : "second")} speaker's region has no token after the crop to "
                    + $"{width}x{height}; no part of the video is tied to that speaker.");
        return new LongCatSpeakers(tracks, regions, gridHeight, gridWidth, masks);
    }

    private static LongCatSpeakerBox Box(VideoSpeakerBox box) => new(box.YMin, box.XMin, box.YMax, box.XMax);

    private static LongCatSpeakerSpeech? Speech(VideoSpeaker speaker)
    {
        if (speaker.Audio is not { } clip)
            return null;
        float[] recording = VideoRecipeUtils.DecodeMono16k(clip);
        return new LongCatSpeakerSpeech(speaker.SeparatedSpeech16k ?? recording, recording);
    }
}
