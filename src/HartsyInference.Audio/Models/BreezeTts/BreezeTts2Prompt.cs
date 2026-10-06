namespace HartsyInference.Audio.Models.BreezeTts;

/// <summary>One piece of a Breeze prompt: a text segment (encoded on its own by the T5Gemma2 encoder) or an audio segment
/// (frames of 16 codes, each placed as one backbone position).</summary>
public abstract record BreezeSegment;

public sealed record BreezeTextSegment(string Text) : BreezeSegment;

/// <param name="Frames"><c>[T][numCodebooks]</c> codes of a reference clip.</param>
/// <param name="AppendEos">Adds the <c>&lt;|audio_eos|&gt;</c> position after the frames (all codes = codebook EOS id).</param>
/// <param name="DropLastFrame">Removes the last frame (not used by the shipped templates).</param>
public sealed record BreezeAudioSegment(int[][] Frames, bool AppendEos = true, bool DropLastFrame = false) : BreezeSegment;

/// <summary>Port of <c>breeze_infer/templates.py</c>: picks the template from the request fields and builds the positive
/// and negative (classifier-free guidance) segment lists.</summary>
public static class BreezeTts2Templates
{
    public const string InstructionBos = "<ins_bos>";
    public const string InstructionEos = "<ins_eos>";

    public enum Kind { TtsPlain, TtsInstruction, RefCloneTata, RefEditTata }

    public sealed record Request
    {
        public required string Text { get; init; }
        public string? Instruction { get; init; }
        public int[][]? RefFrames { get; init; }
        public string? RefText { get; init; }
        /// <summary>Speaker tag; <c>S0</c> by default, an already bracketed <c>[S1]</c> is kept, empty = none.</summary>
        public string Speaker { get; init; } = "S0";
    }

    public static Kind Select(Request r)
    {
        bool hasRef = r.RefFrames is { Length: > 0 }, hasRefText = !string.IsNullOrWhiteSpace(r.RefText);
        if (hasRef != hasRefText) throw new ArgumentException("A reference clip and its transcript must be provided together.");
        bool ins = !string.IsNullOrWhiteSpace(r.Instruction);
        return hasRef ? (ins ? Kind.RefEditTata : Kind.RefCloneTata) : (ins ? Kind.TtsInstruction : Kind.TtsPlain);
    }

    private static string Prefix(Request r)
    {
        if (string.IsNullOrEmpty(r.Speaker)) return "";
        return r.Speaker.StartsWith('[') && r.Speaker.EndsWith(']') ? r.Speaker : $"[{r.Speaker}]";
    }

    private static List<BreezeSegment> Plain(Request r) => [new BreezeTextSegment($"{Prefix(r)}{r.Text}")];

    private static List<BreezeSegment> Instruction(Request r) =>
        [new BreezeTextSegment($"{Prefix(r)}{InstructionBos}{r.Instruction}{InstructionEos}{r.Text}")];

    private static List<BreezeSegment> Clone(Request r) =>
    [
        new BreezeTextSegment($"{Prefix(r)}{r.RefText}"),
        new BreezeAudioSegment(r.RefFrames!),
        new BreezeTextSegment($"{Prefix(r)}{r.Text}"),
    ];

    private static List<BreezeSegment> Edit(Request r) =>
    [
        new BreezeTextSegment($"{Prefix(r)}{r.RefText}"),
        new BreezeAudioSegment(r.RefFrames!),
        new BreezeTextSegment($"{Prefix(r)}{InstructionBos}{r.Instruction}{InstructionEos}{r.Text}"),
    ];

    public static List<BreezeSegment> Positive(Request r) => Select(r) switch
    {
        Kind.TtsPlain => Plain(r), Kind.TtsInstruction => Instruction(r), Kind.RefCloneTata => Clone(r), _ => Edit(r),
    };

    /// <summary>The guidance branch, or null when the template defines none (guidance is then rejected upstream).</summary>
    public static List<BreezeSegment>? Negative(Request r) => Select(r) switch
    {
        Kind.TtsInstruction => Plain(r), Kind.RefEditTata => Clone(r), _ => null,
    };
}
