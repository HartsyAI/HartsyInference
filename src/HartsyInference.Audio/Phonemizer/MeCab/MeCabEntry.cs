namespace HartsyInference.Audio.Phonemizer.MeCab;

/// <summary>One lexicon entry of a compiled MeCab dictionary: left and right context ids, word cost, and the offset of
/// its feature string.</summary>
internal readonly record struct MeCabEntry(ushort LeftId, ushort RightId, short Cost, uint FeatureOffset);
