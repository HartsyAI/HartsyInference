namespace HartsyInference.Vision.Detection.GroundingDino;

/// <summary>Caption token layout for Grounding DINO: the block-diagonal text self-attention mask and the reset
/// position ids that make each dotted phrase ("a cat. a remote control.") an independent sub-sentence. Faithful port
/// of <c>generate_masks_with_special_tokens_and_transfer_map</c> — tokens attend only within the span bounded by the
/// nearest enclosing special tokens (<c>[CLS] [SEP] . ?</c>), always including self; position ids restart at 0 after
/// each delimiter.</summary>
public static class GroundingDinoTextPrompt
{
    /// <summary>BERT ids for <c>[CLS]</c>, <c>[SEP]</c>, <c>.</c>, <c>?</c> — the phrase delimiters.</summary>
    public static readonly int[] SpecialTokens = [101, 102, 1012, 1029];

    /// <summary>Builds the <c>[T, T]</c> boolean attend mask (true = attend) and the <c>[T]</c> position ids.</summary>
    public static (bool[] AttendMask, int[] PositionIds) Build(ReadOnlySpan<int> inputIds)
    {
        int t = inputIds.Length;
        bool[] special = new bool[t];
        for (int i = 0; i < t; i++)
            special[i] = Array.IndexOf(SpecialTokens, inputIds[i]) >= 0;

        int[] prevSpecial = new int[t];
        int running = -1;
        for (int i = 0; i < t; i++)
        {
            if (special[i]) running = i;
            prevSpecial[i] = running;
        }

        int[] nextSpecial = new int[t];
        int runningNext = t;
        for (int i = t - 1; i >= 0; i--)
        {
            if (special[i]) runningNext = i;
            nextSpecial[i] = runningNext;
        }

        bool[] validBlock = new bool[t];
        for (int i = 0; i < t; i++)
            validBlock[i] = nextSpecial[i] != 0 && nextSpecial[i] != t - 1 && nextSpecial[i] != t;

        bool[] attend = new bool[(long)t * t];
        for (int i = 0; i < t; i++)
            for (int j = 0; j < t; j++)
            {
                bool block = nextSpecial[i] == nextSpecial[j] && validBlock[j];
                attend[(long)i * t + j] = block || i == j;
            }

        int[] positionIds = new int[t];
        for (int i = 0; i < t; i++)
        {
            int p = validBlock[i] ? i - prevSpecial[i] - 1 : 0;
            positionIds[i] = p < 0 ? 0 : p;
        }

        return (attend, positionIds);
    }
}
