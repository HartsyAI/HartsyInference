using HartsyInference.Core.Exceptions;

namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>
/// Skeleton-tree tokenizer of the articulation-xl 256 checkpoint: 256 coordinate bins over [-1, 1], then branch, BOS,
/// EOS, PAD, spring, part and class tokens. Ids follow upstream's <c>TokenizerPart</c> for the shipped config.
/// </summary>
public sealed class UniRigTokenizer
{
    /// <summary>Number of coordinate bins.</summary>
    public const int NumDiscrete = 256;

    /// <summary>Marks a joint whose parent is given explicitly.</summary>
    public const int BranchToken = NumDiscrete;

    public const int BosToken = NumDiscrete + 1;
    public const int EosToken = NumDiscrete + 2;
    public const int PadToken = NumDiscrete + 3;
    public const int SpringToken = NumDiscrete + 4;
    public const int ClsNoneToken = FirstPartToken + 2;

    /// <summary>Token of the <c>articulationxl</c> class, the only class the shipped checkpoint was trained on.</summary>
    public const int ArticulationXlToken = FirstClsToken + 2;

    /// <summary>Total vocabulary: 267 for the shipped config.</summary>
    public const int VocabSize = FirstClsToken + 3;

    private const int FirstPartToken = NumDiscrete + 5;
    private const int FirstClsToken = ClsNoneToken + 1;

    private static readonly string[] PartNames = ["body", "hand"];
    private static readonly string[] ClassNames = ["vroid", "mixamo", "articulationxl"];

    /// <summary>Token of a part name (<c>body</c> or <c>hand</c>).</summary>
    public static int PartToken(string part)
    {
        int index = Array.IndexOf(PartNames, part);
        return index < 0 ? throw new HartsyInferenceException($"UniRig tokenizer has no part '{part}'.") : FirstPartToken + index;
    }

    /// <summary>Token of a class name; unknown names map to the no-class token like upstream.</summary>
    public static int ClsToken(string? cls)
    {
        int index = cls is null ? -1 : Array.IndexOf(ClassNames, cls);
        return index < 0 ? ClsNoneToken : FirstClsToken + index;
    }

    /// <summary>True for a part or spring token.</summary>
    public static bool IsPart(int token) => token is SpringToken or (>= FirstPartToken and < ClsNoneToken);

    /// <summary>True for a class token or the no-class token.</summary>
    public static bool IsCls(int token) => token >= ClsNoneToken && token < VocabSize;

    /// <summary>Bin of a coordinate: <c>clip(round((t + 1) / 2 * 256), 0, 255)</c>.</summary>
    public static int Discretize(float value)
    {
        double scaled = ((double)value + 1.0) / 2.0 * NumDiscrete;
        return (int)Math.Clamp(Math.Round(scaled, MidpointRounding.ToEven), 0, NumDiscrete - 1);
    }

    /// <summary>Bin centre: <c>(bin + 0.5) / 256 * 2 - 1</c>.</summary>
    public static float Undiscretize(int bin) => ((float)bin + 0.5f) / NumDiscrete * 2f - 1f;

    /// <summary>Encodes a skeleton into <c>BOS, class, [part], [branch, parent xyz], joint xyz, … EOS</c>.</summary>
    public static int[] Tokenize(UniRigTokenizeInput input)
    {
        ArgumentNullException.ThrowIfNull(input);
        int count = input.Branch.Length;
        if (input.Joints.Length != 3 * count || input.ParentJoints.Length != 3 * count)
        {
            throw new HartsyInferenceException("UniRig tokenize: joint arrays must hold 3 floats per joint.");
        }

        List<int> tokens = [BosToken, ClsToken(input.Cls)];
        for (int i = 0; i < count; i++)
        {
            if (input.PartsBias.TryGetValue(i, out string? part))
            {
                tokens.Add(part is null ? SpringToken : PartToken(part));
            }

            if (input.Branch[i])
            {
                tokens.Add(BranchToken);
                for (int axis = 0; axis < 3; axis++)
                {
                    tokens.Add(Discretize(input.ParentJoints[3 * i + axis]));
                }
            }

            for (int axis = 0; axis < 3; axis++)
            {
                tokens.Add(Discretize(input.Joints[3 * i + axis]));
            }
        }

        tokens.Add(EosToken);
        return [.. tokens];
    }

    /// <summary>Decodes a full token sequence (BOS … EOS, trailing PAD allowed) into a skeleton, throwing on a malformed sequence.</summary>
    public static UniRigSkeleton Detokenize(ReadOnlySpan<int> ids)
    {
        if (ids.Length == 0 || ids[0] != BosToken)
        {
            throw new HartsyInferenceException("UniRig detokenize: first token is not BOS.");
        }

        int end = ids.Length;
        while (end > 0 && ids[end - 1] == PadToken)
        {
            end--;
        }

        if (end < 2 || ids[end - 1] != EosToken)
        {
            throw new HartsyInferenceException("UniRig detokenize: last token is not EOS.");
        }

        ReadOnlySpan<int> body = ids[1..(end - 1)];
        List<float> joints = [];
        List<float> parentJoints = [];
        Dictionary<int, float[]> tails = [];
        List<string?> parts = [];
        float[]? lastJoint = null;
        bool isBranch = false;
        int bones = 0;
        int i = 0;
        while (i < body.Length)
        {
            int token = body[i];
            if (token < NumDiscrete)
            {
                int need = isBranch ? 6 : 3;
                if (i + need > body.Length || !AllBins(body.Slice(i, need)))
                {
                    throw new HartsyInferenceException("UniRig detokenize: truncated or interrupted joint coordinates.");
                }

                float[] current = [Undiscretize(body[i + need - 3]), Undiscretize(body[i + need - 2]), Undiscretize(body[i + need - 1])];
                float[] parent;
                if (isBranch)
                {
                    parent = [Undiscretize(body[i]), Undiscretize(body[i + 1]), Undiscretize(body[i + 2])];
                }
                else if (bones == 0)
                {
                    parent = current;
                }
                else
                {
                    parent = lastJoint ?? throw new HartsyInferenceException("UniRig detokenize: joint without a preceding joint.");
                }

                joints.AddRange(current);
                parentJoints.AddRange(parent);
                i += need;
                if (lastJoint is not null)
                {
                    tails[bones - 1] = current;
                }

                lastJoint = current;
                bones++;
                isBranch = false;
            }
            else if (token == BranchToken)
            {
                isBranch = true;
                lastJoint = null;
                i++;
            }
            else if (IsPart(token))
            {
                parts.Add(token == SpringToken ? null : PartNames[token - FirstPartToken]);
                i++;
            }
            else if (IsCls(token))
            {
                i++;
            }
            else
            {
                throw new HartsyInferenceException($"UniRig detokenize: unexpected token {token}.");
            }
        }

        if (bones == 0)
        {
            throw new HartsyInferenceException("UniRig detokenize: sequence holds no joints.");
        }

        return UniRigSkeletonBuilder.Build([.. joints], [.. parentJoints], tails, [.. parts]);
    }

    private static bool AllBins(ReadOnlySpan<int> tokens)
    {
        foreach (int token in tokens)
        {
            if (token >= NumDiscrete)
            {
                return false;
            }
        }

        return true;
    }
}
