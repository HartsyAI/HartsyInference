namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>
/// Incremental form of upstream's <c>next_posible_token</c>: the token-sequence state machine that decides which ids may
/// follow. Feed it every token with <see cref="Advance"/>; <see cref="Allows"/> then answers for the next position.
/// </summary>
public sealed class UniRigGrammar
{
    private enum State
    {
        ExpectBos,
        ExpectClsOrPartOrJoint,
        ExpectPartOrJoint,
        ExpectJoint,
        ExpectJoint2,
        ExpectJoint3,
        ExpectBranchOrPartOrJoint,
    }

    private State _state = State.ExpectBos;

    /// <summary>Restarts the machine before a new sequence.</summary>
    public void Reset() => _state = State.ExpectBos;

    /// <summary>Consumes one token. A token the current state forbids is a caller error and throws.</summary>
    public void Advance(int token)
    {
        if (!Allows(token))
        {
            throw new InvalidOperationException($"UniRig grammar: token {token} is not allowed in state {_state}.");
        }

        _state = _state switch
        {
            State.ExpectBos => State.ExpectClsOrPartOrJoint,
            State.ExpectClsOrPartOrJoint => token < UniRigTokenizer.NumDiscrete ? State.ExpectJoint2
                : UniRigTokenizer.IsCls(token) ? State.ExpectPartOrJoint
                : State.ExpectJoint,
            State.ExpectPartOrJoint => token < UniRigTokenizer.NumDiscrete ? State.ExpectJoint2 : State.ExpectPartOrJoint,
            State.ExpectJoint => State.ExpectJoint2,
            State.ExpectJoint2 => State.ExpectJoint3,
            State.ExpectJoint3 => State.ExpectBranchOrPartOrJoint,
            State.ExpectBranchOrPartOrJoint => token == UniRigTokenizer.BranchToken ? State.ExpectJoint
                : token < UniRigTokenizer.NumDiscrete ? State.ExpectJoint2
                : State.ExpectJoint,
            _ => _state,
        };
    }

    /// <summary>True when <paramref name="token"/> may come next.</summary>
    public bool Allows(int token)
    {
        bool coordinate = token is >= 0 and < UniRigTokenizer.NumDiscrete;
        return _state switch
        {
            State.ExpectBos => token == UniRigTokenizer.BosToken,
            State.ExpectClsOrPartOrJoint => coordinate || UniRigTokenizer.IsCls(token) || UniRigTokenizer.IsPart(token),
            State.ExpectPartOrJoint => coordinate || UniRigTokenizer.IsPart(token) || token == UniRigTokenizer.EosToken,
            State.ExpectBranchOrPartOrJoint => coordinate || UniRigTokenizer.IsPart(token)
                || token is UniRigTokenizer.BranchToken or UniRigTokenizer.EosToken,
            _ => coordinate,
        };
    }

    /// <summary>Writes 0 for allowed ids and <see cref="float.NegativeInfinity"/> for the rest over <paramref name="mask"/>'s vocabulary.</summary>
    public void FillMask(Span<float> mask)
    {
        for (int token = 0; token < mask.Length; token++)
        {
            mask[token] = Allows(token) ? 0f : float.NegativeInfinity;
        }
    }
}
