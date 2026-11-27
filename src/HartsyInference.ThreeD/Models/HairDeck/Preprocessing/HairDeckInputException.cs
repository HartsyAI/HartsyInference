namespace HartsyInference.ThreeD.Models.HairDeck.Preprocessing;

/// <summary>A HairDeck input that cannot be preprocessed; <see cref="Failure"/> says which rule it broke.</summary>
public sealed class HairDeckInputException : Exception
{
    /// <summary>Creates the exception for one failure kind.</summary>
    public HairDeckInputException(HairDeckInputFailure failure, string message)
        : base(message)
    {
        Failure = failure;
    }

    /// <summary>The failure kind.</summary>
    public HairDeckInputFailure Failure { get; }
}
