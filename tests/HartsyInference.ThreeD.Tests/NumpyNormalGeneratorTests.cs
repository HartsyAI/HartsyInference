using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Tests;

/// <summary>Draws checked against numpy 2.5.4: np.random.default_rng(0).standard_normal and .normal(0.93, 0.07).</summary>
public sealed class NumpyNormalGeneratorTests
{
    private static readonly double[] FirstBrightnessDraws =
    [
        0.9388011154765376, 0.9207526595696089, 0.9748295855310298, 0.9373430082007128,
        0.8925031438787223, 0.955311653843664, 1.0212800031591096, 0.996295667419047,
    ];

    [Fact]
    public void Seed_zero_first_standard_normal_matches_numpy()
    {
        NumpyNormalGenerator generator = new(0);

        Assert.Equal(0.1257302210933933, generator.Normal(0.0, 1.0), 15);
    }

    [Fact]
    public void Seed_zero_scaled_draws_match_numpy_normal()
    {
        NumpyNormalGenerator generator = new(0);

        for (int i = 0; i < FirstBrightnessDraws.Length; i++)
            Assert.Equal(FirstBrightnessDraws[i], generator.Normal(0.93, 0.07), 14);
    }

    [Fact]
    public void Long_stream_matches_numpy_checksums()
    {
        NumpyNormalGenerator generator = new(0);
        double sum = 0;
        double sumOfSquares = 0;
        double last = 0;
        int tails = 0;
        for (int i = 0; i < 200000; i++)
        {
            last = generator.Normal(0.0, 1.0);
            sum += last;
            sumOfSquares += last * last;
            if (Math.Abs(last) > 3.6541528853610088)
                tails++;
        }

        Assert.Equal(26.135110527473202, sum, 9);
        Assert.Equal(200480.26556428347, sumOfSquares, 6);
        Assert.Equal(-0.26553977625065545, last, 15);
        Assert.Equal(58, tails);
    }
}
