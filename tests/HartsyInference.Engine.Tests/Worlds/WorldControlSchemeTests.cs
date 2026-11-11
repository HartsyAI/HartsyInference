using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Worlds;
using HartsyInference.World.Camera;

namespace HartsyInference.Engine.Tests.Worlds;

public sealed class WorldControlSchemeTests
{
    [Fact]
    public void SanaWm_maps_held_keys_to_camera_letters_in_dsl_order()
    {
        IWorldControlScheme scheme = SanaWmControlScheme.Instance;

        Assert.Equal("none", scheme.ActionFor([]));
        Assert.Equal("w", scheme.ActionFor(["W"]));
        Assert.Equal("wa", scheme.ActionFor(["A", "W"]));
        Assert.Equal("ijl", scheme.ActionFor(["L", "J", "I"]));
        Assert.Equal("w", scheme.ActionFor(["w"]));
        Assert.Equal("none", scheme.ActionFor(["Space"]));
        Assert.Equal(WorldControlCadence.OnChange, scheme.Cadence);
    }

    [Fact]
    public void SanaWm_every_key_combination_is_a_valid_camera_script_keyset()
    {
        IWorldControlScheme scheme = SanaWmControlScheme.Instance;
        (string Key, SanaWmControls Flag)[] map =
        [
            ("W", SanaWmControls.Forward), ("S", SanaWmControls.Back), ("A", SanaWmControls.YawLeft), ("D", SanaWmControls.YawRight),
            ("I", SanaWmControls.PitchUp), ("K", SanaWmControls.PitchDown), ("J", SanaWmControls.StrafeLeft), ("L", SanaWmControls.StrafeRight),
        ];
        for (int mask = 0; mask < 1 << map.Length; mask++)
        {
            List<string> held = [];
            SanaWmControls expected = SanaWmControls.None;
            for (int bit = 0; bit < map.Length; bit++)
            {
                if ((mask & 1 << bit) != 0)
                {
                    held.Add(map[bit].Key);
                    expected |= map[bit].Flag;
                }
            }
            Assert.Equal(expected, SanaWmAction.ParseKeys(scheme.ActionFor(held)));
        }
    }

    [Fact]
    public void Oasis_joins_keys_in_order_and_adds_the_camera_term()
    {
        IWorldControlScheme scheme = OasisControlScheme.Instance;

        Assert.Equal("none", scheme.ActionFor([]));
        Assert.Equal("forward+jump+camera:1,-1", scheme.ActionFor(["Up", "Space", "Right", "W"]));
        Assert.Equal("back+sprint", scheme.ActionFor(["LeftCtrl", "S"]));
        Assert.Equal("none", scheme.ActionFor(["Left", "Right"]));
        Assert.Equal("camera:-1,1", scheme.ActionFor(["Left", "Down"]));
        Assert.Equal(WorldControlCadence.PerStep, scheme.Cadence);
    }

    [Fact]
    public void Oasis_legend_and_actions_come_from_one_table()
    {
        IWorldControlScheme scheme = OasisControlScheme.Instance;

        // Keys at the start, middle and end of the legend: each names the action its legend row carries.
        Assert.Equal("forward", scheme.ActionFor(["W"]));
        Assert.Equal("sprint", scheme.ActionFor(["LeftCtrl"]));
        Assert.Equal("inventory", scheme.ActionFor(["Tab"]));
        Assert.Equal(["W", "S", "A", "D", "Space", "LeftShift", "LeftCtrl", "F", "E", "G", "Tab", "Left", "Right", "Up", "Down"],
            scheme.Controls.Select(control => control.Key).ToArray());
        Assert.Equal("look left", scheme.Controls[11].Label);
        // The arrow keys add no term of their own; they only feed the camera.
        Assert.Equal("camera:-1,0", scheme.ActionFor(["Left"]));
    }

    [Fact]
    public void Diamond_gives_one_discrete_action_per_step()
    {
        IWorldControlScheme scheme = DiamondControlScheme.Instance;

        Assert.Equal("noop", scheme.ActionFor([]));
        Assert.Equal("fire", scheme.ActionFor(["Space"]));
        Assert.Equal("left", scheme.ActionFor(["Space", "A"]));
        Assert.Equal("right", scheme.ActionFor(["Right", "Space"]));
        Assert.Equal("noop", scheme.ActionFor(["Left", "Right"]));
        Assert.Equal(WorldControlCadence.PerStep, scheme.Cadence);
    }

    [Fact]
    public void Live_models_declare_their_scheme_and_stage_one_declares_none()
    {
        Assert.IsType<SanaWmControlScheme>(Entry("sana-wm-streaming").World?.LiveControls);
        Assert.IsType<OasisControlScheme>(Entry("oasis").World?.LiveControls);
        Assert.IsType<DiamondControlScheme>(Entry("diamond").World?.LiveControls);
        Assert.Null(Entry("sana-wm").World?.LiveControls);
    }

    private static CatalogEntry Entry(string id) =>
        ModelCatalog.All.Single(entry => string.Equals(entry.Id, id, StringComparison.OrdinalIgnoreCase));
}
