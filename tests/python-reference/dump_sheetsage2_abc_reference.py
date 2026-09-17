"""Dumps the string-exact ABC the released SheetSage2 serializer writes for synthetic event streams.

Gate S7. The serializer's output is a dialect an external validator checks, so the only useful reference is the
text itself, character for character. The streams below are hand-built rather than decoded from a checkpoint
because a real transcription exercises almost none of the hard paths: a 4/4 pop song never changes meter, never
carries a pickup, and never needs a flat-root slash chord. Each case is aimed at one of them.

The module depends only on numpy, so it is imported directly rather than stubbed:

    python3 dump_sheetsage2_abc_reference.py --source /path/to/ComfyUI/comfy/audio_encoders/sheetsage2_abc.py

Pass --real-events to fold in a transcription the released model actually produced, which pins the event shapes
it really emits rather than the ones this file guesses at:

    python3 dump_sheetsage2_abc_reference.py --source .../sheetsage2_abc.py --real-events .../ref_events.json

Writes sheetsage2_reference/abc.json. Also pinned per case is the intermediate score, because the inferred
measure table carries flags (pickup, partial, inferred, pad_before) that never reach the ABC text and so cannot
be caught by comparing it.

Never point --source at a running ComfyUI install's live service; a checkout is fine, it is only read.
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

BEAT = 0.5
SUBBEAT = BEAT / 4


def load_reference(source: Path):
    spec = importlib.util.spec_from_file_location("sheetsage2_abc_reference", source)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: the module's dataclasses resolve their own annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def event(time, *, meter=None, eighth=None, key=None, structure=None, chord=None, melody=()):
    """One timed event. melody entries are (pitch, track, end_time)."""
    values = {}
    if meter is not None or eighth is not None:
        rhythm = {}
        if meter is not None:
            rhythm["meter"] = meter
        if eighth is not None:
            rhythm["eighth_position"] = eighth
        values["rhythm"] = rhythm
    if key is not None:
        values["key"] = key
    if structure is not None:
        values["structure"] = structure
    if chord is not None:
        values["chord"] = chord
    if melody:
        values["melody"] = [{"pitch": p, "track": t, "end_time": e} for p, t, e in melody]
    return {"time": time, "values": values}


def merge(events, extra):
    """Folds loose field events onto the beat event at the same time, keeping one event per instant."""
    by_time = {}
    order = []
    for source in list(events) + list(extra):
        key = round(source["time"], 9)
        if key in by_time:
            target = by_time[key]["values"]
            for field, value in source["values"].items():
                if field == "melody":
                    target.setdefault("melody", []).extend(value)
                elif field == "rhythm":
                    target.setdefault("rhythm", {}).update(value)
                else:
                    target[field] = value
        else:
            by_time[key] = {"time": source["time"], "values": dict(source["values"])}
            order.append(key)
    return [by_time[key] for key in sorted(order)]


def beats(start, meter, eighths, period=BEAT):
    """A run of beat events, one per entry of eighths, spaced by period."""
    return [event(start + index * period, meter=meter, eighth=value) for index, value in enumerate(eighths)]


def common_time(start, bars, period=BEAT):
    return beats(start, (4, 4), [0, 2, 4, 6] * bars, period)


def case_common_time():
    """Four plain 4/4 bars: chords per beat, both voices busy, one length that has to split and one note that
    crosses a barline."""
    grid = common_time(0.0, 4)
    extra = [
        event(0.0, key="C:major", structure="verse", chord="C:maj",
              melody=[(64, 0, 0.5), (48, 1, 2.0)]),
        event(0.5, melody=[(67, 0, 1.0)]),
        event(1.0, chord="A:min", melody=[(69, 0, 2.0)]),
        event(2.0, chord="F:maj", melody=[(65, 0, 2.75), (53, 1, 4.0)]),
        event(2.75, melody=[(64, 0, 3.0)]),
        event(3.0, chord="G:7", melody=[(62, 0, 4.0)]),
        # 13 subbeats: not a single ABC length, so it is written 12 + 1 and tied.
        event(4.0, structure="chorus", chord="C:maj", melody=[(72, 0, 5.625), (43, 1, 6.0)]),
        # Crosses the barline at 6.0, so it is split there and tied.
        event(5.75, melody=[(71, 0, 6.5)]),
        event(6.0, chord="G:7", melody=[(48, 1, 8.0)]),
        event(6.5, melody=[(67, 0, 8.0)]),
    ]
    return merge(grid, extra), 8.0


def case_meter_change():
    """A declared meter change: two 4/4 bars then two 3/4 bars."""
    grid = common_time(0.0, 2) + beats(4.0, (3, 4), [0, 2, 4] * 2)
    extra = [
        event(0.0, key="G:major", structure="intro", chord="G:maj", melody=[(67, 0, 2.0), (55, 1, 4.0)]),
        event(2.0, chord="D:maj", melody=[(74, 0, 4.0)]),
        event(4.0, structure="verse", chord="E:min", melody=[(76, 0, 5.5), (52, 1, 7.0)]),
        event(5.5, chord="C:maj", melody=[(71, 0, 7.0)]),
    ]
    return merge(grid, extra), 7.0


def case_pickup_and_keys():
    """A two-beat pickup that has to be padded out to a full bar, a key change on a downbeat, and a second one
    inside a bar."""
    grid = beats(0.0, (4, 4), [4, 6]) + common_time(1.0, 3)
    extra = [
        event(0.0, key="C:major", structure="intro", chord="G:7", melody=[(67, 0, 1.0)]),
        event(1.0, chord="C:maj", melody=[(60, 0, 2.0), (36, 1, 3.0)]),
        event(2.0, chord="A:min", melody=[(64, 0, 3.0)]),
        event(3.0, key="A:minor", structure="verse", chord="A:min", melody=[(69, 0, 4.25), (45, 1, 5.0)]),
        event(4.25, key="Eb:major", chord="Eb:maj", melody=[(63, 0, 5.0)]),
        event(5.0, chord="Bb:7", melody=[(70, 0, 7.0), (46, 1, 7.0)]),
    ]
    return merge(grid, extra), 7.0


def case_irregular_bars():
    """A three-beat bar in the middle of 4/4 (the model declared 4/4 and put the downbeat early), and a final
    bar the clip cuts short."""
    grid = (common_time(0.0, 1)
            + beats(2.0, (4, 4), [0, 2, 4])
            + common_time(3.5, 1)
            + beats(5.5, (4, 4), [0, 2, 4]))
    extra = [
        event(0.0, key="D:major", structure="verse", chord="D:maj", melody=[(62, 0, 2.0), (50, 1, 3.5)]),
        event(2.0, chord="A:maj", melody=[(69, 0, 3.5)]),
        event(3.5, chord="B:min", melody=[(71, 0, 5.5), (47, 1, 7.0)]),
        event(5.5, chord="G:maj", melody=[(67, 0, 7.0)]),
    ]
    return merge(grid, extra), 7.0


def case_denominator_conflict():
    """One bar whose beats disagree about the beat value; the mode wins at the boundary."""
    grid = [
        event(0.0, meter=(4, 4), eighth=0),
        event(0.5, meter=(4, 4), eighth=2),
        event(1.0, meter=(4, 8), eighth=2),
        event(1.5, meter=(4, 8), eighth=3),
    ] + common_time(2.0, 2)
    extra = [
        event(0.0, key="C:major", structure="instrumental", chord="C:maj", melody=[(60, 0, 2.0), (48, 1, 4.0)]),
        event(2.0, chord="F:maj", melody=[(65, 0, 4.0)]),
        event(4.0, chord="G:maj", melody=[(67, 0, 6.0), (43, 1, 6.0)]),
    ]
    return merge(grid, extra), 6.0


def case_chord_spellings():
    """Every quality the vocabulary can name, plus flat roots and slash basses the decoder never emits but the
    serializer accepts, and a key that has no ABC signature under its own spelling."""
    grid = common_time(0.0, 4)
    chords = ["C:maj/3", "A#:min7/b3", "Db:maj7/7", "Gb:min/5", "F#:7/b7", "E:hdim7", "B:sus4(b7)",
              "G:minmaj7", "D:dim7", "A:maj6", "E:min6", "C:sus2", "F:aug", "Bb:dim", "N", "Cb:maj"]
    extra = [event(index * BEAT, chord=chord) for index, chord in enumerate(chords)]
    extra += [
        event(0.0, key="F:major", structure="bridge", melody=[(65, 0, 4.0), (41, 1, 8.0)]),
        event(4.0, key="A#:major", melody=[(70, 0, 6.0)]),
        event(6.0, key="G#:minor", melody=[(68, 0, 8.0)]),
    ]
    return merge(grid, extra), 8.0


def case_silent_voices():
    """Six bars where each voice falls silent for a stretch, which is what multi-bar Z rests are for."""
    grid = common_time(0.0, 6)
    extra = [
        event(0.0, key="C:major", structure="verse", chord="C:maj", melody=[(60, 0, 2.0)]),
        event(2.0, chord="F:maj"),
        event(4.0, chord="G:maj"),
        event(6.0, chord="C:maj"),
        event(8.0, structure="chorus", chord="A:min", melody=[(45, 1, 10.0)]),
        event(10.0, chord="F:maj", melody=[(53, 1, 12.0)]),
    ]
    return merge(grid, extra), 12.0


def case_free_tempo():
    """Beats that do not land on round numbers, and a clip that runs past the last decoded beat so the grid has
    to be continued at the estimated period."""
    period = 0.4137
    grid = common_time(0.0, 3, period=period)
    extra = [
        event(0.0, key="Bb:major", structure="solo", chord="Bb:maj",
              melody=[(70, 0, 4 * period), (46, 1, 8 * period)]),
        event(4 * period, chord="Eb:maj", melody=[(75, 0, 8 * period)]),
        event(8 * period, chord="F:7", melody=[(74, 0, 12 * period)]),
    ]
    return merge(grid, extra), 12 * period + 1.6


def case_overlapping_notes():
    """Notes that collide on one voice. The serializer resolves them by trimming, never by refusing: two notes
    starting together leave only the one the sort puts last, and a note running into the next onset is cut back
    to it."""
    grid = common_time(0.0, 3)
    extra = [
        # Same onset, so the lower pitch is trimmed to zero length and dropped.
        event(0.0, key="C:major", structure="verse", chord="C:maj",
              melody=[(60, 0, 2.0), (64, 0, 1.0), (48, 1, 3.0)]),
        # Declares two and a half beats but the next note starts in one, so it is cut back.
        event(1.5, chord="F:maj", melody=[(67, 0, 3.0)]),
        event(2.0, melody=[(69, 0, 2.5)]),
        event(2.5, chord="G:maj", melody=[(71, 0, 6.0), (50, 1, 6.0)]),
    ]
    return merge(grid, extra), 6.0


CASES = {
    "common_time": case_common_time,
    "meter_change": case_meter_change,
    "pickup_and_keys": case_pickup_and_keys,
    "irregular_bars": case_irregular_bars,
    "denominator_conflict": case_denominator_conflict,
    "chord_spellings": case_chord_spellings,
    "silent_voices": case_silent_voices,
    "free_tempo": case_free_tempo,
    "overlapping_notes": case_overlapping_notes,
}

# Chord labels the decoder cannot produce but the serializer accepts, so the flat and slash-bass branches are
# pinned rather than left to a live transcription that never reaches them.
CHORD_SYMBOLS = [
    "N", "X", "?", "C:maj", "C:min", "C:dim", "C:aug", "C:7", "C:maj7", "C:min7", "C:dim7", "C:hdim7",
    "C:sus4", "C:sus2", "C:maj6", "C:min6", "C:sus4(b7)", "C:minmaj7", "C:maj/3", "C:maj/5", "C:min/b3",
    "C:7/b7", "C:maj7/7", "A#:min/b3", "Bb:maj/3", "Db:7/b7", "Gb:maj/5", "Cb:maj", "B#:min", "Fbb:maj",
    "G##:maj", "C:maj/E", "C:maj/Bb", "C:maj/9", "C:maj/13", "C:maj/#11", "F#:maj/bb7",
]

CHORD_FAILURES = ["Cmaj", "C:power", "C:maj/x", "H:maj", "C:maj/0", "C:maj/14"]

KEY_SYMBOLS = [
    "C:major", "A:minor", "F:major", "Bb:major", "Eb:major", "A#:major", "D#:major", "G#:minor",
    "Cb:major", "F#:major", "C#:major", "Gb:major", "Db:minor", "E#:major", "Bbb:major", "Cm", "F#m", "G",
]

KEY_FAILURES = ["C:dorian", "H:major", "C#:lydian"]

# Every pitch class across three octaves, under signatures from seven flats to seven sharps, which is what pins
# the key-relative spelling table and the Cb/B# octave crossing.
NOTE_KEYS = ["C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb", "Am", "G#m"]


def load_real_case(path):
    """Reads a transcription the released model actually produced, so the fixture is not limited to the event
    shapes this file guessed at. Its ABC renderings sit next to it and are checked rather than trusted."""
    payload = json.loads(path.read_text())
    events = [{"time": source["time"], "values": source["values"]} for source in payload["events"]]
    expected = {}
    for mode in ("melody", "full"):
        sibling = path.with_name(f"ref_abc_{mode}.abc")
        if sibling.exists():
            expected[mode] = sibling.read_text()
    return events, payload["duration"], expected


def strip_chords(text):
    """Removes quoted chord symbols, leaving header and comment lines alone — the voice headers carry quotes of
    their own, so a blanket substitution would mangle them."""
    lines = []
    for line in text.split("\n"):
        if line.startswith("%") or re.match(r"^[A-Za-z]:", line):
            lines.append(line)
        else:
            lines.append(re.sub(r'"[^"]*"', "", line))
    return "\n".join(lines)


def rle(values):
    """Run-length encodes a per-subbeat array as [count, value] pairs. Lossless, and it keeps the fixture small
    enough to read: a chord track is one pair per chord rather than four entries per beat."""
    runs = []
    for value in values:
        if runs and runs[-1][1] == value:
            runs[-1][0] += 1
        else:
            runs.append([1, value])
    return runs


def measure_row(measure):
    return {
        "index": measure.index,
        "startBeat": measure.start_beat,
        "endBeat": measure.end_beat,
        "numerator": measure.numerator,
        "denominator": measure.denominator,
        "pickup": measure.pickup,
        "partial": measure.partial,
        "inferred": measure.inferred,
        "notatedNumerator": measure.notated_numerator,
        "notatedDenominator": measure.notated_denominator,
        "padBefore": measure.pad_before,
        "abcNumerator": measure.abc_numerator,
        "abcDenominator": measure.abc_denominator,
    }


def dump_events(events):
    rows = []
    for source in events:
        values = source["values"]
        rhythm = values.get("rhythm", {})
        rows.append({
            "time": source["time"],
            "meter": list(rhythm["meter"]) if "meter" in rhythm else None,
            "eighthPosition": rhythm.get("eighth_position"),
            "key": values.get("key"),
            "structure": values.get("structure"),
            "chord": values.get("chord"),
            "melody": [{"pitch": n["pitch"], "track": n["track"], "endTime": n["end_time"]}
                       for n in values.get("melody", [])],
        })
    return rows


def run_case(module, events, duration):
    captured = []
    original = module.score_to_abc

    def capture(score):
        captured.append(score)
        return original(score)

    module.score_to_abc = capture
    try:
        melody = module.events_to_abc(events, duration, melody_only=True)
        full = module.events_to_abc(events, duration, melody_only=False)
    finally:
        module.score_to_abc = original
    score = captured[-1]
    stripped = strip_chords(full)
    return {
        "duration": duration,
        "events": dump_events(events),
        "melodyOnly": melody,
        "full": full,
        "strippedFull": stripped,
        "strippedFullEqualsMelodyOnly": stripped == melody,
        "score": {
            "beatCount": len(score.beats),
            "subbeatCount": len(score.subbeat_times),
            "unitDenominator": module.abc_unit_denominator(score),
            "tempo": module.estimate_tempo(score),
            "diagnostics": list(score.diagnostics),
            "measures": [measure_row(measure) for measure in score.measures],
            "subbeatTimes": [float(value) for value in score.subbeat_times],
            "subbeatQuarters": [float(value) for value in score.subbeat_quarters],
            "subbeatDenominators": rle([int(value) for value in score.subbeat_denominators]),
            "keyArr": rle([str(value) for value in score.key_arr]),
            "chordArr": rle([str(value) for value in score.chord_arr]),
            "structureEvents": [[int(t), str(label)] for t, label in score.structure_events],
            "voiceArrs": {voice: rle([int(value) for value in array]) for voice, array in score.voice_arrs.items()},
        },
    }


def failure_row(module, call):
    try:
        call()
    except module.AbcRebuildError as error:
        return type(error).__name__
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Path to the released sheetsage2_abc.py")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("sheetsage2_reference"))
    parser.add_argument("--real-events", type=Path, default=None,
                        help="ref_events.json from a real transcription, added as the real_piano30 case")
    args = parser.parse_args()

    module = load_reference(args.source)
    payload = {"subbeatDivision": module.SUBBEAT_DIVISION, "voiceIds": list(module.VOICE_IDS)}

    payload["cases"] = {}
    for name, build in CASES.items():
        events, duration = build()
        payload["cases"][name] = run_case(module, events, duration)

    if args.real_events is not None:
        events, duration, expected = load_real_case(args.real_events)
        case = run_case(module, events, duration)
        # The renderings that shipped with the transcription have to come back out of it unchanged, or the case
        # is pinning this script's reading of the events rather than the model's own output.
        for mode, text in expected.items():
            rendered = case["melodyOnly"] if mode == "melody" else case["full"]
            if rendered != text:
                raise SystemExit(f"{args.real_events.with_name(f'ref_abc_{mode}.abc')} is not what the serializer "
                                 f"produces from ref_events.json; refusing to write a fixture that hides that")
        payload["cases"]["real_piano30"] = case

    payload["chordSymbols"] = {chord: module.chord_symbol_to_abc(chord) for chord in CHORD_SYMBOLS}
    payload["chordFailures"] = {
        chord: failure_row(module, lambda chord=chord: module.chord_symbol_to_abc(chord))
        for chord in CHORD_FAILURES
    }
    payload["keySymbols"] = {key: module.key_symbol_to_abc(key) for key in KEY_SYMBOLS}
    payload["keyFailures"] = {
        key: failure_row(module, lambda key=key: module.key_symbol_to_abc(key)) for key in KEY_FAILURES
    }
    payload["keyAccidentals"] = {
        key: module.get_key_accidentals(key)
        for key in sorted(module._KEY_SIGNATURE_ACCIDENTALS)
    }
    payload["noteSpellings"] = {}
    for key in NOTE_KEYS:
        accidentals = module.get_key_accidentals(key)
        payload["noteSpellings"][key] = [
            module.note_to_abc(note, accidentals, {}) for note in range(48, 84)
        ]
    # The same bar state reused across notes, which is what suppresses a repeated accidental.
    running = {}
    payload["noteSpellingsRunning"] = [
        module.note_to_abc(note, module.get_key_accidentals("Eb"), running)
        for note in (63, 75, 62, 74, 63, 60, 61)
    ]
    payload["durationSplits"] = {
        str(value): module._split_duration_units(value)
        for value in (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17, 23, 31, 48, 49, 63, 64, 96, 100, 127, 256)
    }
    payload["compressibleFullRest"] = {
        text: module._is_compressible_full_rest(text)
        for text in ("z16", "z8z8", "z12z3z", "z", "", "c16", "z8c8", '"C"z16', "[K:C]z16", "z16-", "z8 z8",
                     "z8x", "Z", "z4z4z4z4")
    }

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "abc.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    equal = sum(1 for case in payload["cases"].values() if case["strippedFullEqualsMelodyOnly"])
    print(f"wrote {path} ({len(payload['cases'])} cases, {equal} of them where stripping the chords out of the "
          f"full rendering reproduces the melody-only rendering)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
