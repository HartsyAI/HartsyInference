"""Dumps SheetSage2's window stitching from the released implementation, for the C# port to be checked against.

Covers the three pure functions that join a long transcription's windows into one stream — overlap_prefix,
event_time_map and stitched_window_events — plus whole simulated clips driven through sliding_window_plan. None
of it needs weights: the decodes are scripted token streams standing in for what the model would have written,
so the stitching arithmetic is exercised exactly as it would be on a real song.

Why it earns a gate of its own: an off-by-one in the accepted span duplicates or drops a bar at every seam, and
a seam only exists past five minutes of audio. The tolerance on a span's edge is asymmetric on purpose, so an
event landing exactly on a seam belongs to precisely one window; a port that symmetrised it would still pass
every short-clip test.

The source is read out of a ComfyUI checkout rather than imported, so nothing here needs comfy installed; torch
is stubbed, but numpy is real because event_time_map's median and interpolation are what is being pinned.

    /path/to/ComfyUI/venv/bin/python3 dump_sheetsage2_stitch_reference.py \
        --source /path/to/ComfyUI/comfy/audio_encoders/sheetsage2.py

Writes sheetsage2_reference/stitch.json. Never point --source at a running ComfyUI install's live service; a
checkout is fine, it is only read.
"""
import argparse
import copy
import json
import re
import sys
import types
from pathlib import Path

import numpy as np

# Clip lengths worth pinning, sharing the ones dump_sheetsage2_reference.py already pins for the plan itself:
# inside one window, exactly one window, one second over, two windows, and several seams.
CLIP_DURATIONS = (120.0, 300.0, 301.0, 450.0, 905.0)

# Seconds between synthetic events, and subbeats per gap. 2.5 s at the 0.125 s per subbeat the reference assumes
# when a window says nothing better. Every event time stays a multiple of 0.25 s, which is exact in binary AND an
# exact number of hundredths, so a timestamp token round trips without introducing rounding of its own.
GRID_SECONDS = 2.5
GRID_SUBBEATS = 20
SUBBEAT_SECONDS = 0.125


def load_tokenizer_source(source: Path) -> str:
    """The constants and ScoreTokenizer, sliced out of the released module."""
    src = source.read_text()
    blocks = []
    for block in re.split(r"\n(?=class |[A-Z_]+ = |def )", src[src.index("CHROMATIC_SHARPS = ("):]):
        if block.startswith(("class ScoreTokenizer", "FIELD_TO_INDEX")) or re.match(r"^[A-Z_]+ = ", block):
            blocks.append(block)
    return "\n".join(blocks)


def load_reference(source: Path):
    """Executes the tokenizer and the four stitching functions out of the released module."""
    src = source.read_text()
    torch = types.ModuleType("torch")
    sys.modules["torch"] = torch
    namespace = {"torch": torch, "np": np, "copy": copy}
    exec(load_tokenizer_source(source), namespace)
    exec(src[src.index("def sliding_window_plan"):src.index("class PromptGrammarState")], namespace)
    return namespace


def closure_of(function) -> dict:
    """The released event_time_map keeps its tempo in a closure cell; read it rather than re-deriving it."""
    return dict(zip(function.__code__.co_freevars, (cell.cell_contents for cell in function.__closure__ or ())))


class Script:
    """Builds token streams the way the model would have written them."""

    def __init__(self, tokenizer, namespace):
        self.tokenizer = tokenizer
        self.namespace = namespace

    def token(self, field: str, offset: int) -> int:
        return getattr(self.tokenizer, f"{field}_token_start") + offset

    def shifts(self, gap: int) -> list:
        """A position change, spelled as the encoder spells it."""
        step = self.tokenizer.subbeat_shift_token_end - self.tokenizer.subbeat_shift_token_start - 1
        out = []
        while gap > step:
            out.append(self.tokenizer.subbeat_shift_token_end - 1)
            gap -= step
        out.append(self.token("subbeat_shift", gap))
        return out

    def stream(self, events, prefix=None, end=True) -> list:
        """A whole decode: a prompt, then (subbeat, field tokens) pairs, then an end token."""
        tokens = list(self.tokenizer.prompt_prefix()) if prefix is None else list(prefix)
        previous = 0 if prefix is None else self.last_subbeat(tokens)
        for subbeat, fields in events:
            tokens.extend(self.shifts(subbeat - previous))
            tokens.extend(fields)
            previous = subbeat
        if end:
            tokens.append(self.tokenizer.eos_token)
        return tokens

    def last_subbeat(self, tokens) -> int:
        decoded = self.tokenizer.decode_sequence(list(tokens) + [self.tokenizer.eos_token])["events"]
        return decoded[-1]["subbeat"] if decoded else 0

    def fields_at(self, index: int, seconds: float, timestamp: bool = True) -> list:
        """One event's fields, in the order the grammar demands them.

        The pattern is arbitrary but fixed: it puts a time signature on some events and a bare bar position on
        others (the case where a replayed prefix has to be told the meter), leaves every third event without a
        timestamp (so the time map has to interpolate rather than read it off), and leaves the occasional event
        with no fields at all (which the decoder drops, merging its shift into the next event's).
        """
        fields = []
        if timestamp:
            fields.append(self.token("time", round(seconds * 100)))
        if index % 8 == 0:
            fields.extend([self.token("meter", 20), self.token("eighth_position", (index // 8) % 8)])
        elif index % 4 == 0:
            fields.append(self.token("eighth_position", (index % 32) // 4))
        if index % 12 == 0:
            fields.append(self.token("structure", (index // 12) % 5))
        if index == 0:
            fields.append(self.token("key", 5))
        if index % 6 == 0:
            fields.append(self.token("full_chord", 1 + (index // 6) % 40))
        if index % 2 == 1:
            fields.extend([self.token("pitch", 60 + (index % 12)), self.token("duration", 4 + (index % 6))])
        return fields


def window_stream(script, window, duration, prefix, base, positions, offset):
    """What the model would return for one window: its replayed prompt, then everything it reads after it.

    Positions maps a global subbeat to its absolute time, so a window's own subbeats can be worked out from the
    base its prompt was re-based to — which is the only thing tying successive windows' coordinates together.
    """
    start = window["start"]
    stop = window["generation_stop"] if window["generation_stop"] is not None else min(duration - start, 300.0)
    if prefix is None:
        anchor = None
        cutoff = start - 1e-9
    else:
        anchor = positions[base]
        cutoff = anchor + script.last_subbeat(prefix) * SUBBEAT_SECONDS + 1e-9
    times = []
    step = 0
    while True:
        seconds = offset + step * GRID_SECONDS
        step += 1
        if seconds > start + stop + 1e-9:
            break
        if seconds > cutoff:
            times.append(seconds)
    events = []
    for order, seconds in enumerate(times):
        if anchor is None:
            anchor = seconds
        subbeat = round((seconds - anchor) / SUBBEAT_SECONDS)
        index = round((seconds - offset) / GRID_SECONDS)
        # The real decode stops on the first timestamp past its limit, so its last event is that bare timestamp.
        last = order == len(times) - 1
        fields = [script.token("time", round((seconds - start) * 100))] if last \
            else script.fields_at(index, seconds - start, timestamp=index % 3 != 1)
        events.append((subbeat, fields))
        positions[base + subbeat] = seconds
    return script.stream(events, prefix=prefix)


def simulate(namespace, script, duration, offset=0.0, truncate_window=None):
    """Runs a whole clip through the released stitching, with scripted decodes standing in for the model."""
    plan, overlap_prefix = namespace["sliding_window_plan"], namespace["overlap_prefix"]
    event_time_map, window_events = namespace["event_time_map"], namespace["stitched_window_events"]
    stitched, positions, out = [], {}, []
    for index, window in enumerate(plan(duration)):
        prefix, base = overlap_prefix(stitched, script.tokenizer, window["start"], window["prefix_end"])
        stream = window_stream(script, window, duration, prefix, base, positions, offset)
        if index == truncate_window:
            # A window cut at its generation stop ends mid-run: trailing shifts, no end token.
            stream = stream[:-1] + script.shifts(GRID_SUBBEATS * 15)
        decoded = script.tokenizer.decode_sequence(stream)
        lookup = event_time_map(decoded, 300.0)
        accepted = window_events(decoded, lookup, window["start"], window["accept_start"],
                                 window["accept_end"], duration, global_subbeat_base=base)
        stitched.extend(accepted)
        out.append({
            "start": window["start"], "acceptStart": window["accept_start"], "acceptEnd": window["accept_end"],
            "prefixEnd": window["prefix_end"], "stream": stream, "prefixTokens": prefix, "base": base,
            "period": closure_of(lookup).get("period"), "anchorCount": len(closure_of(lookup).get("steps", ())),
            "accepted": [serialize(event) for event in accepted],
        })
    return out


def serialize(event) -> dict:
    return {
        "subbeat": event["subbeat"],
        "seconds": event["time"],
        "globalSubbeat": event["global_subbeat"],
        "timestamp": event["values"].get("timestamp"),
        "noteEnds": [note["end_time"] for note in event["values"].get("melody", ())],
    }


def stitched_from(tokenizer, stream, seconds, globals_) -> list:
    """Hand-built accepted events, decoded from real tokens so their shape is the pipeline's own."""
    events = tokenizer.decode_sequence(stream)["events"]
    if len(events) != len(seconds) or len(events) != len(globals_):
        raise ValueError(f"{len(events)} events against {len(seconds)} times and {len(globals_)} positions")
    out = []
    for event, time, position in zip(events, seconds, globals_):
        placed = copy.deepcopy(event)
        placed["time"] = time
        placed["global_subbeat"] = position
        if "timestamp" in placed["values"]:
            placed["values"]["timestamp"] = time
        out.append(placed)
    return out


def prefix_cases(script) -> dict:
    """Hand-built overlap cases, each isolating one decision overlap_prefix makes."""
    tokenizer = script.tokenizer
    time, meter, eighth = script.token("time", 0), script.token("meter", 20), script.token("eighth_position", 3)
    key, chord, structure = script.token("key", 5), script.token("full_chord", 7), script.token("structure", 3)
    pitch, length = script.token("pitch", 60), script.token("duration", 5)
    cases = {}

    def case(name, events, seconds, globals_, start, prefix_end):
        cases[name] = {"stream": script.stream(events), "seconds": list(seconds),
                       "globalSubbeats": list(globals_), "start": start, "prefixEnd": prefix_end}

    # Nothing accepted yet, which is every clip's first window.
    cases["empty"] = {"stream": None, "seconds": [], "globalSubbeats": [], "start": 0.0, "prefixEnd": 0.0}
    # Events in range, but not one of them states a position to re-base the rest against.
    case("noPosition", [(0, [chord]), (20, [structure]), (40, [chord, pitch, length])],
         [100.0, 102.5, 105.0], [800, 820, 840], 100.0, 200.0)
    # The fields of the dropped leading events are what the first kept one inherits.
    case("leadingContextOnly", [(0, [structure]), (20, [key]), (40, [chord]), (60, [time + 750, pitch, length])],
         [100.0, 102.5, 105.0, 107.5], [800, 820, 840, 860], 100.0, 200.0)
    # A key stated by an earlier window, before this window's audio even starts.
    case("carryBackBeforeStart", [(0, [key, chord]), (20, [time + 250, pitch, length]), (40, [time + 500])],
         [50.0, 102.5, 105.0], [400, 820, 840], 100.0, 200.0)
    # A bar position that has lost its time signature gets the one in force back.
    case("meterInjected", [(0, [meter, eighth]), (20, [time + 250, eighth]), (40, [time + 500])],
         [50.0, 102.5, 105.0], [400, 820, 840], 100.0, 200.0)
    # One that still has its own is left alone.
    case("meterPresent", [(0, [script.token("meter", 30), eighth]), (20, [time + 250, meter, eighth])],
         [50.0, 102.5], [400, 820], 100.0, 200.0)
    # A first event with no bar position at all gets no meter, even though one is in force.
    case("meterNotInjectedWithoutPosition", [(0, [meter, eighth]), (20, [time + 250, pitch, length])],
         [50.0, 102.5], [400, 820], 100.0, 200.0)
    # Two events at one position, which tie on both sort keys; the accepted order has to break the tie.
    case("duplicatePosition", [(0, [time + 250, chord]), (0, [structure, pitch, length]), (20, [time + 500])],
         [102.5, 102.5, 105.0], [820, 820, 840], 100.0, 200.0)
    # Same position, different times.
    case("duplicatePositionDifferentTime", [(0, [time + 260, chord]), (0, [time + 250, structure]), (20, [time + 500])],
         [102.6, 102.5, 105.0], [820, 820, 840], 100.0, 200.0)
    # Positions and times that disagree on the order. A window's time map is not guaranteed monotonic, and the
    # prompt is written in position order, not time order.
    case("positionsAgainstTimes", [(0, [time + 250, chord]), (20, [time + 750]), (40, [time + 500, structure])],
         [102.5, 107.5, 105.0], [820, 840, 860], 100.0, 200.0)
    # Handed to the stitcher out of order, as a non-monotonic time map would leave it.
    case("outOfOrderInput", [(0, [time + 500, chord]), (20, [time + 250, structure]), (40, [time + 750])],
         [105.0, 102.5, 107.5], [840, 820, 860], 100.0, 200.0)
    # Inside the tolerance BEFORE the window starts: its timestamp rounds below zero and is clamped up.
    case("timestampBeforeStart", [(0, [time + 1000, eighth]), (20, [time + 250])],
         [100.0 - 1e-4, 102.5], [800, 820], 100.0, 200.0)
    # Exactly half a hundredth, where rounding has to go to even rather than away from zero.
    case("timestampHalfTick", [(0, [time + 1000, eighth]), (20, [time + 1100]), (40, [time + 1200])],
         [102.125, 102.375, 102.625], [800, 820, 840], 100.0, 200.0)
    # Further past the window's start than the timestamp range can name. Unreachable on a real plan, whose
    # overlap is at most a window long, so it pins the clamp rather than a behaviour anything depends on.
    case("timestampPastRange", [(0, [time + 1000, eighth]), (20, [time + 250])],
         [100.0, 500.0], [800, 820], 100.0, 600.0)
    # Everything carried at once, onto a first event that states none of it.
    case("allFieldsCarried", [(0, [structure, key, chord]), (20, [time + 250, pitch, length])],
         [50.0, 102.5], [400, 820], 100.0, 200.0)
    # Events on both sides of the range, so only the overlap is replayed.
    case("outsideRange", [(0, [time + 0, structure]), (20, [time + 250]), (40, [time + 500]), (60, [time + 750])],
         [99.9, 102.5, 199.9, 205.0], [790, 820, 1590, 1640], 100.0, 200.0)

    for name, spec in cases.items():
        stitched = [] if spec["stream"] is None else stitched_from(
            tokenizer, spec["stream"], spec["seconds"], spec["globalSubbeats"])
        tokens, base = script.namespace["overlap_prefix"](stitched, tokenizer, spec["start"], spec["prefixEnd"])
        spec["tokens"], spec["base"] = tokens, base
    return cases


def time_map_cases(script) -> dict:
    """Hand-built time maps, each isolating one branch of the tempo or the interpolation."""
    time = script.token("time", 0)
    chord, pitch, length = script.token("full_chord", 7), script.token("pitch", 60), script.token("duration", 5)
    probes = [-40, -1, 0, 1, 10, 20, 30, 40, 50, 55, 60, 80, 100, 1000, 2000, 100000]
    specs = {
        # Nothing to anchor on: a fixed tempo, clipped at both ends.
        "noAnchor": [(0, [chord]), (20, [pitch, length]), (40, [script.token("structure", 3)])],
        # One anchor: the same fixed tempo, but hung off that anchor.
        "single": [(20, [time + 500])],
        "uniform": [(0, [time + 0]), (20, [time + 250]), (40, [time + 500]), (60, [time + 750])],
        # An odd number of gaps, so the median is a gap rather than the mean of two.
        "threeGaps": [(0, [time + 0]), (20, [time + 250]), (40, [time + 700]), (60, [time + 750])],
        # An even number, so it is the mean of the middle two.
        "fourGaps": [(0, [time + 0]), (20, [time + 250]), (40, [time + 700]), (60, [time + 750]),
                     (80, [time + 1400])],
        # One wild timestamp, which a median survives and a mean would not.
        "outlier": [(0, [time + 0]), (20, [time + 250]), (40, [time + 29000]), (60, [time + 750])],
        # Every timestamp the same: no tempo at all, so the fixed one takes over.
        "flat": [(0, [time + 500]), (20, [time + 500]), (40, [time + 500])],
        # Timestamps running backwards: a negative tempo, which also falls back.
        "backwards": [(0, [time + 1000]), (20, [time + 500]), (40, [time + 0])],
        # Two events at one position, where the later timestamp is the one that counts.
        "repeatedPosition": [(0, [time + 0]), (20, [time + 250]), (20, [time + 400]), (40, [time + 1000])],
        # A single enormous gap, so extrapolation past the last anchor runs well past the window.
        "sparse": [(0, [time + 0]), (2000, [time + 25000])],
    }
    # The same anchors under a ceiling that sits between them: only the two extrapolations are clipped, so
    # interpolation is free to report a time the map would otherwise never name.
    specs["belowTheCeiling"] = specs["uniform"]
    specs["sparseBelowTheCeiling"] = specs["sparse"]
    targets = {"belowTheCeiling": 5.0, "sparseBelowTheCeiling": 5.0}
    cases = {}
    for name, events in specs.items():
        target = targets.get(name, 300.0)
        stream = script.stream(events)
        decoded = script.tokenizer.decode_sequence(stream)
        lookup = script.namespace["event_time_map"](decoded, target)
        closure = closure_of(lookup)
        cases[name] = {
            "stream": stream, "targetSeconds": target, "period": closure.get("period"),
            "anchorCount": len(closure.get("steps", ())),
            "probes": [{"step": step, "seconds": lookup(step)} for step in probes],
        }
    return cases


def boundary_cases(script) -> dict:
    """Events sitting exactly on an accepted span's edges, which is where a seam duplicates or drops a bar."""
    time, pitch = script.token("time", 0), script.token("pitch", 60)
    # Five events, a grid apart, at 100.0 to 110.0 once the window's own start is added back.
    events = [(index * GRID_SUBBEATS, [time + index * 250]) for index in range(5)]
    stream = script.stream(events)
    # The same five, each carrying a note, for the end-time cases.
    melodic = script.stream([(index * GRID_SUBBEATS, [time + index * 250, pitch, script.token("duration", bin_)])
                             for index, bin_ in enumerate((0, 3, 11, 15, 19))])
    # A tempo of a fortieth of a second per subbeat, where the shortest note maps to less than the floor.
    fast = script.stream([(index * GRID_SUBBEATS, [time + index * 50, pitch, script.token("duration", 0)])
                          for index in range(5)])
    middle = 105.0
    specs = {
        "acceptStartOnEvent": dict(stream=stream, acceptStart=middle, acceptEnd=200.0),
        "acceptStartATolerancePast": dict(stream=stream, acceptStart=middle + 1e-4, acceptEnd=200.0),
        "acceptStartTwoTolerancesPast": dict(stream=stream, acceptStart=middle + 2e-4, acceptEnd=200.0),
        "acceptStartAToleranceBefore": dict(stream=stream, acceptStart=middle - 1e-4, acceptEnd=200.0),
        "acceptEndOnEvent": dict(stream=stream, acceptStart=0.0, acceptEnd=middle),
        "acceptEndATolerancePast": dict(stream=stream, acceptStart=0.0, acceptEnd=middle + 1e-4),
        "acceptEndTwoTolerancesPast": dict(stream=stream, acceptStart=0.0, acceptEnd=middle + 2e-4),
        "acceptEndAToleranceBefore": dict(stream=stream, acceptStart=0.0, acceptEnd=middle - 1e-4),
        "durationOnEvent": dict(stream=stream, acceptStart=0.0, acceptEnd=200.0, duration=middle),
        "durationATolerancePast": dict(stream=stream, acceptStart=0.0, acceptEnd=200.0, duration=middle + 1e-4),
        "durationTwoTolerancesPast": dict(stream=stream, acceptStart=0.0, acceptEnd=200.0, duration=middle + 2e-4),
        "spanAcceptsNothing": dict(stream=stream, acceptStart=150.0, acceptEnd=150.0),
        "spanRunsBackwards": dict(stream=stream, acceptStart=150.0, acceptEnd=120.0),
        "basedPastZero": dict(stream=stream, acceptStart=0.0, acceptEnd=200.0, base=4321),
        "notesToTheirLengths": dict(stream=melodic, acceptStart=0.0, acceptEnd=200.0),
        # Long notes, against a clip that ends before they do.
        "notesClippedByTheClip": dict(stream=melodic, acceptStart=0.0, acceptEnd=200.0, duration=112.0),
        # Notes shorter than the floor, at a tempo fast enough to put them there.
        "notesBelowTheFloor": dict(stream=fast, acceptStart=0.0, acceptEnd=200.0),
    }
    cases = {}
    for name, spec in specs.items():
        start, duration = 100.0, spec.get("duration", 1200.0)
        base = spec.get("base", 0)
        decoded = script.tokenizer.decode_sequence(spec["stream"])
        lookup = script.namespace["event_time_map"](decoded, 300.0)
        accepted = script.namespace["stitched_window_events"](
            decoded, lookup, start, spec["acceptStart"], spec["acceptEnd"], duration, global_subbeat_base=base)
        cases[name] = {
            "stream": spec["stream"], "start": start, "acceptStart": spec["acceptStart"],
            "acceptEnd": spec["acceptEnd"], "duration": duration, "base": base,
            "accepted": [serialize(event) for event in accepted],
        }
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Path to the released sheetsage2.py")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("sheetsage2_reference"))
    args = parser.parse_args()

    namespace = load_reference(args.source)
    script = Script(namespace["ScoreTokenizer"](), namespace)

    payload = {
        "gridSeconds": GRID_SECONDS,
        "timeMaps": time_map_cases(script),
        "prefixes": prefix_cases(script),
        "boundaries": boundary_cases(script),
        "clips": {f"{duration:g}": {"duration": duration, "windows": simulate(namespace, script, duration)}
                  for duration in CLIP_DURATIONS},
    }
    # The same seams read by events that never land on them, and one window cut off mid-stream.
    payload["clips"]["450offset"] = {"duration": 450.0,
                                     "windows": simulate(namespace, script, 450.0, offset=0.25)}
    payload["clips"]["450truncated"] = {"duration": 450.0,
                                        "windows": simulate(namespace, script, 450.0, truncate_window=0)}

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "stitch.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    seams = sum(len(clip["windows"]) - 1 for clip in payload["clips"].values())
    print(f"wrote {path} ({len(payload['timeMaps'])} time maps, {len(payload['prefixes'])} overlaps, "
          f"{len(payload['boundaries'])} boundaries, {len(payload['clips'])} clips, {seams} seams)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
