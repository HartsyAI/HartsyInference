"""Dump upstream Spark-TTS prompt token ids for the engine's prompt-parity fixture.

Builds prompts with the unmodified upstream SparkTTS.process_prompt / process_prompt_control logic
(SparkAudio/Spark-TTS cli/SparkTTS.py) and tokenizes them exactly as SparkTTS.inference does:
AutoTokenizer.from_pretrained(LLM dir)([prompt]). Upstream applies no text normalization.

Zero-shot clone prompts need BiCodec global (and, with a transcript, semantic) tokens of a reference clip.
The prompt ids depend only on those token ids, not on the audio, so fixed synthetic ids stand in for them.

Usage:
  python spark_prompt_dump.py --spark-src <Spark-TTS checkout> --tokenizer <dir with tokenizer.json>
      [--slow-tokenizer <dir with vocab.json/merges.txt only>] --out spark_prompt_ids.json
"""

import argparse
import json
import subprocess
import sys

TEXTS = [
    "Yes.",
    "Okay, see you.",
    "The weather is nice today, so we might take a walk.",
    "After the meeting ended, the team gathered in the kitchen to talk about the new product launch, "
    "the budget, and the schedule for the coming quarter.",
    "On March 3rd, 2026 at 4:30 PM, 1,250 people paid $99.95 each.",
    "Wait... what?! No -- really; (yes), 'quoted' & \"double\" -- ok?",
    "No no no no no no no no no no no no.",
    "Every morning the old lighthouse keeper climbed the narrow spiral stairs, polished the great brass lamp, "
    "and wrote a short note in his logbook about the wind, the waves, and the ships that passed. Nobody read "
    "those notes, but he believed that someday somebody would want to know.",
]

# Styles the engine can request: gender from the voice field, speed bucketed from the rate multiplier.
STYLES = [("female", "moderate", "moderate"), ("male", "moderate", "very_high"), ("male", "low", "very_low")]

GLOBAL_IDS = [(i * 1237 + 11) % 4096 for i in range(32)]
SEMANTIC_IDS = [(i * 3571 + 5) % 8192 for i in range(25)]
PROMPT_TEXT = "This is the reference speaker."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spark-src", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--slow-tokenizer")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.spark_src)
    from sparktts.utils.token_parser import GENDER_MAP, LEVELS_MAP, TASK_TOKEN_MAP
    from transformers import AutoTokenizer

    # Verbatim from cli/SparkTTS.py (the class itself imports torch and BiCodec at module load).
    def process_prompt_control(gender, pitch, speed, text):
        attribte_tokens = "".join([f"<|gender_{GENDER_MAP[gender]}|>", f"<|pitch_label_{LEVELS_MAP[pitch]}|>",
                                   f"<|speed_label_{LEVELS_MAP[speed]}|>"])
        return "".join([TASK_TOKEN_MAP["controllable_tts"], "<|start_content|>", text, "<|end_content|>",
                        "<|start_style_label|>", attribte_tokens, "<|end_style_label|>"])

    def process_prompt(text, global_ids, prompt_text, semantic_ids):
        global_tokens = "".join([f"<|bicodec_global_{i}|>" for i in global_ids])
        if prompt_text is not None:
            semantic_tokens = "".join([f"<|bicodec_semantic_{i}|>" for i in semantic_ids])
            inputs = [TASK_TOKEN_MAP["tts"], "<|start_content|>", prompt_text, text, "<|end_content|>",
                      "<|start_global_token|>", global_tokens, "<|end_global_token|>",
                      "<|start_semantic_token|>", semantic_tokens]
        else:
            inputs = [TASK_TOKEN_MAP["tts"], "<|start_content|>", text, "<|end_content|>",
                      "<|start_global_token|>", global_tokens, "<|end_global_token|>"]
        return "".join(inputs)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    slow = AutoTokenizer.from_pretrained(args.slow_tokenizer, use_fast=False) if args.slow_tokenizer else None

    cases = []
    for text in TEXTS:
        for gender, pitch, speed in STYLES:
            cases.append({"mode": "controllable", "text": text, "gender": gender, "pitch": pitch, "speed": speed,
                          "prompt": process_prompt_control(gender, pitch, speed, text)})
        cases.append({"mode": "zero_shot", "text": text, "global_ids": GLOBAL_IDS,
                      "prompt": process_prompt(text, GLOBAL_IDS, None, None)})
    cases.append({"mode": "zero_shot", "text": TEXTS[2], "global_ids": GLOBAL_IDS, "prompt_text": PROMPT_TEXT,
                  "semantic_ids": SEMANTIC_IDS, "prompt": process_prompt(TEXTS[2], GLOBAL_IDS, PROMPT_TEXT,
                                                                         SEMANTIC_IDS)})

    for case in cases:
        ids = tok([case["prompt"]], return_tensors=None)["input_ids"][0]
        case["ids"] = ids
        if slow is not None:
            slow_ids = slow([case["prompt"]])["input_ids"][0]
            if slow_ids != ids:
                raise SystemExit(f"slow/fast tokenizer disagree on {case['prompt']!r}")

    try:
        commit = subprocess.check_output(["git", "-C", args.spark_src, "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    import transformers
    out = {"source": "SparkAudio/Spark-TTS cli/SparkTTS.py process_prompt / process_prompt_control",
           "spark_commit": commit, "transformers": transformers.__version__, "cases": cases}
    # One case per line keeps the committed fixture diffable.
    body = ",\n  ".join(json.dumps(c, ensure_ascii=False) for c in out.pop("cases"))
    head = json.dumps(out, ensure_ascii=False)[:-1]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"{head}, \"cases\": [\n  {body}\n]}}\n")
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
