"""Generate CosyVoice 2 reference audio with the upstream Python implementation (CPU).

    python tools/parity/cosyvoice_reference_tts.py <CosyVoice repo> <model dir> <prompt.wav> "<prompt transcript>" <out dir> [seed]

The model dir must hold llm.pt, flow.pt, hift.pt, cosyvoice2.yaml, campplus.onnx, speech_tokenizer_v2.onnx and
CosyVoice-BlankEN/. Texts are the regression-matrix lines; one WAV per line, named like the engine matrix rows.
"""
import os
import sys

REPO, MODEL, PROMPT_WAV, PROMPT_TEXT, OUT = sys.argv[1:6]
SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 0
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party", "Matcha-TTS"))

# wetext 0.0.4 calls ModelScope's snapshot_download on every Normalizer() even when the FSTs are cached, and the
# ModelScope API rate-limits; point it at the local snapshot when one exists so runs are reproducible offline.
import os as _os
import modelscope as _modelscope
_WETEXT_CACHE = _os.path.expanduser("~/.cache/modelscope/models/pengzhendong--wetext/snapshots/master")
if _os.path.isdir(_WETEXT_CACHE):
    _modelscope.snapshot_download = lambda *a, **k: _WETEXT_CACHE
import torch  # noqa: E402
import torchaudio  # noqa: E402
from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: E402
from cosyvoice.utils.common import set_all_random_seed  # noqa: E402
from cosyvoice.utils.file_utils import load_wav  # noqa: E402

TEXTS = [
    ("01_single_word", "Yes."),
    ("02_two_words", "Okay, see you."),
    ("03_eight_words", "The weather is nice today, so we might take a walk."),
    ("04_twenty_words", "After the meeting ended, the team gathered in the kitchen to talk about the new product launch, the budget, and the schedule for the coming quarter."),
    ("05_paragraph", "Our water bill arrived late this month, so I called the office to ask why. The clerk said a meter reading was missed in March. She promised to fix the record and send a corrected statement within two weeks. I thanked her, hung up, and wrote the date on my calendar so I would remember to check. Last time this happened it took three phone calls, so I am hoping this one goes faster."),
    ("06_numbers_dates", "On March 3rd, 2026 at 4:30 PM, 1,250 people paid $99.95 each."),
    ("07_punctuation", "Wait... what?! No -- really; (yes), 'quoted' & \"double\" -- ok?"),
    ("08_repeated", "No no no no no no no no no no no no."),
]

os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
model = CosyVoice2(MODEL, load_jit=False, load_trt=False, fp16=False)
prompt = load_wav(PROMPT_WAV, 16000)
for tag, text in TEXTS:
    set_all_random_seed(SEED)
    pieces = [out["tts_speech"] for out in model.inference_zero_shot(text, PROMPT_TEXT, prompt, stream=False)]
    torchaudio.save(os.path.join(OUT, f"{tag}.wav"), torch.cat(pieces, dim=1), model.sample_rate)
    print(tag, "done", flush=True)
