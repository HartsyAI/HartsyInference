"""Encodes sample records with the official `encode_record` and the real tokenizer, for ClefRecordEncoderTests.

Usage: python encode_reference.py <release dir (joint_schema_model.py + tokenizer.json)> <output json>
(needs torch, transformers 5.10)
"""
import importlib.util, json, sys
from pathlib import Path
from transformers import AutoTokenizer

release, out = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("jsm", release / "joint_schema_model.py")
jsm = importlib.util.module_from_spec(spec); sys.modules["jsm"] = jsm; spec.loader.exec_module(jsm)
tok = AutoTokenizer.from_pretrained(str(release))

records = [
    {"state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue", "lines": [1, 2.5, 1e-7, 3e20]}},
     "questions": {
         "status": {"type": "choice", "instructions": "What is the invoice status?",
                    "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."}},
         "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"}}},
    {"state": "Our checkout started returning errors and orders are blocked. Ünïcödé \"quotes\" \n newline 12345",
     "questions": {
         "department": {"type": "choice", "instructions": "Which team should handle the message?",
                        "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
         "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
         "outage": {"type": "noul", "instructions": "Is a service down?", "criteria": {"true": "Yes, down."}}}},
    {"state": [True, None, {"b": 1, "a": {"z": 0.1, "y": -2}}],
     "questions": {"q": {"type": "choice", "instructions": "", "criteria": {"b": "B", "a": "A", "B": "upper"}}}},
]
golden = []
for r in records:
    e = jsm.encode_record(tok, r, processor=None)
    golden.append({"record": r, "input_ids": list(e.input_ids),
                   "questions": [{"id": q.question_id, "type": q.question_type, "question": list(q.question_span),
                                  "options": [list(s) for s in q.option_spans], "optionIds": list(q.option_ids)} for q in e.questions]})
json.dump(golden, open(out, "w"))
print([len(g["input_ids"]) for g in golden])
