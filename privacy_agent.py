"""
privacy_agent.py  -  Agent 1: Sensitive-Data Detection, Relevance Check & Sanitization
=====================================================================================

What it does (runs 100% locally, nothing is sent anywhere):
  1. Detects sensitive info in the prompt AND in every attachment
     (Presidio + spaCy NER + regex + validators + your custom dictionary).
  2. Works out what the TASK is, then checks
        a) whether each detected entity is actually needed for that task
        b) whether each attachment is relevant to that task
  3. Produces
        - a sanitized prompt (span-based replacement, never global str.replace)
        - attachment_decisions.json  -> which attachments to REMOVE / SANITIZE / KEEP / REVIEW
        - report.json                -> full report (contains NO raw sensitive values)
        - vault.json                 -> local-only placeholder -> original mapping

Works with graceful fallbacks:
  Presidio+spaCy  ->  spaCy only  ->  regex/heuristics only

Usage:
  python privacy_agent.py --prompt "Summarize the attached resume" --attach resume.pdf photo.jpg
  python privacy_agent.py --prompt-file prompt.txt --attach a.docx b.png c.mp4 --strict
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import stat
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Optional dependencies (script still works if they are missing)
# --------------------------------------------------------------------------- #
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    HAS_PRESIDIO = True
except Exception:  # pragma: no cover
    HAS_PRESIDIO = False

try:
    import spacy
    HAS_SPACY = True
except Exception:  # pragma: no cover
    HAS_SPACY = False

for _n in ("presidio-analyzer", "tldextract", "tldextract.cache"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

# --------------------------------------------------------------------------- #
# Policy tables
# --------------------------------------------------------------------------- #
# category -> severity (0-10)
SEVERITY = {
    "SECRET": 10,      # API keys, passwords, private keys, tokens, DB URLs
    "GOV_ID": 9,       # Aadhaar, PAN, SSN, passport, driving licence
    "FINANCIAL": 8,    # card numbers, bank a/c, IBAN, IFSC, UPI
    "METADATA": 7,     # EXIF GPS etc.
    "CUSTOM": 6,       # your own dictionary terms
    "CONTACT": 5,      # email, phone, IP
    "PERSON": 4,
    "LOCATION": 3,
    "ORG": 2,
}
CRITICAL_SEVERITY = 9  # >= this is "critical"

# Presidio entity -> (entity_type, category)
PRESIDIO_MAP = {
    "PERSON": ("PERSON", "PERSON"),
    "LOCATION": ("LOCATION", "LOCATION"),
    "ORGANIZATION": ("ORG", "ORG"),
    "EMAIL_ADDRESS": ("EMAIL", "CONTACT"),
    "PHONE_NUMBER": ("PHONE", "CONTACT"),
    "IP_ADDRESS": ("IP_ADDRESS", "CONTACT"),
    "CREDIT_CARD": ("CREDIT_CARD", "FINANCIAL"),
    "IBAN_CODE": ("IBAN", "FINANCIAL"),
    "US_BANK_NUMBER": ("BANK_ACCOUNT", "FINANCIAL"),
    "CRYPTO": ("CRYPTO_WALLET", "FINANCIAL"),
    "US_SSN": ("SSN", "GOV_ID"),
    "US_PASSPORT": ("PASSPORT", "GOV_ID"),
    "US_DRIVER_LICENSE": ("DRIVING_LICENSE", "GOV_ID"),
    "US_ITIN": ("ITIN", "GOV_ID"),
    "MEDICAL_LICENSE": ("MEDICAL_LICENSE", "GOV_ID"),
    "IN_AADHAAR": ("AADHAAR", "GOV_ID"),
    "IN_PAN": ("PAN", "GOV_ID"),
    "IN_PASSPORT": ("PASSPORT", "GOV_ID"),
    "IN_VOTER": ("VOTER_ID", "GOV_ID"),
    "IN_VEHICLE_REGISTRATION": ("VEHICLE_REG", "GOV_ID"),
}

# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class Entity:
    start: int
    end: int
    text: str
    entity_type: str
    category: str
    score: float
    detector: str

    @property
    def severity(self) -> int:
        return SEVERITY.get(self.category, 1)


# --------------------------------------------------------------------------- #
# Validators
# --------------------------------------------------------------------------- #
_VD = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],[3,4,0,1,2,8,9,5,6,7],
       [4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],[6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],
       [8,7,6,5,9,3,2,1,0,4],[9,8,7,6,5,4,3,2,1,0]]
_VP = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],[8,9,1,6,0,4,3,5,2,7],
       [9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],[2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]


def verhoeff_ok(num: str) -> bool:
    digits = re.sub(r"\D", "", num)
    if len(digits) != 12:
        return False
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VD[c][_VP[i % 8][int(ch)]]
    return c == 0


def luhn_ok(num: str) -> bool:
    digits = re.sub(r"\D", "", num)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iban_ok(s: str) -> bool:
    s = re.sub(r"\s", "", s).upper()
    if not 15 <= len(s) <= 34:
        return False
    rearranged = s[4:] + s[:4]
    try:
        return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# Regex layer  (name, entity_type, category, regex, group, validator, score)
# --------------------------------------------------------------------------- #
def _rx(p, flags=0):
    return re.compile(p, flags)


REGEX_RULES = [
    # ---- secrets -----------------------------------------------------------
    ("private_key", "PRIVATE_KEY", "SECRET",
     _rx(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"), 0, None, 1.0),
    ("aws_key", "AWS_ACCESS_KEY", "SECRET", _rx(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), 0, None, 1.0),
    ("github_token", "GITHUB_TOKEN", "SECRET", _rx(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"), 0, None, 1.0),
    ("llm_key", "API_KEY", "SECRET", _rx(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_\-]{20,}\b"), 0, None, 0.95),
    ("slack_token", "SLACK_TOKEN", "SECRET", _rx(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"), 0, None, 1.0),
    ("google_key", "API_KEY", "SECRET", _rx(r"\bAIza[0-9A-Za-z_\-]{35}\b"), 0, None, 1.0),
    ("jwt", "JWT", "SECRET",
     _rx(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"), 0, None, 0.95),
    ("conn_string", "CREDENTIAL_URL", "SECRET",
     _rx(r"\b[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s:/@]+:[^\s@/]+@[^\s]+"), 0, None, 0.95),
    ("kv_secret", "PASSWORD_OR_TOKEN", "SECRET",
     _rx(r"(?i)\b(?:password|passwd|pwd|passcode|secret|api[_\-]?key|access[_\-]?token|auth[_\-]?token|"
         r"client[_\-]?secret|token)\b[\"']?\s*[:=]\s*[\"']?([^\s\"',;]{4,})"), 1, None, 0.9),
    ("said_secret", "PASSWORD_OR_PIN", "SECRET",
     _rx(r"(?i)\b(?:password|passcode|pin|otp|cvv)\s+(?:is|was|:)\s*[\"']?([^\s\"',;]{3,})"), 1, None, 0.9),
    # ---- government IDs ----------------------------------------------------
    ("aadhaar", "AADHAAR", "GOV_ID", _rx(r"(?<!\d)[2-9]\d{3}[ \-]?\d{4}[ \-]?\d{4}(?!\d)"), 0, verhoeff_ok, 0.95),
    ("pan", "PAN", "GOV_ID", _rx(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), 0, None, 0.9),
    ("ssn", "SSN", "GOV_ID", _rx(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"), 0, None, 0.85),
    ("passport_ctx", "PASSPORT", "GOV_ID",
     _rx(r"(?i)passport(?:\s*(?:no|number|num|#))?\.?\s*[:\-]?\s*([A-Z0-9]{6,9})\b"), 1, None, 0.85),
    ("dl_ctx", "DRIVING_LICENSE", "GOV_ID",
     _rx(r"(?i)(?:driving\s*licen[cs]e|\bDL)(?:\s*(?:no|number|num|#))?\.?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\- ]{8,17}[A-Z0-9])"),
     1, None, 0.8),
    # ---- financial ---------------------------------------------------------
    ("card", "CREDIT_CARD", "FINANCIAL", _rx(r"(?<!\d)(?:\d[ \-]?){12,18}\d(?!\d)"), 0, luhn_ok, 0.95),
    ("iban", "IBAN", "FINANCIAL", _rx(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}[ ]?[A-Z0-9]{1,4}\b"), 0, iban_ok, 0.9),
    ("ifsc", "IFSC", "FINANCIAL", _rx(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), 0, None, 0.85),
    ("upi", "UPI_ID", "FINANCIAL",
     _rx(r"\b[\w.\-]{2,}@(?:ok(?:axis|sbi|hdfcbank|icici)|ybl|paytm|upi|ibl|axl|apl|sbi|icici|hdfcbank)(?![\w\-]|\.\w)"),
     0, None, 0.9),
    ("acct_ctx", "BANK_ACCOUNT", "FINANCIAL",
     _rx(r"(?i)(?:a/c|acct|account)(?:\s*(?:no|number|num|#))?\.?\s*[:\-]?\s*(\d{9,18})\b"), 1, None, 0.9),
    # ---- contact -----------------------------------------------------------
    ("email", "EMAIL", "CONTACT", _rx(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), 0, None, 0.95),
    ("phone_in", "PHONE", "CONTACT",
     _rx(r"(?<![\w\-])(?:\+91[\s\-]?|0)?[6-9]\d{4}[\s\-]?\d{5}(?!\d)"), 0, None, 0.85),
    ("phone_intl", "PHONE", "CONTACT",
     _rx(r"(?<![\w\-])\+\d{1,3}[\s\-]?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}(?!\d)"), 0, None, 0.8),
    ("ipv4", "IP_ADDRESS", "CONTACT",
     _rx(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), 0, None, 0.8),
]

# Used ONLY when neither Presidio nor spaCy is installed
_NAME = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
_TITLE = r"(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+"
NAME_HEURISTICS = [
    _rx(r"\b" + _TITLE + _NAME),
    _rx(r"(?:\bmy name is|\bDear|\bRegards,?|\bSincerely,?|\bThanks,?)\s+(?:" + _TITLE + r")?" + _NAME),
]


# NER often mislabels tech words as ORG/PERSON. These are never treated as sensitive.
DEFAULT_ALLOW = set("""python java javascript typescript sql mysql postgres postgresql mongodb redis docker kubernetes linux
windows macos ubuntu android ios aws azure gcp google github gitlab git react angular vue node nodejs django flask fastapi
pandas numpy tensorflow pytorch excel word powerpoint pdf json xml html css api rest http https pan aadhaar ssn otp""".split())


def pick_spacy_model() -> Optional[str]:
    if not HAS_SPACY:
        return None
    for m in ("en_core_web_lg", "en_core_web_md", "en_core_web_sm"):
        if spacy.util.is_package(m):
            return m
    return None


# --------------------------------------------------------------------------- #
# Detector (Agent-1 step 1)
# --------------------------------------------------------------------------- #
class Detector:
    MAX_NER_CHARS = 400_000

    def __init__(self, dictionary_path: Optional[str] = None):
        self.model = pick_spacy_model()
        self.analyzer = None
        self.nlp = None
        self.engine = "regex+heuristics"
        self.custom_terms: dict[str, str] = {}
        self.allow = set(DEFAULT_ALLOW)

        if dictionary_path:
            data = json.loads(Path(dictionary_path).read_text(encoding="utf-8"))
            # {"terms": {"Project Phoenix": "PROJECT", "Acme Corp": "CLIENT"}}  or  {"terms": ["Project Phoenix"]}
            terms = data.get("terms", data)
            self.custom_terms = {t: "CUSTOM" for t in terms} if isinstance(terms, list) else dict(terms)

        if HAS_PRESIDIO and self.model:
            try:
                provider = NlpEngineProvider(nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": self.model}]})
                self.analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
                self.engine = f"presidio+spacy({self.model})+regex"
            except Exception as e:  # pragma: no cover
                print(f"[warn] Presidio init failed ({e}); falling back.", file=sys.stderr)
        if self.analyzer is None and self.model:
            self.nlp = spacy.load(self.model)
            self.engine = f"spacy({self.model})+regex"

    # ---- layers -----------------------------------------------------------
    def _regex(self, text: str) -> list[Entity]:
        out = []
        for name, etype, cat, rx, grp, validator, score in REGEX_RULES:
            for m in rx.finditer(text):
                s, e = m.span(grp)
                if s < 0 or s == e:
                    continue
                val = text[s:e]
                if validator and not validator(val):
                    continue
                out.append(Entity(s, e, val, etype, cat, score, f"regex:{name}"))
        for term, label in self.custom_terms.items():
            for m in re.finditer(re.escape(term), text, flags=re.IGNORECASE):
                out.append(Entity(m.start(), m.end(), m.group(0), str(label).upper(), "CUSTOM", 1.0, "dictionary"))
        return out

    def _presidio(self, text: str) -> list[Entity]:
        out = []
        for r in self.analyzer.analyze(text=text[: self.MAX_NER_CHARS], language="en", score_threshold=0.5):
            if r.entity_type in PRESIDIO_MAP:
                etype, cat = PRESIDIO_MAP[r.entity_type]
                out.append(Entity(r.start, r.end, text[r.start:r.end], etype, cat, float(r.score), "presidio"))
        return out

    def _spacy(self, text: str) -> list[Entity]:
        out = []
        label_map = {"PERSON": ("PERSON", "PERSON"), "GPE": ("LOCATION", "LOCATION"),
                     "LOC": ("LOCATION", "LOCATION"), "FAC": ("LOCATION", "LOCATION"), "ORG": ("ORG", "ORG")}
        self.nlp.max_length = max(self.nlp.max_length, self.MAX_NER_CHARS + 10)
        for ent in self.nlp(text[: self.MAX_NER_CHARS]).ents:
            if ent.label_ in label_map:
                etype, cat = label_map[ent.label_]
                out.append(Entity(ent.start_char, ent.end_char, ent.text, etype, cat, 0.8, "spacy"))
        return out

    def _heuristic(self, text: str) -> list[Entity]:
        return [Entity(m.start(1), m.end(1), m.group(1), "PERSON", "PERSON", 0.5, "heuristic")
                for rx in NAME_HEURISTICS for m in rx.finditer(text)]

    @staticmethod
    def _resolve(ents: list[Entity]) -> list[Entity]:
        """Overlap resolution: higher severity > higher score > longer span."""
        accepted: list[Entity] = []
        for e in sorted(ents, key=lambda x: (-x.severity, -x.score, -(x.end - x.start))):
            if all(e.end <= a.start or e.start >= a.end for a in accepted):
                accepted.append(e)
        return sorted(accepted, key=lambda x: x.start)

    def detect(self, text: str) -> list[Entity]:
        if not text.strip():
            return []
        ents = self._regex(text)
        if self.analyzer is not None:
            ents += self._presidio(text)
        elif self.nlp is not None:
            ents += self._spacy(text)
        else:
            ents += self._heuristic(text)
        ents = [e for e in ents if not (e.category in ("PERSON", "ORG", "LOCATION") and e.text.strip().lower() in self.allow)]
        return self._resolve(ents)


# --------------------------------------------------------------------------- #
# Task analysis (Agent-1 step 2a)
# --------------------------------------------------------------------------- #
# (task_type, regex, categories the task genuinely needs)
TASK_RULES = [
    ("contact_or_send", r"\b(send|call|text|whatsapp|dm|contact|forward|phone|ring|reach out)\b", {"CONTACT", "PERSON", "ORG"}),
    ("summarize", r"\b(summari[sz]e|summary|tl;?dr|key points|gist|brief me)\b", set()),
    ("translate", r"\b(translate|translation)\b", set()),
    ("proofread_rewrite", r"\b(proofread|rewrite|rephrase|paraphrase|polish|grammar|improve (?:this|my|the))\b", set()),
    ("draft_message", r"\b(draft|write|compose|reply|respond)\b[^.\n]{0,40}\b(email|e-mail|mail|message|letter|cover letter|post)\b",
     {"PERSON", "ORG"}),
    ("code_help", r"\b(debug|refactor|stack ?trace|traceback|compile|function|bug|code|script|sql|exception|error)\b", set()),
    ("data_analysis", r"\b(analy[sz]e|analysis|chart|plot|trend|statistics|insights?|forecast|calculate|dataset)\b", set()),
    ("extract_info", r"\b(extract|find|list|identify|pull out|parse)\b", {"PERSON", "ORG", "LOCATION"}),
    ("question_answer", r"\b(what|why|how|when|who|which|explain|difference|compare)\b", {"PERSON", "ORG", "LOCATION"}),
]


def analyze_task(prompt: str) -> dict:
    matched = [(t, needs) for t, rx, needs in TASK_RULES if re.search(rx, prompt, re.I)]
    if not matched:
        return {"task_type": "general", "all_matches": [], "needs_categories": []}
    return {"task_type": matched[0][0], "all_matches": [m[0] for m in matched],
            "needs_categories": sorted(set().union(*[m[1] for m in matched]))}


def entity_policy(ent: Entity, task: dict) -> dict:
    """Decide: needed for task? and what to do with it."""
    needs = set(task["needs_categories"])
    cat = ent.category
    if cat == "SECRET":
        return {"needed": False, "action": "remove",
                "reason": "Credentials/secrets are never required by a cloud LLM."}
    if cat in ("GOV_ID", "FINANCIAL"):
        return {"needed": False, "action": "redact",
                "reason": "Government/financial identifiers must stay on-device; the task only needs the structure."}
    if cat == "CONTACT":
        if "CONTACT" in needs:
            return {"needed": True, "action": "pseudonymize",
                    "reason": "Task involves contacting someone; placeholder keeps the structure, real value restored locally."}
        return {"needed": False, "action": "redact", "reason": "Contact details are not needed for this task."}
    if cat == "CUSTOM":
        return {"needed": cat in needs, "action": "pseudonymize",
                "reason": "Matches your custom sensitive dictionary; replaced with a consistent placeholder."}
    # PERSON / ORG / LOCATION
    needed = cat in needs
    return {"needed": needed, "action": "pseudonymize",
            "reason": ("Referenced by the task; " if needed else "Not essential to the task; ")
                      + "replaced with a consistent placeholder so the text still reads coherently."}


# --------------------------------------------------------------------------- #
# Sanitizer + local vault (Agent-1 step 3)
# --------------------------------------------------------------------------- #
class Sanitizer:
    def __init__(self):
        self.by_value: dict[tuple, str] = {}
        self.counters: Counter = Counter()
        self.vault: dict[str, str] = {}        # placeholder -> original (local only)

    def placeholder(self, ent: Entity, action: str) -> str:
        if action == "remove":
            return "[SECRET_REMOVED]"
        if action == "redact":
            return f"[REDACTED_{ent.entity_type}]"
        key = (ent.entity_type, re.sub(r"\s+", " ", ent.text.strip().lower()))
        if key not in self.by_value:
            self.counters[ent.entity_type] += 1
            ph = f"<{ent.entity_type}_{self.counters[ent.entity_type]}>"
            self.by_value[key] = ph
            self.vault[ph] = ent.text
        return self.by_value[key]

    def apply(self, text: str, ents: list[Entity], task: dict) -> tuple[str, list[dict]]:
        """Span-based rebuild: every replacement uses the detected offsets (no global replace)."""
        parts, cursor, report = [], 0, []
        for e in sorted(ents, key=lambda x: x.start):
            pol = entity_policy(e, task)
            ph = self.placeholder(e, pol["action"])
            parts.append(text[cursor:e.start])
            parts.append(ph)
            cursor = e.end
            report.append({"span": [e.start, e.end], "entity_type": e.entity_type, "category": e.category,
                           "severity": e.severity, "confidence": round(e.score, 2), "detector": e.detector,
                           "preview": mask_preview(e.text), "needed_for_task": pol["needed"],
                           "action": pol["action"], "replaced_with": ph, "reason": pol["reason"]})
        parts.append(text[cursor:])
        return "".join(parts), report


def mask_preview(s: str) -> str:
    s = s.strip()
    return "*" * len(s) if len(s) <= 3 else s[0] + "*" * (min(len(s), 12) - 2) + s[-1]


# --------------------------------------------------------------------------- #
# Attachment reading
# --------------------------------------------------------------------------- #
KIND_BY_EXT = {
    **{e: "text" for e in (".txt", ".md", ".csv", ".tsv", ".json", ".log", ".py", ".js", ".ts", ".java", ".c", ".cpp",
                           ".html", ".xml", ".yaml", ".yml", ".ini", ".env", ".sql", ".rtf")},
    ".pdf": "pdf", ".docx": "docx", ".xlsx": "xlsx",
    **{e: "image" for e in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff", ".heic")},
    **{e: "video" for e in (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")},
    **{e: "audio" for e in (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac")},
}


def read_attachment(path: Path) -> dict:
    """Return {kind, text, inspectable, notes, meta_entities}"""
    kind = KIND_BY_EXT.get(path.suffix.lower(), "other")
    res = {"kind": kind, "text": "", "inspectable": False, "notes": [], "gps": False}
    try:
        if kind == "text":
            res["text"] = path.read_text(encoding="utf-8", errors="ignore")
            res["inspectable"] = True

        elif kind == "pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            res["text"] = "\n".join((p.extract_text() or "") for p in reader.pages)
            res["inspectable"] = len(res["text"].strip()) > 20
            if not res["inspectable"]:
                res["notes"].append("PDF has no extractable text (scanned?). Cannot verify content.")

        elif kind == "docx":
            import docx
            d = docx.Document(str(path))
            chunks = [p.text for p in d.paragraphs]
            for t in d.tables:
                for row in t.rows:
                    chunks.append(" | ".join(c.text for c in row.cells))
            res["text"] = "\n".join(chunks)
            res["inspectable"] = True

        elif kind == "xlsx":
            import openpyxl
            wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
            rows = []
            for ws in wb.worksheets:
                for row in ws.iter_rows(values_only=True):
                    rows.append(" | ".join("" if c is None else str(c) for c in row))
            res["text"] = "\n".join(rows)
            res["inspectable"] = True

        elif kind == "image":
            from PIL import Image
            img = Image.open(path)
            try:
                gps = img.getexif().get_ifd(0x8825)
                res["gps"] = bool(gps)
            except Exception:
                pass
            try:
                import pytesseract
                res["text"] = pytesseract.image_to_string(img)
                res["inspectable"] = len(res["text"].strip()) > 5
                if not res["inspectable"]:
                    res["notes"].append("No readable text found; visual content (faces, ID cards, screens) cannot be verified.")
            except Exception:
                res["notes"].append("OCR unavailable (install pytesseract + Tesseract) - image text cannot be verified.")

        elif kind in ("video", "audio"):
            res["notes"].append(f"{kind} content cannot be inspected locally by this script.")
        else:
            res["notes"].append("Unknown file type - cannot inspect.")
    except Exception as e:
        res["notes"].append(f"Could not read file: {e}")
    return res


# --------------------------------------------------------------------------- #
# Relevance (Agent-1 step 2b)
# --------------------------------------------------------------------------- #
STOP = set("""a an the and or but if then than that this those these there here for with from into onto over under about
above below of on in at to by as is are was were be been being am do does did done have has had having can could would
should will shall may might must not no yes it its i me my mine we our you your he she they them his her their please
just also very more most some any all each every other such only own same so too s t don now get got let make made help
tell need want like using use used give show write summarize summarise translate rewrite extract analyze analyse explain
attached attachment attach file files document documents doc pdf image photo picture screenshot video clip here""".split())

TYPE_WORDS = {
    "pdf": r"\bpdf\b", "docx": r"\b(?:docx|word doc(?:ument)?)\b", "xlsx": r"\b(?:xlsx|excel|spreadsheet|sheet)\b",
    "image": r"\b(?:image|photo|picture|screenshot|png|jpe?g|selfie|scan)\b",
    "video": r"\b(?:video|clip|recording|mp4)\b", "audio": r"\b(?:audio|voice note|recording|mp3)\b",
    "text": r"\b(?:csv|txt|log|script|code)\b",
}
GENERIC_REF = re.compile(r"\b(attach(?:ed|ment|ments)?|this|these|the (?:file|document|doc|report|resume|cv|contract))\b", re.I)


def stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "ly", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def terms(text: str) -> set[str]:
    text = re.sub(r"[<\[][A-Z_0-9]+[>\]]", " ", text)  # drop placeholders
    return {stem(w) for w in re.findall(r"[a-z][a-z0-9]{2,}", text.lower()) if w not in STOP}


def relevance(prompt: str, path: Path, info: dict, n_attachments: int = 1) -> tuple[float, list[str]]:
    why, score = [], 0.0
    task_terms = terms(prompt)
    doc_terms = terms(info["text"]) if info["inspectable"] else set()

    if task_terms and doc_terms:
        cov = len(task_terms & doc_terms) / len(task_terms)
        score += cov
        if cov:
            why.append(f"{len(task_terms & doc_terms)}/{len(task_terms)} task keywords appear in the file")

    stem_name = re.sub(r"[_\-.]+", " ", path.stem.lower()).strip()
    if stem_name and stem_name in re.sub(r"[_\-.]+", " ", prompt.lower()):
        score += 0.6
        why.append("file is named in the prompt")

    mentioned = {k for k, rx in TYPE_WORDS.items() if re.search(rx, prompt, re.I)}
    is_media = info["kind"] in ("image", "video", "audio")
    cov_zero = not (task_terms and doc_terms and (task_terms & doc_terms))
    if info["kind"] in mentioned or (info["kind"] == "text" and "text" in mentioned):
        score += 0.35
        why.append(f"prompt refers to a {info['kind']}")
    elif not (mentioned - {"text"}) and GENERIC_REF.search(prompt):
        if is_media and n_attachments > 1:
            why.append("generic 'attached' wording is not enough to link this media file to the task")
        elif info["inspectable"] and task_terms and cov_zero:
            score += 0.15
            why.append("generic 'attached' wording, but the content shares no keywords with the task")
        else:
            score += 0.35
            why.append("prompt refers generically to 'attached/this' content")
    elif mentioned:
        why.append(f"prompt refers to {sorted(mentioned)}, not to a {info['kind']}")
    return min(1.0, round(score, 2)), why


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
REL_MIN = 0.25


def risk_summary(entities: list[dict]) -> dict:
    if not entities:
        return {"level": "NONE", "score": 0, "max_severity": 0, "by_category": {}}
    mx = max(e["severity"] for e in entities)
    score = min(100, mx * 8 + min(20, len(entities) * 2))
    level = "CRITICAL" if mx >= CRITICAL_SEVERITY else "HIGH" if mx >= 7 else "MEDIUM" if mx >= 4 else "LOW"
    return {"level": level, "score": score, "max_severity": mx,
            "by_category": dict(Counter(e["category"] for e in entities))}


def main() -> int:
    ap = argparse.ArgumentParser(description="Agent 1 - sensitive data detection, relevance and sanitization")
    ap.add_argument("--prompt", help="the user prompt (or use --prompt-file / stdin)")
    ap.add_argument("--prompt-file", help="read prompt from a text file")
    ap.add_argument("--attach", nargs="*", default=[], help="attachment paths (pdf/docx/xlsx/txt/image/video...)")
    ap.add_argument("--dictionary", help='JSON file of custom sensitive terms: {"terms": ["Project Phoenix", "Acme"]}')
    ap.add_argument("--allow", nargs="*", default=[], help="extra words that must NOT be treated as sensitive names/orgs")
    ap.add_argument("--out-dir", default="privacy_output")
    ap.add_argument("--strict", action="store_true",
                    help="any critical item (secret/gov-ID) inside an attachment => remove the whole attachment")
    ap.add_argument("--unverifiable", choices=["review", "remove"], default="review",
                    help="policy for attachments that cannot be inspected locally but are mentioned in the prompt")
    ap.add_argument("--no-vault", action="store_true", help="do not write the local mapping vault")
    ap.add_argument("--json", action="store_true", help="print full report JSON to stdout")
    args = ap.parse_args()

    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    elif args.prompt is not None:
        prompt = args.prompt
    else:
        print("Enter prompt (Ctrl-D / Ctrl-Z+Enter to finish):", file=sys.stderr)
        prompt = sys.stdin.read()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    detector = Detector(args.dictionary)
    detector.allow |= {w.lower() for w in args.allow}
    sanitizer = Sanitizer()
    task = analyze_task(prompt)

    # ---- 1. prompt ---------------------------------------------------------
    p_ents = detector.detect(prompt)
    sanitized, p_report = sanitizer.apply(prompt, p_ents, task)
    sanitized = re.sub(r"[ \t]+", " ", sanitized).strip()

    leftover_words = re.findall(r"[A-Za-z]{2,}", re.sub(r"[<\[][A-Z_0-9]+[>\]]", " ", sanitized))
    if any(e["severity"] >= CRITICAL_SEVERITY or e["category"] == "SECRET" for e in p_report) and len(leftover_words) < 3:
        prompt_action = "block"
    elif p_report:
        prompt_action = "sanitize"
    else:
        prompt_action = "allow"

    # ---- 2. attachments ------------------------------------------------------
    att_details = []
    for a in args.attach:
        path = Path(a)
        if not path.exists():
            att_details.append({"file": a, "action": "remove", "reasons": ["File not found"]})
            continue
        info = read_attachment(path)
        rel, rel_why = relevance(prompt, path, info, len(args.attach))
        ents = detector.detect(info["text"]) if info["inspectable"] else []
        ents_report_text, ents_report = (info["text"], [])
        if ents:
            ents_report_text, ents_report = sanitizer.apply(info["text"], ents, task)
        if info["gps"]:
            ents_report.append({"entity_type": "GPS_EXIF", "category": "METADATA", "severity": SEVERITY["METADATA"],
                                "action": "strip_metadata", "reason": "Image carries GPS location metadata."})

        critical = [e for e in ents_report if e["severity"] >= CRITICAL_SEVERITY]
        reasons = list(rel_why) + info["notes"]
        action, sanitized_copy = "keep", None

        if not info["inspectable"]:
            referenced = rel >= REL_MIN
            if not referenced:
                action = "remove"
                reasons.append("Cannot be inspected and is not referenced by the task.")
            else:
                action = "remove" if args.unverifiable == "remove" else "review"
                reasons.append("Referenced by the task but cannot be inspected locally - confirm it has no sensitive content.")
        elif rel < REL_MIN:
            action = "remove"
            reasons.append(f"Not relevant to the task (relevance {rel} < {REL_MIN}).")
        elif critical and (args.strict or len(critical) > 2):
            action = "remove"
            reasons.append(f"Contains {len(critical)} critical item(s) (secrets / government IDs).")
        elif ents_report:
            action = "sanitize"
            reasons.append(f"Relevant but contains {len(ents_report)} sensitive item(s); send the sanitized copy only.")
            if info["kind"] in ("image",) and ents:
                action = "remove"  # cannot redact pixels
                reasons.append("Sensitive text is inside an image; pixels cannot be redacted here, so remove it.")
            elif info["kind"] == "image" and info["gps"]:
                action = "sanitize"
        else:
            reasons.append("Relevant and no sensitive data detected.")

        if action == "sanitize" and info["kind"] != "image":
            copy_path = out / f"{path.stem}.sanitized.txt"
            copy_path.write_text(ents_report_text, encoding="utf-8")
            sanitized_copy = str(copy_path)
        elif action == "sanitize" and info["kind"] == "image":
            try:
                from PIL import Image
                im = Image.open(path)
                clean = Image.new(im.mode, im.size)
                clean.paste(im)
                copy_path = out / f"{path.stem}.nometadata{path.suffix}"
                clean.save(copy_path)
                sanitized_copy = str(copy_path)
            except Exception as e:
                action = "remove"
                reasons.append(f"Could not strip metadata ({e}).")

        att_details.append({
            "file": str(path), "type": info["kind"], "inspectable": info["inspectable"],
            "relevance_score": rel, "relevant": rel >= REL_MIN,
            "risk": risk_summary(ents_report), "action": action, "reasons": reasons,
            "sanitized_copy": sanitized_copy,
            "sensitive_items": [{k: v for k, v in e.items() if k != "reason"} for e in ents_report],
        })

    decisions = {
        "attachments_to_remove": [d["file"] for d in att_details if d["action"] == "remove"],
        "attachments_to_sanitize": [{"original": d["file"], "send_instead": d["sanitized_copy"]}
                                    for d in att_details if d["action"] == "sanitize"],
        "attachments_needing_user_review": [d["file"] for d in att_details if d["action"] == "review"],
        "attachments_ok_to_send": [d["file"] for d in att_details if d["action"] == "keep"],
        "details": [{"file": d["file"], "action": d["action"], "relevance_score": d.get("relevance_score"),
                     "reasons": d["reasons"]} for d in att_details],
    }

    report = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "detection_engine": detector.engine,
        "task": task,
        "prompt": {
            "action": prompt_action,
            "risk": risk_summary(p_report),
            "sensitive_items": p_report,
            "sanitized_prompt": sanitized,
            "cloud_eligible": prompt_action != "block",
        },
        "attachments": att_details,
        "note": "This report intentionally contains only masked previews of sensitive values.",
    }

    (out / "sanitized_prompt.txt").write_text(sanitized, encoding="utf-8")
    (out / "attachment_decisions.json").write_text(json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    final_texts = sanitized + "\n" + "\n".join(
        Path(d["sanitized_copy"]).read_text(encoding="utf-8") for d in att_details
        if d.get("sanitized_copy") and d["type"] != "image")
    sanitizer.vault = {k: v for k, v in sanitizer.vault.items() if k in final_texts}
    if not args.no_vault and sanitizer.vault:
        vp = out / "vault.json"
        vp.write_text(json.dumps(sanitizer.vault, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            os.chmod(vp, stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            pass

    # ---- console summary -----------------------------------------------------
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"\nEngine           : {detector.engine}")
        print(f"Task type        : {task['task_type']}   (needs: {task['needs_categories'] or 'no personal data'})")
        r = report["prompt"]["risk"]
        print(f"Prompt risk      : {r['level']} (score {r['score']})  -> action: {prompt_action.upper()}")
        for e in p_report:
            print(f"   - {e['entity_type']:<14} {e['preview']:<14} needed={str(e['needed_for_task']):<5} -> {e['replaced_with']}")
        print(f"\nSanitized prompt :\n   {sanitized}\n")
        if att_details:
            print("Attachments:")
            for d in att_details:
                print(f"   [{d['action'].upper():<8}] {d['file']}  (relevance {d.get('relevance_score', '-')}, "
                      f"risk {d.get('risk', {}).get('level', '-')})")
                for why in d["reasons"]:
                    print(f"       · {why}")
        print(f"\nFiles written to : {out.resolve()}")
        print("   sanitized_prompt.txt | attachment_decisions.json | report.json"
              + (" | vault.json (LOCAL ONLY - never upload)" if sanitizer.vault and not args.no_vault else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
