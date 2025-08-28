# main.py — refine + validate (auto map RAW -> Canonical)
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import json, pathlib, os, re

# Import your MVP helpers
from guardrail_mvp import schema_validate, rule_validate, map_raw_to_canonical

# -------- Optional LLM (refine fallback; auto-disabled if no key) --------
LLM_AVAILABLE = False
try:
    import openai  # pip install openai
    if os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

app = FastAPI(title="CERBERUS Guardrail Service", version="1.2.0")

# -------- CORS --------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Models --------
class PromptInput(BaseModel):
    text: str = Field(..., description="Plain-English prompt from the user")

# -------- Load external files --------
def _load_schema() -> Dict[str, Any]:
    path = pathlib.Path("canonical_schema.json")
    if not path.exists():
        raise FileNotFoundError("canonical_schema.json not found next to main.py")
    return json.loads(path.read_text(encoding="utf-8"))

def _load_policies() -> Dict[str, Any]:
    path = pathlib.Path("prompt_policies.json")
    if not path.exists():
        return {
            "min_length": 15,
            "banned_words": [],
            "require_any_keywords": [
                "dfd","data flow diagram","actor","actors","process","processes",
                "data store","datastore","database","db","flow","flows","external","user","system","server"
            ],
            "hints": [
                "Describe at least one actor (e.g., User or External System).",
                "List one or more processes (e.g., Login, Search).",
                "Include a data store (e.g., Database, Movie DB).",
                "Mention at least one flow connecting components."
            ]
        }
    return json.loads(path.read_text(encoding="utf-8"))

SCHEMA: Dict[str, Any] = _load_schema()
POLICY: Dict[str, Any] = _load_policies()

# -------- Prompt refinement helpers (no arrows) --------
_FILLER_PATTERNS = [
    r"\bplease\b", r"\bkindly\b", r"\bcan you\b", r"\bcould you\b", r"\bwould you\b",
    r"\bgive me\b", r"\bshow me\b", r"\bgenerate\b", r"\bcreate\b",
    r"\bdiagram of\b", r"\bdiagram\b", r"\bdata flow diagram\b"
]
_NORMALIZE_PAIRS = [
    (r"\bshow\s+how\b", "how"),
    (r"\bshow\s+the\b", "the"),
    (r"\bshow\s+me\b", ""),
    (r"\bgo(es)?\s+to\b", "proceeds to"),
    (r"\bsend(s)?\s+to\b", "is sent to"),
    (r"\bflows\s+to\b", "flows to"),
    (r"\binto\s+the\b", "into the"),
]

def _regex_cleanup(raw: str) -> str:
    text = raw.strip()
    for pat in _FILLER_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    for pat, repl in _NORMALIZE_PAIRS:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    text = re.sub(r"^[\s,.;:-]+", "", text)
    text = re.sub(r"\s+", " ", text).strip(" .,:;")
    return text

def _looks_dfdy(text: str) -> bool:
    if not text:
        return False
    req_any = [k.lower() for k in POLICY.get("require_any_keywords", [])]
    low = text.lower()
    return any(k in low for k in req_any) if req_any else True

def _too_short(text: str) -> bool:
    return len(text.strip()) < int(POLICY.get("min_length", 15))

def _has_banned(text: str) -> bool:
    banned = [w.lower() for w in POLICY.get("banned_words", [])]
    low = text.lower()
    return any(b and b in low for b in banned)

def _llm_refine(raw: str) -> Optional[str]:
    if not LLM_AVAILABLE:
        return None
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content":
                 "You are a guardrail that refines user input for DFD creation. "
                 "Return one sentence starting with 'Create a DFD illustrating ...'. "
                 "Use formal wording, no arrows, no bullets. Remove filler and unrelated content. "
                 "If input is not about DFDs, return 'INVALID'."},
                {"role": "user", "content": raw}
            ],
            max_tokens=80,
            temperature=0.2,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return None

def _finalize_refined(text: str) -> str:
    core = text.strip().rstrip(".")
    core = re.sub(r"^(show(ing)?|describe|explore|explain)\s*[:\-]?\s*", "", core, flags=re.IGNORECASE)
    core = re.sub(r"\bshow(ing)?\b\s*[:\-]?\s*", "", core, flags=re.IGNORECASE)
    return f"Create a DFD illustrating {core}."

# =========================
# 1) /refine-prompt
# =========================
@app.post("/refine-prompt")
def refine_prompt(input_data: PromptInput):
    """
    Returns:
      - { "valid": false, "errors": [...], "hints": [...] }
      - { "valid": true,  "refined_prompt": "Create a DFD illustrating ..." }
    """
    raw = (input_data.text or "").strip()
    hints: List[str] = POLICY.get("hints", [])
    errors: List[Dict[str, str]] = []

    if _has_banned(raw):
        return {"valid": False, "errors": [{"type":"content","message":"Prompt contains disallowed language."}], "hints": hints}
    if _too_short(raw):
        errors.append({"type":"quality","message":f"Prompt too short (min {POLICY.get('min_length',15)} characters)."})

    cleaned = _regex_cleanup(raw)

    if not _looks_dfdy(cleaned):
        llm_out = _llm_refine(raw)
        if llm_out:
            if llm_out.strip().upper() == "INVALID":
                return {"valid": False, "errors":[{"type":"quality","message":"Prompt unrelated to DFD."}], "hints": hints}
            llm_core = re.sub(r"^create a dfd illustrating\s*", "", llm_out, flags=re.IGNORECASE)
            refined = _finalize_refined(llm_core)
            return {"valid": True, "refined_prompt": refined}
        errors.append({"type":"quality","message":"Ambiguous or incomplete. Include actors, processes, data stores, and at least one flow."})

    if errors and (not _looks_dfdy(cleaned) or _too_short(cleaned)):
        return {"valid": False, "errors": errors, "hints": hints}

    refined = _finalize_refined(cleaned)
    return {"valid": True, "refined_prompt": refined}

# =========================
# 2) /validate-json (RAW or Canonical)
# =========================
@app.post("/validate-json")
def validate_json(body: Dict[str, Any] = Body(...)):
    """
    Accepts ANY of:
      - Bare RAW editor JSON:
          { "components": [...], "connections": [...] }
      - Wrapped RAW:
          { "raw": { "components": [...], "connections": [...] } }
      - Canonical:
          { "dfd": { ... } }  or  { "json_data": { ... } }
      - Bare Canonical (advanced): starts with version/nodes/flows...

    Behavior:
      1) Detect RAW vs Canonical
      2) If RAW -> map_raw_to_canonical()
      3) Validate canonical (schema + rules)
      4) Return { valid, errors[], warnings[], normalized }
    """
    # Step 1: pick the candidate object from common wrappers or use body itself
    candidate = (
        body.get("dfd")
        or body.get("json_data")
        or body.get("raw")
        or body.get("editor_json")
        or body
    )

    # Step 2: detect RAW by keys
    is_raw_like = isinstance(candidate, dict) and "components" in candidate and "connections" in candidate

    # Step 3: normalize to canonical
    try:
        canonical = map_raw_to_canonical(candidate) if is_raw_like else candidate
    except Exception as ex:
        return {
            "valid": False,
            "errors": [{"type": "mapping", "message": f"Failed to map RAW to canonical: {ex}"}],
            "warnings": [],
            "normalized": {}
        }

    # Step 4: validate canonical JSON
    schema_errs = schema_validate(canonical, SCHEMA)
    rule_errs, warns = rule_validate(canonical)
    valid = (len(schema_errs) + len(rule_errs)) == 0

    return {
        "valid": valid,
        "errors": schema_errs + rule_errs,
        "warnings": warns,
        "normalized": canonical
    }
