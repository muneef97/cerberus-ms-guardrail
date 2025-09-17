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
