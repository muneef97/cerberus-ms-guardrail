#!/usr/bin/env python3
"""
CERBERUS Guardrail MVP (raw -> canonical -> validate)
- Reads RAW DFD (components/connections)
- Maps to CANONICAL (uses data.tb for trust boundaries when present)
- Builds trust_boundaries from all TBs seen (fallback tb-default)
- Validates against external canonical_schema.json (draft-07)
- Always outputs JSON: { valid, errors, warnings, normalized }
"""

import json
import sys
import pathlib
from typing import Any, Dict, List, Set, Tuple

# ---------- utilities ----------

def load_json(path: str) -> Dict[str, Any]:
    p = pathlib.Path(path)
    if not p.exists():
        return {"__error__": f"File not found: {path}"}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as ex:
        return {"__error__": f"Failed to parse JSON from {path}: {ex}"}

def as_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2))

# ---------- raw -> canonical mapping ----------

SHAPE_TO_NODE = {
    "oval": "external_entities",
    "rect": "processes",
    "document": "data_stores",
}

def infer_protocol(label_a: str, label_b: str) -> str:
    s = f"{(label_a or '').lower()} {(label_b or '').lower()}"
    if "https://" in s or " tls" in s:
        return "HTTPS"
    if "http://" in s:
        return "HTTP"
    if "grpc" in s:
        return "gRPC"
    if "amqp" in s:
        return "AMQP"
    if "udp" in s:
        return "UDP"
    if "tcp" in s:
        return "TCP"
    return "UNSPECIFIED"

def map_raw_to_canonical(raw: Dict[str, Any]) -> Dict[str, Any]:
    nodes = {"processes": [], "data_stores": [], "external_entities": []}
    seen_tb_ids: Set[str] = set()

    # Map components -> nodes (use data.tb if present)
    for comp in raw.get("components", []):
        bucket = SHAPE_TO_NODE.get(comp.get("type"))
        if not bucket:
            # Unknown shape -> skip in MVP
            continue
        data = comp.get("data") or {}
        name = (data.get("text") or "").strip()
        tb_from_raw = (data.get("tb") or "").strip()
        tb_id = tb_from_raw if tb_from_raw else "tb-default"
        seen_tb_ids.add(tb_id)

        nodes[bucket].append({
            "id": (comp.get("id") or "").strip(),
            "name": name,
            "trust_boundary_id": tb_id
        })

    # Map connections -> flows
    flows = []
    for conn in raw.get("connections", []):
        d = conn.get("data") or {}
        start = (d.get("startLabel") or "").strip()
        end = (d.get("endLabel") or "").strip()
        data_label = start if start else end
        if start and end and start != end:
            data_label = f"{start} | {end}"
        flows.append({
            "id": (conn.get("id") or "").strip(),
            "source": (conn.get("from") or "").strip(),
            "target": (conn.get("to") or "").strip(),
            "data": data_label,
            "protocol": infer_protocol(start, end)
        })

    # Build trust_boundaries from what we actually used
    trust_boundaries = [{"id": tb, "name": tb if tb != "tb-default" else "Default"} for tb in sorted(seen_tb_ids)]
    if not trust_boundaries:
        trust_boundaries = [{"id": "tb-default", "name": "Default"}]

    canonical = {
        "version": "1.0.0",
        "metadata": {"name": raw.get("name", "")},
        "nodes": nodes,
        "trust_boundaries": trust_boundaries,
        "flows": flows
    }
    return canonical

# ---------- validation (schema + rules) ----------

def schema_validate(canon: Dict[str, Any], schema: Dict[str, Any]) -> List[Dict[str, str]]:
    try:
        from jsonschema import Draft7Validator
        v = Draft7Validator(schema)
        errs = []
        for e in sorted(v.iter_errors(canon), key=lambda x: list(x.absolute_path)):
            path = "$" + "".join([f".{p}" if isinstance(p, str) else f"[{p}]" for p in e.absolute_path])
            errs.append({"type": "schema", "path": path or "$", "message": e.message})
        return errs
    except Exception as ex:
        return [{"type": "schema", "path": "$", "message": f"Schema validation error: {ex}"}]

# guardrail_mvp.py

def rule_validate(canon: dict) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Return (errors, warnings) after rule checks.

    Accepts canonical DFD where `nodes` may be either:
      - dict-of-lists with plural or singular bucket keys, or
      - a flat list of node dicts.
    """
    errs: List[Dict[str, Any]] = []
    warns: List[Dict[str, Any]] = []

    # ---- 1) Normalize nodes into flat list + id map
    nodes_field = canon.get("nodes", {})
    flat_nodes: List[Dict[str, Any]] = []

    if isinstance(nodes_field, dict):
        # support plural and singular keys
        bucket_keys = (
            "processes", "process",
            "data_stores", "data_store",
            "external_entities", "external_entity",
        )
        for key in bucket_keys:
            items = nodes_field.get(key) or []
            if isinstance(items, list):
                for n in items:
                    if isinstance(n, dict):
                        flat_nodes.append(n)
    elif isinstance(nodes_field, list):
        for n in nodes_field:
            if isinstance(n, dict):
                flat_nodes.append(n)

    id2node: Dict[str, Dict[str, Any]] = {}
    dup = set()
    for n in flat_nodes:
        nid = n.get("id")
        if not nid:
            errs.append({"type": "schema", "message": "Node missing id"})
            continue
        if nid in id2node:
            dup.add(nid)
        id2node[nid] = n
    for d in dup:
        errs.append({"type": "schema", "message": f"Duplicate node id '{d}'"})

    # ---- 2) Flows basic checks
    flows = canon.get("flows") or []
    if not isinstance(flows, list):
        flows = []

    connected_ids = set()
    for f in flows:
        if not isinstance(f, dict):
            continue
        sid = f.get("source")
        tid = f.get("target")
        if sid:
            connected_ids.add(sid)
        if tid:
            connected_ids.add(tid)

        if sid not in id2node:
            errs.append({"type": "rule", "message": f"Unknown source '{sid}'"})
        if tid not in id2node:
            errs.append({"type": "rule", "message": f"Unknown target '{tid}'"})
        if sid and tid and sid == tid:
            errs.append({"type": "rule", "message": f"Self-flow not allowed: '{sid}'"})

        data_val = f.get("data")
        if not isinstance(data_val, str) or not data_val.strip():
            errs.append({"type": "rule", "message": f"Flow '{f.get('id')}' must include non-empty 'data'"})

    # ---- 3) Orphan warnings
    for nid in id2node.keys():
        if nid not in connected_ids:
            warns.append({"type": "rule", "message": f"Orphan node '{nid}'"})

    # ---- 4) Cross-boundary HTTPS preference (optional warning)
    node2tb: Dict[str, str] = {}
    for n in flat_nodes:
        nid = n.get("id")
        tb = n.get("trust_boundary_id")
        if nid and tb:
            node2tb[nid] = tb

    for f in flows:
        if not isinstance(f, dict):
            continue
        proto = (f.get("protocol") or "").upper()
        sid, tid = f.get("source"), f.get("target")
        if sid in node2tb and tid in node2tb and node2tb[sid] != node2tb[tid]:
            if proto != "HTTPS":
                warns.append({
                    "type": "security",
                    "message": f"Cross-boundary flow '{f.get('id')}' uses '{proto or 'UNSPECIFIED'}'; prefer HTTPS"
                })

    return errs, warns




# ---------- main (CLI) ----------

def main() -> None:
    # Usage: python guardrail_mvp.py --raw dfd.json --schema canonical_schema.json
    if len(sys.argv) < 5 or sys.argv[1] != "--raw" or sys.argv[3] != "--schema":
        print("Usage: python guardrail_mvp.py --raw dfd.json --schema canonical_schema.json", file=sys.stderr)
        sys.exit(2)

    raw_path = sys.argv[2]
    schema_path = sys.argv[4]

    raw = load_json(raw_path)
    if "__error__" in raw:
        as_json({
            "valid": False,
            "errors": [{"type": "parse", "path": raw_path, "message": raw["__error__"]}],
            "warnings": [],
            "normalized": None
        })
        sys.exit(0)

    schema = load_json(schema_path)
    if "__error__" in schema:
        as_json({
            "valid": False,
            "errors": [{"type": "schema_load", "path": schema_path, "message": schema["__error__"]}],
            "warnings": [],
            "normalized": None
        })
        sys.exit(0)

    canonical = map_raw_to_canonical(raw)
    schema_errors = schema_validate(canonical, schema)
    rule_errors, warnings = rule_validate(canonical)

    valid = (len(schema_errors) + len(rule_errors)) == 0
    result = {
        "valid": valid,
        "errors": schema_errors + rule_errors,
        "warnings": warnings,
        "normalized": canonical
    }
    as_json(result)

if __name__ == "__main__":
    main()
