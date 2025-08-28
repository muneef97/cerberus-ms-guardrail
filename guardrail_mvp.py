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

def rule_validate(canon: Dict[str, Any]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []

    tb_ids: Set[str] = {tb.get("id") for tb in canon.get("trust_boundaries", []) if tb.get("id")}
    node_ids: Set[str] = set()
    node_tb: Dict[str, str] = {}
    node_deg: Dict[str, int] = {}

    buckets = {
        "processes": "nodes.processes",
        "data_stores": "nodes.data_stores",
        "external_entities": "nodes.external_entities",
    }

    # Nodes
    for bucket, prefix in buckets.items():
        for i, n in enumerate(canon.get("nodes", {}).get(bucket, [])):
            nid = n.get("id", "")
            if nid in node_ids:
                errors.append({"type": "rule", "path": f"{prefix}[{i}].id", "message": f"Duplicate node id '{nid}'"})
            else:
                node_ids.add(nid)
            if not (n.get("name") or "").strip():
                errors.append({"type": "rule", "path": f"{prefix}[{i}].name", "message": "Name is required"})
            tbid = n.get("trust_boundary_id")
            if not tbid:
                errors.append({"type": "rule", "path": f"{prefix}[{i}].trust_boundary_id", "message": "trust_boundary_id is required"})
            elif tbid not in tb_ids:
                errors.append({"type": "rule", "path": f"{prefix}[{i}].trust_boundary_id", "message": f"Unknown trust_boundary_id '{tbid}'"})
            node_tb[nid] = tbid or ""
            node_deg[nid] = 0

    # Flows
    flow_ids: Set[str] = set()
    pair_seen: Set[Tuple[str, str, str]] = set()

    for i, f in enumerate(canon.get("flows", [])):
        fpath = f"flows[{i}]"
        fid = f.get("id", "")
        if fid in flow_ids:
            errors.append({"type": "rule", "path": f"{fpath}.id", "message": f"Duplicate flow id '{fid}'"})
        else:
            flow_ids.add(fid)

        src = f.get("source", "")
        tgt = f.get("target", "")
        if src == tgt and src:
            errors.append({"type": "rule", "path": fpath, "message": "Self-flow is not allowed (source == target)"})
        if src not in node_ids:
            errors.append({"type": "rule", "path": f"{fpath}.source", "message": f"Unknown source '{src}'"})
        if tgt not in node_ids:
            errors.append({"type": "rule", "path": f"{fpath}.target", "message": f"Unknown target '{tgt}'"})

        if src in node_deg:
            node_deg[src] += 1
        if tgt in node_deg:
            node_deg[tgt] += 1

        key = (src, tgt, f.get("data", ""))
        if key in pair_seen:
            warnings.append({"path": fpath, "message": "Duplicate flow pair (same source, target, data)"})
        else:
            pair_seen.add(key)

    # Orphan nodes
    for nid, deg in node_deg.items():
        if deg == 0:
            warnings.append({"path": f"$[node:'{nid}']", "message": "Orphan node (no incoming or outgoing flows)"})

    # Cross-boundary without HTTPS
    for i, f in enumerate(canon.get("flows", [])):
        src, tgt = f.get("source"), f.get("target")
        tb_src, tb_tgt = node_tb.get(src), node_tb.get(tgt)
        if tb_src and tb_tgt and tb_src != tb_tgt:
            if f.get("protocol") != "HTTPS":
                warnings.append({"path": f"flows[{i}]", "message": "Cross-boundary flow without HTTPS"})

    return errors, warnings

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
