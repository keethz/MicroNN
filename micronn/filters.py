"""Attribute predicates for hybrid queries (PLAN.md section 4).

A filter is a conjunction (AND) of simple predicates on attribute columns:
    [eq("city", "Seattle"), between("year", 2019, 2021)]
"""
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

OPS = {"=", "<", "<=", ">", ">=", "IN", "BETWEEN"}


@dataclass(frozen=True)
class Pred:
    col: str
    op: str
    value: Any  # scalar; tuple for IN; (lo, hi) for BETWEEN


Filter = Sequence[Pred]


def eq(col, v): return Pred(col, "=", v)
def lt(col, v): return Pred(col, "<", v)
def le(col, v): return Pred(col, "<=", v)
def gt(col, v): return Pred(col, ">", v)
def ge(col, v): return Pred(col, ">=", v)
def isin(col, vs): return Pred(col, "IN", tuple(vs))
def between(col, lo, hi): return Pred(col, "BETWEEN", (lo, hi))


def to_sql(f: Filter, alias: str, columns: Sequence[str]) -> Tuple[str, List[Any]]:
    """Translate to a parameterized WHERE fragment. Column names are checked against the
    schema (they cannot be bound as parameters), values are always bound."""
    allowed = set(columns)
    parts, params = [], []
    for p in f:
        if p.col not in allowed:
            raise ValueError(f"unknown attribute column {p.col!r}")
        if p.op not in OPS:
            raise ValueError(f"unsupported operator {p.op!r}")
        c = f"{alias}.{p.col}"
        if p.op == "IN":
            if not p.value:
                parts.append("0")
                continue
            parts.append(f"{c} IN ({', '.join('?' * len(p.value))})")
            params.extend(p.value)
        elif p.op == "BETWEEN":
            parts.append(f"{c} BETWEEN ? AND ?")
            params.extend(p.value)
        else:
            parts.append(f"{c} {p.op} ?")
            params.append(p.value)
    return (" AND ".join(parts) or "1"), params


def matches(f: Filter, row: dict) -> bool:
    """Evaluate in Python (used by tests and ground truth)."""
    for p in f:
        x = row[p.col]
        if x is None:
            return False
        if p.op == "=" and not x == p.value: return False
        if p.op == "<" and not x < p.value: return False
        if p.op == "<=" and not x <= p.value: return False
        if p.op == ">" and not x > p.value: return False
        if p.op == ">=" and not x >= p.value: return False
        if p.op == "IN" and x not in p.value: return False
        if p.op == "BETWEEN" and not p.value[0] <= x <= p.value[1]: return False
    return True
