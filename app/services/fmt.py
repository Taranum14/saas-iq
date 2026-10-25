"""Number formatting for an Indian audience: lakh/crore grouping, true minus sign."""
import math

MINUS = "−"


def _bad(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def _group(n):
    """12345678 -> '1,23,45,678'"""
    s = str(abs(int(n)))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def inr(v, signed=False):
    """Full rupee amount: ₹16,17,596"""
    if _bad(v):
        return "—"
    n = int(round(v))
    sign = MINUS if n < 0 else ("+" if signed and n > 0 else "")
    return f"{sign}₹{_group(n)}"


def inr_short(v):
    """Spoken-style amount: ₹16.18 lakh, ₹1.62 crore, otherwise the full figure."""
    if _bad(v):
        return "—"
    a = abs(v)
    sign = MINUS if v < 0 else ""
    if a >= 1e7:
        return f"{sign}₹{a / 1e7:.2f} crore"
    if a >= 1e5:
        return f"{sign}₹{a / 1e5:.2f} lakh"
    return inr(v)


def inr_axis(v):
    """Compact tick label: ₹12L, ₹40k."""
    sign = MINUS if v < 0 else ""
    a = abs(v)
    if a >= 1e7:
        return f"{sign}₹{a / 1e7:g}Cr"
    if a >= 1e5:
        return f"{sign}₹{a / 1e5:g}L"
    if a >= 1e3:
        return f"{sign}₹{a / 1e3:g}k"
    return f"{sign}₹{a:g}"


def num(v):
    return "—" if _bad(v) else _group(round(v)) if v >= 0 else MINUS + _group(round(v))


def pct(v, d=1):
    return "—" if _bad(v) else f"{v * 100:.{d}f}%"


def signed_pct(v, d=1):
    if _bad(v):
        return "—"
    sign = MINUS if v < 0 else "+"
    return f"{sign}{abs(v) * 100:.{d}f}%"


def times(v, d=1):
    return "—" if _bad(v) else f"{v:.{d}f}×"


FILTERS = dict(inr=inr, inr_short=inr_short, num=num, pct=pct, signed_pct=signed_pct, times=times)
