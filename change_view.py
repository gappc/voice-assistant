"""HTML snippets for showing corrections in the dialogs: one line per change,
removed words struck through in red, added words bold green."""

import html

import text_corrections as tc

REMOVED = "color:#c0392b; text-decoration:line-through"
ADDED = "color:#1e8449; font-weight:bold"
MUTED = "color:gray"


def diff(before, after):
    out = []
    for op, text in tc.inline_diff(before, after):
        text = html.escape(text)
        if op == "=":
            out.append(text)
        else:
            out.append(f'<span style="{REMOVED if op == "-" else ADDED}">{text}</span>')
    return "".join(out)


def line(inner, prefix="", suffix=""):
    """One paragraph; prefix/suffix are shown muted (e.g. a date, a rule)."""
    pre = f'<span style="{MUTED}">{html.escape(prefix)}</span> ' if prefix else ""
    post = f' <span style="{MUTED}">{html.escape(suffix)}</span>' if suffix else ""
    return f'<p style="margin:0 0 5px 0">{pre}{inner}{post}</p>'


def heading(text):
    return f'<p style="margin:8px 0 4px 0"><b>{html.escape(text)}</b></p>'


def text(value):
    return line(html.escape(value))


def failures(items, label):
    """Examples that fail: what was expected vs. what the rules produce."""
    return "".join(
        line(f"{html.escape(src)}<br>&nbsp;&nbsp;expected → actual: {diff(want, got)}", label)
        for src, want, got in items)


def date(when):
    return when.strftime("%Y-%m-%d") if when else ""
