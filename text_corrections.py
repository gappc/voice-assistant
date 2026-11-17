"""Fix words the speech recognizer keeps getting wrong.

Rules live in a TOML file (default ~/.config/voice-assistant/corrections.toml):

    [[rule]]
    replace = ["Link", "linken"]          # misrecognized variants (whole words, any case)
    with = "Lynq"                         # what they should be
    followed_by = ["API", "Projekt"]      # optional: only before one of these words
    preceded_by = ["das"]                 # optional: only after one of these words
    examples = [["Die alte Link API.", "Die alte Lynq API."]]   # input -> expected
    keep = ["Gib mir einen Link."]                              # must stay unchanged

For cases plain words can't express, `regex = '...'` (Python, case-insensitive)
takes the place of `replace`, and `with` is then an re.sub template.

Rules run in file order on every dictation. `examples` and `keep` are checked
against *all* rules, so a new rule that breaks an older example shows up in
`check`. The running app reloads the file when it changes.

Command line (`uv run text_corrections.py --help`):
    check                 verify every example
    try TEXT              show what the rules make of TEXT
    history [REGEX]       apply the rules to past dictations from the journal
    add WRONG RIGHT       derive a rule from a corrected sentence and append it
"""

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


def default_path():
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "voice-assistant" / "corrections.toml"


@dataclass(frozen=True)
class Rule:
    pattern: re.Pattern
    replace: str  # re.sub template
    examples: tuple = field(default=())  # (input, expected) pairs

    @classmethod
    def make(cls, regex, template, examples=()):
        rule = cls(re.compile(regex, re.IGNORECASE), template, tuple(examples))
        rule.pattern.sub(rule.replace, "")  # validates group references
        return rule

    @classmethod
    def from_spec(cls, spec, examples=()):
        """Rule from a rule table (see module docstring). ValueError if invalid."""
        try:
            return cls.make(*compile_spec(spec), examples)
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc}") from None


SPEC_KEYS = ("replace", "regex", "with", "followed_by", "preceded_by")


def _words(spec, key):
    value = spec.get(key, [])
    if isinstance(value, str):
        value = [value]
    if not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
        raise ValueError(f"'{key}' must be a list of strings")
    return [v.strip() for v in value if v.strip()]


def _phrase(text):
    """Regex for `text` as whole words; any run of spaces matches any spaces."""
    parts = text.split()
    rx = r"\s+".join(re.escape(p) for p in parts)
    if re.match(r"\w", parts[0]):
        rx = r"(?<!\w)" + rx
    if re.search(r"\w$", parts[-1]):
        rx += r"(?!\w)"
    return rx


def _any(phrases):
    return "(?:" + "|".join(_phrase(p) for p in phrases) + ")"


def compile_spec(spec):
    """(regex, re.sub template) for a rule table."""
    if not isinstance(spec.get("with"), str):
        raise ValueError("needs 'with' (the correct text)")
    if "regex" in spec:
        if not isinstance(spec["regex"], str) or not spec["regex"]:
            raise ValueError("'regex' must be a non-empty string")
        if "replace" in spec:
            raise ValueError("use either 'replace' or 'regex', not both")
        return spec["regex"], spec["with"]
    variants = _words(spec, "replace")
    if not variants:
        raise ValueError("needs 'replace' (the misrecognized words)")
    regex = _any(variants)
    template = spec["with"].replace("\\", "\\\\")
    if after := _words(spec, "followed_by"):
        regex += r"(?=[\s-]+" + _any(after) + ")"
    if before := _words(spec, "preceded_by"):
        regex = r"(?P<before>" + _any(before) + r"[\s-]+)" + regex
        template = r"\g<before>" + template
    return regex, template


def load_rules(path):
    """Rules from `path`; [] if it doesn't exist. ValueError if it is invalid."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path}: {exc}") from None
    rules = []
    for n, entry in enumerate(data.get("rule", []), 1):
        where = f"{path}: rule {n}"
        examples = []
        for ex in entry.get("examples", []):
            if not (isinstance(ex, list) and len(ex) == 2 and all(isinstance(s, str) for s in ex)):
                raise ValueError(f"{where}: each example must be [input, expected]")
            examples.append(tuple(ex))
        for keep in entry.get("keep", []):
            if not isinstance(keep, str):
                raise ValueError(f"{where}: 'keep' entries must be strings")
            examples.append((keep, keep))
        try:
            rules.append(Rule.from_spec(entry, examples))
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from None
    return rules


def apply(rules, text):
    for rule in rules:
        text = rule.pattern.sub(rule.replace, text)
    return text


def check(rules):
    """(input, expected, actual) for every example the full rule set gets wrong."""
    failures = []
    for rule in rules:
        for src, want in rule.examples:
            got = apply(rules, src)
            if got != want:
                failures.append((src, want, got))
    return failures


def affected(rules, history):
    """(before, after) for each past dictation the rules would change."""
    return [(h, out) for h in history if (out := apply(rules, h)) != h]


class Corrector:
    """Applies the rules file, reloading it when it changes. A broken file is
    reported and the previous rules stay in force, so dictation keeps working."""

    def __init__(self, path=None):
        self.path = Path(path) if path else default_path()
        self._rules = []
        self._stamp = None

    def _refresh(self):
        try:
            st = self.path.stat()
            stamp = (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            stamp = None
        if stamp == self._stamp:
            return
        self._stamp = stamp
        try:
            self._rules = load_rules(self.path)
            print(f"[corrections] loaded {len(self._rules)} rules from {self.path}")
        except ValueError as exc:
            print(f"[corrections] keeping previous rules, file is invalid: {exc}", file=sys.stderr)

    def correct(self, text):
        self._refresh()
        return apply(self._rules, text)


# --- Deriving rules from corrected sentences ---------------------------

_TOKEN = re.compile(r"\w+|[^\w\s]")


@dataclass(frozen=True)
class Change:
    """One suggested rule and its example: the sentence fragment around the
    change, cut short of any other change so the examples of separately saved
    rules don't depend on each other."""
    spec: dict
    before: str
    after: str


def _token_spans(text):
    return [(m.start(), m.end()) for m in _TOKEN.finditer(text)]


def suggest_changes(text, fixed):
    """One Change per place where `fixed` differs from `text`, in order.
    Unchanged words between two edits separate them ("komm mitten und den"
    -> "committen und in" gives two rules, not one)."""
    a, b = _token_spans(text), _token_spans(fixed)
    if not a:
        return []
    opcodes = difflib.SequenceMatcher(
        None, [text[s:e] for s, e in a], [fixed[s:e] for s, e in b], autojunk=False,
    ).get_opcodes()
    groups, current = [], None
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            current = None
        elif current is None:
            current = [i1, i2, j1, j2]
            groups.append(current)
        else:
            current[1], current[3] = i2, j2

    spans = []
    for i1, i2, j1, j2 in groups:
        if i1 == i2:  # pure insertion: anchor on a neighbouring word
            if i1 > 0:
                i1, j1 = i1 - 1, j1 - 1
            else:
                i2, j2 = i2 + 1, j2 + 1
        elif j1 == j2:  # pure deletion: likewise, so no double space is left
            if i2 < len(a):
                i2, j2 = i2 + 1, j2 + 1
            else:
                i1, j1 = i1 - 1, j1 - 1
        # Grow to whole space-separated words ("Link" in "Link.ing" ->
        # "Link.ing"), then drop punctuation at the edges ("Link." -> "Link").
        first, last = i1, i2
        while i1 > 0 and j1 > 0 and a[i1 - 1][1] == a[i1][0] and b[j1 - 1][1] == b[j1][0]:
            i1, j1 = i1 - 1, j1 - 1
        while (i2 < len(a) and j2 < len(b)
               and a[i2 - 1][1] == a[i2][0] and b[j2 - 1][1] == b[j2][0]):
            i2, j2 = i2 + 1, j2 + 1
        punct = lambda span: not re.match(r"\w", text[span[0]:span[1]])
        while i1 < first and punct(a[i1]):
            i1, j1 = i1 + 1, j1 + 1
        while i2 > last and punct(a[i2 - 1]):
            i2, j2 = i2 - 1, j2 - 1
        if spans and i1 < spans[-1][1]:  # grew into the previous change: merge
            p = spans.pop()
            i1, j1 = p[0], p[2]
        spans.append((i1, i2, j1, j2))

    def start(tokens, index):  # where the text after the previous change begins
        return tokens[index][0] if index < len(tokens) else None

    def end(tokens, index):  # where the text before the next change ends
        return tokens[index - 1][1] if index > 0 else 0

    changes = []
    for k, (i1, i2, j1, j2) in enumerate(spans):
        prev = spans[k - 1] if k > 0 else None
        nxt = spans[k + 1] if k + 1 < len(spans) else None
        before = text[start(a, prev[1]) if prev else 0:end(a, nxt[0]) if nxt else len(text)]
        after = fixed[(start(b, prev[3]) or len(fixed)) if prev else 0:
                      end(b, nxt[2]) if nxt else len(fixed)]
        old = text[a[i1][0]:a[i2 - 1][1]]
        new = fixed[b[j1][0]:b[j2 - 1][1]] if j2 > j1 else ""
        changes.append(Change({"replace": [old], "with": new}, before.strip(), after.strip()))
    return changes


def suggest_rule(raw, fixed):
    """Rule table for the first change from `raw` to `fixed`, or None."""
    changes = suggest_changes(raw, fixed)
    return changes[0].spec if changes else None


def describe(spec):
    """One line for humans, e.g. 'Link, linken -> Lynq (followed by API)'."""
    what = f"regex {spec['regex']}" if "regex" in spec else ", ".join(spec.get("replace", []))
    text = f"{what} -> {spec.get('with', '')}"
    if spec.get("followed_by"):
        text += f" (followed by {', '.join(spec['followed_by'])})"
    if spec.get("preceded_by"):
        text += f" (preceded by {', '.join(spec['preceded_by'])})"
    return text


@dataclass
class Evaluation:
    error: str | None = None          # invalid rule
    result: str = ""                  # what all rules make of the raw text
    produces_fix: bool = False        # result == the corrected sentence
    side_effects: list = field(default_factory=list)     # (before, after) in history
    broken_examples: list = field(default_factory=list)  # (input, expected, actual)


def evaluate_new_rule(existing, raw, fixed, spec, history):
    """How a candidate rule, appended after `existing`, would behave."""
    try:
        rule = Rule.from_spec(spec)
    except ValueError as exc:
        return Evaluation(error=str(exc))
    rules = [*existing, rule]
    result = apply(rules, raw)
    side_effects = [(h, out) for h in history
                    if h != raw and (out := apply(rules, h)) != apply(existing, h)]
    broken = check(rules)  # the new rule has no examples, so: existing ones it breaks
    return Evaluation(None, result, result == fixed, side_effects, broken)


def _toml_str(s):
    if "'" not in s and "\n" not in s:
        return f"'{s}'"
    return json.dumps(s, ensure_ascii=False)


def _toml_list(items):
    return "[" + ", ".join(_toml_str(i) for i in items) + "]"


def append_rule(path, spec, examples):
    """Append rule table `spec` with `examples` [(input, expected), ...] to `path`."""
    Rule.from_spec(spec)  # refuse to write a rule that wouldn't load
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["", "[[rule]]"]
    for key in SPEC_KEYS:
        value = spec.get(key)
        if isinstance(value, list) and value:
            lines.append(f"{key} = {_toml_list(value)}")
        elif isinstance(value, str) and (value or key == "with"):
            lines.append(f"{key} = {_toml_str(value)}")
    lines.append("examples = [" + ", ".join(_toml_list(ex) for ex in examples) + "]")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --- Past dictations ---------------------------------------------------

def dictations_from_journal(journal_text):
    """Raw recognized texts, oldest first, without duplicates. Since rules
    exist, each dictation logs 'Recognized:' (raw) then 'Injecting:'
    (corrected); older logs only have 'Injecting:', which was raw then."""
    seen, out, after_recognized = set(), [], False
    for line in journal_text.splitlines():
        m = re.search(r"\b(Recognized|Injecting): (.*)$", line)
        if not m:
            continue
        kind, text = m.groups()
        if kind == "Injecting" and after_recognized:
            after_recognized = False
            continue
        after_recognized = kind == "Recognized"
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def load_history():
    try:
        journal = subprocess.run(
            ["journalctl", "--user", "-o", "cat", "--no-pager", "--grep", "^(Recognized|Injecting): "],
            capture_output=True, text=True, check=False,
        ).stdout
    except FileNotFoundError:
        return []
    return dictations_from_journal(journal)


# --- Command line ------------------------------------------------------

def _show_affected(changes, limit=15):
    for before, after in changes[:limit]:
        print(f"  - {before}\n  + {after}")
    if len(changes) > limit:
        print(f"  ... and {len(changes) - limit} more")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=default_path())
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p_try = sub.add_parser("try")
    p_try.add_argument("text")
    p_hist = sub.add_parser("history")
    p_hist.add_argument("regex", nargs="?", default="")
    p_add = sub.add_parser("add")
    p_add.add_argument("wrong")
    p_add.add_argument("right")
    args = parser.parse_args(argv)

    try:
        rules = load_rules(args.file)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1

    if args.cmd == "check":
        failures = check(rules)
        for src, want, got in failures:
            print(f"FAIL  {src}\n  want {want}\n  got  {got}")
        n = sum(len(r.examples) for r in rules)
        print(f"{len(rules)} rules, {n} examples, {len(failures)} failing")
        return 1 if failures else 0

    if args.cmd == "try":
        print(apply(rules, args.text))
        return 0

    if args.cmd == "history":
        pattern = re.compile(args.regex, re.IGNORECASE)
        for text in load_history():
            if pattern.search(text):
                out = apply(rules, text)
                print(f"  {text}" if out == text else f"- {text}\n+ {out}")
        return 0

    # add: one rule per separate change, each with its own fragment as example
    changes = suggest_changes(apply(rules, args.wrong), args.right)
    if not changes:
        print("nothing changed")
        return 1
    history = load_history()
    for change in changes:
        append_rule(args.file, change.spec, [(change.before, change.after)])
        print(f"added: {describe(change.spec)}")
        side = affected([Rule.from_spec(change.spec)], history)
        if side:
            print(f"  this rule alone would change {len(side)} past dictations:")
            _show_affected(side)
    failures = check(load_rules(args.file))
    for src, want, got in failures:
        print(f"FAIL  {src}\n  want {want}\n  got  {got}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
