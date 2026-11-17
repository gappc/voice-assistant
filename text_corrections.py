"""Fix words the speech recognizer keeps getting wrong.

Rules live in a TOML file (default ~/.config/voice-assistant/corrections.toml):

    [[rule]]
    match = '(?<!\\w)Link(?= (API|Projekt|Webseite)\\b)'   # Python regex, case-insensitive
    replace = 'Lynq'                                      # re.sub template (\\1 etc. allowed)
    examples = [["Die alte Link API.", "Die alte Lynq API."]]   # input -> expected
    keep = ["Gib mir einen Link."]                          # must stay unchanged

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
    replace: str
    examples: tuple = field(default=())  # (input, expected) pairs

    @classmethod
    def make(cls, match, replace, examples=()):
        return cls(re.compile(match, re.IGNORECASE), replace, tuple(examples))


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
        if not isinstance(entry.get("match"), str) or not isinstance(entry.get("replace"), str):
            raise ValueError(f"{where}: needs string 'match' and 'replace'")
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
            rule = Rule.make(entry["match"], entry["replace"], examples)
            rule.pattern.sub(rule.replace, "")  # validates group references
        except re.error as exc:
            raise ValueError(f"{where}: {exc}") from None
        rules.append(rule)
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


def suggest_rule(raw, fixed):
    """(match, replace) turning the changed words of `raw` into `fixed`, or
    None if nothing changed. Changes spread over a sentence become one rule
    for the span from the first to the last change."""
    a = [(m.start(), m.end()) for m in _TOKEN.finditer(raw)]
    b = [(m.start(), m.end()) for m in _TOKEN.finditer(fixed)]
    ops = [op for op in difflib.SequenceMatcher(
        None, [raw[s:e] for s, e in a], [fixed[s:e] for s, e in b], autojunk=False,
    ).get_opcodes() if op[0] != "equal"]
    if not ops:
        return None
    i1, i2, j1, j2 = ops[0][1], ops[-1][2], ops[0][3], ops[-1][4]
    if i1 == i2:  # pure insertion: anchor on the word before it
        i1, j1 = max(i1 - 1, 0), max(j1 - 1, 0)
        i2, j2 = max(i2, i1 + 1), max(j2, j1 + 1)
    old_words = [raw[s:e] for s, e in a[i1:i2]]
    new_text = fixed[b[j1][0]:b[j2 - 1][1]] if j2 > j1 else ""
    match = r"\s+".join(re.escape(w) for w in old_words)
    if re.match(r"\w", old_words[0]):
        match = r"(?<!\w)" + match
    if re.search(r"\w$", old_words[-1]):
        match += r"(?!\w)"
    return match, new_text.replace("\\", "\\\\")


@dataclass
class Evaluation:
    error: str | None = None          # invalid regex/template
    result: str = ""                  # what all rules make of the raw text
    produces_fix: bool = False        # result == the corrected sentence
    side_effects: list = field(default_factory=list)     # (before, after) in history
    broken_examples: list = field(default_factory=list)  # (input, expected, actual)


def evaluate_new_rule(existing, raw, fixed, match, replace, history):
    """How a candidate rule, appended after `existing`, would behave."""
    try:
        rule = Rule.make(match, replace, [(raw, fixed)])
        rule.pattern.sub(rule.replace, "")
    except re.error as exc:
        return Evaluation(error=str(exc))
    rules = [*existing, rule]
    result = apply(rules, raw)
    side_effects = [(h, out) for h in history
                    if h != raw and (out := apply(rules, h)) != apply(existing, h)]
    broken = check([*existing, Rule.make(match, replace)])  # existing examples only
    return Evaluation(None, result, result == fixed, side_effects, broken)


def _toml_str(s):
    if "'" not in s and "\n" not in s:
        return f"'{s}'"
    return json.dumps(s, ensure_ascii=False)


def append_rule(path, match, replace, example):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    src, want = example
    block = (f"\n[[rule]]\nmatch = {_toml_str(match)}\nreplace = {_toml_str(replace)}\n"
             f"examples = [[{_toml_str(src)}, {_toml_str(want)}]]\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write(block)


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

    # add
    suggestion = suggest_rule(args.wrong, args.right)
    if suggestion is None:
        print("nothing changed")
        return 1
    match, replace = suggestion
    append_rule(args.file, match, replace, (args.wrong, args.right))
    print(f"added: {match}  ->  {replace}")
    changes = affected([Rule.make(match, replace)], load_history())
    if changes:
        print(f"this rule alone would change {len(changes)} past dictations:")
        _show_affected(changes)
    failures = check(load_rules(args.file))
    for src, want, got in failures:
        print(f"FAIL  {src}\n  want {want}\n  got  {got}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
