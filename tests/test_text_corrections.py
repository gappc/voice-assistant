import os

import pytest

import text_corrections as tc


def write(path, content):
    path.write_text(content, encoding="utf-8")
    return path


RULES = r"""
[[rule]]
match = '\b(lync|lynch)\b'
replace = 'Lynq'
examples = [
  ["Die alte Lync API.", "Die alte Lynq API."],
]
keep = ["Gib mir einen Link."]

[[rule]]
match = '\blink(?= (api|projekt)\b)'
replace = 'Lynq'
examples = [["Die alte Link API.", "Die alte Lynq API."]]
"""


def test_apply_runs_rules_in_order_case_insensitively(tmp_path):
    rules = tc.load_rules(write(tmp_path / "c.toml", RULES))
    assert tc.apply(rules, "LYNCH und link Projekt") == "Lynq und Lynq Projekt"


def test_apply_leaves_unmatched_text_alone(tmp_path):
    rules = tc.load_rules(write(tmp_path / "c.toml", RULES))
    assert tc.apply(rules, "Gib mir einen Link.") == "Gib mir einen Link."


def test_replacement_may_use_groups(tmp_path):
    rules = tc.load_rules(write(tmp_path / "c.toml", r"""
[[rule]]
match = '\bgit (pool)\b'
replace = 'Git Pull'
[[rule]]
match = '\b(ge)?merch(t|en)?\b'
replace = '\1merg\2'
"""))
    assert tc.apply(rules, "gemercht nach git pool") == "gemergt nach Git Pull"


def test_missing_file_means_no_rules(tmp_path):
    assert tc.load_rules(tmp_path / "missing.toml") == []


@pytest.mark.parametrize("content", [
    "[[rule]]\nmatch = '('\nreplace = 'x'\n",          # bad regex
    "[[rule]]\nreplace = 'x'\n",                        # no match
    "[[rule]]\nmatch = 'a'\n",                          # no replace
    "[[rule]]\nmatch = 'a'\nreplace = 'b'\nexamples = [['only one']]\n",
    "this is = = not toml",
])
def test_invalid_files_raise_value_error(tmp_path, content):
    with pytest.raises(ValueError):
        tc.load_rules(write(tmp_path / "c.toml", content))


def test_check_reports_examples_that_fail_with_all_rules_applied(tmp_path):
    rules = tc.load_rules(write(tmp_path / "c.toml", RULES + r"""
[[rule]]
match = '\blink\b'
replace = 'Lynq'
"""))
    failures = tc.check(rules)
    # The greedy third rule breaks the first rule's "keep" sentence.
    assert failures == [("Gib mir einen Link.", "Gib mir einen Link.", "Gib mir einen Lynq.")]


def test_check_passes_when_all_examples_hold(tmp_path):
    assert tc.check(tc.load_rules(write(tmp_path / "c.toml", RULES))) == []


def test_corrector_reloads_when_the_file_changes(tmp_path):
    path = write(tmp_path / "c.toml", RULES)
    corrector = tc.Corrector(path)
    assert corrector.correct("Lync") == "Lynq"

    write(path, "[[rule]]\nmatch = 'lync'\nreplace = 'LYNQ'\n")
    os.utime(path, ns=(1, path.stat().st_mtime_ns + 10**9))
    assert corrector.correct("Lync") == "LYNQ"


def test_corrector_keeps_previous_rules_when_the_file_breaks(tmp_path, capsys):
    path = write(tmp_path / "c.toml", RULES)
    corrector = tc.Corrector(path)
    corrector.correct("x")

    write(path, "[[rule]]\nmatch = '('\nreplace = 'x'\n")
    os.utime(path, ns=(1, path.stat().st_mtime_ns + 10**9))
    assert corrector.correct("Lync") == "Lynq"
    assert "corrections" in capsys.readouterr().err


def test_corrector_without_file_returns_text_unchanged(tmp_path):
    assert tc.Corrector(tmp_path / "none.toml").correct("Lync") == "Lync"


def test_history_lines_prefer_raw_recognized_text():
    journal = "\n".join([
        "Nov 17 voice[1]: Injecting: alt vor dem Feature",
        "Nov 17 voice[1]: Recognized: Link API",
        "Nov 17 voice[1]: Injecting: Lynq API",
        "Nov 17 voice[1]: Recognized: Link API",
    ])
    assert tc.dictations_from_journal(journal) == ["alt vor dem Feature", "Link API"]


def test_cli_check_exit_code(tmp_path, capsys):
    good = write(tmp_path / "good.toml", RULES)
    assert tc.main(["--file", str(good), "check"]) == 0
    bad = write(tmp_path / "bad.toml", RULES.replace('"Die alte Lynq API."]]', '"falsch"]]'))
    assert tc.main(["--file", str(bad), "check"]) == 1
    assert "falsch" in capsys.readouterr().out


def test_cli_try(tmp_path, capsys):
    path = write(tmp_path / "c.toml", RULES)
    assert tc.main(["--file", str(path), "try", "Die alte Link API."]) == 0
    assert "Die alte Lynq API." in capsys.readouterr().out


@pytest.mark.parametrize("raw, fixed, match, replace", [
    ("Für das Projekt Link.", "Für das Projekt Lynq.", r"(?<!\w)Link(?!\w)", "Lynq"),
    ("Squash Merch für 307", "Squash Merge für 307", r"(?<!\w)Merch(?!\w)", "Merge"),
    ("einen Git Pool machen", "einen Git Pull machen", r"(?<!\w)Pool(?!\w)", "Pull"),
    # Two changed words in one span become one rule covering the span.
    ("Link Ing gebaut", "lynq.ing gebaut", r"(?<!\w)Link\s+Ing(?!\w)", "lynq.ing"),
])
def test_suggest_rule_from_a_corrected_sentence(raw, fixed, match, replace):
    assert tc.suggest_rule(raw, fixed) == (match, replace)
    assert tc.apply([tc.Rule.make(match, replace)], raw) == fixed


def test_suggest_rule_returns_none_when_nothing_changed():
    assert tc.suggest_rule("gleich", "gleich") is None


def test_affected_lists_history_that_a_rule_would_change():
    rule = tc.Rule.make(r"(?<!\w)Link(?!\w)", "Lynq")
    history = ["Gib mir einen Link.", "Kein Treffer.", "Link API"]
    assert tc.affected([rule], history) == [
        ("Gib mir einen Link.", "Gib mir einen Lynq."), ("Link API", "Lynq API")]


def test_append_rule_writes_a_loadable_rule_with_its_example(tmp_path):
    path = tmp_path / "sub" / "c.toml"
    tc.append_rule(path, r"(?<!\w)Link(?= API)", "Lynq", ("Die Link API", "Die Lynq API"))
    tc.append_rule(path, "it's", 'say "x"', ("it's", 'say "x"'))
    rules = tc.load_rules(path)
    assert [r.replace for r in rules] == ["Lynq", 'say "x"']
    assert tc.check(rules) == []


def test_cli_add_appends_and_checks(tmp_path, capsys):
    path = tmp_path / "c.toml"
    assert tc.main(["--file", str(path), "add", "Für das Projekt Link.", "Für das Projekt Lynq."]) == 0
    assert tc.apply(tc.load_rules(path), "Projekt Link") == "Projekt Lynq"


def test_evaluate_new_rule_runs_after_existing_rules_and_lists_side_effects():
    existing = [tc.Rule.make(r"\blync\b", "Lynq", [("Lync", "Lynq")])]
    history = ["Gib mir einen Link.", "Link API und Lync", "Für das Projekt Link."]
    ev = tc.evaluate_new_rule(existing, "Für das Projekt Link.", "Für das Projekt Lynq.",
                              r"(?<!\w)Link(?!\w)", "Lynq", history)
    assert ev.error is None and ev.produces_fix
    # The sentence being fixed is not a side effect; the other two are.
    assert ev.side_effects == [("Gib mir einen Link.", "Gib mir einen Lynq."),
                               ("Link API und Lync", "Lynq API und Lynq")]
    assert ev.broken_examples == []


def test_evaluate_new_rule_reports_bad_regex_and_broken_examples():
    existing = [tc.Rule.make(r"x", "x", [("Ein Link.", "Ein Link.")])]
    assert tc.evaluate_new_rule(existing, "a", "b", "(", "b", []).error
    ev = tc.evaluate_new_rule(existing, "Projekt Link", "Projekt Lynq", r"\bLink\b", "Lynq", [])
    assert ev.broken_examples == [("Ein Link.", "Ein Link.", "Ein Lynq.")]
    ev = tc.evaluate_new_rule(existing, "Projekt Link", "Projekt Lynq", r"\bFoo\b", "Lynq", [])
    assert not ev.produces_fix
