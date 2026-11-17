import os

import pytest

import text_corrections as tc


def write(path, content):
    path.write_text(content, encoding="utf-8")
    return path


def rule(**spec):
    return tc.Rule.from_spec(spec)


RULES = """
[[rule]]
replace = ["Lync", "Lynch"]
with = "Lynq"
examples = [
  ["Die alte Lync API.", "Die alte Lynq API."],
]
keep = ["Gib mir einen Link."]

[[rule]]
replace = ["Link"]
with = "Lynq"
followed_by = ["API", "Projekt"]
examples = [["Die alte Link API.", "Die alte Lynq API."]]
"""


# --- Simple rules -------------------------------------------------------

def test_replace_matches_whole_words_in_any_case():
    r = [rule(**{"replace": ["Merch"], "with": "Merge"})]
    assert tc.apply(r, "Squash merch, MERCH! Merchandise") == "Squash Merge, Merge! Merchandise"


def test_replace_accepts_several_variants_and_multiword_phrases():
    r = [rule(**{"replace": ["Link Ing", "Link.ing", "Link Punkt ing"], "with": "lynq.ing"})]
    assert tc.apply(r, "Link  Ing, Link.ing und link punkt ING") == "lynq.ing, lynq.ing und lynq.ing"


def test_followed_by_limits_the_rule_to_words_before_given_words():
    r = [rule(**{"replace": ["Link", "linken"], "with": "Lynq",
                 "followed_by": ["API", "Domain"]})]
    assert tc.apply(r, "Die Link API, die linken Domain, ein Link, Link APIs") == \
        "Die Lynq API, die Lynq Domain, ein Link, Link APIs"
    assert tc.apply(r, "Link-API") == "Lynq-API"


def test_preceded_by_limits_the_rule_to_words_after_given_words():
    r = [rule(**{"replace": ["Link"], "with": "Lynq", "preceded_by": ["Projekt", "Project"]})]
    assert tc.apply(r, "Projekt  Link, project link, Projektlink, ein Link") == \
        "Projekt  Lynq, project Lynq, Projektlink, ein Link"


def test_with_is_taken_literally():
    r = [rule(**{"replace": ["x"], "with": r"a\1b"})]
    assert tc.apply(r, "x") == r"a\1b"


def test_regex_rules_remain_available_for_special_cases():
    r = [rule(**{"regex": r"\b(ge)?merch(t|en)?\b", "with": r"\1merg\2"})]
    assert tc.apply(r, "gemercht und merchen") == "gemergt und mergen"


# --- Loading ------------------------------------------------------------

def test_load_and_apply_rules_in_order(tmp_path):
    rules = tc.load_rules(write(tmp_path / "c.toml", RULES))
    assert tc.apply(rules, "LYNCH und link Projekt") == "Lynq und Lynq Projekt"
    assert tc.apply(rules, "Gib mir einen Link.") == "Gib mir einen Link."


def test_missing_file_means_no_rules(tmp_path):
    assert tc.load_rules(tmp_path / "missing.toml") == []


@pytest.mark.parametrize("content", [
    "[[rule]]\nregex = '('\nwith = 'x'\n",                         # bad regex
    "[[rule]]\nwith = 'x'\n",                                      # nothing to replace
    "[[rule]]\nreplace = ['a']\n",                                 # no with
    "[[rule]]\nreplace = []\nwith = 'x'\n",                        # empty variants
    "[[rule]]\nreplace = ['a']\nregex = 'a'\nwith = 'x'\n",        # both
    "[[rule]]\nreplace = ['a']\nwith = 'b'\nfollowed_by = 3\n",
    "[[rule]]\nreplace = ['a']\nwith = 'b'\nexamples = [['only one']]\n",
    "this is = = not toml",
])
def test_invalid_files_raise_value_error(tmp_path, content):
    with pytest.raises(ValueError):
        tc.load_rules(write(tmp_path / "c.toml", content))


# --- Checking -------------------------------------------------------------

def test_check_reports_examples_that_fail_with_all_rules_applied(tmp_path):
    rules = tc.load_rules(write(tmp_path / "c.toml", RULES + """
[[rule]]
replace = ["Link"]
with = "Lynq"
"""))
    # The greedy third rule breaks the first rule's "keep" sentence.
    assert tc.check(rules) == [("Gib mir einen Link.", "Gib mir einen Link.", "Gib mir einen Lynq.")]


def test_check_passes_when_all_examples_hold(tmp_path):
    assert tc.check(tc.load_rules(write(tmp_path / "c.toml", RULES))) == []


# --- Corrector --------------------------------------------------------------

def test_corrector_reloads_when_the_file_changes(tmp_path):
    path = write(tmp_path / "c.toml", RULES)
    corrector = tc.Corrector(path)
    assert corrector.correct("Lync") == "Lynq"

    write(path, "[[rule]]\nreplace = ['lync']\nwith = 'LYNQ'\n")
    os.utime(path, ns=(1, path.stat().st_mtime_ns + 10**9))
    assert corrector.correct("Lync") == "LYNQ"


def test_corrector_keeps_previous_rules_when_the_file_breaks(tmp_path, capsys):
    path = write(tmp_path / "c.toml", RULES)
    corrector = tc.Corrector(path)
    corrector.correct("x")

    write(path, "[[rule]]\nregex = '('\nwith = 'x'\n")
    os.utime(path, ns=(1, path.stat().st_mtime_ns + 10**9))
    assert corrector.correct("Lync") == "Lynq"
    assert "corrections" in capsys.readouterr().err


def test_corrector_without_file_returns_text_unchanged(tmp_path):
    assert tc.Corrector(tmp_path / "none.toml").correct("Lync") == "Lync"


# --- History ----------------------------------------------------------------

def test_history_lines_prefer_raw_recognized_text():
    journal = "\n".join([
        "Nov 17 voice[1]: Injecting: alt vor dem Feature",
        "Nov 17 voice[1]: Recognized: Link API",
        "Nov 17 voice[1]: Injecting: Lynq API",
        "Nov 17 voice[1]: Recognized: Link API",
    ])
    assert tc.dictations_from_journal(journal) == ["alt vor dem Feature", "Link API"]


# --- Deriving rules ---------------------------------------------------------

@pytest.mark.parametrize("raw, fixed, old, new", [
    ("Für das Projekt Link.", "Für das Projekt Lynq.", "Link", "Lynq"),
    ("Squash Merch für 307", "Squash Merge für 307", "Merch", "Merge"),
    # Two changed words in one span become one rule covering the span.
    ("Link Ing gebaut", "lynq.ing gebaut", "Link Ing", "lynq.ing"),
    ("die Link.ing Seite", "die lynq.ing Seite", "Link.ing", "lynq.ing"),
])
def test_suggest_rule_from_a_corrected_sentence(raw, fixed, old, new):
    spec = tc.suggest_rule(raw, fixed)
    assert spec == {"replace": [old], "with": new}
    assert tc.apply([tc.Rule.from_spec(spec)], raw) == fixed


def test_suggest_rule_returns_none_when_nothing_changed():
    assert tc.suggest_rule("gleich", "gleich") is None


def test_describe():
    assert tc.describe({"replace": ["Link", "linken"], "with": "Lynq",
                        "followed_by": ["API"]}) == "Link, linken -> Lynq (followed by API)"


def test_affected_lists_history_that_a_rule_would_change():
    r = rule(**{"replace": ["Link"], "with": "Lynq"})
    history = ["Gib mir einen Link.", "Kein Treffer.", "Link API"]
    assert tc.affected([r], history) == [
        ("Gib mir einen Link.", "Gib mir einen Lynq."), ("Link API", "Lynq API")]


def test_append_rule_writes_a_loadable_rule_with_its_example(tmp_path):
    path = tmp_path / "sub" / "c.toml"
    tc.append_rule(path, {"replace": ["Link"], "with": "Lynq", "followed_by": ["API"]},
                   [("Die Link API", "Die Lynq API"), ("Link API", "Lynq API")])
    tc.append_rule(path, {"replace": ["it's"], "with": 'say "x"'}, [("it's", 'say "x"')])
    text = path.read_text()
    assert "replace = ['Link']" in text and "followed_by = ['API']" in text
    rules = tc.load_rules(path)
    assert len(rules) == 2 and len(rules[0].examples) == 2
    assert tc.check(rules) == []


def test_append_rule_refuses_an_invalid_rule(tmp_path):
    with pytest.raises(ValueError):
        tc.append_rule(tmp_path / "c.toml", {"replace": [], "with": "x"}, [("a", "b")])
    assert not (tmp_path / "c.toml").exists()


def test_evaluate_new_rule_runs_after_existing_rules_and_lists_side_effects():
    existing = [tc.Rule.from_spec({"replace": ["lync"], "with": "Lynq"}, [("Lync", "Lynq")])]
    history = ["Gib mir einen Link.", "Link API und Lync", "Für das Projekt Link."]
    ev = tc.evaluate_new_rule(existing, "Für das Projekt Link.", "Für das Projekt Lynq.",
                              {"replace": ["Link"], "with": "Lynq"}, history)
    assert ev.error is None and ev.produces_fix
    # The sentence being fixed is not a side effect; the other two are.
    assert ev.side_effects == [("Gib mir einen Link.", "Gib mir einen Lynq."),
                               ("Link API und Lync", "Lynq API und Lynq")]
    assert ev.broken_examples == []


def test_evaluate_new_rule_reports_invalid_rules_and_broken_examples():
    existing = [tc.Rule.from_spec({"replace": ["x"], "with": "x"}, [("Ein Link.", "Ein Link.")])]
    assert tc.evaluate_new_rule(existing, "a", "b", {"replace": [], "with": "b"}, []).error
    ev = tc.evaluate_new_rule(existing, "Projekt Link", "Projekt Lynq",
                              {"replace": ["Link"], "with": "Lynq"}, [])
    assert ev.broken_examples == [("Ein Link.", "Ein Link.", "Ein Lynq.")]
    ev = tc.evaluate_new_rule(existing, "Projekt Link", "Projekt Lynq",
                              {"replace": ["Foo"], "with": "Lynq"}, [])
    assert not ev.produces_fix


# --- Command line -----------------------------------------------------------

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


def test_cli_add_appends_and_checks(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tc, "load_history", lambda: ["Ein Link hier."])
    path = tmp_path / "c.toml"
    assert tc.main(["--file", str(path), "add", "Für das Projekt Link.", "Für das Projekt Lynq."]) == 0
    assert tc.apply(tc.load_rules(path), "Projekt Link") == "Projekt Lynq"
    out = capsys.readouterr().out
    assert "added: Link -> Lynq" in out and "Ein Lynq hier." in out


@pytest.mark.parametrize("raw, fixed, old, new", [
    ("Am Ende ein Link.", "Am Ende ein Lynq.", "Link", "Lynq"),
    ("(Link) dort", "(Lynq) dort", "Link", "Lynq"),
    ("Siehe Link.ing.", "Siehe lynq.ing.", "Link.ing", "lynq.ing"),
])
def test_suggest_rule_grows_to_whole_words_but_not_edge_punctuation(raw, fixed, old, new):
    assert tc.suggest_rule(raw, fixed) == {"replace": [old], "with": new}


# --- Separate changes -------------------------------------------------------

def test_separate_changes_become_separate_rules_with_independent_examples():
    raw = "Ja, bitte komm mitten und den Main mergen und pushen."
    fixed = "Ja, bitte committen und in Main mergen und pushen."
    first, second = tc.suggest_changes(raw, fixed)
    assert first == tc.Change({"replace": ["komm mitten"], "with": "committen"},
                              "Ja, bitte komm mitten und", "Ja, bitte committen und")
    assert second == tc.Change({"replace": ["den"], "with": "in"},
                               "und den Main mergen und pushen.", "und in Main mergen und pushen.")
    # Either rule's example holds whether or not the other rule exists.
    r1, r2 = (tc.Rule.from_spec(c.spec, [(c.before, c.after)]) for c in (first, second))
    assert tc.check([r1]) == [] and tc.check([r2]) == [] and tc.check([r1, r2]) == []
    assert tc.apply([r1, r2], raw) == fixed


@pytest.mark.parametrize("raw, fixed", [
    ("x a b", "a b"), ("a b x", "a b"), ("a x b", "a b"),       # deletions
    ("a b", "a b c"), ("b c", "a b c"),                          # insertions
])
def test_deletions_and_insertions_anchor_on_a_neighbouring_word(raw, fixed):
    [change] = tc.suggest_changes(raw, fixed)
    rule = tc.Rule.from_spec(change.spec)
    assert tc.apply([rule], raw) == fixed
    assert tc.apply([rule], change.before) == change.after


def test_cli_add_creates_one_rule_per_separate_change(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tc, "load_history", lambda: [])
    path = tmp_path / "c.toml"
    assert tc.main(["--file", str(path), "add", "bitte komm mitten und den Main mergen",
                    "bitte committen und in Main mergen"]) == 0
    rules = tc.load_rules(path)
    assert len(rules) == 2 and tc.check(rules) == []
    out = capsys.readouterr().out
    assert "komm mitten -> committen" in out and "den -> in" in out
