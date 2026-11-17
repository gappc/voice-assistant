import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QDialogButtonBox

import text_corrections as tc
from correction_dialog import CorrectionDialog


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def save_button(dlg):
    return dlg.buttons.button(QDialogButtonBox.Save)


def test_editing_should_be_suggests_a_plain_word_rule(app, tmp_path):
    path = tmp_path / "c.toml"
    history = ["Gib mir einen Link.", "Für das Projekt Link."]
    dlg = CorrectionDialog("Für das Projekt Link.", "Für das Projekt Link.", path, history)
    assert not save_button(dlg).isEnabled()

    dlg.fixed.setText("Für das Projekt Lynq.")
    assert (dlg.wrong.text(), dlg.right.text()) == ("Link", "Lynq")
    assert "1 of 2" in dlg.status.text()
    assert dlg.shown_side_effects == [("Gib mir einen Link.", "Gib mir einen Lynq.")]

    dlg.preceded_by.setText("Projekt")  # narrow the rule
    assert "none of 2" in dlg.status.text()
    assert dlg.fixed.text() == "Für das Projekt Lynq."
    save_button(dlg).click()

    text = path.read_text()
    assert "replace = ['Link']" in text and "preceded_by = ['Projekt']" in text
    rules = tc.load_rules(path)
    assert tc.apply(rules, "Gib mir einen Link. Projekt Link") == "Gib mir einen Link. Projekt Lynq"
    assert tc.check(rules) == []


def test_filling_in_the_rule_updates_should_be_and_allows_saving(app, tmp_path):
    path = tmp_path / "c.toml"
    raw = "Wurde die Read Me bereits abgedatet?"
    dlg = CorrectionDialog(raw, raw, path, ["Ich habe den Server abgedatet."])

    dlg.wrong.setText("abgedatet, abgedated")
    dlg.right.setText("upgedatet")
    assert dlg.fixed.text() == "Wurde die Read Me bereits upgedatet?"
    assert save_button(dlg).isEnabled()
    save_button(dlg).click()
    rules = tc.load_rules(path)
    assert tc.apply(rules, "abgedated") == "upgedatet"
    assert tc.check(rules) == []


def test_rule_that_changes_nothing_cannot_be_saved(app, tmp_path):
    dlg = CorrectionDialog("a b", "a b", tmp_path / "c.toml", [])
    dlg.wrong.setText("zzz")
    dlg.right.setText("y")
    assert "changes nothing" in dlg.status.text()
    assert not save_button(dlg).isEnabled()


def test_rule_that_breaks_an_existing_example_cannot_be_saved(app, tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[[rule]]\nreplace = ['Lync']\nwith = 'Lynq'\nkeep = ['Gib mir einen Link.']\n")
    raw = "Die Link Doku"
    dlg = CorrectionDialog(raw, raw, path, [])
    dlg.fixed.setText("Die Lynq Doku")
    assert "Breaks 1" in dlg.status.text()
    assert not save_button(dlg).isEnabled()
    dlg.followed_by.setText("Doku")
    assert save_button(dlg).isEnabled()


def test_check_button_verifies_the_rules_file(app, tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[[rule]]\nreplace = ['Lync']\nwith = 'Lynq'\n"
                    "examples = [['Lync', 'Lynq']]\nkeep = ['Link']\n")
    dlg = CorrectionDialog("a", "a", path, [])
    dlg.check_button.click()
    assert "1 rule" in dlg.status.text() and "all 2 examples pass" in dlg.status.text()

    path.write_text("[[rule]]\nreplace = ['Lync']\nwith = 'Lynq'\nexamples = [['Lync', 'falsch']]\n")
    dlg.check_button.click()
    assert "1 of 1 examples fail" in dlg.status.text()
    assert "falsch" in dlg.preview.toPlainText()

    path.write_text("[[rule]]\nwith = 'x'\n")
    dlg.check_button.click()
    assert "invalid" in dlg.status.text()


def test_separate_changes_are_saved_one_rule_at_a_time(app, tmp_path):
    path = tmp_path / "c.toml"
    raw = "Ja, bitte komm mitten und den Main mergen und pushen."
    history = ["Er soll den Main mergen.", "Wir komm mitten rein."]
    dlg = CorrectionDialog(raw, raw, path, history)
    dlg.fixed.setText("Ja, bitte committen und in Main mergen und pushen.")

    assert (dlg.wrong.text(), dlg.right.text()) == ("komm mitten", "committen")
    assert "Rule 1 of 2" in dlg.rule_header.text()
    assert "Next: “den” → “in”" in dlg.rule_header.text()
    assert save_button(dlg).text() == "Save && Next"
    save_button(dlg).click()
    assert "Rule 2 of 2" in dlg.rule_header.text() and "Next:" not in dlg.rule_header.text()
    assert save_button(dlg).text() == "Save"

    # Dialog stays open with the next change; its side effects are visible.
    assert dlg.result() == 0
    assert (dlg.wrong.text(), dlg.right.text()) == ("den", "in")
    assert "✓ Saved rule 1: komm mitten -> committen" in dlg.status.text()
    assert ("Er soll den Main mergen.", "Er soll in Main mergen.") in dlg.shown_side_effects
    dlg.reject()  # "den" -> "in" was a one-off: no rule

    rules = tc.load_rules(path)
    assert len(rules) == 1 and tc.check(rules) == []
    assert rules[0].examples == (("Ja, bitte komm mitten und", "Ja, bitte committen und"),)
    assert dlg.saved == [{"replace": ["komm mitten"], "with": "committen"}]


def test_saving_the_last_change_closes_the_dialog(app, tmp_path):
    path = tmp_path / "c.toml"
    raw = "bitte komm mitten und den Main mergen"
    dlg = CorrectionDialog(raw, raw, path, [])
    dlg.fixed.setText("bitte committen und in Main mergen")
    save_button(dlg).click()
    dlg.followed_by.setText("Main")
    save_button(dlg).click()
    assert dlg.result() == 1
    rules = tc.load_rules(path)
    assert len(rules) == 2 and tc.check(rules) == []
    assert tc.apply(rules, raw) == "bitte committen und in Main mergen"
    assert tc.apply(rules, "den Hund") == "den Hund"


def test_tuning_a_suggested_rule_keeps_the_target_sentence(app, tmp_path):
    raw = "die Link Doku"
    dlg = CorrectionDialog(raw, raw, tmp_path / "c.toml", [])
    dlg.fixed.setText("die Lynq Doku")
    dlg.wrong.setText("Link, Lync")
    assert dlg.fixed.text() == "die Lynq Doku"


def test_single_change_shows_no_step_counter(app, tmp_path):
    dlg = CorrectionDialog("die Link Doku", "die Link Doku", tmp_path / "c.toml", [])
    dlg.fixed.setText("die Lynq Doku")
    assert "of" not in dlg.rule_header.text()
    assert save_button(dlg).text() == "Save"


def test_variant_of_an_existing_rule_is_offered_and_merged(app, tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("# commit\n[[rule]]\nreplace = ['komm mitten']\nwith = 'committen'\n"
                    "examples = [['bitte komm mitten', 'bitte committen']]\n")
    raw = "Ja, Kommitten und weiter."
    dlg = CorrectionDialog(raw, raw, path, [])
    dlg.fixed.setText("Ja, committen und weiter.")
    assert dlg.merge_box.isVisibleTo(dlg) and dlg.merge_box.isChecked()
    assert "komm mitten -> committen" in dlg.merge_box.text()

    dlg.merge_box.setChecked(False)          # a manual uncheck survives further edits
    dlg.followed_by.setText("und")
    dlg.followed_by.setText("")
    assert not dlg.merge_box.isChecked()
    dlg.merge_box.setChecked(True)
    save_button(dlg).click()

    rules = tc.load_rules(path)
    assert len(rules) == 1
    assert rules[0].spec["replace"] == ["komm mitten", "Kommitten"]
    assert len(rules[0].examples) == 2 and tc.check(rules) == []
    assert path.read_text().startswith("# commit\n")


def test_unchecked_merge_appends_a_new_rule(app, tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[[rule]]\nreplace = ['komm mitten']\nwith = 'committen'\n")
    dlg = CorrectionDialog("Kommitten", "Kommitten", path, [])
    dlg.fixed.setText("committen")
    dlg.merge_box.setChecked(False)
    save_button(dlg).click()
    assert len(tc.load_rules(path)) == 2


def test_ordinary_word_without_context_is_warned_about(app, tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "dictionary_words", lambda: frozenset({"link"}))
    dlg = CorrectionDialog("die Link Doku", "die Link Doku", tmp_path / "c.toml", [])
    dlg.fixed.setText("die Lynq Doku")
    assert "“Link” is an ordinary word" in dlg.status.text()
    dlg.followed_by.setText("Doku")
    assert "ordinary word" not in dlg.status.text()
