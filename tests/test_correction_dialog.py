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
    assert "+ Gib mir einen Lynq." in dlg.preview.toPlainText()

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
