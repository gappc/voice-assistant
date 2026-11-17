import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QDialogButtonBox

import text_corrections as tc
from correction_dialog import CorrectionDialog


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_dialog_suggests_previews_and_saves_a_rule(app, tmp_path):
    path = tmp_path / "c.toml"
    history = ["Gib mir einen Link.", "Für das Projekt Link."]
    dlg = CorrectionDialog("Für das Projekt Link.", "Für das Projekt Link.", path, history)
    save = dlg.buttons.button(QDialogButtonBox.Save)
    assert not save.isEnabled()

    dlg.fixed.setText("Für das Projekt Lynq.")
    assert dlg.match.text() == r"(?<!\w)Link(?!\w)" and dlg.replace.text() == "Lynq"
    assert "1 of 2" in dlg.status.text()
    assert "+ Gib mir einen Lynq." in dlg.preview.toPlainText()

    dlg.match.setText(r"(?<=Projekt )Link(?!\w)")  # narrowed by hand
    assert "none of 2" in dlg.status.text()
    assert save.isEnabled()
    save.click()

    rules = tc.load_rules(path)
    assert tc.apply(rules, "Gib mir einen Link. Projekt Link") == "Gib mir einen Link. Projekt Lynq"
    assert tc.check(rules) == []


def test_dialog_refuses_invalid_regex(app, tmp_path):
    dlg = CorrectionDialog("a b", "a b", tmp_path / "c.toml", [])
    dlg.fixed.setText("a c")
    dlg.match.setText("(")
    assert "Invalid regex" in dlg.status.text()
    assert not dlg.buttons.button(QDialogButtonBox.Save).isEnabled()


def test_editing_the_rule_directly_updates_should_be_and_allows_saving(app, tmp_path):
    path = tmp_path / "c.toml"
    raw = "Wurde die Read Me bereits abgedatet?"
    dlg = CorrectionDialog(raw, raw, path, ["Ich habe den Server abgedatet."])
    save = dlg.buttons.button(QDialogButtonBox.Save)

    dlg.match.setText("abgedatet")
    dlg.replace.setText("upgedatet")
    assert dlg.fixed.text() == "Wurde die Read Me bereits upgedatet?"
    assert save.isEnabled()
    save.click()
    assert tc.apply(tc.load_rules(path), raw) == "Wurde die Read Me bereits upgedatet?"
    assert tc.check(tc.load_rules(path)) == []


def test_rule_that_changes_nothing_cannot_be_saved(app, tmp_path):
    dlg = CorrectionDialog("a b", "a b", tmp_path / "c.toml", [])
    dlg.match.setText("zzz")
    dlg.replace.setText("y")
    assert "changes nothing" in dlg.status.text()
    assert not dlg.buttons.button(QDialogButtonBox.Save).isEnabled()
