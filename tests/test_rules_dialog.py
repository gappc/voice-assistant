import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from datetime import datetime
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

import text_corrections as tc
from rules_dialog import RulesDialog

RULES = """
[[rule]]
replace = ["Lync"]
with = "Lynq"
examples = [["Die Lync API", "Die Lynq API"]]

[[rule]]
replace = ["Link"]
with = "Lynq"

[[rule]]
replace = ["Nimmermehr"]
with = "x"
"""


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(app, tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "dictionary_words", lambda: frozenset({"link"}))
    path = tmp_path / "c.toml"
    path.write_text(RULES)
    history = [tc.Dictation("Die Lync API", datetime(2026, 11, 3)),
               tc.Dictation("ein Link", datetime(2026, 11, 5)),
               tc.Dictation("noch ein Link", datetime(2026, 11, 7))]
    return RulesDialog(path, history)


def cell(dlg, row, col):
    return dlg.table.item(row, col).text()


def test_table_shows_rules_with_hits_last_hit_and_notes(dialog):
    assert dialog.table.rowCount() == 3
    assert [cell(dialog, r, 3) for r in range(3)] == ["1", "2", "0"]
    assert cell(dialog, 1, 4) == "2026-11-07" and cell(dialog, 2, 4) == "never"
    assert cell(dialog, 1, 5) == "ordinary word: Link" and cell(dialog, 0, 5) == ""
    assert "3 rules" in dialog.summary.text() and "1 never fired" in dialog.summary.text()
    assert "all 1 examples pass" in dialog.summary.text()


def test_without_selection_all_changes_are_shown_newest_first(dialog):
    assert dialog.selected() is None
    assert dialog.shown == [("noch ein Link", "noch ein Lynq"), ("ein Link", "ein Lynq"),
                            ("Die Lync API", "Die Lynq API")]
    text = dialog.details.toPlainText()
    assert "All changes: 3 earlier dictations" in text
    assert "2026-11-07" in text and "← #2 Link → Lynq" in text


def test_selecting_a_rule_shows_only_its_changes_then_examples(dialog):
    dialog.table.selectRow(0)
    assert dialog.shown == [("Die Lync API", "Die Lynq API")]
    text = dialog.details.toPlainText()
    assert text.index("changes in 1 earlier") < text.index("Examples")
    assert dialog.delete_button.isEnabled()
    dialog.all_button.click()
    assert dialog.selected() is None and len(dialog.shown) == 3


def test_reload_keeps_the_selection_and_its_details(dialog):
    dialog.table.selectRow(1)
    dialog.reload_button.click()
    assert dialog.selected() == 1
    assert dialog.shown == [("noch ein Link", "noch ein Lynq"), ("ein Link", "ein Lynq")]
    assert dialog.delete_button.isEnabled()


def test_delete_removes_the_rule_after_confirmation(dialog, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    dialog.table.selectRow(1)
    dialog.delete_button.click()
    assert [r.spec["replace"] for r in tc.load_rules(dialog.path)] == [["Lync"], ["Nimmermehr"]]
    assert dialog.table.rowCount() == 2


def test_delete_can_be_cancelled(dialog, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    dialog.table.selectRow(0)
    dialog.delete_button.click()
    assert len(tc.load_rules(dialog.path)) == 3


def test_invalid_file_is_reported(dialog):
    Path(dialog.path).write_text("[[rule]]\nwith = 'x'\n")
    dialog.reload()
    assert "invalid" in dialog.summary.text() and dialog.table.rowCount() == 0


def test_row_heights_fit_the_shown_width_and_survive_reload(app, tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[[rule]]\nreplace = ["Link"]\nwith = "Lynq"\nfollowed_by = '
                    '["API", "Projekt", "Project", "Webseite", "Website", "Homepage", '
                    '"Domain", "Client", "Plugin", "Browser"]\n')
    dlg = RulesDialog(path, [])
    dlg.show()
    app.processEvents()
    shown = dlg.table.rowHeight(0)
    dlg.reload()
    app.processEvents()
    assert dlg.table.rowHeight(0) == shown
    dlg.close()
