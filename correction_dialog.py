""""Fix Last Dictation" dialog: turn a corrected sentence into a rule.

The user edits the last dictation into what they meant; the dialog derives a
rule (text_corrections.suggest_rule), shows what else in the dictation history
it would change, and appends it to the rules file with the sentence as its
example. The logic lives in text_corrections; this module is only the UI.
"""

import html

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QPlainTextEdit, QVBoxLayout,
)

import text_corrections as tc

SHOWN_SIDE_EFFECTS = 30


class CorrectionDialog(QDialog):
    def __init__(self, raw, injected, path, history, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fix Last Dictation")
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.resize(760, 520)
        self.raw, self.injected, self.path, self.history = raw, injected, path, history
        try:
            self.existing = tc.load_rules(path)
            load_error = None
        except ValueError as exc:
            self.existing, load_error = [], str(exc)

        recognized = QLabel(html.escape(raw))
        recognized.setTextInteractionFlags(Qt.TextSelectableByMouse)
        recognized.setWordWrap(True)
        self.fixed = QLineEdit(injected)
        self.match = QLineEdit()
        self.replace = QLineEdit()
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.preview = QPlainTextEdit(readOnly=True)

        form = QFormLayout()
        form.addRow("Recognized:", recognized)
        form.addRow("Should be:", self.fixed)
        form.addRow("Rule – match (regex):", self.match)
        form.addRow("Rule – replace with:", self.replace)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._save)
        self.buttons.rejected.connect(self.reject)

        hint = QLabel(f"Rules file: {path}  ·  edit it for context rules; "
                      "<code>uv run text_corrections.py check</code> verifies it.")
        hint.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.status)
        layout.addWidget(self.preview)
        layout.addWidget(hint)
        layout.addWidget(self.buttons)

        self.fixed.textChanged.connect(self._suggest)
        self.match.textChanged.connect(self._evaluate)
        self.replace.textChanged.connect(self._evaluate)
        self._suggest()
        if load_error:
            self.status.setText(f"⚠ Rules file is invalid, fix it first: {load_error}")
            self.buttons.button(QDialogButtonBox.Save).setEnabled(False)
        self.fixed.setFocus()
        self.fixed.selectAll()

    def _suggest(self):
        # Derive from the injected text: existing rules already ran on it,
        # and the new rule runs after them.
        suggestion = tc.suggest_rule(self.injected, self.fixed.text())
        match, replace = suggestion or ("", "")
        for widget, value in ((self.match, match), (self.replace, replace)):
            widget.blockSignals(True)
            widget.setText(value)
            widget.blockSignals(False)
        self._evaluate()

    def _evaluate(self):
        save = self.buttons.button(QDialogButtonBox.Save)
        self.evaluation = None
        if not self.match.text():
            self.status.setText("Edit “Should be” into what you said.")
            self.preview.setPlainText("")
            save.setEnabled(False)
            return
        ev = tc.evaluate_new_rule(self.existing, self.raw, self.fixed.text(),
                                  self.match.text(), self.replace.text(), self.history)
        if ev.error:
            self.status.setText(f"⚠ Invalid regex: {ev.error}")
            self.preview.setPlainText("")
            save.setEnabled(False)
            return
        lines, notes = [], []
        if not ev.produces_fix:
            notes.append(f"⚠ The rules turn the dictation into: “{ev.result}”")
        if ev.broken_examples:
            notes.append(f"⚠ Breaks {len(ev.broken_examples)} existing example(s).")
            for src, want, got in ev.broken_examples:
                lines += [f"BROKEN  {src}", f"   want {want}", f"   got  {got}", ""]
        if ev.side_effects:
            notes.append(f"Would also change {len(ev.side_effects)} of "
                         f"{len(self.history)} earlier dictations – narrow the match if any are wrong.")
            for before, after in ev.side_effects[:SHOWN_SIDE_EFFECTS]:
                lines += [f"- {before}", f"+ {after}", ""]
            if len(ev.side_effects) > SHOWN_SIDE_EFFECTS:
                lines.append(f"… and {len(ev.side_effects) - SHOWN_SIDE_EFFECTS} more")
        else:
            notes.append(f"Changes none of {len(self.history)} earlier dictations.")
        self.status.setText("\n".join(notes))
        self.preview.setPlainText("\n".join(lines))
        self.evaluation = ev
        save.setEnabled(ev.produces_fix)

    def _save(self):
        tc.append_rule(self.path, self.match.text(), self.replace.text(),
                       (self.raw, self.fixed.text()))
        self.accept()
