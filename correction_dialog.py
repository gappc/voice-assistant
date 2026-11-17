""""Fix Last Dictation" dialog: turn a corrected sentence into a rule.

Two ways in: edit "Should be" into what you said and the rule is derived
(text_corrections.suggest_rule), or fill in the rule fields and "Should be"
follows. Either way the dialog shows what else in the dictation history the
rule would change, and saving appends it to the rules file with the sentence
as its example. The logic lives in text_corrections; this module is only UI.
"""

import html

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QPlainTextEdit, QVBoxLayout,
)

import text_corrections as tc

SHOWN_SIDE_EFFECTS = 30


def _split(text):
    return [part.strip() for part in text.split(",") if part.strip()]


class CorrectionDialog(QDialog):
    def __init__(self, raw, injected, path, history, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fix Last Dictation")
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.resize(760, 560)
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
        self.wrong = QLineEdit()
        self.wrong.setPlaceholderText("e.g. Link, linken")
        self.right = QLineEdit()
        self.followed_by = QLineEdit()
        self.followed_by.setPlaceholderText("optional, e.g. API, Projekt")
        self.preceded_by = QLineEdit()
        self.preceded_by.setPlaceholderText("optional, e.g. Projekt")
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.preview = QPlainTextEdit(readOnly=True)

        form = QFormLayout()
        form.addRow("Recognized:", recognized)
        form.addRow("Should be:", self.fixed)
        form.addRow(QLabel("<b>Rule</b> – words are matched whole, in any case; "
                           "separate alternatives with commas"))
        form.addRow("Replace:", self.wrong)
        form.addRow("With:", self.right)
        form.addRow("Only if followed by:", self.followed_by)
        form.addRow("Only if preceded by:", self.preceded_by)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._save)
        self.buttons.rejected.connect(self.reject)

        hint = QLabel(f"Rules file: {html.escape(str(path))} – "
                      "<code>uv run text_corrections.py check</code> verifies it.")
        hint.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.status)
        layout.addWidget(self.preview)
        layout.addWidget(hint)
        layout.addWidget(self.buttons)

        self.fixed.textChanged.connect(self._suggest)
        for field in (self.wrong, self.right, self.followed_by, self.preceded_by):
            field.textChanged.connect(self._rule_edited)
        self._suggest()
        if load_error:
            self.status.setText(f"⚠ Rules file is invalid, fix it first: {load_error}")
            self.buttons.button(QDialogButtonBox.Save).setEnabled(False)
        self.fixed.setFocus()
        self.fixed.selectAll()

    def spec(self):
        spec = {"replace": _split(self.wrong.text()), "with": self.right.text().strip()}
        if after := _split(self.followed_by.text()):
            spec["followed_by"] = after
        if before := _split(self.preceded_by.text()):
            spec["preceded_by"] = before
        return spec

    def _set(self, field, value):
        field.blockSignals(True)
        field.setText(value)
        field.blockSignals(False)

    def _suggest(self):
        # Derive from the injected text: existing rules already ran on it,
        # and the new rule runs after them. Context fields are left as set.
        suggestion = tc.suggest_rule(self.injected, self.fixed.text())
        self._set(self.wrong, ", ".join(suggestion["replace"]) if suggestion else "")
        self._set(self.right, suggestion["with"] if suggestion else "")
        self._evaluate()

    def _rule_edited(self):
        # Rule typed by hand: "Should be" follows the rule's result.
        try:
            rule = tc.Rule.from_spec(self.spec())
        except ValueError:
            rule = None
        if rule:
            self._set(self.fixed, tc.apply([*self.existing, rule], self.raw))
        self._evaluate()

    def _evaluate(self):
        save = self.buttons.button(QDialogButtonBox.Save)
        save.setEnabled(False)
        self.preview.setPlainText("")
        if not _split(self.wrong.text()):
            self.status.setText("Edit “Should be” into what you said, or fill in the rule.")
            return
        ev = tc.evaluate_new_rule(self.existing, self.raw, self.fixed.text(), self.spec(),
                                  self.history)
        if ev.error:
            self.status.setText(f"⚠ {ev.error}")
            return
        if ev.result == tc.apply(self.existing, self.raw):
            self.status.setText("This rule changes nothing in the dictation.")
            return
        lines, notes = [], []
        if not ev.produces_fix:
            notes.append(f"⚠ The rules turn the dictation into: “{ev.result}”")
        if ev.broken_examples:
            notes.append(f"⚠ Breaks {len(ev.broken_examples)} existing example(s) – narrow "
                         "the rule with “Only if followed/preceded by” to save it.")
            for src, want, got in ev.broken_examples:
                lines += [f"BROKEN  {src}", f"   want {want}", f"   got  {got}", ""]
        if ev.side_effects:
            notes.append(f"Would also change {len(ev.side_effects)} of {len(self.history)} "
                         "earlier dictations. If any are wrong, add words to "
                         "“Only if followed/preceded by”.")
            for before, after in ev.side_effects[:SHOWN_SIDE_EFFECTS]:
                lines += [f"- {before}", f"+ {after}", ""]
            if len(ev.side_effects) > SHOWN_SIDE_EFFECTS:
                lines.append(f"… and {len(ev.side_effects) - SHOWN_SIDE_EFFECTS} more")
        else:
            notes.append(f"Changes none of {len(self.history)} earlier dictations.")
        self.status.setText("\n".join(notes))
        self.preview.setPlainText("\n".join(lines))
        save.setEnabled(ev.produces_fix and not ev.broken_examples)

    def _save(self):
        tc.append_rule(self.path, self.spec(), (self.raw, self.fixed.text()))
        self.accept()
