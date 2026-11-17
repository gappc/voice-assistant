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
    QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QTextEdit, QVBoxLayout,
)

import change_view as cv
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
        self.saved = []            # rule specs saved in this dialog
        self.target_edited = False  # has the user typed into "Should be"?
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
        self.merge_box = QCheckBox()  # "add as variant to an existing rule"
        self.merge_box.setVisible(False)
        self.merge_target = None   # rule index the variants would go to
        self._offered = None       # last offered target, to keep a manual uncheck
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.preview = QTextEdit(readOnly=True)
        self.shown_side_effects = []  # (before, after) listed in the preview

        form = QFormLayout()
        form.addRow("Recognized:", recognized)
        form.addRow("Should be:", self.fixed)
        self.rule_header = QLabel()
        self.rule_header.setWordWrap(True)
        form.addRow(self.rule_header)
        form.addRow("Replace:", self.wrong)
        form.addRow("With:", self.right)
        form.addRow("Only if followed by:", self.followed_by)
        form.addRow("Only if preceded by:", self.preceded_by)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._save)
        self.buttons.rejected.connect(self.reject)
        self.check_button = self.buttons.addButton("Check All Rules", QDialogButtonBox.ActionRole)
        self.check_button.setToolTip("Verify every example in the rules file against all rules")
        self.check_button.clicked.connect(self._check_all)

        hint = QLabel(f"Rules file: {html.escape(str(path))}")
        hint.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.merge_box)
        layout.addWidget(self.status)
        layout.addWidget(self.preview)
        layout.addWidget(hint)
        layout.addWidget(self.buttons)

        self.fixed.textChanged.connect(self._on_target_edited)
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

    @property
    def current(self):
        """The dictation as the saved rules (including this dialog's) leave it."""
        return tc.apply(self.existing, self.raw)

    def _on_target_edited(self):
        self.target_edited = True
        self._suggest()

    def _suggest(self):
        # One rule per separate change; the first remaining one is shown.
        # Context fields are left as set, so narrowing survives retyping.
        changes = tc.suggest_changes(self.current, self.fixed.text())
        spec = changes[0].spec if changes else {"replace": [], "with": ""}
        self._set(self.wrong, ", ".join(spec["replace"]))
        self._set(self.right, spec["with"])
        self._evaluate()

    def _rule_edited(self):
        # Rule typed by hand without a target sentence: "Should be" follows it.
        if not self.target_edited:
            try:
                rule = tc.Rule.from_spec(self.spec())
            except ValueError:
                rule = None
            if rule:
                self._set(self.fixed, tc.apply([*self.existing, rule], self.raw))
        self._evaluate()

    def _examples(self, rule):
        """Examples for `rule`: the fragment around each place it changes,
        or the whole dictation if a fragment alone doesn't reproduce it."""
        rules = [*self.existing, rule]
        result = tc.apply(rules, self.raw)
        # Fragments are cut at the other changes towards "Should be", so this
        # example stays valid whether or not rules for those get saved.
        target = self.fixed.text() if self.target_edited else result
        examples = [(c.before, c.after) for c in tc.suggest_changes(self.current, target)
                    if tc.apply(rules, c.before) == c.after
                    and tc.apply(self.existing, c.before) != c.after]
        return examples or [(self.raw, result)]

    def _show_step(self, remaining):
        """Header "Rule n of m" plus the changes still to come, and a
        "Save & Next" button while more follow."""
        header = "<b>Rule</b>"
        if self.saved or remaining:
            n = len(self.saved) + 1
            header = f"<b>Rule {n} of {n + len(remaining)}</b>"
        header += " – words are matched whole, in any case; separate alternatives with commas"
        if remaining:
            todo = ", ".join(f"“{html.escape(c.spec['replace'][0])}” → “{html.escape(c.spec['with'])}”"
                             for c in remaining)
            header += (f"<br>Next: {todo} <i>(close the dialog to skip what "
                       "needs no rule)</i>")
        self.rule_header.setText(header)
        self.buttons.button(QDialogButtonBox.Save).setText(
            "Save && Next" if remaining else "Save")

    def _evaluate(self):
        save = self.buttons.button(QDialogButtonBox.Save)
        save.setEnabled(False)
        self._show_step([])
        self.merge_target = None
        self.merge_box.setVisible(False)
        self.preview.clear()
        self.shown_side_effects = []
        if not _split(self.wrong.text()):
            if self.saved and self.current == self.fixed.text():
                self.status.setText("✓ The dictation is fully corrected.")
            else:
                self.status.setText("Edit “Should be” into what you said, or fill in the rule.")
            return
        ev = tc.evaluate_new_rule(self.existing, self.raw, self.fixed.text(), self.spec(),
                                  self.history)
        if ev.error:
            self.status.setText(f"⚠ {ev.error}")
            return
        if ev.result == self.current:
            self.status.setText("This rule changes nothing in the dictation.")
            return
        parts, notes = [], []
        if self.saved:
            notes.append(f"✓ Saved rule {len(self.saved)}: {tc.describe(self.saved[-1])}")
        if ev.broken_examples:
            notes.append(f"⚠ Breaks {len(ev.broken_examples)} existing example(s) – narrow "
                         "the rule with “Only if followed/preceded by” to save it.")
            parts += [cv.heading("Existing examples this rule breaks:"),
                      cv.failures(ev.broken_examples, "BROKEN")]
        if common := tc.common_variants(self.spec()):
            words = ", ".join(f"“{w}”" for w in common)
            notes.append(f"⚠ {words} is an ordinary word, so the rule changes it everywhere. "
                         "Add a word to “Only if followed/preceded by” unless that is what you want.")
        if ev.side_effects:
            notes.append(f"Would also change {len(ev.side_effects)} of {len(self.history)} "
                         "earlier dictations. If any are wrong, add words to "
                         "“Only if followed/preceded by”.")
            self.shown_side_effects = ev.side_effects[:SHOWN_SIDE_EFFECTS]
            parts.append(cv.heading("Earlier dictations this rule would also change:"))
            parts += [cv.line(cv.diff(b, a)) for b, a in self.shown_side_effects]
            if len(ev.side_effects) > SHOWN_SIDE_EFFECTS:
                parts.append(cv.text(f"… and {len(ev.side_effects) - SHOWN_SIDE_EFFECTS} more"))
        else:
            notes.append(f"Changes none of {len(self.history)} earlier dictations.")
        self._show_step(tc.suggest_changes(ev.result, self.fixed.text()))
        self.status.setText("\n".join(notes))
        self.preview.setHtml("".join(parts))
        self._offer_merge()
        save.setEnabled(not ev.broken_examples)

    def _check_all(self):
        """Same as `text_corrections.py check`, shown in the dialog. Rereads
        the file, so it also covers edits made in an editor meanwhile."""
        try:
            rules = tc.load_rules(self.path)
        except ValueError as exc:
            self.status.setText(f"⚠ Rules file is invalid: {exc}")
            self.preview.clear()
            return
        failures = tc.check(rules)
        total = sum(len(r.examples) for r in rules)
        counts = f"{len(rules)} rule{'s' if len(rules) != 1 else ''}"
        if failures:
            self.status.setText(f"⚠ Rules file: {counts}, {len(failures)} of {total} examples fail.")
            self.preview.setHtml(cv.failures(failures, "FAIL"))
        else:
            self.status.setText(f"✓ Rules file: {counts}, all {total} examples pass.")
            self.preview.clear()

    def _offer_merge(self):
        """Offer to add the variants to an existing rule with the same result
        and context, which keeps the rules file short."""
        spec = self.spec()
        target = tc.variant_target(self.existing, spec)
        if target is not None:
            known = {v.lower() for v in self.existing[target].spec["replace"]}
            if all(v.lower() in known for v in spec["replace"]):
                target = None
        if target is not None and target != self._offered:
            self.merge_box.setChecked(True)
            self._offered = target
        self.merge_target = target
        self.merge_box.setVisible(target is not None)
        if target is not None:
            self.merge_box.setText("Add as variant to the existing rule: "
                                   + tc.describe(self.existing[target].spec))

    def _save(self):
        spec = self.spec()
        rule = tc.Rule.from_spec(spec)
        examples = self._examples(rule)
        if self.merge_target is not None and self.merge_box.isChecked():
            old = self.existing[self.merge_target].spec["replace"]
            new = [v for v in spec["replace"] if v.lower() not in {o.lower() for o in old}]
            tc.update_rule(self.path, self.merge_target, {"replace": [*old, *new]},
                           add_examples=examples)
        else:
            tc.append_rule(self.path, spec, examples)
        self.saved.append(spec)
        self.existing = tc.load_rules(self.path)
        if not tc.suggest_changes(self.current, self.fixed.text()):
            self.accept()
            return
        # More separate changes left: suggest the next one, context cleared.
        self._set(self.followed_by, "")
        self._set(self.preceded_by, "")
        self._suggest()
