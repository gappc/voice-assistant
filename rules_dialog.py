""""Correction Rules" window: every rule with how often it fired on past
dictations, its examples and matches, plus delete / open-in-editor.

Statistics come from the journal (text_corrections.rule_stats), so nothing
extra is stored; "hits" counts past dictations the rule changes when all
rules run in file order, including dictations from before the rule existed.
"""

import html

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QHeaderView, QLabel, QMessageBox,
    QSplitter, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout,
)

import change_view as cv
import text_corrections as tc

COLUMNS = ("Replace", "With", "Only if", "Hits", "Last hit", "Note")
SHOWN_CHANGES = 50


def _context(spec):
    parts = []
    if spec.get("followed_by"):
        parts.append("followed by " + ", ".join(spec["followed_by"]))
    if spec.get("preceded_by"):
        parts.append("preceded by " + ", ".join(spec["preceded_by"]))
    return "; ".join(parts)


class RulesDialog(QDialog):
    def __init__(self, path, dictations, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Correction Rules")
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.resize(900, 620)
        self.path, self.dictations = path, dictations
        self.rules, self.stats = [], []

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        for col in (0, 2):  # the long text columns share the width and wrap
            header.setSectionResizeMode(col, QHeaderView.Stretch)
        self.table.setWordWrap(True)
        # Row heights follow the column widths: computed once, they would fit
        # the narrow columns before the window is shown and change on reload.
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.details = QTextEdit(readOnly=True)
        self.shown = []  # (before, after) listed in the details, newest first

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.table)
        splitter.addWidget(self.details)
        splitter.setSizes([320, 260])

        self.buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.all_button = self.buttons.addButton("All Changes", QDialogButtonBox.ActionRole)
        self.all_button.setToolTip("Show the changes of all rules (clears the selection)")
        self.all_button.clicked.connect(self.table.clearSelection)
        self.delete_button = self.buttons.addButton("Delete Rule", QDialogButtonBox.ActionRole)
        self.open_button = self.buttons.addButton("Open in Editor", QDialogButtonBox.ActionRole)
        self.reload_button = self.buttons.addButton("Reload", QDialogButtonBox.ActionRole)
        self.delete_button.clicked.connect(self._delete)
        self.open_button.clicked.connect(self._open)
        self.reload_button.clicked.connect(self.reload)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addWidget(splitter)
        layout.addWidget(QLabel(f"Rules file: {path}"))
        layout.addWidget(self.buttons)

        self.table.itemSelectionChanged.connect(self._show_details)
        self.reload()

    def reload(self):
        keep = self.selected()
        try:
            self.rules = tc.load_rules(self.path)
        except ValueError as exc:
            self.rules, self.stats = [], []
            self.table.setRowCount(0)
            self.summary.setText(f"⚠ The rules file is invalid: {exc}")
            self.details.clear()
            self.shown = []
            self.delete_button.setEnabled(False)
            return
        self.stats = tc.rule_stats(self.rules, self.dictations)
        failures = tc.check(self.rules)
        examples = sum(len(r.examples) for r in self.rules)
        check = (f"⚠ {len(failures)} of {examples} examples fail (see “Check All Rules” in "
                 "Fix Last Dictation)" if failures else f"✓ all {examples} examples pass")
        unused = sum(1 for s in self.stats if not s.hits)
        self.summary.setText(
            f"{len(self.rules)} rules · checked against {len(self.dictations)} earlier "
            f"dictations · {check}" + (f" · {unused} never fired" if unused else ""))

        self.table.setRowCount(len(self.rules))
        for row, (rule, st) in enumerate(zip(self.rules, self.stats)):
            spec = rule.spec
            replace = f"regex: {spec['regex']}" if "regex" in spec else ", ".join(spec["replace"])
            common = tc.common_variants(spec)
            cells = (replace, spec["with"], _context(spec), str(st.hits),
                     st.last.strftime("%Y-%m-%d") if st.last else "never",
                     f"ordinary word: {', '.join(common)}" if common else "")
            for col, value in enumerate(cells):
                item = QTableWidgetItem(value)
                if col == 3:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row, col, item)
        # Keep the selected rule selected; Qt reports no selection change when
        # the same row stays selected, so refresh the details explicitly.
        self.table.blockSignals(True)
        if keep is not None and keep < len(self.rules):
            self.table.selectRow(keep)
        else:
            self.table.clearSelection()
        self.table.blockSignals(False)
        self._show_details()

    def selected(self):
        rows = self.table.selectionModel().selectedRows()
        return rows[0].row() if rows else None

    def _label(self, index):
        spec = self.rules[index].spec
        what = "regex" if "regex" in spec else ", ".join(spec["replace"])
        return f"#{index + 1} {what} → {spec['with']}"

    def _show_details(self):
        row = self.selected()
        self.delete_button.setEnabled(row is not None)
        if row is None:
            self._show_all()
            return
        rule, st = self.rules[row], self.stats[row]
        hits = st.changes[::-1]  # newest first
        self.shown = [(h.before, h.after) for h in hits[:SHOWN_CHANGES]]
        parts = [cv.heading(f"{self._label(row)}: changes in {st.hits} earlier "
                            "dictation(s), newest first")]
        parts += [cv.line(cv.diff(h.before, h.after), cv.date(h.when))
                  for h in hits[:SHOWN_CHANGES]]
        if st.hits > SHOWN_CHANGES:
            parts.append(cv.text(f"… and {st.hits - SHOWN_CHANGES} older"))
        if not st.hits:
            parts.append(cv.text("None – this rule never fired on past dictations."))
        if rule.examples:
            parts.append(cv.heading("Examples"))
            for src, want in rule.examples:
                parts.append(cv.line(cv.diff(src, want)) if src != want
                             else cv.line(html.escape(src), "keep:"))
        self.details.setHtml("".join(parts))

    def _show_all(self):
        changes = tc.all_changes(self.rules, self.dictations)
        self.shown = [(d.text, out) for d, out, _ in changes[:SHOWN_CHANGES]]
        parts = [cv.heading(f"All changes: {len(changes)} earlier dictations, newest first "
                            "(select a rule to see only its changes)")]
        for d, out, fired in changes[:SHOWN_CHANGES]:
            rules = "; ".join(self._label(i) for i in fired)
            parts.append(cv.line(cv.diff(d.text, out), cv.date(d.when), f"← {rules}"))
        if len(changes) > SHOWN_CHANGES:
            parts.append(cv.text(f"… and {len(changes) - SHOWN_CHANGES} older"))
        self.details.setHtml("".join(parts))

    def _delete(self):
        row = self.selected()
        if row is None:
            return
        answer = QMessageBox.question(
            self, "Delete Rule", f"Delete this rule?\n\n{tc.describe(self.rules[row].spec)}")
        if answer != QMessageBox.Yes:
            return
        try:
            tc.delete_rule(self.path, row)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Delete Rule", str(exc))
        self.reload()

    def _open(self):
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("# Correction rules, see the README.\n", encoding="utf-8")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.path)))
