import voice_assistant
import text_corrections


def test_correct_dictation_applies_rules_and_remembers_raw_text(tmp_path, capsys):
    rules = tmp_path / "c.toml"
    rules.write_text("[[rule]]\nmatch = '\\blync\\b'\nreplace = 'Lynq'\n")
    va = voice_assistant.VoiceAssistant.__new__(voice_assistant.VoiceAssistant)
    va.corrector = text_corrections.Corrector(rules)
    va.last_dictation = None

    assert va.correct_dictation("Die Lync API") == "Die Lynq API"
    assert va.last_dictation == ("Die Lync API", "Die Lynq API")
    assert "Recognized: Die Lync API" in capsys.readouterr().out


def test_correct_dictation_ignores_empty_text(tmp_path):
    va = voice_assistant.VoiceAssistant.__new__(voice_assistant.VoiceAssistant)
    va.corrector = text_corrections.Corrector(tmp_path / "none.toml")
    va.last_dictation = ("a", "a")
    assert va.correct_dictation("") == ""
    assert va.last_dictation == ("a", "a")
