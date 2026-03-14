import os

from src.rag.file_type_detector import detect_file_type


def test_detects_text_file_with_unknown_extension(tmp_path):
    sample = tmp_path / "abalone.data"
    sample.write_text("rings,weight,height\n1,2,3\n4,5,6\n", encoding="utf-8")

    detected = detect_file_type(str(sample))

    assert detected.ingestible is True
    assert detected.source_type == "csv"


def test_detects_extensionless_plain_text_file(tmp_path):
    sample = tmp_path / "README"
    sample.write_text("This is a simple text file with several readable lines.\nAnother line.\n", encoding="utf-8")

    detected = detect_file_type(str(sample))

    assert detected.ingestible is True
    assert detected.source_type == "txt"
