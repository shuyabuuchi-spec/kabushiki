"""Unit tests for AES-256-GCM encryption & decryption utilities."""

from pathlib import Path
import json
import pytest
from cryptography.exceptions import InvalidTag

from kabushiki.crypto_utils import encrypt_bytes, decrypt_bytes, encrypt_file, decrypt_file


def test_encrypt_decrypt_bytes_roundtrip():
    secret = "MyStrongPassphrase!2026"
    payload = json.dumps({"equity": 101250, "name": "トヨタ自動車"}, ensure_ascii=False).encode("utf-8")

    env = encrypt_bytes(payload, secret, iterations=10_000)
    assert env["encrypted"] is True
    assert env["alg"] == "AES-256-GCM"
    assert env["kdf"] == "PBKDF2-SHA256"
    assert "トヨタ" not in json.dumps(env, ensure_ascii=False)

    restored = decrypt_bytes(env, secret)
    assert restored == payload

    with pytest.raises(InvalidTag):
        decrypt_bytes(env, "WrongPassword")


def test_encrypt_decrypt_file_roundtrip(tmp_path: Path):
    src = tmp_path / "sample.db"
    enc = tmp_path / "sample.db.enc"
    dec = tmp_path / "restored.db"

    original = b"\x00\x01SQLite format 3\x00binary-test-data" * 50
    src.write_bytes(original)

    encrypt_file(src, enc, "Passphrase123!")
    assert enc.exists()
    assert original not in enc.read_bytes()

    decrypt_file(enc, dec, "Passphrase123!")
    assert dec.read_bytes() == original
