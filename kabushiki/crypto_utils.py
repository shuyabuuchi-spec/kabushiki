"""AES-256-GCM + PBKDF2-HMAC-SHA256 encryption compatible with browser Web Crypto API."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

DEFAULT_ITERATIONS = 310_000
SALT_BYTES = 16
IV_BYTES = 12
KEY_BYTES = 32


def _derive_key(passphrase: str, salt: bytes, iterations: int = DEFAULT_ITERATIONS) -> bytes:
    if not passphrase:
        raise ValueError("Passphrase must not be empty")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_BYTES,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(passphrase.encode("utf-8"))


def encrypt_bytes(plaintext: bytes, passphrase: str, iterations: int = DEFAULT_ITERATIONS) -> dict[str, Any]:
    """Encrypt arbitrary bytes into a Web-Crypto-compatible AES-256-GCM JSON envelope."""
    salt = os.urandom(SALT_BYTES)
    iv = os.urandom(IV_BYTES)
    key = _derive_key(passphrase, salt, iterations)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(iv, plaintext, None)
    return {
        "encrypted": True,
        "alg": "AES-256-GCM",
        "kdf": "PBKDF2-SHA256",
        "iterations": iterations,
        "salt": base64.b64encode(salt).decode("ascii"),
        "iv": base64.b64encode(iv).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
    }


def decrypt_bytes(envelope: dict[str, Any], passphrase: str) -> bytes:
    """Decrypt a Web-Crypto-compatible AES-256-GCM JSON envelope back to raw bytes."""
    if not envelope.get("encrypted"):
        raise ValueError("Payload is not marked as encrypted")
    salt = base64.b64decode(envelope["salt"])
    iv = base64.b64decode(envelope["iv"])
    ciphertext = base64.b64decode(envelope["ciphertext"])
    iterations = int(envelope.get("iterations", DEFAULT_ITERATIONS))
    key = _derive_key(passphrase, salt, iterations)
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(iv, ciphertext, None)


def encrypt_file(src_path: Path, dst_path: Path, passphrase: str) -> Path:
    raw = src_path.read_bytes()
    envelope = encrypt_bytes(raw, passphrase)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    return dst_path


def decrypt_file(src_path: Path, dst_path: Path, passphrase: str) -> Path:
    envelope = json.loads(src_path.read_text(encoding="utf-8"))
    raw = decrypt_bytes(envelope, passphrase)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_bytes(raw)
    return dst_path
