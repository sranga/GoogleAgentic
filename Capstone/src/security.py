"""
Security Module

Production security features for the V-Access system:
- Secure encrypted storage
- Input validation and sanitization
- PII detection and masking
- Rate limiting
- Session token management
"""

import os
import re
import hashlib
import secrets
import time
import threading
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, UTC
from collections import defaultdict

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


# ============================================================================
# SECURE STORAGE
# ============================================================================

class SecureStorage:
    """
    Encrypted file storage for sensitive data using Fernet (AES-128).
    """

    def __init__(self, base_dir: str = "./secure_data", key_file: str = ".encryption_key"):
        self.base_dir = Path(base_dir)
        self.key_file = Path(key_file)

        self.base_dir.mkdir(exist_ok=True, mode=0o700)
        self.cipher = self._get_or_create_cipher()

        logger.info("SecureStorage initialized", extra={"base_dir": str(self.base_dir)})

    def _get_or_create_cipher(self) -> Fernet:
        """Load existing key or generate new one."""
        if self.key_file.exists():
            key = self.key_file.read_bytes()
        else:
            key = Fernet.generate_key()
            self.key_file.write_bytes(key)
            self.key_file.chmod(0o600)
            logger.warning("Generated new encryption key - backup securely!")

        return Fernet(key)

    def save_confirmation(self, user_id: str, confirmation: Dict[str, Any]) -> str:
        """Save encrypted confirmation."""
        hashed_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"conf_{hashed_id}_{timestamp}.enc"
        filepath = self.base_dir / filename

        if not self._is_safe_path(filepath):
            raise ValueError("Invalid file path")

        data = {
            "confirmation": confirmation,
            "user_id_hash": hashed_id,
            "saved_at": datetime.now(UTC).isoformat(),
        }

        encrypted = self.cipher.encrypt(json.dumps(data).encode())
        filepath.write_bytes(encrypted)
        filepath.chmod(0o600)

        logger.info("Confirmation saved", extra={"filename": filename})
        return str(filepath)

    def load_confirmation(self, user_id: str, filename: str) -> Optional[Dict[str, Any]]:
        """Load and decrypt confirmation."""
        filepath = self.base_dir / filename

        if not filepath.exists() or not self._is_safe_path(filepath):
            return None

        try:
            decrypted = self.cipher.decrypt(filepath.read_bytes())
            data = json.loads(decrypted.decode())

            expected_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]
            if data.get("user_id_hash") != expected_hash:
                logger.error("User ID hash mismatch")
                return None

            return data["confirmation"]
        except Exception as e:
            logger.error("Decryption failed", extra={"error": str(e)})
            return None

    def delete_confirmation(self, user_id: str, filename: str) -> bool:
        """Securely delete confirmation file."""
        if self.load_confirmation(user_id, filename) is None:
            return False

        filepath = self.base_dir / filename
        filepath.unlink()
        logger.info("Confirmation deleted", extra={"filename": filename})
        return True

    def _is_safe_path(self, filepath: Path) -> bool:
        """Check path is within base directory."""
        try:
            filepath.resolve().relative_to(self.base_dir.resolve())
            return True
        except ValueError:
            return False


# ============================================================================
# INPUT VALIDATION
# ============================================================================

class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class InputValidator:
    """Input validation and sanitization."""

    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    PHONE_PATTERN = re.compile(r'^\+?1?\d{9,15}$')

    SQL_PATTERNS = [
        r"(\bUNION\b|\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b)",
        r"(--|;|'|\"|\*|=)",
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"onerror=",
        r"onload=",
    ]

    @staticmethod
    def validate_user_id(user_id: str) -> str:
        """Validate user ID."""
        if not user_id or not isinstance(user_id, str):
            raise ValidationError("user_id must be a non-empty string")

        if len(user_id) > 128:
            raise ValidationError("user_id too long (max 128 characters)")

        if len(user_id) < 3:
            raise ValidationError("user_id too short (min 3 characters)")

        if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
            raise ValidationError("user_id contains invalid characters")

        return user_id.strip()

    @staticmethod
    def validate_location_query(query: str) -> str:
        """Validate location query."""
        if not query or not isinstance(query, str):
            raise ValidationError("location_query must be a non-empty string")

        if len(query) > 200:
            raise ValidationError("location_query too long (max 200 characters)")

        query = query.strip()

        for pattern in InputValidator.SQL_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                raise ValidationError("Invalid characters in location_query")

        for pattern in InputValidator.XSS_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                raise ValidationError("Invalid characters in location_query")

        return query

    @staticmethod
    def validate_email(email: str) -> str:
        """Validate email address."""
        if not email or not isinstance(email, str):
            raise ValidationError("Invalid email")

        email = email.strip().lower()

        if not InputValidator.EMAIL_PATTERN.match(email):
            raise ValidationError("Invalid email format")

        return email

    @staticmethod
    def validate_phone(phone: str) -> str:
        """Validate phone number."""
        if not phone:
            raise ValidationError("Invalid phone number")

        phone = re.sub(r'[\s\-\(\)\.]+', '', phone)

        if not InputValidator.PHONE_PATTERN.match(phone):
            raise ValidationError("Invalid phone format")

        return phone


# ============================================================================
# PII MASKING
# ============================================================================

class PIIMasker:
    """Detect and mask PII in data."""

    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_PATTERN = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    CC_PATTERN = re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b')

    @staticmethod
    def mask_email(email: str) -> str:
        if not email or '@' not in email:
            return email
        local, domain = email.split('@', 1)
        return f"{local[0]}***@{domain}"

    @staticmethod
    def mask_phone(phone: str) -> str:
        digits = re.sub(r'\D', '', phone)
        return f"***-***-{digits[-4:]}" if len(digits) >= 4 else "***-***-****"

    @staticmethod
    def mask_ssn(ssn: str) -> str:
        parts = ssn.split('-')
        return f"***-**-{parts[2]}" if len(parts) == 3 else "***-**-****"

    @staticmethod
    def mask_credit_card(cc: str) -> str:
        digits = re.sub(r'\D', '', cc)
        return f"****-****-****-{digits[-4:]}" if len(digits) >= 4 else "****-****-****-****"

    @staticmethod
    def sanitize_for_logging(data: Any) -> Any:
        """Recursively sanitize data for logging."""
        if isinstance(data, dict):
            return {k: PIIMasker.sanitize_for_logging(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [PIIMasker.sanitize_for_logging(item) for item in data]
        elif isinstance(data, str):
            text = data
            text = PIIMasker.EMAIL_PATTERN.sub(lambda m: PIIMasker.mask_email(m.group()), text)
            text = PIIMasker.PHONE_PATTERN.sub(lambda m: PIIMasker.mask_phone(m.group()), text)
            text = PIIMasker.SSN_PATTERN.sub(lambda m: PIIMasker.mask_ssn(m.group()), text)
            text = PIIMasker.CC_PATTERN.sub(lambda m: PIIMasker.mask_credit_card(m.group()), text)
            return text
        return data


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, rate: int = 10, per_seconds: int = 60, burst: int = 20):
        self.rate = rate
        self.per_seconds = per_seconds
        self.burst = burst
        self.refill_rate = rate / per_seconds

        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, user_id: str, cost: float = 1.0) -> bool:
        """Check if request is allowed."""
        now = time.time()

        with self._lock:
            if user_id not in self._buckets:
                self._buckets[user_id] = (self.burst, now)

            tokens, last_refill = self._buckets[user_id]
            elapsed = now - last_refill
            tokens = min(self.burst, tokens + elapsed * self.refill_rate)

            if tokens >= cost:
                self._buckets[user_id] = (tokens - cost, now)
                return True
            else:
                self._buckets[user_id] = (tokens, now)
                logger.warning("Rate limit exceeded", extra={"user_id": user_id[:8]})
                return False

    def reset(self, user_id: str):
        """Reset rate limit for user."""
        with self._lock:
            if user_id in self._buckets:
                del self._buckets[user_id]


# ============================================================================
# SESSION TOKENS
# ============================================================================

class SessionTokenManager:
    """Secure session token management."""

    def __init__(self, token_length: int = 32, default_ttl_seconds: int = 3600):
        self.token_length = token_length
        self.default_ttl = default_ttl_seconds

        self._tokens: Dict[str, Tuple[str, float]] = {}
        self._lock = threading.Lock()

    def generate_token(self, user_id: str, ttl_seconds: Optional[int] = None) -> str:
        """Generate a new session token."""
        token = secrets.token_urlsafe(self.token_length)
        ttl = ttl_seconds or self.default_ttl
        expires_at = time.time() + ttl

        with self._lock:
            self._tokens[token] = (user_id, expires_at)

        return token

    def validate_token(self, token: str) -> Optional[str]:
        """Validate token and return user_id if valid."""
        now = time.time()

        with self._lock:
            if token not in self._tokens:
                return None

            user_id, expires_at = self._tokens[token]

            if now > expires_at:
                del self._tokens[token]
                return None

            return user_id

    def revoke_token(self, token: str):
        """Revoke a token."""
        with self._lock:
            self._tokens.pop(token, None)

    def cleanup_expired(self):
        """Remove expired tokens."""
        now = time.time()
        with self._lock:
            self._tokens = {
                tok: (uid, exp) for tok, (uid, exp) in self._tokens.items()
                if exp > now
            }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "SecureStorage",
    "InputValidator",
    "ValidationError",
    "PIIMasker",
    "RateLimiter",
    "SessionTokenManager",
]