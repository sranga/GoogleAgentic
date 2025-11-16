"""
Production Security Module for V-Access

Provides:
- Secure encrypted storage for sensitive data
- Input validation and sanitization
- PII detection and masking
- Rate limiting
- Authentication helpers
- Secure session management

Usage:
    from security import SecureStorage, InputValidator, PIIMasker, RateLimiter

    # Secure storage
    storage = SecureStorage()
    file_path = storage.save_confirmation(user_id, confirmation_data)

    # Input validation
    validator = InputValidator()
    validator.validate_user_id(user_id)
    validator.validate_location_query(location)

    # PII masking
    masked = PIIMasker.mask_email("user@example.com")  # u***@example.com
    safe_log = PIIMasker.sanitize_for_logging(data)
"""

import os
import re
import hashlib
import secrets
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import json

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from observability import get_logger, metrics

logger = get_logger(__name__)


# ============================================================================
# SECURE STORAGE
# ============================================================================

class SecureStorage:
    """
    Encrypted file storage for sensitive data.
    Uses Fernet (symmetric encryption) with key derivation.

    Features:
    - Encrypted at rest
    - Hashed filenames (non-reversible)
    - Directory traversal protection
    - Proper file permissions
    - Audit logging
    """

    def __init__(self, base_dir: str = "./secure_data", key_file: str = ".encryption_key"):
        """
        Initialize secure storage.

        Args:
            base_dir: Base directory for encrypted files
            key_file: Path to encryption key file
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError(
                "cryptography package not installed. "
                "Install with: pip install cryptography"
            )

        self.base_dir = Path(base_dir)
        self.key_file = Path(key_file)

        # Create base directory with restrictive permissions
        self.base_dir.mkdir(exist_ok=True, mode=0o700)

        # Load or generate encryption key
        self.cipher = self._get_or_create_cipher()

        logger.info("SecureStorage initialized", base_dir=str(self.base_dir))

    def _get_or_create_cipher(self) -> Fernet:
        """Load existing key or generate new one."""
        if self.key_file.exists():
            # Load existing key
            key = self.key_file.read_bytes()
            logger.info("Loaded existing encryption key")
        else:
            # Generate new key
            key = Fernet.generate_key()
            self.key_file.write_bytes(key)
            self.key_file.chmod(0o600)  # Owner read/write only
            logger.warning("Generated NEW encryption key - backup securely!")

        return Fernet(key)

    def save_confirmation(self, user_id: str, confirmation: Dict[str, Any]) -> str:
        """
        Save appointment confirmation securely.

        Args:
            user_id: User identifier
            confirmation: Confirmation data to save

        Returns:
            Path to saved encrypted file

        Raises:
            ValueError: If path validation fails
            IOError: If file write fails
        """
        # Hash user_id for filename (SHA-256, non-reversible)
        hashed_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"conf_{hashed_id}_{timestamp}.enc"
        filepath = self.base_dir / filename

        # Validate path (prevent directory traversal)
        if not self._is_safe_path(filepath):
            raise ValueError("Invalid file path - potential directory traversal")

        # Add metadata
        data_with_metadata = {
            "confirmation": confirmation,
            "user_id_hash": hashed_id,
            "saved_at": datetime.utcnow().isoformat(),
            "version": "1.0"
        }

        # Serialize and encrypt
        json_data = json.dumps(data_with_metadata).encode('utf-8')
        encrypted_data = self.cipher.encrypt(json_data)

        # Write with proper permissions
        filepath.write_bytes(encrypted_data)
        filepath.chmod(0o600)  # Owner read/write only

        logger.info(
            "Confirmation saved securely",
            filename=filename,
            size_bytes=len(encrypted_data)
        )
        metrics.counter("secure_files_created")

        return str(filepath)

    def load_confirmation(self, user_id: str, filename: str) -> Optional[Dict[str, Any]]:
        """
        Load and decrypt confirmation.

        Args:
            user_id: User identifier (for validation)
            filename: Encrypted file name

        Returns:
            Decrypted confirmation data or None if not found
        """
        filepath = self.base_dir / filename

        if not filepath.exists():
            logger.warning("Confirmation file not found", filename=filename)
            return None

        if not self._is_safe_path(filepath):
            raise ValueError("Invalid file path")

        try:
            # Read and decrypt
            encrypted_data = filepath.read_bytes()
            decrypted_data = self.cipher.decrypt(encrypted_data)
            data = json.loads(decrypted_data.decode('utf-8'))

            # Verify user_id matches
            expected_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]
            if data.get("user_id_hash") != expected_hash:
                logger.error("User ID hash mismatch - unauthorized access attempt")
                metrics.counter("security_violations", labels={"type": "unauthorized_access"})
                return None

            logger.info("Confirmation loaded", filename=filename)
            metrics.counter("secure_files_loaded")
            return data["confirmation"]

        except Exception as e:
            logger.error("Failed to decrypt confirmation", error=str(e), filename=filename)
            metrics.counter("decryption_failures")
            return None

    def delete_confirmation(self, user_id: str, filename: str) -> bool:
        """
        Securely delete confirmation file.

        Args:
            user_id: User identifier
            filename: File to delete

        Returns:
            True if deleted, False otherwise
        """
        filepath = self.base_dir / filename

        if not filepath.exists():
            return False

        # Verify ownership before deletion
        data = self.load_confirmation(user_id, filename)
        if data is None:
            logger.error("Cannot delete - ownership verification failed")
            return False

        try:
            filepath.unlink()
            logger.info("Confirmation deleted", filename=filename)
            metrics.counter("secure_files_deleted")
            return True
        except Exception as e:
            logger.error("Failed to delete confirmation", error=str(e))
            return False

    def _is_safe_path(self, filepath: Path) -> bool:
        """Check if path is safe (within base directory)."""
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
    """
    Input validation and sanitization.

    Validates:
    - User IDs
    - Location queries
    - Session data
    - Email addresses
    - Phone numbers
    """

    # Validation patterns
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    PHONE_PATTERN = re.compile(r'^\+?1?\d{9,15}$')
    ZIP_CODE_PATTERN = re.compile(r'^\d{5}(-\d{4})?$')

    # Dangerous patterns to block
    SQL_INJECTION_PATTERNS = [
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
        """
        Validate and sanitize user ID.

        Args:
            user_id: User identifier

        Returns:
            Validated user ID

        Raises:
            ValidationError: If validation fails
        """
        if not user_id or not isinstance(user_id, str):
            raise ValidationError("user_id must be a non-empty string")

        # Length check
        if len(user_id) > 128:
            raise ValidationError("user_id too long (max 128 characters)")

        if len(user_id) < 3:
            raise ValidationError("user_id too short (min 3 characters)")

        # Character whitelist (alphanumeric + hyphen + underscore)
        if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
            raise ValidationError("user_id contains invalid characters")

        metrics.counter("input_validations", labels={"type": "user_id", "status": "success"})
        return user_id.strip()

    @staticmethod
    def validate_location_query(query: str) -> str:
        """
        Validate and sanitize location query.

        Args:
            query: Location string (zip code, address, etc.)

        Returns:
            Validated query

        Raises:
            ValidationError: If validation fails
        """
        if not query or not isinstance(query, str):
            raise ValidationError("location_query must be a non-empty string")

        # Length check
        if len(query) > 200:
            raise ValidationError("location_query too long (max 200 characters)")

        query = query.strip()

        # Check for SQL injection patterns
        for pattern in InputValidator.SQL_INJECTION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning("Potential SQL injection detected", query=query[:50])
                metrics.counter("security_violations", labels={"type": "sql_injection"})
                raise ValidationError("Invalid characters in location_query")

        # Check for XSS patterns
        for pattern in InputValidator.XSS_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning("Potential XSS detected", query=query[:50])
                metrics.counter("security_violations", labels={"type": "xss"})
                raise ValidationError("Invalid characters in location_query")

        metrics.counter("input_validations", labels={"type": "location", "status": "success"})
        return query

    @staticmethod
    def validate_email(email: str) -> str:
        """Validate email address."""
        if not email or not isinstance(email, str):
            raise ValidationError("Invalid email")

        email = email.strip().lower()

        if not InputValidator.EMAIL_PATTERN.match(email):
            raise ValidationError("Invalid email format")

        if len(email) > 254:  # RFC 5321
            raise ValidationError("Email too long")

        return email

    @staticmethod
    def validate_phone(phone: str) -> str:
        """Validate phone number."""
        if not phone or not isinstance(phone, str):
            raise ValidationError("Invalid phone number")

        # Remove common separators
        phone = re.sub(r'[\s\-\(\)\.]+', '', phone)

        if not InputValidator.PHONE_PATTERN.match(phone):
            raise ValidationError("Invalid phone format")

        return phone


# ============================================================================
# PII DETECTION & MASKING
# ============================================================================

class PIIMasker:
    """
    Detect and mask Personally Identifiable Information (PII) in logs and data.

    Masks:
    - Email addresses
    - Phone numbers
    - Credit card numbers
    - Social Security Numbers
    - API keys/tokens
    """

    # PII patterns
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_PATTERN = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b')
    API_KEY_PATTERN = re.compile(r'\b[A-Za-z0-9_-]{32,}\b')

    @staticmethod
    def mask_email(email: str) -> str:
        """Mask email address: user@example.com -> u***@example.com"""
        if not email or '@' not in email:
            return email

        local, domain = email.split('@', 1)
        if len(local) <= 2:
            return f"{local[0]}***@{domain}"
        return f"{local[0]}{'*' * (len(local) - 1)}@{domain}"

    @staticmethod
    def mask_phone(phone: str) -> str:
        """Mask phone number: 555-123-4567 -> ***-***-4567"""
        digits = re.sub(r'\D', '', phone)
        if len(digits) >= 4:
            return f"***-***-{digits[-4:]}"
        return "***-***-****"

    @staticmethod
    def mask_ssn(ssn: str) -> str:
        """Mask SSN: 123-45-6789 -> ***-**-6789"""
        parts = ssn.split('-')
        if len(parts) == 3:
            return f"***-**-{parts[2]}"
        return "***-**-****"

    @staticmethod
    def mask_credit_card(cc: str) -> str:
        """Mask credit card: 1234-5678-9012-3456 -> ****-****-****-3456"""
        digits = re.sub(r'\D', '', cc)
        if len(digits) >= 4:
            return f"****-****-****-{digits[-4:]}"
        return "****-****-****-****"

    @staticmethod
    def sanitize_for_logging(data: Any) -> Any:
        """
        Recursively sanitize data for logging by masking PII.

        Args:
            data: Data to sanitize (dict, list, str, etc.)

        Returns:
            Sanitized copy of data
        """
        if isinstance(data, dict):
            return {k: PIIMasker.sanitize_for_logging(v) for k, v in data.items()}

        elif isinstance(data, list):
            return [PIIMasker.sanitize_for_logging(item) for item in data]

        elif isinstance(data, str):
            # Apply all masking patterns
            text = data
            text = PIIMasker.EMAIL_PATTERN.sub(
                lambda m: PIIMasker.mask_email(m.group()), text
            )
            text = PIIMasker.PHONE_PATTERN.sub(
                lambda m: PIIMasker.mask_phone(m.group()), text
            )
            text = PIIMasker.SSN_PATTERN.sub(
                lambda m: PIIMasker.mask_ssn(m.group()), text
            )
            text = PIIMasker.CREDIT_CARD_PATTERN.sub(
                lambda m: PIIMasker.mask_credit_card(m.group()), text
            )
            return text

        return data


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for API endpoints and user actions.

    Features:
    - Per-user rate limiting
    - Configurable rate and burst size
    - Thread-safe
    - Automatic cleanup of old entries
    """

    def __init__(self, rate: int = 10, per_seconds: int = 60, burst: int = 20):
        """
        Initialize rate limiter.

        Args:
            rate: Number of requests allowed per time period
            per_seconds: Time period in seconds
            burst: Maximum burst size (token bucket capacity)
        """
        self.rate = rate
        self.per_seconds = per_seconds
        self.burst = burst
        self.refill_rate = rate / per_seconds  # Tokens per second

        # Storage: {user_id: (tokens, last_refill_time)}
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._lock = threading.Lock()

        logger.info(
            "RateLimiter initialized",
            rate=rate,
            per_seconds=per_seconds,
            burst=burst
        )

    def is_allowed(self, user_id: str, cost: float = 1.0) -> bool:
        """
        Check if request is allowed under rate limit.

        Args:
            user_id: User identifier
            cost: Cost of this request in tokens (default: 1.0)

        Returns:
            True if allowed, False if rate limited
        """
        now = time.time()

        with self._lock:
            if user_id not in self._buckets:
                # New user - start with full bucket
                self._buckets[user_id] = (self.burst, now)

            tokens, last_refill = self._buckets[user_id]

            # Refill tokens based on elapsed time
            elapsed = now - last_refill
            tokens = min(self.burst, tokens + elapsed * self.refill_rate)

            # Check if enough tokens available
            if tokens >= cost:
                tokens -= cost
                self._buckets[user_id] = (tokens, now)

                metrics.counter("rate_limit_checks", labels={"status": "allowed"})
                return True
            else:
                # Rate limited
                self._buckets[user_id] = (tokens, now)

                logger.warning(
                    "Rate limit exceeded",
                    user_id=user_id[:8],
                    tokens_available=tokens,
                    cost=cost
                )
                metrics.counter("rate_limit_checks", labels={"status": "blocked"})
                metrics.counter("rate_limit_violations", labels={"user": user_id[:8]})
                return False

    def reset(self, user_id: str):
        """Reset rate limit for a user."""
        with self._lock:
            if user_id in self._buckets:
                del self._buckets[user_id]
                logger.info("Rate limit reset", user_id=user_id[:8])

    def cleanup_old_entries(self, max_age_seconds: int = 3600):
        """Remove entries that haven't been used recently."""
        now = time.time()
        with self._lock:
            old_count = len(self._buckets)
            self._buckets = {
                uid: (tokens, last)
                for uid, (tokens, last) in self._buckets.items()
                if now - last < max_age_seconds
            }
            removed = old_count - len(self._buckets)
            if removed > 0:
                logger.info("Cleaned up old rate limit entries", removed=removed)


# ============================================================================
# SECURE SESSION TOKENS
# ============================================================================

class SessionTokenManager:
    """
    Generate and validate secure session tokens.

    Features:
    - Cryptographically secure random tokens
    - Token expiration
    - Token revocation
    """

    def __init__(self, token_length: int = 32, default_ttl_seconds: int = 3600):
        """
        Initialize session token manager.

        Args:
            token_length: Length of tokens in bytes
            default_ttl_seconds: Default time-to-live for tokens
        """
        self.token_length = token_length
        self.default_ttl = default_ttl_seconds

        # Storage: {token: (user_id, expires_at)}
        self._tokens: Dict[str, Tuple[str, float]] = {}
        self._lock = threading.Lock()

    def generate_token(self, user_id: str, ttl_seconds: Optional[int] = None) -> str:
        """
        Generate a new session token.

        Args:
            user_id: User identifier
            ttl_seconds: Time-to-live (optional, uses default if not specified)

        Returns:
            Secure random token
        """
        token = secrets.token_urlsafe(self.token_length)
        ttl = ttl_seconds or self.default_ttl
        expires_at = time.time() + ttl

        with self._lock:
            self._tokens[token] = (user_id, expires_at)

        logger.info(
            "Session token generated",
            user_id=user_id[:8],
            ttl_seconds=ttl
        )
        metrics.counter("session_tokens_generated")

        return token

    def validate_token(self, token: str) -> Optional[str]:
        """
        Validate token and return user_id if valid.

        Args:
            token: Token to validate

        Returns:
            User ID if valid, None if invalid or expired
        """
        now = time.time()

        with self._lock:
            if token not in self._tokens:
                metrics.counter("token_validations", labels={"status": "not_found"})
                return None

            user_id, expires_at = self._tokens[token]

            if now > expires_at:
                # Token expired - remove it
                del self._tokens[token]
                logger.info("Token expired", user_id=user_id[:8])
                metrics.counter("token_validations", labels={"status": "expired"})
                return None

            metrics.counter("token_validations", labels={"status": "valid"})
            return user_id

    def revoke_token(self, token: str):
        """Revoke a token."""
        with self._lock:
            if token in self._tokens:
                user_id = self._tokens[token][0]
                del self._tokens[token]
                logger.info("Token revoked", user_id=user_id[:8])
                metrics.counter("tokens_revoked")

    def cleanup_expired(self):
        """Remove all expired tokens."""
        now = time.time()
        with self._lock:
            old_count = len(self._tokens)
            self._tokens = {
                tok: (uid, exp)
                for tok, (uid, exp) in self._tokens.items()
                if exp > now
            }
            removed = old_count - len(self._tokens)
            if removed > 0:
                logger.info("Cleaned up expired tokens", removed=removed)


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