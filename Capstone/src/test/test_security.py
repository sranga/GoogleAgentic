"""
Tests for Security Module
"""

import pytest
import sys
import os
import time
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from security import (
    SecureStorage,
    InputValidator,
    ValidationError,
    PIIMasker,
    RateLimiter,
    SessionTokenManager,
)


# ============================================================================
# SECURE STORAGE TESTS
# ============================================================================

@pytest.fixture
def temp_dir():
    """Temporary directory for storage tests."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def secure_storage(temp_dir):
    """SecureStorage with temp directory."""
    key_file = os.path.join(temp_dir, ".test_key")
    return SecureStorage(base_dir=temp_dir, key_file=key_file)


class TestSecureStorage:
    def test_initialization(self, temp_dir):
        key_file = os.path.join(temp_dir, ".key")
        storage = SecureStorage(base_dir=temp_dir, key_file=key_file)
        assert storage.base_dir.exists()
        assert storage.key_file.exists()

    def test_key_file_permissions(self, temp_dir):
        key_file = Path(temp_dir) / ".key"
        SecureStorage(base_dir=temp_dir, key_file=str(key_file))
        assert oct(key_file.stat().st_mode)[-3:] == '600'

    def test_save_confirmation(self, secure_storage):
        filepath = secure_storage.save_confirmation("user1", {"id": "CONF-1"})
        assert os.path.exists(filepath)
        assert filepath.endswith(".enc")

    def test_save_and_load_confirmation(self, secure_storage):
        data = {"confirmation_id": "CONF-123", "clinic": "Test Clinic"}
        filepath = secure_storage.save_confirmation("user1", data)
        filename = os.path.basename(filepath)

        loaded = secure_storage.load_confirmation("user1", filename)

        assert loaded is not None
        assert loaded["confirmation_id"] == "CONF-123"

    def test_load_wrong_user_fails(self, secure_storage):
        data = {"id": "CONF-1"}
        filepath = secure_storage.save_confirmation("user1", data)
        filename = os.path.basename(filepath)

        loaded = secure_storage.load_confirmation("wrong_user", filename)
        assert loaded is None

    def test_load_nonexistent_returns_none(self, secure_storage):
        loaded = secure_storage.load_confirmation("user1", "nonexistent.enc")
        assert loaded is None

    def test_delete_confirmation(self, secure_storage):
        filepath = secure_storage.save_confirmation("user1", {"id": "C1"})
        filename = os.path.basename(filepath)

        result = secure_storage.delete_confirmation("user1", filename)

        assert result is True
        assert not os.path.exists(filepath)


# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================

class TestInputValidatorUserId:
    def test_valid_user_ids(self):
        assert InputValidator.validate_user_id("user123") == "user123"
        assert InputValidator.validate_user_id("user_abc") == "user_abc"
        assert InputValidator.validate_user_id("user-123") == "user-123"

    def test_too_short(self):
        with pytest.raises(ValidationError):
            InputValidator.validate_user_id("ab")

    def test_too_long(self):
        with pytest.raises(ValidationError):
            InputValidator.validate_user_id("a" * 129)

    def test_invalid_characters(self):
        with pytest.raises(ValidationError):
            InputValidator.validate_user_id("user@123")

    def test_empty(self):
        with pytest.raises(ValidationError):
            InputValidator.validate_user_id("")

    def test_none(self):
        with pytest.raises(ValidationError):
            InputValidator.validate_user_id(None)


class TestInputValidatorLocation:
    def test_valid_locations(self):
        assert InputValidator.validate_location_query("94110") == "94110"
        assert InputValidator.validate_location_query("San Francisco") == "San Francisco"

    def test_too_long(self):
        with pytest.raises(ValidationError):
            InputValidator.validate_location_query("a" * 201)

    def test_sql_injection(self):
        with pytest.raises(ValidationError):
            InputValidator.validate_location_query("94110' OR '1'='1")

    def test_xss(self):
        with pytest.raises(ValidationError):
            InputValidator.validate_location_query("<script>alert('xss')</script>")


class TestInputValidatorEmail:
    def test_valid_emails(self):
        assert InputValidator.validate_email("test@example.com") == "test@example.com"
        assert InputValidator.validate_email("Test@Example.COM") == "test@example.com"

    def test_invalid_email(self):
        with pytest.raises(ValidationError):
            InputValidator.validate_email("not-an-email")


class TestInputValidatorPhone:
    def test_valid_phones(self):
        assert InputValidator.validate_phone("5551234567") == "5551234567"
        assert InputValidator.validate_phone("555-123-4567") == "5551234567"
        assert InputValidator.validate_phone("(555) 123-4567") == "5551234567"

    def test_invalid_phone(self):
        with pytest.raises(ValidationError):
            InputValidator.validate_phone("123")


# ============================================================================
# PII MASKING TESTS
# ============================================================================

class TestPIIMasker:
    def test_mask_email(self):
        assert PIIMasker.mask_email("user@example.com") == "u***@example.com"

    def test_mask_phone(self):
        assert PIIMasker.mask_phone("555-123-4567") == "***-***-4567"

    def test_mask_ssn(self):
        assert PIIMasker.mask_ssn("123-45-6789") == "***-**-6789"

    def test_mask_credit_card(self):
        assert PIIMasker.mask_credit_card("1234-5678-9012-3456") == "****-****-****-3456"

    def test_sanitize_dict(self):
        data = {"email": "user@example.com", "name": "John"}
        sanitized = PIIMasker.sanitize_for_logging(data)
        assert "u***@example.com" in sanitized["email"]
        assert sanitized["name"] == "John"

    def test_sanitize_nested(self):
        data = {"user": {"email": "test@example.com"}}
        sanitized = PIIMasker.sanitize_for_logging(data)
        assert "t***@example.com" in sanitized["user"]["email"]

    def test_sanitize_list(self):
        data = ["user@example.com", "555-123-4567"]
        sanitized = PIIMasker.sanitize_for_logging(data)
        assert "u***@example.com" in sanitized[0]
        assert "***-***-4567" in sanitized[1]


# ============================================================================
# RATE LIMITER TESTS
# ============================================================================

class TestRateLimiter:
    def test_allows_within_limit(self):
        limiter = RateLimiter(rate=10, per_seconds=60, burst=10)
        for _ in range(10):
            assert limiter.is_allowed("user1") is True

    def test_blocks_over_limit(self):
        limiter = RateLimiter(rate=2, per_seconds=60, burst=2)
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is False

    def test_per_user_isolation(self):
        limiter = RateLimiter(rate=1, per_seconds=60, burst=1)
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is False
        assert limiter.is_allowed("user2") is True

    def test_refills_tokens(self):
        limiter = RateLimiter(rate=10, per_seconds=1, burst=10)
        for _ in range(10):
            limiter.is_allowed("user1")

        assert limiter.is_allowed("user1") is False
        time.sleep(1.1)
        assert limiter.is_allowed("user1") is True

    def test_reset(self):
        limiter = RateLimiter(rate=1, per_seconds=60, burst=1)
        limiter.is_allowed("user1")
        assert limiter.is_allowed("user1") is False

        limiter.reset("user1")
        assert limiter.is_allowed("user1") is True


# ============================================================================
# SESSION TOKEN TESTS
# ============================================================================

class TestSessionTokenManager:
    def test_generate_token(self):
        manager = SessionTokenManager()
        token = manager.generate_token("user1")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_validate_token(self):
        manager = SessionTokenManager()
        token = manager.generate_token("user1")
        user_id = manager.validate_token(token)
        assert user_id == "user1"

    def test_invalid_token_returns_none(self):
        manager = SessionTokenManager()
        assert manager.validate_token("invalid") is None

    def test_token_expiration(self):
        manager = SessionTokenManager(default_ttl_seconds=1)
        token = manager.generate_token("user1")
        assert manager.validate_token(token) == "user1"

        time.sleep(1.5)
        assert manager.validate_token(token) is None

    def test_revoke_token(self):
        manager = SessionTokenManager()
        token = manager.generate_token("user1")
        assert manager.validate_token(token) == "user1"

        manager.revoke_token(token)
        assert manager.validate_token(token) is None

    def test_cleanup_expired(self):
        manager = SessionTokenManager(default_ttl_seconds=1)
        token1 = manager.generate_token("user1")
        token2 = manager.generate_token("user2", ttl_seconds=10)

        time.sleep(1.5)
        manager.cleanup_expired()

        assert manager.validate_token(token1) is None
        assert manager.validate_token(token2) == "user2"

    def test_token_uniqueness(self):
        manager = SessionTokenManager()
        tokens = [manager.generate_token("user1") for _ in range(100)]
        assert len(set(tokens)) == 100