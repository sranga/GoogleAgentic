"""
Unit tests for Security Module

Tests:
- SecureStorage (encryption, decryption, file permissions)
- InputValidator (validation, sanitization, injection prevention)
- PIIMasker (PII detection and masking)
- RateLimiter (token bucket, rate limiting)
- SessionTokenManager (token generation, validation, expiration)
"""

import pytest
import os
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

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
def temp_storage_dir():
    """Create temporary directory for storage tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def secure_storage(temp_storage_dir):
    """Create SecureStorage instance with temp directory."""
    key_file = os.path.join(temp_storage_dir, ".test_key")
    storage = SecureStorage(base_dir=temp_storage_dir, key_file=key_file)
    return storage


def test_secure_storage_initialization(temp_storage_dir):
    """Test SecureStorage initializes correctly."""
    key_file = os.path.join(temp_storage_dir, ".test_key")
    storage = SecureStorage(base_dir=temp_storage_dir, key_file=key_file)

    assert storage.base_dir.exists()
    assert storage.key_file.exists()
    assert storage.cipher is not None


def test_secure_storage_creates_key_file(temp_storage_dir):
    """Test encryption key file is created."""
    key_file = Path(temp_storage_dir) / ".test_key"
    storage = SecureStorage(base_dir=temp_storage_dir, key_file=str(key_file))

    assert key_file.exists()
    # Key file should have restrictive permissions
    stat_info = key_file.stat()
    assert oct(stat_info.st_mode)[-3:] == '600'


def test_secure_storage_reuses_existing_key(temp_storage_dir):
    """Test existing key is reused."""
    key_file = Path(temp_storage_dir) / ".test_key"

    # Create first storage instance
    storage1 = SecureStorage(base_dir=temp_storage_dir, key_file=str(key_file))
    key_content1 = key_file.read_bytes()

    # Create second instance
    storage2 = SecureStorage(base_dir=temp_storage_dir, key_file=str(key_file))
    key_content2 = key_file.read_bytes()

    # Key should be the same
    assert key_content1 == key_content2


def test_save_confirmation(secure_storage):
    """Test saving confirmation data."""
    user_id = "test_user_123"
    confirmation = {
        "confirmation_id": "CONF-123",
        "clinic_id": "clinic_1",
        "time": "2025-12-01T10:00:00Z"
    }

    filepath = secure_storage.save_confirmation(user_id, confirmation)

    assert os.path.exists(filepath)
    assert filepath.endswith(".enc")


def test_save_and_load_confirmation(secure_storage):
    """Test saving and loading confirmation."""
    user_id = "test_user_456"
    confirmation = {
        "confirmation_id": "CONF-456",
        "clinic_id": "clinic_2",
        "time": "2025-12-02T14:00:00Z",
        "patient_name": "Test Patient"
    }

    # Save
    filepath = secure_storage.save_confirmation(user_id, confirmation)
    filename = os.path.basename(filepath)

    # Load
    loaded = secure_storage.load_confirmation(user_id, filename)

    assert loaded is not None
    assert loaded["confirmation_id"] == "CONF-456"
    assert loaded["clinic_id"] == "clinic_2"
    assert loaded["patient_name"] == "Test Patient"


def test_load_confirmation_wrong_user(secure_storage):
    """Test loading confirmation with wrong user_id fails."""
    user_id = "user_abc"
    confirmation = {"confirmation_id": "CONF-789"}

    filepath = secure_storage.save_confirmation(user_id, confirmation)
    filename = os.path.basename(filepath)

    # Try to load with different user_id
    loaded = secure_storage.load_confirmation("wrong_user", filename)

    assert loaded is None  # Should fail authentication


def test_load_nonexistent_confirmation(secure_storage):
    """Test loading non-existent confirmation."""
    loaded = secure_storage.load_confirmation("user_123", "nonexistent.enc")

    assert loaded is None


def test_delete_confirmation(secure_storage):
    """Test deleting confirmation."""
    user_id = "user_delete"
    confirmation = {"confirmation_id": "CONF-DEL"}

    filepath = secure_storage.save_confirmation(user_id, confirmation)
    filename = os.path.basename(filepath)

    # Delete
    result = secure_storage.delete_confirmation(user_id, filename)

    assert result is True
    assert not os.path.exists(filepath)


def test_delete_confirmation_wrong_user(secure_storage):
    """Test deleting confirmation with wrong user_id fails."""
    user_id = "user_owner"
    confirmation = {"confirmation_id": "CONF-OWN"}

    filepath = secure_storage.save_confirmation(user_id, confirmation)
    filename = os.path.basename(filepath)

    # Try to delete with wrong user_id
    result = secure_storage.delete_confirmation("wrong_user", filename)

    assert result is False
    assert os.path.exists(filepath)  # File should still exist


def test_path_traversal_prevention(secure_storage):
    """Test prevention of directory traversal attacks."""
    user_id = "user_test"
    confirmation = {"confirmation_id": "CONF-TEST"}

    # Try to save/load with path traversal
    with pytest.raises(ValueError):
        secure_storage.load_confirmation(user_id, "../../../etc/passwd")


# ============================================================================
# INPUT VALIDATOR TESTS
# ============================================================================

def test_validate_user_id_valid():
    """Test validation of valid user IDs."""
    assert InputValidator.validate_user_id("user123") == "user123"
    assert InputValidator.validate_user_id("user_abc") == "user_abc"
    assert InputValidator.validate_user_id("user-123-abc") == "user-123-abc"


def test_validate_user_id_too_short():
    """Test validation fails for too short user_id."""
    with pytest.raises(ValidationError) as exc:
        InputValidator.validate_user_id("ab")
    assert "too short" in str(exc.value)


def test_validate_user_id_too_long():
    """Test validation fails for too long user_id."""
    long_id = "a" * 129
    with pytest.raises(ValidationError) as exc:
        InputValidator.validate_user_id(long_id)
    assert "too long" in str(exc.value)


def test_validate_user_id_invalid_chars():
    """Test validation fails for invalid characters."""
    with pytest.raises(ValidationError) as exc:
        InputValidator.validate_user_id("user@123")
    assert "invalid characters" in str(exc.value)


def test_validate_user_id_empty():
    """Test validation fails for empty user_id."""
    with pytest.raises(ValidationError):
        InputValidator.validate_user_id("")


def test_validate_user_id_none():
    """Test validation fails for None user_id."""
    with pytest.raises(ValidationError):
        InputValidator.validate_user_id(None)


def test_validate_location_query_valid():
    """Test validation of valid location queries."""
    assert InputValidator.validate_location_query("94110") == "94110"
    assert InputValidator.validate_location_query("San Francisco, CA") == "San Francisco, CA"
    assert InputValidator.validate_location_query("123 Main St") == "123 Main St"


def test_validate_location_query_too_long():
    """Test validation fails for too long location."""
    long_query = "a" * 201
    with pytest.raises(ValidationError) as exc:
        InputValidator.validate_location_query(long_query)
    assert "too long" in str(exc.value)


def test_validate_location_query_sql_injection():
    """Test validation blocks SQL injection attempts."""
    sql_queries = [
        "94110' OR '1'='1",
        "94110; DROP TABLE users;",
        "94110' UNION SELECT * FROM passwords--",
    ]

    for query in sql_queries:
        with pytest.raises(ValidationError) as exc:
            InputValidator.validate_location_query(query)
        assert "invalid characters" in str(exc.value).lower()


def test_validate_location_query_xss():
    """Test validation blocks XSS attempts."""
    xss_queries = [
        "94110<script>alert('xss')</script>",
        "94110 javascript:alert(1)",
        "<img src=x onerror=alert(1)>",
    ]

    for query in xss_queries:
        with pytest.raises(ValidationError):
            InputValidator.validate_location_query(query)


def test_validate_email_valid():
    """Test validation of valid emails."""
    assert InputValidator.validate_email("user@example.com") == "user@example.com"
    assert InputValidator.validate_email("test.user@sub.example.com") == "test.user@sub.example.com"


def test_validate_email_invalid():
    """Test validation fails for invalid emails."""
    invalid_emails = ["not-an-email", "@example.com", "user@", "user @example.com"]

    for email in invalid_emails:
        with pytest.raises(ValidationError):
            InputValidator.validate_email(email)


def test_validate_phone_valid():
    """Test validation of valid phone numbers."""
    phones = ["5551234567", "555-123-4567", "(555) 123-4567", "1-555-123-4567"]

    for phone in phones:
        result = InputValidator.validate_phone(phone)
        assert result.isdigit()
        assert len(result) >= 10


def test_validate_phone_invalid():
    """Test validation fails for invalid phone numbers."""
    with pytest.raises(ValidationError):
        InputValidator.validate_phone("123")


# ============================================================================
# PII MASKER TESTS
# ============================================================================

def test_mask_email():
    """Test email masking."""
    assert PIIMasker.mask_email("user@example.com") == "u***@example.com"
    assert PIIMasker.mask_email("ab@example.com") == "a***@example.com"
    assert PIIMasker.mask_email("a@example.com") == "a***@example.com"


def test_mask_phone():
    """Test phone number masking."""
    assert PIIMasker.mask_phone("555-123-4567") == "***-***-4567"
    assert PIIMasker.mask_phone("5551234567") == "***-***-4567"


def test_mask_ssn():
    """Test SSN masking."""
    assert PIIMasker.mask_ssn("123-45-6789") == "***-**-6789"


def test_mask_credit_card():
    """Test credit card masking."""
    assert PIIMasker.mask_credit_card("1234-5678-9012-3456") == "****-****-****-3456"
    assert PIIMasker.mask_credit_card("1234567890123456") == "****-****-****-3456"


def test_sanitize_for_logging_dict():
    """Test sanitizing dict with PII."""
    data = {
        "email": "user@example.com",
        "phone": "555-123-4567",
        "message": "Contact me at user@example.com or 555-123-4567"
    }

    sanitized = PIIMasker.sanitize_for_logging(data)

    assert "u***@example.com" in sanitized["email"]
    assert "***-***-4567" in sanitized["phone"]
    assert "u***@example.com" in sanitized["message"]
    assert "***-***-4567" in sanitized["message"]


def test_sanitize_for_logging_nested():
    """Test sanitizing nested structures."""
    data = {
        "user": {
            "email": "test@example.com",
            "contacts": ["phone: 555-123-4567"]
        }
    }

    sanitized = PIIMasker.sanitize_for_logging(data)

    assert "t***@example.com" in sanitized["user"]["email"]
    assert "***-***-4567" in sanitized["user"]["contacts"][0]


def test_sanitize_preserves_non_pii():
    """Test sanitization preserves non-PII data."""
    data = {
        "name": "Test User",
        "age": 30,
        "city": "San Francisco"
    }

    sanitized = PIIMasker.sanitize_for_logging(data)

    assert sanitized["name"] == "Test User"
    assert sanitized["age"] == 30
    assert sanitized["city"] == "San Francisco"


# ============================================================================
# RATE LIMITER TESTS
# ============================================================================

def test_rate_limiter_initialization():
    """Test rate limiter initializes correctly."""
    limiter = RateLimiter(rate=10, per_seconds=60, burst=20)

    assert limiter.rate == 10
    assert limiter.per_seconds == 60
    assert limiter.burst == 20


def test_rate_limiter_allows_within_limit():
    """Test requests within rate limit are allowed."""
    limiter = RateLimiter(rate=10, per_seconds=60, burst=10)

    # Should allow up to burst size
    for i in range(10):
        assert limiter.is_allowed("user1") is True


def test_rate_limiter_blocks_over_limit():
    """Test requests over rate limit are blocked."""
    limiter = RateLimiter(rate=2, per_seconds=60, burst=2)

    # First 2 should succeed
    assert limiter.is_allowed("user1") is True
    assert limiter.is_allowed("user1") is True

    # Third should be blocked
    assert limiter.is_allowed("user1") is False


def test_rate_limiter_refills_tokens():
    """Test tokens refill over time."""
    limiter = RateLimiter(rate=10, per_seconds=1, burst=10)  # 10 tokens per second

    # Use up tokens
    for _ in range(10):
        limiter.is_allowed("user1")

    # Should be rate limited
    assert limiter.is_allowed("user1") is False

    # Wait for refill
    time.sleep(1.1)

    # Should be allowed again
    assert limiter.is_allowed("user1") is True


def test_rate_limiter_per_user():
    """Test rate limiting is per-user."""
    limiter = RateLimiter(rate=2, per_seconds=60, burst=2)

    # User1 uses their quota
    assert limiter.is_allowed("user1") is True
    assert limiter.is_allowed("user1") is True
    assert limiter.is_allowed("user1") is False

    # User2 should still have their quota
    assert limiter.is_allowed("user2") is True
    assert limiter.is_allowed("user2") is True


def test_rate_limiter_cost():
    """Test rate limiting with different costs."""
    limiter = RateLimiter(rate=10, per_seconds=60, burst=10)

    # Expensive operation costs 5 tokens
    assert limiter.is_allowed("user1", cost=5.0) is True

    # Should have 5 tokens left
    assert limiter.is_allowed("user1", cost=5.0) is True

    # Should be rate limited now
    assert limiter.is_allowed("user1", cost=1.0) is False


def test_rate_limiter_reset():
    """Test resetting rate limit for a user."""
    limiter = RateLimiter(rate=2, per_seconds=60, burst=2)

    # Use up quota
    limiter.is_allowed("user1")
    limiter.is_allowed("user1")
    assert limiter.is_allowed("user1") is False

    # Reset
    limiter.reset("user1")

    # Should be allowed again
    assert limiter.is_allowed("user1") is True


# ============================================================================
# SESSION TOKEN MANAGER TESTS
# ============================================================================

def test_session_token_generation():
    """Test session token generation."""
    manager = SessionTokenManager()

    token = manager.generate_token("user123")

    assert isinstance(token, str)
    assert len(token) > 0


def test_session_token_validation():
    """Test session token validation."""
    manager = SessionTokenManager()

    token = manager.generate_token("user123")
    user_id = manager.validate_token(token)

    assert user_id == "user123"


def test_session_token_invalid():
    """Test validation of invalid token."""
    manager = SessionTokenManager()

    user_id = manager.validate_token("invalid_token")

    assert user_id is None


def test_session_token_expiration():
    """Test token expiration."""
    manager = SessionTokenManager(default_ttl_seconds=1)

    token = manager.generate_token("user123")

    # Should be valid immediately
    assert manager.validate_token(token) == "user123"

    # Wait for expiration
    time.sleep(1.5)

    # Should be expired
    assert manager.validate_token(token) is None


def test_session_token_revocation():
    """Test token revocation."""
    manager = SessionTokenManager()

    token = manager.generate_token("user123")
    assert manager.validate_token(token) == "user123"

    # Revoke token
    manager.revoke_token(token)

    # Should no longer be valid
    assert manager.validate_token(token) is None


def test_session_token_cleanup():
    """Test cleanup of expired tokens."""
    manager = SessionTokenManager(default_ttl_seconds=1)

    # Generate tokens
    token1 = manager.generate_token("user1", ttl_seconds=1)
    token2 = manager.generate_token("user2", ttl_seconds=10)

    # Wait for first to expire
    time.sleep(1.5)

    # Cleanup
    manager.cleanup_expired()

    # token1 should be removed, token2 should still exist
    assert manager.validate_token(token1) is None
    assert manager.validate_token(token2) == "user2"


def test_session_token_uniqueness():
    """Test that generated tokens are unique."""
    manager = SessionTokenManager()

    tokens = [manager.generate_token("user1") for _ in range(100)]

    # All tokens should be unique
    assert len(set(tokens)) == 100


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_secure_storage_with_pii_masking(secure_storage):
    """Test storing sensitive data with PII masking for logs."""
    user_id = "user_integration"
    confirmation = {
        "confirmation_id": "CONF-INT",
        "patient_email": "patient@example.com",
        "patient_phone": "555-123-4567"
    }

    # Mask PII before logging
    safe_log = PIIMasker.sanitize_for_logging(confirmation)

    # Save actual data (encrypted)
    filepath = secure_storage.save_confirmation(user_id, confirmation)

    # Load and verify
    filename = os.path.basename(filepath)
    loaded = secure_storage.load_confirmation(user_id, filename)

    # Actual data should be intact
    assert loaded["patient_email"] == "patient@example.com"

    # Logged data should be masked
    assert "p***@example.com" in safe_log["patient_email"]


def test_rate_limiter_with_validation(secure_storage):
    """Test rate limiter combined with input validation."""
    limiter = RateLimiter(rate=5, per_seconds=60, burst=5)

    # Validate and rate limit
    try:
        user_id = InputValidator.validate_user_id("test_user")
        if limiter.is_allowed(user_id):
            # Proceed with operation
            confirmation = {"test": "data"}
            secure_storage.save_confirmation(user_id, confirmation)
        else:
            pytest.fail("Should not be rate limited yet")
    except ValidationError:
        pytest.fail("Validation should not fail for valid input")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])