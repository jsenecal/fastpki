"""Unit tests for the password policy (issue #11)."""

import pytest
from pydantic import ValidationError

from app.core import password_policy
from app.core.config import Settings, settings
from app.core.password_policy import (
    BCRYPT_MAX_BYTES,
    PasswordPolicyError,
    load_common_passwords,
    validate_password,
)
from app.schemas.user import UserCreate, UserUpdate

STRONG = "correct horse battery staple"


class TestLength:
    def test_too_short_rejected(self):
        with pytest.raises(PasswordPolicyError, match="at least 12 characters"):
            validate_password("Xq7!vR2#mZ9")  # 11 chars

    def test_min_length_accepted(self):
        assert validate_password("Xq7!vR2#mZ9$") == "Xq7!vR2#mZ9$"  # 12 chars

    def test_max_bytes_accepted(self):
        pw = "k" * 30 + "Q" * (BCRYPT_MAX_BYTES - 30)
        assert validate_password(pw) == pw

    def test_too_long_rejected(self):
        with pytest.raises(PasswordPolicyError, match="at most 72 bytes"):
            validate_password("k" * 30 + "Q" * (BCRYPT_MAX_BYTES - 29))

    def test_multibyte_limit_counts_bytes(self):
        # 37 two-byte characters = 74 bytes, only 37 characters long.
        with pytest.raises(PasswordPolicyError, match="at most 72 bytes"):
            validate_password("é" * 37)

    def test_min_length_follows_settings(self, monkeypatch):
        monkeypatch.setattr(settings, "PASSWORD_MIN_LENGTH", 16)
        with pytest.raises(PasswordPolicyError, match="at least 16 characters"):
            validate_password("Xq7!vR2#mZ9$abc")  # 15 chars

    def test_no_composition_rules(self):
        # All lowercase letters, no digits or symbols: fine under NIST rules.
        assert validate_password("purple monkey dishwasher")


class TestCommonPasswords:
    def test_list_loaded_and_cached(self):
        common = load_common_passwords()
        assert isinstance(common, frozenset)
        assert len(common) > 10_000
        assert load_common_passwords() is common

    @pytest.mark.parametrize("pw", ["qwerty123456", "QWERTY123456", "1q2w3e4r5t6y"])
    def test_common_rejected_case_insensitive(self, pw):
        with pytest.raises(PasswordPolicyError, match="too common"):
            validate_password(pw)

    def test_common_short_password_rejected_when_floor_lowered(self, monkeypatch):
        monkeypatch.setattr(settings, "PASSWORD_MIN_LENGTH", 8)
        with pytest.raises(PasswordPolicyError, match="too common"):
            validate_password("password1")


class TestPersonalInfo:
    def test_equal_to_username_rejected(self):
        with pytest.raises(PasswordPolicyError, match="username"):
            validate_password("AliceWonderland", username="alicewonderland")

    def test_contains_username_rejected(self):
        with pytest.raises(PasswordPolicyError, match="username"):
            validate_password("my-JSMITH-is-great", username="jsmith")

    def test_contains_email_local_part_rejected(self):
        with pytest.raises(PasswordPolicyError, match="email"):
            validate_password("zz-bobby.tables-zz", email="Bobby.Tables@example.com")

    def test_short_identifier_not_checked_for_containment(self):
        # A 2-char username must not forbid every password containing it.
        assert validate_password("absolutely fabulous", username="ab")
        assert validate_password("absolutely fabulous", email="ab@example.com")

    def test_three_char_identifier_checked(self):
        with pytest.raises(PasswordPolicyError, match="username"):
            validate_password("absolutely fabulous", username="Abs")

    def test_no_identifiers(self):
        assert validate_password(STRONG, username=None, email=None) == STRONG

    def test_error_message_does_not_echo_password(self):
        pw = "secret-jsmith-value"
        with pytest.raises(PasswordPolicyError) as exc:
            validate_password(pw, username="jsmith")
        assert pw not in str(exc.value)


class TestSettings:
    def test_min_length_default(self):
        assert Settings().PASSWORD_MIN_LENGTH == 12

    def test_min_length_floor(self):
        with pytest.raises(ValidationError, match="PASSWORD_MIN_LENGTH"):
            Settings(PASSWORD_MIN_LENGTH=7)

    def test_min_length_cannot_exceed_bcrypt_limit(self):
        with pytest.raises(ValidationError, match="PASSWORD_MIN_LENGTH"):
            Settings(PASSWORD_MIN_LENGTH=73)


class TestSchemas:
    def test_user_create_valid(self):
        u = UserCreate(username="alice", email="alice@example.com", password=STRONG)
        assert u.password == STRONG

    def test_user_create_rejects_username(self):
        with pytest.raises(ValidationError, match="username"):
            UserCreate(
                username="alice",
                email="a@example.com",
                password="alice-in-wonderland",
            )

    def test_user_create_rejects_email_local_part(self):
        with pytest.raises(ValidationError, match="email"):
            UserCreate(
                username="zed",
                email="carol.danvers@example.com",
                password="carol.danvers-rocks",
            )

    def test_user_update_rejects_short(self):
        with pytest.raises(ValidationError, match="at least 12"):
            UserUpdate(password="short")

    def test_user_update_rejects_new_email_local_part(self):
        with pytest.raises(ValidationError, match="email"):
            UserUpdate(
                email="carol.danvers@example.com", password="carol.danvers-rocks"
            )

    def test_user_update_none_password_allowed(self):
        assert UserUpdate(email="x@example.com").password is None
        # An explicit null runs the validator and must pass through.
        assert UserUpdate(password=None).password is None


def test_module_exposes_path():
    assert password_policy.COMMON_PASSWORDS_PATH.is_file()
