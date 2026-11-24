"""Password policy (NIST SP 800-63B style).

Applied whenever a password is *set or changed* (user creation and password
update). It is deliberately **not** applied at login, so users whose stored
password predates the policy can still authenticate.

Rules:

* Length between ``settings.PASSWORD_MIN_LENGTH`` characters (default 12) and
  :data:`BCRYPT_MAX_BYTES` bytes once UTF-8 encoded. bcrypt only consumes the
  first 72 bytes of its input (and bcrypt>=5 raises on longer input), so a
  longer password would either be silently truncated or fail to hash.
* Not present (case-insensitively) in the bundled common/breached password
  list.
* Not equal to, and not containing, the username or the local part of the
  email address.
* No character-class composition rules.

The common password list (``resources/common_passwords.txt``) is the union of
SecLists ``Passwords/Common-Credentials/100k-most-used-passwords-NCSC.txt``
and ``Passwords/Common-Credentials/10k-most-common.txt`` (MIT License, see
``resources/common_passwords.LICENSE``), case-folded, de-duplicated and
reduced to entries of at least 8 characters (shorter entries can never pass
the minimum-length rule). It is loaded once and cached; no network access
happens at runtime.

Error messages never include the password itself.
"""

from functools import lru_cache
from pathlib import Path

from app.core.config import settings

COMMON_PASSWORDS_PATH = Path(__file__).parent / "resources" / "common_passwords.txt"

BCRYPT_MAX_BYTES = 72

# Identifiers shorter than this are not checked for containment; otherwise a
# two-letter username would forbid every password containing those letters.
_MIN_CONTAINMENT_LEN = 3


class PasswordPolicyError(ValueError):
    """Raised when a password does not satisfy the password policy."""


@lru_cache(maxsize=1)
def load_common_passwords() -> frozenset[str]:
    """Return the case-folded common password list (read once, then cached)."""
    with COMMON_PASSWORDS_PATH.open(encoding="utf-8") as fh:
        return frozenset(line.strip().casefold() for line in fh if line.strip())


def _matches_identifier(folded_password: str, identifier: str | None) -> bool:
    if not identifier:
        return False
    # Containment covers equality; identifiers too short to be checked can
    # never equal a password that already passed the minimum-length rule.
    ident = identifier.casefold()
    return len(ident) >= _MIN_CONTAINMENT_LEN and ident in folded_password


def validate_password(
    password: str, *, username: str | None = None, email: str | None = None
) -> str:
    """Validate ``password`` against the policy and return it unchanged.

    Raises :class:`PasswordPolicyError` (a ``ValueError``, so Pydantic turns it
    into a 422 validation error) describing the first violated rule.
    """
    min_length = settings.PASSWORD_MIN_LENGTH
    if len(password) < min_length:
        raise PasswordPolicyError(  # noqa: TRY003
            f"Password must be at least {min_length} characters long"
        )
    if len(password.encode("utf-8")) > BCRYPT_MAX_BYTES:
        raise PasswordPolicyError(  # noqa: TRY003
            f"Password must be at most {BCRYPT_MAX_BYTES} bytes when UTF-8 encoded"
        )

    folded = password.casefold()
    if folded in load_common_passwords():
        raise PasswordPolicyError(  # noqa: TRY003
            "Password is too common; choose a less predictable password"
        )
    if _matches_identifier(folded, username):
        raise PasswordPolicyError("Password must not contain the username")  # noqa: TRY003
    local_part = email.rsplit("@", 1)[0] if email else None
    if _matches_identifier(folded, local_part):
        raise PasswordPolicyError("Password must not contain the email address")  # noqa: TRY003
    return password
