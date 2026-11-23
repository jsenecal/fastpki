# Authentication

FastPKI uses JWT (JSON Web Tokens) with OAuth2 password flow for authentication.

## Login Flow

1. **POST credentials** to the token endpoint as form-encoded data.
2. **Receive a JWT** access token in the response.
3. **Send the token** as a `Bearer` header on every subsequent request.

### Obtain a Token

```bash
curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=admin&password=tangerine-orbit-velvet-42"
```

??? example "Response"

    ```json
    {
        "access_token": "eyJhbGciOiJIUzI1NiIs...",
        "token_type": "bearer"
    }
    ```

!!! note
    The token endpoint uses `application/x-www-form-urlencoded` (standard OAuth2), **not** JSON.

### Use the Token

Pass the token in the `Authorization` header:

```bash
curl -s http://localhost:8000/api/v1/users/me \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..."
```

### Shell Variable Shortcut

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=admin&password=tangerine-orbit-velvet-42" \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Use in subsequent requests
curl -s http://localhost:8000/api/v1/cas/ \
  -H "Authorization: Bearer $TOKEN"
```

## Token Lifecycle

FastPKI issues short-lived access tokens and long-lived refresh tokens.

| Token | Default TTL | Configurable via |
|-------|-------------|-----------------|
| Access token | 15 minutes | `ACCESS_TOKEN_EXPIRE_MINUTES` |
| Refresh token | 24 hours (1440 min) | `REFRESH_TOKEN_EXPIRE_MINUTES` |

### Token Rotation on Refresh

Each call to `POST /api/v1/auth/refresh` consumes the submitted refresh token and issues a new access token together with a new refresh token. The old refresh token is immediately invalidated, preventing replay attacks.

### Logout (Single Session)

`POST /api/v1/auth/logout` invalidates the refresh token associated with the current session and immediately revokes the caller's access token by adding it to the blocklist. Both the access token and refresh token become unusable as soon as logout completes.

### Invalidate All Sessions

`POST /api/v1/auth/invalidate` invalidates **all** refresh tokens belonging to the authenticated user. Use this to force sign-out from every device or client at once.

### Automatic Revocation

Refresh tokens are automatically invalidated in the following situations:

- **Password change** — all existing refresh tokens for the user are revoked when the password is updated.
- **Account deactivation** — all refresh tokens are revoked when a user's `is_active` flag is set to `false`.

## Token Details

| Property | Value |
|----------|-------|
| Algorithm | HS256 (configurable via `ALGORITHM`) |
| Signing key | `SECRET_KEY` environment variable |
| Access token expiration | `ACCESS_TOKEN_EXPIRE_MINUTES` (default 15 minutes) |
| Refresh token expiration | `REFRESH_TOKEN_EXPIRE_MINUTES` (default 1440 minutes = 24 hours) |

### Token Payload

The JWT payload contains:

| Claim | Description |
|-------|-------------|
| `sub` | Username |
| `id` | User ID |
| `role` | User role (`superuser`, `admin`, `user`) |
| `exp` | Expiration timestamp |
| `iat` | Issued-at timestamp (used for mass invalidation) |
| `jti` | Unique token ID (used for per-token blocklisting) |

## Unauthenticated Endpoints

The following operations can be performed without a token:

1. **`POST /api/v1/auth/token`** — login
2. **`POST /api/v1/users/`** — create the first user (bootstrap only, by default)
3. **`GET /crl/{slug}`** and **`GET /ca/{slug}.crt`** — public PKI endpoints (CRL and CA certificate downloads)

By default, unauthenticated user registration is **disabled** after the first user is created. Set `ALLOW_UNAUTHENTICATED_REGISTRATION=true` to allow unauthenticated creation of regular user accounts. Superuser and admin accounts always require authentication.

All other API endpoints require a valid bearer token.

## Password Policy

FastPKI follows [NIST SP 800-63B](https://pages.nist.gov/800-63-4/sp800-63b.html) guidance: favour length and block known-bad passwords instead of imposing composition rules. The policy is enforced whenever a password is **set or changed** — `POST /api/v1/users/` and `PATCH /api/v1/users/{user_id}` with a `password` field.

| Rule | Detail |
|------|--------|
| Minimum length | 12 characters (configurable via `PASSWORD_MIN_LENGTH`, allowed range 8–72) |
| Maximum length | 72 bytes once UTF-8 encoded (72 ASCII characters, fewer with multi-byte characters). bcrypt only hashes the first 72 bytes, so longer passwords are rejected rather than silently truncated. |
| Common passwords | Rejected if found (case-insensitively) in a bundled list of ~47,000 common/breached passwords, derived from the [SecLists](https://github.com/danielmiessler/SecLists) `100k-most-used-passwords-NCSC.txt` and `10k-most-common.txt` lists (MIT License). The list ships with the server; no network lookup is made. |
| Personal information | Rejected if it equals or contains the username or the local part of the email address (case-insensitive; identifiers shorter than 3 characters are ignored). |
| Composition | **No** character-class requirements (upper/lower/digit/symbol). Passphrases such as `tangerine-orbit-velvet-42` are encouraged. |

A violation returns `422 Unprocessable Entity` with a message describing the rule, for example:

```json
{
    "detail": [
        {
            "type": "value_error",
            "loc": ["body", "password"],
            "msg": "Value error, Password is too common; choose a less predictable password",
            "input": "[redacted]"
        }
    ]
}
```

Password values (and refresh tokens) are always redacted from `422` responses.

!!! note "Existing passwords"
    The policy is not applied at login. Users whose password was set before the policy was introduced can still log in, and are only held to the policy the next time they change it.

## Failed Login Attempts

Failed login attempts are recorded in the [audit log](../guides/audit-logs.md) with action `login_failure`, along with the attempted username. Successful logins are recorded with `login_success`.
