# CLI Reference

The `fastpki` CLI is installed with `pip install fastpki[cli]`.

## Command Overview

| Command | Description |
|---------|-------------|
| `fastpki auth login` | Authenticate and store access token |
| `fastpki auth logout` | Clear stored token |
| `fastpki auth status` | Show current auth status |
| `fastpki ca list` | List certificate authorities |
| `fastpki ca show <id>` | Show CA details |
| `fastpki ca create` | Create a new CA |
| `fastpki ca assign-org <id>` | Assign a CA to an organization (superuser) |
| `fastpki ca delete <id>` | Delete a CA |
| `fastpki ca chain <id>` | Show CA chain to root |
| `fastpki ca children <id>` | List child CAs |
| `fastpki ca private-key <id>` | Show CA with private key |
| `fastpki cert list` | List certificates |
| `fastpki cert show <id>` | Show certificate details |
| `fastpki cert create` | Issue a new certificate |
| `fastpki cert sign-csr <csr-file>` | Sign a PEM-encoded CSR |
| `fastpki cert revoke <id>` | Revoke a certificate |
| `fastpki cert private-key <id>` | Show certificate with private key |
| `fastpki org list` | List organizations |
| `fastpki org show <id>` | Show organization details |
| `fastpki org create` | Create an organization |
| `fastpki org update <id>` | Update an organization |
| `fastpki org delete <id>` | Delete an organization |
| `fastpki org add-user <org_id> <user_id>` | Add user to organization |
| `fastpki org remove-user <org_id> <user_id>` | Remove user from organization |
| `fastpki org users <id>` | List users in organization |
| `fastpki user list` | List all users |
| `fastpki user me` | Show current user |
| `fastpki user show <id>` | Show user details |
| `fastpki user create` | Create a user |
| `fastpki user update <id>` | Update a user |
| `fastpki user delete <id>` | Delete a user |
| `fastpki export ca-cert <id>` | Download CA certificate PEM |
| `fastpki export ca-key <id>` | Download CA private key PEM |
| `fastpki export cert <id>` | Download certificate PEM |
| `fastpki export cert-key <id>` | Download certificate private key PEM |
| `fastpki export cert-chain <id>` | Download certificate chain PEM |
| `fastpki audit list` | List audit log entries |
| `fastpki config show` | Show all configuration |
| `fastpki config get <key>` | Get a config value |
| `fastpki config set <key> <value>` | Set a config value |
| `fastpki config unset <key>` | Remove a config value |
| `fastpki config path` | Show config file path |

## Configuration Keys

Settings are stored in `$XDG_CONFIG_HOME/fastpki/config.json`.

| Key | Type | Description |
|-----|------|-------------|
| `server.url` | string | API server URL (default: `http://localhost:8000`) |
| `auth.token` | string | Stored JWT access token (set by `auth login`) |
| `defaults.output_format` | string | Default output format: `table` or `json` |
| `defaults.ca_key_size` | int | Default CA key size in bits |
| `defaults.ca_valid_days` | int | Default CA validity in days |
| `defaults.cert_key_size` | int | Default certificate key size in bits |
| `defaults.cert_valid_days` | int | Default certificate validity in days |

## API-to-CLI Mapping

Every API endpoint has a corresponding CLI command:

| API Endpoint | CLI Command |
|-------------|-------------|
| `POST /api/v1/auth/token` | `fastpki auth login` |
| `GET /api/v1/users/me` | `fastpki user me` |
| `POST /api/v1/users/` | `fastpki user create` |
| `GET /api/v1/users/` | `fastpki user list` |
| `GET /api/v1/users/{id}` | `fastpki user show` |
| `PATCH /api/v1/users/{id}` | `fastpki user update` |
| `DELETE /api/v1/users/{id}` | `fastpki user delete` |
| `POST /api/v1/cas/` | `fastpki ca create` |
| `GET /api/v1/cas/` | `fastpki ca list` |
| `GET /api/v1/cas/{id}` | `fastpki ca show` |
| `GET /api/v1/cas/{id}/private-key` | `fastpki ca private-key` |
| `GET /api/v1/cas/{id}/chain` | `fastpki ca chain` |
| `GET /api/v1/cas/{id}/children` | `fastpki ca children` |
| `PATCH /api/v1/cas/{id}` | `fastpki ca assign-org` |
| `DELETE /api/v1/cas/{id}` | `fastpki ca delete` |
| `POST /api/v1/certificates/` | `fastpki cert create` |
| `POST /api/v1/certificates/sign-csr` | `fastpki cert sign-csr` |
| `GET /api/v1/certificates/` | `fastpki cert list` |
| `GET /api/v1/certificates/{id}` | `fastpki cert show` |
| `GET /api/v1/certificates/{id}/private-key` | `fastpki cert private-key` |
| `POST /api/v1/certificates/{id}/revoke` | `fastpki cert revoke` |
| `GET /api/v1/export/ca/{id}/certificate` | `fastpki export ca-cert` |
| `GET /api/v1/export/ca/{id}/private-key` | `fastpki export ca-key` |
| `GET /api/v1/export/certificate/{id}` | `fastpki export cert` |
| `GET /api/v1/export/certificate/{id}/private-key` | `fastpki export cert-key` |
| `GET /api/v1/export/certificate/{id}/chain` | `fastpki export cert-chain` |
| `POST /api/v1/organizations/` | `fastpki org create` |
| `GET /api/v1/organizations/` | `fastpki org list` |
| `GET /api/v1/organizations/{id}` | `fastpki org show` |
| `PUT /api/v1/organizations/{id}` | `fastpki org update` |
| `DELETE /api/v1/organizations/{id}` | `fastpki org delete` |
| `POST /api/v1/organizations/{id}/users/{uid}` | `fastpki org add-user` |
| `DELETE /api/v1/organizations/{id}/users/{uid}` | `fastpki org remove-user` |
| `GET /api/v1/organizations/{id}/users` | `fastpki org users` |
| `GET /api/v1/audit-logs/` | `fastpki audit list` |
