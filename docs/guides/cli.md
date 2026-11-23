# CLI Tool

FastPKI ships with a command-line interface that provides full access to the API from your terminal.

## Installation

Install from PyPI:

```bash
pip install fastpki
```

Or with uv:

```bash
uv tool install fastpki
```

This installs the `fastpki` command.

## Configuration

The CLI stores settings in `$XDG_CONFIG_HOME/fastpki/config.json` (defaults to `~/.config/fastpki/config.json`).

### Set the server URL

```bash
fastpki config set server.url http://localhost:8000
```

### Set default values

```bash
fastpki config set defaults.ca_key_size 4096
fastpki config set defaults.ca_valid_days 3650
fastpki config set defaults.cert_key_size 2048
fastpki config set defaults.cert_valid_days 365
fastpki config set defaults.output_format table   # or json
```

### View configuration

```bash
fastpki config show
```

### Other config commands

```bash
fastpki config get server.url     # Get a single value
fastpki config unset defaults.ca_key_size   # Remove a value
fastpki config path               # Show config file location
```

## Authentication

### Login

```bash
fastpki auth login -u admin -p tangerine-orbit-velvet-42
```

You can also set the server during login:

```bash
fastpki auth login -u admin -p tangerine-orbit-velvet-42 -s https://pki.example.com
```

If you omit `-u` or `-p`, you will be prompted interactively (password input is hidden).

### Check status

```bash
fastpki auth status
```

Shows the current server, user, role, and whether the token is still valid.

### Logout

```bash
fastpki auth logout
```

## Certificate Authorities

### List CAs

```bash
fastpki ca list
```

### Create a root CA

```bash
fastpki ca create \
  --name "Production Root CA" \
  --subject-dn "CN=Production Root CA,O=Acme Corp,C=US" \
  --key-size 4096 \
  --valid-days 3650
```

### Create an intermediate CA

```bash
fastpki ca create \
  --name "Issuing CA" \
  --subject-dn "CN=Issuing CA,O=Acme Corp,C=US" \
  --parent 1 \
  --valid-days 1825
```

### Show CA details

```bash
fastpki ca show 1
```

### Show CA with private key

```bash
fastpki ca private-key 1
```

### View CA chain

```bash
fastpki ca chain 2
```

### List child CAs

```bash
fastpki ca children 1
```

### Delete a CA

```bash
fastpki ca delete 1          # prompts for confirmation
fastpki ca delete 1 --force  # skip confirmation
```

## Certificates

### List certificates

```bash
fastpki cert list
fastpki cert list --ca 1        # filter by issuing CA
fastpki cert list --limit 50    # pagination
```

### Issue a certificate

```bash
fastpki cert create \
  --ca 1 \
  --cn web.example.com \
  --subject-dn "CN=web.example.com,O=Acme Corp,C=US" \
  --type server
```

Other options:

| Flag | Description |
|------|-------------|
| `--type` / `-t` | `server`, `client`, or `ca` (default: `server`) |
| `--key-size` / `-k` | RSA key size (uses config default) |
| `--valid-days` / `-v` | Validity period (uses config default) |
| `--no-private-key` | Don't generate a private key |
| `--san-dns` | DNS SAN entry (repeatable) |
| `--san-ip` | IP SAN entry (repeatable) |
| `--san-email` | Email SAN entry (repeatable) |

### Sign a CSR

Submit a CSR produced elsewhere (the private key never leaves the requesting host).

```bash
fastpki cert sign-csr ./api.example.com.csr \
  --ca-name "Acme Issuing CA" \
  --type server
```

Use `--ca <id>` to select the CA by ID instead of name. Defaults (subject, SANs, public key) are extracted from the CSR; pass `--cn`, `--subject-dn`, `--san-dns`, `--san-ip`, or `--san-email` to override them. The response never includes a private key.

### Show certificate details

```bash
fastpki cert show 1
```

### Show certificate with private key

```bash
fastpki cert private-key 1
```

### Revoke a certificate

```bash
fastpki cert revoke 1 --reason "Key compromised"
fastpki cert revoke 1 --force   # skip confirmation
```

## Organizations

### List organizations

```bash
fastpki org list
```

### Create an organization

```bash
fastpki org create --name "Engineering" --description "Engineering team"
```

### Show organization details

```bash
fastpki org show 1
```

### Update an organization

```bash
fastpki org update 1 --name "Platform Engineering"
```

### Delete an organization

```bash
fastpki org delete 1
```

### Manage members

```bash
fastpki org add-user 1 2       # add user #2 to org #1
fastpki org remove-user 1 2    # remove user #2 from org #1
fastpki org users 1            # list users in org #1
```

## Users

### List users

```bash
fastpki user list
```

### Show current user

```bash
fastpki user me
```

### Create a user

```bash
fastpki user create \
  --username alice \
  --email alice@example.com \
  --password sunflower-harbor-velvet-7 \
  --role user \
  --can-create-cert
```

Capability flags: `--can-create-ca`, `--can-create-cert`, `--can-revoke-cert`, `--can-export-key`, `--can-delete-ca`.

### Update a user

```bash
fastpki user update 2 --role admin --can-create-ca
fastpki user update 2 --active       # activate
fastpki user update 2 --inactive     # deactivate
```

Toggle capabilities off with `--no-*` flags:

```bash
fastpki user update 2 --no-create-ca --no-delete-ca
```

### Delete a user

```bash
fastpki user delete 2
```

## Exporting

Download certificates and keys as PEM files:

```bash
fastpki export ca-cert 1                  # CA certificate
fastpki export ca-key 1                   # CA private key
fastpki export cert 1                     # Certificate
fastpki export cert-key 1                 # Certificate private key
fastpki export cert-chain 1              # Full certificate chain
```

All commands accept `--output` / `-o` to specify the output file path:

```bash
fastpki export ca-cert 1 -o /etc/ssl/ca.pem
```

## Audit Logs

```bash
fastpki audit list
fastpki audit list --action CA_CREATE
fastpki audit list --user 1
fastpki audit list --since 2025-01-01T00:00:00 --until 2025-02-01T00:00:00
fastpki audit list --resource-type certificate --resource-id 5
```

## Output Formats

All list and detail commands support `--output` / `-o` to switch between table and JSON output:

```bash
fastpki ca list -o json
fastpki cert show 1 -o json
```

Set a persistent default:

```bash
fastpki config set defaults.output_format json
```
