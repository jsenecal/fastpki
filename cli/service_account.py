"""Service account commands."""

import typer

from cli import client
from cli.output import (
    display_detail,
    display_list,
    output_option,
    set_format_override,
)

app = typer.Typer(no_args_is_help=True)
token_app = typer.Typer(no_args_is_help=True, help="Manage service-account tokens")
app.add_typer(token_app, name="token")
policy_app = typer.Typer(no_args_is_help=True, help="Manage issuance policies")
app.add_typer(policy_app, name="policy")

BASE = "/api/v1/service-accounts"

SA_LIST_COLUMNS = ["ID", "Name", "Description", "Disabled At"]
SA_LIST_KEYS = ["id", "name", "description", "disabled_at"]

SA_DETAIL_FIELDS = [
    ("ID", "id"),
    ("Name", "name"),
    ("Description", "description"),
    ("Organization", "organization_id"),
    ("Disabled At", "disabled_at"),
    ("Can Create CA", "can_create_ca"),
    ("Can Create Cert", "can_create_cert"),
    ("Can Revoke Cert", "can_revoke_cert"),
    ("Can Export Private Key", "can_export_private_key"),
    ("Can Delete CA", "can_delete_ca"),
    ("Created At", "created_at"),
]

TOKEN_LIST_COLUMNS = ["ID", "Public ID", "Name", "Last Used", "Expires", "Revoked"]
TOKEN_LIST_KEYS = ["id", "public_id", "name", "last_used_at", "expires_at", "revoked"]

TOKEN_DETAIL_FIELDS = [
    ("ID", "id"),
    ("Public ID", "public_id"),
    ("Name", "name"),
    ("Expires At", "expires_at"),
    ("Created At", "created_at"),
]

POLICY_DETAIL_FIELDS = [
    ("Service Account", "service_account_id"),
    ("CN Patterns", "cn_patterns"),
    ("SAN DNS Patterns", "san_dns_patterns"),
    ("SAN IP CIDRs", "san_ip_cidrs"),
    ("SAN Email Domains", "san_email_domains"),
    ("Allowed CA IDs", "allowed_ca_ids"),
    ("Allowed Cert Types", "allowed_certificate_types"),
    ("Max Validity Days", "max_validity_days"),
]


def _callback(output: str | None = output_option()) -> None:
    set_format_override(output)


app.callback(invoke_without_command=True)(_callback)


@app.command("list")
def list_service_accounts() -> None:
    """List service accounts in your organization."""
    data = client.get(f"{BASE}/").json()
    display_list(data, SA_LIST_COLUMNS, keys=SA_LIST_KEYS, title="Service Accounts")


@app.command()
def show(sa_id: int = typer.Argument(..., help="Service account ID")) -> None:
    """Show service account details."""
    data = client.get(f"{BASE}/{sa_id}").json()
    display_detail(data, SA_DETAIL_FIELDS, title=f"Service Account #{sa_id}")


@app.command()
def create(
    name: str = typer.Option(..., "--name", "-n", prompt=True),
    description: str | None = typer.Option(None, "--description", "-d"),
    organization_id: int | None = typer.Option(
        None, "--organization-id", help="Required only for superusers"
    ),
    can_create_ca: bool = typer.Option(False, "--can-create-ca"),
    can_create_cert: bool = typer.Option(False, "--can-create-cert"),
    can_revoke_cert: bool = typer.Option(False, "--can-revoke-cert"),
    can_export_private_key: bool = typer.Option(False, "--can-export-private-key"),
    can_delete_ca: bool = typer.Option(False, "--can-delete-ca"),
) -> None:
    """Create a service account."""
    payload: dict[str, object] = {
        "name": name,
        "can_create_ca": can_create_ca,
        "can_create_cert": can_create_cert,
        "can_revoke_cert": can_revoke_cert,
        "can_export_private_key": can_export_private_key,
        "can_delete_ca": can_delete_ca,
    }
    if description is not None:
        payload["description"] = description
    if organization_id is not None:
        payload["organization_id"] = organization_id
    data = client.post(f"{BASE}/", json=payload).json()
    display_detail(data, SA_DETAIL_FIELDS, title="Service Account Created")


@app.command()
def update(
    sa_id: int = typer.Argument(..., help="Service account ID"),
    name: str | None = typer.Option(None, "--name", "-n"),
    description: str | None = typer.Option(None, "--description", "-d"),
    disabled: bool | None = typer.Option(
        None, "--disable/--enable", help="Disable or enable the account"
    ),
    can_create_ca: bool | None = typer.Option(
        None, "--can-create-ca/--no-can-create-ca"
    ),
    can_create_cert: bool | None = typer.Option(
        None, "--can-create-cert/--no-can-create-cert"
    ),
    can_revoke_cert: bool | None = typer.Option(
        None, "--can-revoke-cert/--no-can-revoke-cert"
    ),
    can_export_private_key: bool | None = typer.Option(
        None, "--can-export-private-key/--no-can-export-private-key"
    ),
    can_delete_ca: bool | None = typer.Option(
        None, "--can-delete-ca/--no-can-delete-ca"
    ),
) -> None:
    """Update a service account. Only provided fields change."""
    fields = {
        "name": name,
        "description": description,
        "disabled": disabled,
        "can_create_ca": can_create_ca,
        "can_create_cert": can_create_cert,
        "can_revoke_cert": can_revoke_cert,
        "can_export_private_key": can_export_private_key,
        "can_delete_ca": can_delete_ca,
    }
    payload: dict[str, object] = {k: v for k, v in fields.items() if v is not None}
    if not payload:
        typer.echo("Nothing to update.")
        raise typer.Exit(1)
    data = client.patch(f"{BASE}/{sa_id}", json=payload).json()
    display_detail(data, SA_DETAIL_FIELDS, title=f"Service Account #{sa_id} Updated")


@app.command()
def delete(
    sa_id: int = typer.Argument(..., help="Service account ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a service account and all of its tokens."""
    if not force:
        typer.confirm(f"Delete service account #{sa_id} and its tokens?", abort=True)
    client.delete(f"{BASE}/{sa_id}")
    typer.echo(f"Service account #{sa_id} deleted.")


@token_app.command("create")
def token_create(
    sa_id: int = typer.Argument(..., help="Service account ID"),
    name: str | None = typer.Option(None, "--name", "-n", help="Token label"),
    expires_at: str | None = typer.Option(
        None, "--expires-at", help="ISO-8601 expiry, e.g. 2027-01-01T00:00:00Z"
    ),
) -> None:
    """Mint a token. The plaintext is shown only once — store it now."""
    payload: dict[str, object] = {}
    if name is not None:
        payload["name"] = name
    if expires_at is not None:
        payload["expires_at"] = expires_at
    data = client.post(f"{BASE}/{sa_id}/tokens", json=payload).json()
    token = data.pop("token")
    display_detail(data, TOKEN_DETAIL_FIELDS, title="Token Created")
    typer.secho(
        "\nStore this token now — it will not be shown again:", fg=typer.colors.YELLOW
    )
    typer.secho(token, fg=typer.colors.GREEN, bold=True)


@token_app.command("list")
def token_list(
    sa_id: int = typer.Argument(..., help="Service account ID"),
) -> None:
    """List tokens for a service account (metadata only)."""
    data = client.get(f"{BASE}/{sa_id}/tokens").json()
    display_list(
        data,
        TOKEN_LIST_COLUMNS,
        keys=TOKEN_LIST_KEYS,
        title=f"Tokens for Service Account #{sa_id}",
    )


@token_app.command("revoke")
def token_revoke(
    sa_id: int = typer.Argument(..., help="Service account ID"),
    token_id: int = typer.Argument(..., help="Token ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Revoke a service-account token."""
    if not force:
        typer.confirm(f"Revoke token #{token_id}?", abort=True)
    client.delete(f"{BASE}/{sa_id}/tokens/{token_id}")
    typer.echo(f"Token #{token_id} revoked.")


@policy_app.command("set")
def policy_set(
    sa_id: int = typer.Argument(..., help="Service account ID"),
    max_validity_days: int = typer.Option(..., "--max-validity-days"),
    cn_pattern: list[str] = typer.Option([], "--cn-pattern", help="Repeatable"),
    san_dns_pattern: list[str] = typer.Option(
        [], "--san-dns-pattern", help="Repeatable"
    ),
    san_ip_cidr: list[str] = typer.Option([], "--san-ip-cidr", help="Repeatable"),
    san_email_domain: list[str] = typer.Option(
        [], "--san-email-domain", help="Repeatable"
    ),
    allowed_ca_id: list[int] = typer.Option([], "--allowed-ca-id", help="Repeatable"),
    cert_type: list[str] = typer.Option(
        [], "--cert-type", help="server or client (repeatable)"
    ),
) -> None:
    """Create or replace the issuance policy (deny-by-default allowlist)."""
    payload: dict[str, object] = {
        "cn_patterns": cn_pattern,
        "san_dns_patterns": san_dns_pattern,
        "san_ip_cidrs": san_ip_cidr,
        "san_email_domains": san_email_domain,
        "allowed_ca_ids": allowed_ca_id,
        "allowed_certificate_types": cert_type,
        "max_validity_days": max_validity_days,
    }
    data = client.put(f"{BASE}/{sa_id}/policy", json=payload).json()
    display_detail(
        data, POLICY_DETAIL_FIELDS, title=f"Policy for Service Account #{sa_id}"
    )


@policy_app.command("show")
def policy_show(sa_id: int = typer.Argument(..., help="Service account ID")) -> None:
    """Show the issuance policy for a service account."""
    data = client.get(f"{BASE}/{sa_id}/policy").json()
    display_detail(
        data, POLICY_DETAIL_FIELDS, title=f"Policy for Service Account #{sa_id}"
    )


@policy_app.command("clear")
def policy_clear(
    sa_id: int = typer.Argument(..., help="Service account ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Clear the issuance policy (reverts the account to deny-all)."""
    if not force:
        typer.confirm(f"Clear policy for service account #{sa_id}?", abort=True)
    client.delete(f"{BASE}/{sa_id}/policy")
    typer.echo(f"Policy for service account #{sa_id} cleared.")
