"""Certificate Authority commands."""

import typer

from cli import client
from cli.config import get_default
from cli.output import (
    display_detail,
    display_list,
    output_option,
    set_format_override,
)

app = typer.Typer(no_args_is_help=True)

CA_LIST_COLUMNS = [
    "ID",
    "Name",
    "Subject DN",
    "Key Size",
    "Valid Days",
    "Parent",
    "Root",
]
CA_LIST_KEYS = [
    "id",
    "name",
    "subject_dn",
    "key_size",
    "valid_days",
    "parent_ca_id",
    "is_root",
]

CA_DETAIL_FIELDS = [
    ("ID", "id"),
    ("Name", "name"),
    ("Description", "description"),
    ("Subject DN", "subject_dn"),
    ("Key Size", "key_size"),
    ("Valid Days", "valid_days"),
    ("Root", "is_root"),
    ("Parent CA", "parent_ca_id"),
    ("Path Length", "path_length"),
    ("Allow Leaf Certs", "allow_leaf_certs"),
    ("CRL Base URL", "crl_base_url"),
    ("Organization", "organization_id"),
    ("Created By", "created_by_user_id"),
    ("Created At", "created_at"),
    ("Updated At", "updated_at"),
]


def _callback(output: str | None = output_option()) -> None:
    set_format_override(output)


app.callback(invoke_without_command=True)(_callback)


@app.command("list")
def list_cas() -> None:
    """List all certificate authorities."""
    data = client.get("/api/v1/cas/").json()
    display_list(
        data, CA_LIST_COLUMNS, keys=CA_LIST_KEYS, title="Certificate Authorities"
    )


@app.command()
def show(ca_id: int = typer.Argument(..., help="CA ID")) -> None:
    """Show details of a certificate authority."""
    data = client.get(f"/api/v1/cas/{ca_id}").json()
    display_detail(data, CA_DETAIL_FIELDS, title=f"CA #{ca_id}")


@app.command()
def create(
    name: str = typer.Option(..., "--name", "-n", prompt=True),
    subject_dn: str = typer.Option(..., "--subject-dn", "-s", prompt=True),
    description: str | None = typer.Option(None, "--description", "-d"),
    key_size: int | None = typer.Option(None, "--key-size", "-k"),
    valid_days: int | None = typer.Option(None, "--valid-days", "-v"),
    parent_ca_id: int | None = typer.Option(
        None, "--parent", "-p", help="Parent CA ID for intermediate CA"
    ),
    path_length: int | None = typer.Option(None, "--path-length"),
    allow_leaf_certs: bool | None = typer.Option(None, "--allow-leaf-certs"),
    crl_base_url: str | None = typer.Option(None, "--crl-base-url"),
) -> None:
    """Create a new certificate authority."""
    payload: dict[str, object] = {"name": name, "subject_dn": subject_dn}
    if description is not None:
        payload["description"] = description
    ks = key_size or get_default("ca_key_size")
    if ks:
        payload["key_size"] = int(ks)
    vd = valid_days or get_default("ca_valid_days")
    if vd:
        payload["valid_days"] = int(vd)
    if parent_ca_id is not None:
        payload["parent_ca_id"] = parent_ca_id
    if path_length is not None:
        payload["path_length"] = path_length
    if allow_leaf_certs is not None:
        payload["allow_leaf_certs"] = allow_leaf_certs
    if crl_base_url is not None:
        payload["crl_base_url"] = crl_base_url

    data = client.post("/api/v1/cas/", json=payload).json()
    fields = [
        *CA_DETAIL_FIELDS,
        ("Certificate", "certificate"),
        ("Private Key", "private_key"),
    ]
    display_detail(data, fields, title="CA Created")


@app.command("assign-org")
def assign_org(
    ca_id: int = typer.Argument(..., help="CA ID"),
    organization_id: int = typer.Option(
        ..., "--org", "-o", help="Organization ID to assign the CA to"
    ),
    cascade: bool = typer.Option(
        False,
        "--cascade",
        help="Also adopt org-less descendant CAs and their issued certificates",
    ),
) -> None:
    """Assign a CA to an organization (superuser only)."""
    payload: dict[str, object] = {"organization_id": organization_id}
    if cascade:
        payload["cascade"] = True
    data = client.patch(f"/api/v1/cas/{ca_id}", json=payload).json()
    display_detail(data, CA_DETAIL_FIELDS, title=f"CA #{ca_id}")


@app.command()
def delete(
    ca_id: int = typer.Argument(..., help="CA ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a certificate authority."""
    if not force:
        typer.confirm(
            f"Delete CA #{ca_id}? This also deletes all its certificates.", abort=True
        )
    client.delete(f"/api/v1/cas/{ca_id}")
    typer.echo(f"CA #{ca_id} deleted.")


@app.command()
def chain(ca_id: int = typer.Argument(..., help="CA ID")) -> None:
    """Show the certificate chain from CA to root."""
    data = client.get(f"/api/v1/cas/{ca_id}/chain").json()
    display_list(
        data, CA_LIST_COLUMNS, keys=CA_LIST_KEYS, title=f"Chain for CA #{ca_id}"
    )


@app.command()
def children(ca_id: int = typer.Argument(..., help="CA ID")) -> None:
    """List direct child CAs."""
    data = client.get(f"/api/v1/cas/{ca_id}/children").json()
    display_list(
        data, CA_LIST_COLUMNS, keys=CA_LIST_KEYS, title=f"Children of CA #{ca_id}"
    )


@app.command("private-key")
def private_key(ca_id: int = typer.Argument(..., help="CA ID")) -> None:
    """Show CA details including private key."""
    data = client.get(f"/api/v1/cas/{ca_id}/private-key").json()
    fields = [
        *CA_DETAIL_FIELDS,
        ("Certificate", "certificate"),
        ("Private Key", "private_key"),
    ]
    display_detail(data, fields, title=f"CA #{ca_id} (with private key)")
