# FastPKI

**API-based PKI management system — an easier alternative to Easy-RSA.**

FastPKI lets you create and manage Certificate Authorities, issue certificates, and revoke them through a RESTful API. It supports full CA hierarchies, CRL generation, organization-based multi-tenancy, and role-based access control.

---

## Features

| Category | Details |
|----------|---------|
| **Certificate Authorities** | Root and intermediate CAs, path length constraints, chain of trust |
| **Certificates** | Server, client, and CA certificates with configurable key sizes and validity |
| **CRL & Public PKI** | CRL generation, public `/crl/` and `/ca/` download endpoints, CDP/AIA extensions embedded in certificates |
| **Access Control** | Three roles (Superuser, Admin, User), per-user capability flags, organization-scoped access |
| **Security** | JWT authentication, optional private key encryption at rest (Fernet), audit logging |
| **Database** | SQLite (development) and PostgreSQL (production), Alembic migrations |
| **CLI Tool** | Full-featured `fastpki` command-line interface, XDG config, table/JSON output |
| **Deployment** | Docker images on ghcr.io, automatic migrations on container startup |

## Quick links

- [Installation](getting-started/installation.md) — get FastPKI running locally or in Docker
- [First Steps](getting-started/first-steps.md) — create your first CA and issue a certificate
- [CRL & Public PKI](guides/crl.md) — CRL generation and public certificate/CRL download endpoints
- [API Reference](reference/api.md) — complete endpoint documentation
- [CLI Tool](guides/cli.md) — manage PKI from the command line
- [Docker Deployment](deployment/docker.md) — container-based deployment
