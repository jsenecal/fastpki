import logging

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings

logger = logging.getLogger("fastpki")


class EncryptionService:
    @staticmethod
    def _get_fernet() -> Fernet | None:
        key = settings.PRIVATE_KEY_ENCRYPTION_KEY
        if key is None:
            return None
        return Fernet(key.encode("utf-8"))

    @staticmethod
    def is_encrypted(data: str) -> bool:
        return not data.startswith("-----BEGIN")

    @staticmethod
    def encrypt_private_key(pem_text: str) -> str:
        fernet = EncryptionService._get_fernet()
        if fernet is None:
            return pem_text
        if EncryptionService.is_encrypted(pem_text):
            return pem_text
        return fernet.encrypt(pem_text.encode("utf-8")).decode("utf-8")

    @staticmethod
    def decrypt_private_key(data: str) -> str:
        if not EncryptionService.is_encrypted(data):
            return data
        fernet = EncryptionService._get_fernet()
        if fernet is None:
            raise ValueError(  # noqa: TRY003
                "Cannot decrypt private key: "
                "PRIVATE_KEY_ENCRYPTION_KEY is not configured"
            )
        try:
            return fernet.decrypt(data.encode("utf-8")).decode("utf-8")
        except InvalidToken as e:
            raise ValueError(  # noqa: TRY003
                "Cannot decrypt private key: wrong encryption key or corrupted data"
            ) from e

    @staticmethod
    def decrypt_optional_private_key(data: str | None) -> str | None:
        if data is None:
            return None
        return EncryptionService.decrypt_private_key(data)


async def encrypt_existing_keys() -> None:
    if settings.PRIVATE_KEY_ENCRYPTION_KEY is None:
        return

    from sqlmodel import select

    from app.db.models import Certificate, CertificateAuthority
    from app.db.session import engine

    async with engine.begin() as conn:
        from sqlalchemy.ext.asyncio import AsyncSession

        async with AsyncSession(bind=conn) as session:
            count = 0

            ca_result = await session.execute(select(CertificateAuthority))
            for ca in ca_result.scalars().all():
                if EncryptionService.is_encrypted(ca.private_key):
                    # Validate we can decrypt with the current key
                    try:
                        EncryptionService.decrypt_private_key(ca.private_key)
                    except ValueError as e:
                        raise RuntimeError(  # noqa: TRY003
                            f"Cannot decrypt private key for "
                            f"CA '{ca.name}' (id={ca.id}). "
                            "The PRIVATE_KEY_ENCRYPTION_KEY "
                            "may have changed."
                        ) from e
                else:
                    ca.private_key = EncryptionService.encrypt_private_key(
                        ca.private_key
                    )
                    session.add(ca)
                    count += 1

            cert_result = await session.execute(select(Certificate))
            for cert in cert_result.scalars().all():
                if cert.private_key is None:
                    continue
                if EncryptionService.is_encrypted(cert.private_key):
                    try:
                        EncryptionService.decrypt_private_key(cert.private_key)
                    except ValueError as e:
                        raise RuntimeError(  # noqa: TRY003
                            "Cannot decrypt private key for "
                            f"certificate '{cert.common_name}'"
                            f" (id={cert.id}). The "
                            "PRIVATE_KEY_ENCRYPTION_KEY "
                            "may have changed."
                        ) from e
                else:
                    cert.private_key = EncryptionService.encrypt_private_key(
                        cert.private_key
                    )
                    session.add(cert)
                    count += 1

            await session.flush()

    if count > 0:
        logger.info("Encrypted %d existing plaintext private keys", count)
