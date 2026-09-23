"""
Object storage abstraction for uploaded files.

Backends:
* **local** -- the container filesystem (development); point ``LOCAL_STORAGE_DIR``
  at a mounted volume in production if you are not using cloud storage.
* **s3**    -- Amazon S3 (also works with MinIO / LocalStack via ``S3_ENDPOINT_URL``)
* **azure** -- Azure Blob Storage

The API never assumes a local disk, which keeps the service stateless.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import NamedTuple, Optional

logger = logging.getLogger("retriva.storage")

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class StoredObject(NamedTuple):
    key: str
    size: int
    backend: str


def build_object_key(filename: str, user_id: str = "global") -> str:
    """Namespaced, collision-free object key: ``<user>/<timestamp>_<rand>_<name>``."""
    safe = _UNSAFE.sub("_", os.path.basename(filename or "document.pdf")).strip("_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{user_id}/{stamp}_{uuid.uuid4().hex[:8]}_{safe}"


class BaseStorage:
    backend = "base"

    def save(self, key: str, data: bytes, content_type: Optional[str] = None) -> StoredObject:
        raise NotImplementedError

    def load(self, key: str) -> bytes:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError


class LocalStorage(BaseStorage):
    """Writes under ``root``; use a mounted volume to make it durable."""

    backend = "local"

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def _path(self, key: str) -> str:
        path = os.path.abspath(os.path.join(self.root, key))
        if not path.startswith(self.root):  # defend against path traversal
            raise ValueError("Invalid object key")
        return path

    def save(self, key, data, content_type=None) -> StoredObject:
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(data)
        return StoredObject(key, len(data), self.backend)

    def load(self, key) -> bytes:
        with open(self._path(key), "rb") as handle:
            return handle.read()

    def exists(self, key) -> bool:
        return os.path.exists(self._path(key))

    def delete(self, key) -> None:
        try:
            os.remove(self._path(key))
        except FileNotFoundError:
            pass


class S3Storage(BaseStorage):
    backend = "s3"

    def __init__(self, bucket: str, prefix: str = "", region: str = "", endpoint_url: str = ""):
        if not bucket:
            raise ValueError("S3_BUCKET is required when STORAGE_BACKEND=s3")
        import boto3

        self.bucket = bucket
        self.prefix = (prefix or "").lstrip("/")
        self.client = boto3.client(
            "s3",
            region_name=region or None,
            endpoint_url=endpoint_url or None,
        )

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}" if self.prefix else key

    def save(self, key, data, content_type=None) -> StoredObject:
        target = self._key(key)
        kwargs = {"Bucket": self.bucket, "Key": target, "Body": data}
        if content_type:
            kwargs["ContentType"] = content_type
        self.client.put_object(**kwargs)
        return StoredObject(target, len(data), self.backend)

    def load(self, key) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        return response["Body"].read()

    def exists(self, key) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except ClientError:
            return False

    def delete(self, key) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=self._key(key))


class AzureBlobStorage(BaseStorage):
    backend = "azure"

    def __init__(self, connection_string: str, container: str, prefix: str = ""):
        if not connection_string:
            raise ValueError(
                "AZURE_STORAGE_CONNECTION_STRING is required when STORAGE_BACKEND=azure"
            )
        from azure.storage.blob import BlobServiceClient

        self.prefix = (prefix or "").lstrip("/")
        self.container = container or "uploads"
        self.service = BlobServiceClient.from_connection_string(connection_string)
        try:
            self.service.create_container(self.container)
        except Exception:  # noqa: BLE001 - already exists
            pass

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}" if self.prefix else key

    def save(self, key, data, content_type=None) -> StoredObject:
        from azure.storage.blob import ContentSettings

        blob = self.service.get_blob_client(self.container, self._key(key))
        blob.upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type)
            if content_type
            else None,
        )
        return StoredObject(self._key(key), len(data), self.backend)

    def load(self, key) -> bytes:
        blob = self.service.get_blob_client(self.container, self._key(key))
        return blob.download_blob().readall()

    def exists(self, key) -> bool:
        blob = self.service.get_blob_client(self.container, self._key(key))
        return bool(blob.exists())

    def delete(self, key) -> None:
        blob = self.service.get_blob_client(self.container, self._key(key))
        try:
            blob.delete_blob()
        except Exception:  # noqa: BLE001
            pass


def build_storage(config) -> BaseStorage:
    """Factory driven by ``STORAGE_BACKEND``."""
    backend = (config.STORAGE_BACKEND or "local").strip().lower()
    if backend == "local":
        logger.info("Object storage: local (%s)", config.LOCAL_STORAGE_DIR)
        return LocalStorage(config.LOCAL_STORAGE_DIR)
    if backend == "s3":
        logger.info("Object storage: s3 (bucket=%s)", config.S3_BUCKET)
        return S3Storage(
            config.S3_BUCKET,
            prefix=config.S3_PREFIX,
            region=config.AWS_REGION,
            endpoint_url=config.S3_ENDPOINT_URL,
        )
    if backend == "azure":
        logger.info("Object storage: azure blob (container=%s)", config.AZURE_STORAGE_CONTAINER)
        return AzureBlobStorage(
            config.AZURE_STORAGE_CONNECTION_STRING,
            config.AZURE_STORAGE_CONTAINER,
            prefix=config.S3_PREFIX,
        )
    raise ValueError(f"Unknown STORAGE_BACKEND: {config.STORAGE_BACKEND!r}")
