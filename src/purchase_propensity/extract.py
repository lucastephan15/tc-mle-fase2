"""Estágio `extract`: do zip baixado da UCI ao CSV bruto.

A UCI distribui o dataset em zip. O `dvc import-url` traz o zip com a URL de origem
registrada, e este estágio grava o CSV de dentro dele sem alterar nenhum byte. A extração
é em Python porque a imagem `python:*-slim` não tem `unzip`.
"""

import zipfile
from typing import BinaryIO

from purchase_propensity.schema import RAW_ARCHIVE_MEMBER, RAW_ARCHIVE_PATH, RAW_DATA_PATH


def read_member(archive: BinaryIO, member: str) -> bytes:
    """Lê um arquivo de dentro do zip, byte a byte.

    Args:
        archive: Zip aberto em modo binário.
        member: Nome do arquivo dentro do zip.

    Returns:
        O conteúdo do arquivo, sem conversão de texto ou de fim de linha.

    Raises:
        KeyError: Se o zip não tem o arquivo, por exemplo se a UCI mudou o pacote.
    """
    with zipfile.ZipFile(archive) as zipped:
        return zipped.read(member)


def main() -> None:
    """Extrai o CSV bruto de `data/raw/` a partir do zip importado."""
    with RAW_ARCHIVE_PATH.open("rb") as archive:
        RAW_DATA_PATH.write_bytes(read_member(archive, RAW_ARCHIVE_MEMBER))
