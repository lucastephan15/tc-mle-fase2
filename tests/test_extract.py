import io
import zipfile

import pytest

from purchase_propensity.extract import read_member


def make_zip(files: dict[str, bytes]) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zipped:
        for name, content in files.items():
            zipped.writestr(name, content)
    buffer.seek(0)
    return buffer


def test_devolve_o_conteudo_byte_a_byte():
    # \r\n e acento: nenhuma conversão de texto pode mudar o MD5 do CSV.
    content = b"Month,Revenue\r\nF\xc3\xa9v,FALSE\r\n"
    assert read_member(make_zip({"dados.csv": content}), "dados.csv") == content


def test_le_so_o_arquivo_pedido():
    archive = make_zip({"a.csv": b"a", "b.csv": b"b"})
    assert read_member(archive, "b.csv") == b"b"


def test_arquivo_ausente_no_zip_levanta_erro():
    with pytest.raises(KeyError):
        read_member(make_zip({"outro.csv": b"x"}), "dados.csv")
