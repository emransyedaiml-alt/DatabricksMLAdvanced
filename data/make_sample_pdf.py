"""Create data/sample_documents/data_retention_policy.pdf without third-party packages.

Run: python data/make_sample_pdf.py
The PDF uses the built-in Helvetica font, so any PDF reader (and pypdf) can extract its text.
"""
from pathlib import Path

LINES = [
    "Data Retention and Access Policy (Version 2.0)",
    "",
    "1. Purpose: this policy defines how long lakehouse data is kept and who may access it.",
    "2. Bronze data is retained for 90 days, then deleted by the scheduled VACUUM job.",
    "3. Silver data is retained for 3 years. Gold data products are retained for 7 years.",
    "4. Personal data (names, e-mail addresses, phone numbers) must be masked in Gold tables.",
    "5. Access to Silver and Gold tables is granted through Unity Catalog groups only.",
    "6. Service principals used by pipelines receive the minimum privileges they need.",
    "7. Pipeline logs must not contain credentials, tokens or personal data.",
    "8. Exceptions require written approval from the Data Governance Board.",
]


def build_pdf(lines: list[str]) -> bytes:
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    text_ops = ["BT", "/F1 11 Tf", "14 TL", "72 760 Td"]
    for line in lines:
        text_ops.append(f"({esc(line)}) Tj T*")
    text_ops.append("ET")
    stream = "\n".join(text_ops).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


if __name__ == "__main__":
    target = Path(__file__).parent / "sample_documents" / "data_retention_policy.pdf"
    target.write_bytes(build_pdf(LINES))
    print("wrote", target)
