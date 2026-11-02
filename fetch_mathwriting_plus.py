"""Download the start of MathWriting+.zip from Zenodo and the zip's file list, the two files prepare_handwriting.py reads.

The zip holds Symbols, Test, Val, Train and Synthetic in that order, so its first 260 MB have all the test and validation inks
and about 59,000 of the training inks, which is more than training needs. Zenodo can be slow per connection and drops long
downloads, so everything comes in small byte ranges over several connections. Run it again to finish after a break.

Usage: python fetch_mathwriting_plus.py --out <folder> [--megabytes 260]
"""
import argparse
import http.client
import json
import pathlib
import struct
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://zenodo.org/records/15091575/files/MathWriting+.zip?download=1"
PIECE = 2 ** 20
CONNECTIONS = 12


def fetch(start, end):
    """Bytes start..end of the zip, trying again after any break."""
    while True:
        request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}", "User-Agent": "curl/8"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if len(data) == end - start + 1:
                return data
        except (OSError, http.client.HTTPException) as error:
            print("trying again:", repr(error)[:80], flush=True)
        time.sleep(5)


def zip_rows(directory):
    """[name, method, packed size, plain size, offset] for each file in a zip's central directory."""
    rows, i = [], 0
    while directory[i:i + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", directory[i + 10:i + 12])[0]
        packed, plain = struct.unpack("<II", directory[i + 20:i + 28])
        name_len, extra_len, note_len = struct.unpack("<HHH", directory[i + 28:i + 34])
        offset = struct.unpack("<I", directory[i + 42:i + 46])[0]
        rows.append([directory[i + 46:i + 46 + name_len].decode("utf-8"), method, packed, plain, offset])
        i += 46 + name_len + extra_len + note_len
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--megabytes", type=int, default=260)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    file_list = out / "directory.json"
    with ThreadPoolExecutor(CONNECTIONS) as pool:
        if not file_list.exists():
            request = urllib.request.Request(URL, headers={"Range": "bytes=0-0", "User-Agent": "curl/8"})
            with urllib.request.urlopen(request, timeout=120) as response:
                total = int(response.headers["Content-Range"].rsplit("/", 1)[1])
            tail = fetch(total - 2 ** 16, total - 1)
            end_record = tail.rfind(b"PK\x05\x06")
            size, start = struct.unpack("<II", tail[end_record + 12:end_record + 20])
            pieces = range(start, start + size, PIECE)
            directory = b"".join(pool.map(lambda at: fetch(at, min(at + PIECE, start + size) - 1), pieces))
            file_list.write_text(json.dumps({"total": total, "entries": zip_rows(directory)}), encoding="utf-8")
            print("file list saved", flush=True)

        blob, part = out / "MathWriting+.start.bin", out / "MathWriting+.start.part"
        have = json.loads((out / "pieces.json").read_text()) if (out / "pieces.json").exists() else []
        wanted = args.megabytes * 10 ** 6
        if not blob.exists():
            if not part.exists():
                with open(part, "wb") as f:
                    f.truncate(wanted)
            todo = [at for at in range(0, wanted, PIECE) if at not in have]
            with open(part, "r+b") as f:
                for done, (at, data) in enumerate(pool.map(lambda at: (at, fetch(at, min(at + PIECE, wanted) - 1)), todo), 1):
                    f.seek(at)
                    f.write(data)
                    have.append(at)
                    (out / "pieces.json").write_text(json.dumps(have))
                    print(f"{done}/{len(todo)} pieces", flush=True)
            part.replace(blob)
            (out / "pieces.json").unlink()
    print("complete:", blob, file_list)


if __name__ == "__main__":
    main()
