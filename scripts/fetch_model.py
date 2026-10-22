#!/usr/bin/env python3
"""Stream model weights from an HF mirror straight into the target files.

huggingface_hub downloads into a temporary file and unlinks it afterwards, which
this sandbox refuses. This script writes each file in place, resumes from
`<name>.part`, and never deletes anything.
"""
import json
import os
import sys
import time
import urllib.request

REPO = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
DEST = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/models/DeepSeek-R1-Distill-Qwen-1.5B"
)
# hf-mirror answers 403 to the default Python-urllib user agent.
UA = "llaisys-setup/1.0"


def open_url(url, **kwargs):
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(request, timeout=kwargs.pop("timeout", 30), **kwargs)


def repo_info():
    url = f"{ENDPOINT}/api/models/{REPO}?blobs=true"
    with open_url(url) as response:
        return json.load(response)


def head_size(name):
    url = f"{ENDPOINT}/{REPO}/resolve/main/{name}"
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=30) as response:
        return int(response.headers.get("Content-Length", 0))


def human(n):
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def fetch(name, size, dest):
    target = os.path.join(dest, name)
    if os.path.exists(target) and os.path.getsize(target) == size:
        print(f"  skip {name} (already {human(size)})", flush=True)
        return

    url = f"{ENDPOINT}/{REPO}/resolve/main/{name}"
    part = target + ".part"
    print(f"  get  {name} ({human(size)})", flush=True)

    for attempt in range(1, 41):
        done = os.path.getsize(part) if os.path.exists(part) else 0
        if done > size:
            done = 0
        if done == size:
            break

        request = urllib.request.Request(url, headers={"User-Agent": UA})
        if done:
            request.add_header("Range", f"bytes={done}-")
        try:
            response = urllib.request.urlopen(request, timeout=60)
            mode = "ab"
            if done and response.status != 206:
                print("       server ignored the range request, restarting", flush=True)
                mode, done = "wb", 0
            start, base = time.time(), done
            with response, open(part, mode) as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    elapsed = max(time.time() - start, 1e-6)
                    print(
                        f"\r       {done / 1e6:8.1f} / {size / 1e6:.1f} MB "
                        f"({(done - base) / elapsed / 1e6:5.1f} MB/s)",
                        end="",
                        flush=True,
                    )
            print(flush=True)
        except Exception as error:  # noqa: BLE001 - mirrors drop connections mid-stream
            print(f"\n       {type(error).__name__}: {error} - retrying from {human(done)}",
                  flush=True)
            time.sleep(min(2 * attempt, 15))

    if os.path.getsize(part) != size:
        raise SystemExit(f"incomplete: {part} has {os.path.getsize(part)}, want {size}")


def main():
    info = repo_info()
    wanted = []
    for sibling in info.get("siblings", []):
        name = sibling["rfilename"]
        if "/" in name or not name.endswith((".json", ".safetensors", ".txt", ".model")):
            continue
        wanted.append((name, sibling.get("size") or head_size(name)))

    os.makedirs(DEST, exist_ok=True)
    print(f"downloading {len(wanted)} files into {DEST}", flush=True)

    for name, size in sorted(wanted):
        fetch(name, size, DEST)

    # Publish the finished files. rename() can be denied in restricted
    # environments; a hard link achieves the same result without copying bytes,
    # at the cost of leaving the .part entry behind.
    for name in sorted(os.listdir(DEST)):
        if not name.endswith(".part"):
            continue
        final = os.path.join(DEST, name[:-5])
        source = os.path.join(DEST, name)
        if os.path.exists(final):
            continue
        try:
            os.rename(source, final)
            print(f"  done {name[:-5]}", flush=True)
        except OSError:
            os.link(source, final)
            print(f"  linked {name[:-5]} (rename denied; {name} can be removed)", flush=True)

    print("files on disk:", flush=True)
    for name in sorted(os.listdir(DEST)):
        path = os.path.join(DEST, name)
        if os.path.isfile(path) and not name.endswith(".part"):
            print(f"  {os.path.getsize(path) / 1e6:9.1f} MB  {name}", flush=True)


if __name__ == "__main__":
    main()
