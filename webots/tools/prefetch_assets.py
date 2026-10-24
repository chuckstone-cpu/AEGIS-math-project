"""Pre-fill Webots' asset cache for a world, so Webots never has to download.

Webots fetches PROTOs/textures/meshes from GitHub at load time and caches each
file as  %LOCALAPPDATA%/Cyberbotics/Webots/cache/assets/<sha1(url)>.
On some networks its downloader stalls ("Downloading assets 0%"), so this
script crawls the world's dependency graph and downloads everything itself.

    python webots/tools/prefetch_assets.py webots/worlds/apex_city.wbt
"""
import concurrent.futures as cf
import hashlib
import os
import posixpath
import re
import sys
import urllib.request

BASE = "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/"
CACHE = os.path.join(os.environ["LOCALAPPDATA"], "Cyberbotics", "Webots", "cache", "assets")
EXT = r"(?:proto|png|jpg|jpeg|hdr|obj|dae|stl|wav|mp3|ogg|bvh|stl)"
QUOTED = re.compile(r'"([^"\s]+\.' + EXT + r')"', re.I)


def cache_path(url):
    return os.path.join(CACHE, hashlib.sha1(url.encode()).hexdigest())


def normalise(ref, parent_url):
    if ref.startswith("webots://"):
        return BASE + ref[len("webots://"):]
    if ref.startswith("http://") or ref.startswith("https://"):
        return ref
    if parent_url is None:
        return None
    base_dir = parent_url.rsplit("/", 1)[0]
    return posixpath.normpath(base_dir + "/" + ref).replace("https:/", "https://", 1)


def fetch(url):
    path = cache_path(url)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return url, open(path, "rb").read(), False
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read()
    except Exception as e:  # missing optional asset (e.g. template-only path)
        return url, None, str(e)
    with open(path, "wb") as f:
        f.write(data)
    return url, data, True


def refs_in(text, parent_url):
    out = set()
    for m in QUOTED.finditer(text):
        u = normalise(m.group(1), parent_url)
        if u and u.startswith("https://"):
            out.add(u)
    return out


def main():
    world = sys.argv[1]
    os.makedirs(CACHE, exist_ok=True)
    todo = refs_in(open(world, encoding="utf-8").read(), None)
    seen, new, failed = set(), 0, []
    with cf.ThreadPoolExecutor(max_workers=16) as pool:
        while todo:
            batch = [u for u in todo if u not in seen]
            seen.update(batch)
            todo = set()
            for url, data, status in pool.map(fetch, batch):
                if data is None:
                    failed.append((url, status))
                    continue
                new += status is True
                if url.lower().endswith(".proto"):
                    todo |= refs_in(data.decode("utf-8", "replace"), url)
    print(f"{len(seen)} assets referenced, {new} downloaded now, "
          f"{len(seen) - new - len(failed)} already cached, {len(failed)} unavailable")
    for u, e in failed[:15]:
        print("  unavailable:", u.replace(BASE, ""), "-", e)


if __name__ == "__main__":
    main()
