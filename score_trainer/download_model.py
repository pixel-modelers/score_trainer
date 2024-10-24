import os
import requests
import tqdm
import hashlib

CHECKSUMS={"state_ep10.net": "a4ef4ef1d97e424eea8ed5665bda0675"}


def dl(url):
    # https://stackoverflow.com/a/16696317/2077270
    # https://stackoverflow.com/a/37573701/2077270
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get("content-length", 0))
    with tqdm.tqdm(total=total_size, unit="B", unit_scale=True, ascii=True) as progress_bar:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                progress_bar.update(len(chunk))
                f.write(chunk)
    return local_filename


def main():
    print("Downloading model file")
    for name in CHECKSUMS:
        f = dl(f"https://smb.slac.stanford.edu/~dermen/{name}")
        md5 = hashlib.md5(open(f, 'rb').read()).hexdigest()
        print(f"Checksum for {f}={md5}")
        assert md5 == CHECKSUMS[name]
        dirname = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        name = os.path.join(dirname, os.path.basename(f))
        os.rename(f, name)
        print(f"Model saved to {name}.")
