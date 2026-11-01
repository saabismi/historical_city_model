#!/usr/bin/env python3
"""
Download a year's MTK GeoPackages from CSC:
https://ftp.csc.fi/pub/sci/geo/geodata/mml/maastotietokanta/
check available years first, from the above link!

Examples:
    python download_mtk.py --year 2009
    python download_mtk.py --year 2009 --exclude suo virtavesi
    python download_mtk.py --year 2009 -o geodata/MTK_2009
    python download_mtk.py --year 2009 --list-only

Arguments:
    --year          Archive year to download.
    -o, --output    Output folder; default MTK_<year> in the working directory.
    --exclude       Additional dataset names to omit, e.g. suo virtavesi.
    --list-only     Show selection without downloading GeoPackages.
    --overwrite     Replace existing completed files.
    --archive       Override the archive root URL.

Always excluded:
    korkeussuhteet, maastokuvionreuna

Outputs:
    Selected GeoPackages and download_manifest.csv.

Existing files with the expected size are skipped. Downloads use .part
files, which are renamed only after successful completion. Interrupted
downloads restart from the beginning; this version does not resume them.

Requires Requests: pip install requests
"""

import argparse
import csv
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Plain HTTP matches the working CSC access method from the earlier scripts.
# HTTP does not authenticate the server or protect download integrity.
DEFAULT_ARCHIVE = (
    "http://ftp.csc.fi/pub/sci/geo/geodata/mml/maastotietokanta/"
)

# These large datasets are deliberately never selected.
ALWAYS_EXCLUDED = {
    "korkeussuhteet",
    "maastokuvionreuna",
}


class LinkParser(HTMLParser):
    """Collect hyperlinks from the archive's HTML directory listing."""

    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attributes):
        if tag.lower() == "a":
            href = dict(attributes).get("href")
            if href:
                self.links.append(href)


def make_session():
    """Create a reusable session with retries for initial HTTP requests."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "MTKGeoPackageDownloader/1.0",
        "Accept-Encoding": "identity",
    })

    # Retries help transient connection/status failures. They do not resume
    # a download whose streamed response fails after data has started arriving.
    retries = Retry(
        total=4,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def discover_datasets(session, directory_url, year):
    """Read actual filenames from the requested year's gpkg directory."""
    response = session.get(directory_url, timeout=(30, 120))
    response.raise_for_status()

    parser = LinkParser()
    parser.feed(response.text)

    # Use the final directory URL if the server redirected the request.
    base_url = response.url
    base = urlsplit(base_url)

    pattern = re.compile(
        rf"^MTK-(?P<dataset>.+)_{year}\.gpkg$",
        re.IGNORECASE,
    )

    datasets = {}

    for href in parser.links:
        url = urljoin(base_url, href)
        parsed = urlsplit(url)

        # Ignore parent links, sorting links, and external resources.
        if parsed.netloc != base.netloc:
            continue
        if parsed.query or parsed.fragment:
            continue
        if not parsed.path.startswith(base.path):
            continue

        relative_path = parsed.path[len(base.path):]

        # Only accept files directly inside the gpkg directory.
        if not relative_path or "/" in relative_path:
            continue

        filename = unquote(relative_path)
        match = pattern.fullmatch(filename)
        if not match:
            continue

        dataset = match.group("dataset").lower()

        if dataset in datasets:
            raise ValueError(
                f"Multiple files found for dataset {dataset!r}."
            )

        datasets[dataset] = {
            "dataset": dataset,
            "filename": filename,
            "url": url,
        }

    if not datasets:
        raise ValueError(
            f"No MTK GeoPackages found for {year} at:\n{directory_url}"
        )

    return datasets


def human_size(byte_count):
    """Format byte counts for progress messages."""
    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024


def download_file(session, url, destination, overwrite):
    """
    Stream one file to disk and publish it after successful completion.

    Existing files are skipped only when their size matches Content-Length.
    Matching size is not a checksum verification.
    """
    partial = destination.with_name(destination.name + ".part")

    try:
        with session.get(
            url,
            stream=True,
            timeout=(30, 180),
        ) as response:
            response.raise_for_status()

            # Requests may decode encoded responses, so only use the length
            # for validation when the server sends unencoded bytes.
            length_header = response.headers.get("Content-Length")
            expected = (
                int(length_header)
                if length_header
                and not response.headers.get("Content-Encoding")
                else None
            )

            if destination.exists() and not overwrite:
                if (
                    expected is not None
                    and destination.stat().st_size == expected
                ):
                    print(
                        f"  Already exists with matching size: "
                        f"{destination.name}"
                    )
                    return "existing", destination.stat().st_size

                raise FileExistsError(
                    f"Existing file cannot be verified by size: "
                    f"{destination}\n"
                    "Use --overwrite to download a replacement."
                )

            downloaded = 0
            start = time.monotonic()
            last_update = start

            # Opening with wb restarts a previous partial download.
            with partial.open("wb") as handle:
                for chunk in response.iter_content(
                    chunk_size=4 * 1024 * 1024
                ):
                    if not chunk:
                        continue

                    handle.write(chunk)
                    downloaded += len(chunk)

                    now = time.monotonic()
                    if now - last_update >= 2:
                        elapsed = max(now - start, 0.001)
                        speed = downloaded / elapsed

                        if expected:
                            progress = (
                                f"{100 * downloaded / expected:5.1f}% "
                                f"({human_size(downloaded)} / "
                                f"{human_size(expected)})"
                            )
                        else:
                            progress = human_size(downloaded)

                        print(
                            f"\r  {progress} — "
                            f"{human_size(speed)}/s",
                            end="",
                            flush=True,
                        )
                        last_update = now

            print()

            if expected is not None and downloaded != expected:
                raise RuntimeError(
                    f"Incomplete download: {destination.name}; "
                    f"expected {expected} bytes, received {downloaded}."
                )

            if downloaded == 0:
                raise RuntimeError(
                    f"Downloaded an empty file: {destination.name}"
                )

        # Existing completed data remains intact until replacement succeeds.
        partial.replace(destination)
        print(f"  Saved {human_size(downloaded)}")
        return "downloaded", downloaded

    except BaseException:
        # Leave the partial file for diagnosis. A subsequent run restarts it.
        if partial.exists():
            print(
                f"\n  Incomplete download retained at: {partial}",
                file=sys.stderr,
            )
        raise


def write_manifest(filename, rows):
    """Save selection and completion status after each download."""
    temporary = filename.with_name(filename.name + ".tmp")

    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "year",
                "dataset",
                "filename",
                "url",
                "local_path",
                "status",
                "size_bytes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(filename)


def main():
    parser = argparse.ArgumentParser(
        description="Download selected MTK GeoPackages for an archive year.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Archive year.",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output folder; default MTK_<year> in the working directory.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help="Additional dataset names to omit, e.g. suo virtavesi.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Show selection without downloading GeoPackages.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing completed GeoPackages.",
    )
    parser.add_argument(
        "--archive",
        default=DEFAULT_ARCHIVE,
        help="Archive root containing year folders.",
    )
    args = parser.parse_args()

    if not 1900 <= args.year <= 2100:
        parser.error("--year must be between 1900 and 2100.")

    output = (
        Path(args.output).resolve()
        if args.output
        else Path.cwd() / f"MTK_{args.year}"
    )

    if output.exists() and not output.is_dir():
        parser.error(f"Output path is not a folder: {output}")

    directory_url = (
        args.archive.rstrip("/") + f"/{args.year}/gpkg/"
    )
    requested_exclusions = {
        name.strip().lower() for name in args.exclude
    }

    with make_session() as session:
        print(f"Reading archive listing: {directory_url}")
        datasets = discover_datasets(
            session, directory_url, args.year
        )

        unknown = requested_exclusions - set(datasets)
        if unknown:
            parser.error(
                "Unknown excluded dataset name(s): "
                + ", ".join(sorted(unknown))
                + "\nAvailable dataset names: "
                + ", ".join(sorted(datasets))
            )

        excluded = ALWAYS_EXCLUDED | requested_exclusions
        selected = [
            entry
            for name, entry in sorted(datasets.items())
            if name not in excluded
        ]

        print("\nSelected:")
        for entry in selected:
            print(f"  {entry['filename']}")

        print("\nExcluded:")
        for name in sorted(set(datasets) & excluded):
            print(f"  {name}")

        if args.list_only:
            print("\nDownload URLs:")
            for entry in selected:
                print(entry["url"])
            return

        if not selected:
            print("\nNothing selected; no files downloaded.")
            return

        output.mkdir(parents=True, exist_ok=True)

        rows = [
            {
                "year": args.year,
                "dataset": entry["dataset"],
                "filename": entry["filename"],
                "url": entry["url"],
                "local_path": str(output / entry["filename"]),
                "status": "pending",
                "size_bytes": "",
            }
            for entry in selected
        ]

        manifest = output / "download_manifest.csv"
        write_manifest(manifest, rows)

        for number, row in enumerate(rows, 1):
            print(
                f"\n[{number}/{len(rows)}] {row['filename']}"
            )
            try:
                status, size = download_file(
                    session,
                    row["url"],
                    Path(row["local_path"]),
                    args.overwrite,
                )
                row["status"] = status
                row["size_bytes"] = size

            except Exception:
                row["status"] = "failed"
                write_manifest(manifest, rows)
                raise

            write_manifest(manifest, rows)

        print(f"\nFinished: {output}")
        print(f"Manifest: {manifest}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nDownload interrupted. Rerunning skips completed files "
            "with matching sizes and restarts partial downloads.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
