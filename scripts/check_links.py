"""Check local Markdown/HTML links in the project's reading and delivery files."""

from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


class HTMLLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        for attribute, value in attrs:
            if attribute in ("href", "src") and value:
                self.urls.append(value)


def check_links():
    files = [ROOT / "README.md", ROOT / "TODO.md"]
    files += [path for directory in ("docs", "deliverables", "results")
              for path in (ROOT / directory).rglob("*")
              if path.suffix in (".md", ".html") and "build" not in path.parts]
    checked = 0
    broken = []
    for document in sorted(files):
        content = document.read_text(encoding="utf-8")
        urls = []
        if document.suffix == ".md":
            content = re.sub(r"(?ms)^```[^\n]*\n.*?^```[ \t]*$", "", content)
            for match in re.finditer(r"\[[^\]\n]*\]\(([^)\n]+)\)", content):
                target = match.group(1).strip()
                urls.append(target[1:target.index(">")] if target.startswith("<") else target.split()[0])
        parser = HTMLLinks()
        parser.feed(content)
        urls.extend(parser.urls)
        for url in urls:
            if url.startswith("data:"):
                continue
            parts = urlsplit(url)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            checked += 1
            target = Path(unquote(parts.path))
            if target.is_absolute():
                broken.append(f"{document.relative_to(ROOT)}: absolute local link {url}")
            elif not (document.parent / target).exists():
                broken.append(f"{document.relative_to(ROOT)}: missing {url}")
    if broken:
        raise SystemExit("Broken local links:\n" + "\n".join(broken))
    print(f"Checked {checked} local links in {len(files)} Markdown/HTML files: all targets exist.")
    return {"files": len(files), "local_links": checked, "broken_links": 0}


if __name__ == "__main__":
    check_links()
