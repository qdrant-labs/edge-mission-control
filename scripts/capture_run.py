"""Drive a full demo run in headless Chrome and screenshot each act.

Assumes the server is running on localhost:8000.
Run: uv run python scripts/capture_run.py
Screenshots land in /tmp/edge-shots/.
"""

from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path("/tmp/edge-shots")
OUT.mkdir(exist_ok=True)

# (seconds after pressing space, name). Boot takes ~8.2s, then video starts.
BOOT = 8.5
SHOTS = [
    (1.5, "01-title-boot"),
    (5.0, "02-boot-terminal"),
    (BOOT + 12, "03-act1-ingest"),
    (BOOT + 27, "04-act1-query-chair"),
    (BOOT + 49, "05-act1-query-stools"),
    (BOOT + 57, "06-act2-open-vocab"),
    (BOOT + 78, "07-act2-query-pool"),
    (BOOT + 93, "08-act2-teach"),
    (BOOT + 105, "09-act2-query-bed"),
    (BOOT + 123, "10-act3-query-bathtub"),
    (BOOT + 137, "11-act3-query-wine"),
    (BOOT + 149, "12-closing"),
]


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto("http://localhost:8000/?auto")
        page.wait_for_timeout(1500)
        page.keyboard.press("Space")

        elapsed = 0.0
        for at, name in SHOTS:
            page.wait_for_timeout(int((at - elapsed) * 1000))
            elapsed = at
            page.screenshot(path=str(OUT / f"{name}.png"))
            print(f"captured {name} at t+{at:.0f}s")

        browser.close()


if __name__ == "__main__":
    main()
