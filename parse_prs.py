import json
import re

with open("prs_final.json") as f:
    data = json.load(f)

print("# Absolute Final Audit Report: Post-Cleanup\n")
open_count = 0
closed_count = 0

for item in data.get("items", []):
    number = item["number"]
    title = item["title"]
    state = item["state"]
    body = item.get("body") or ""
    issues = set(re.findall(r"#(\d+)", body))

    if state == "open":
        open_count += 1
    else:
        closed_count += 1

    print(f"## PR #{number}: {title}")
    print(f"- **State**: {state.upper()}")
    issues_str = ", ".join(issues) if issues else "None"
    print(f"- **Mentioned issues**: {issues_str}")
    print("---")

print(f"\n**Summary:** {open_count} Open PRs | {closed_count} Closed PRs")
