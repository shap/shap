import json
import urllib.request

req = urllib.request.Request(
    "https://api.github.com/search/issues?q=repo:shap/shap+is:pr+author:Dotify71", headers={"User-Agent": "Mozilla/5.0"}
)
with urllib.request.urlopen(req) as response:
    data = json.loads(response.read().decode())

print("# Comprehensive Audit of Submitted PRs\n")
for item in data.get("items", []):
    print(f"## PR #{item['number']}: {item['title']}")
    print(f"- **URL**: {item['html_url']}")
    print(f"- **State**: {item['state']}")
    body = item.get("body") or ""
    # Extract any mentioned issues
    import re

    issues = set(re.findall(r"#(\d+)", body))
    print(f"- **Mentioned Issues**: {', '.join(issues) if issues else 'None'}")
    print("\n")
