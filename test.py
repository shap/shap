import json
import sys

data = json.load(sys.stdin)
for item in data.get("items", []):
    print(f"PR #{item['number']}: {item['title']}")
    print(f"State: {item['state']}")
    print(f"URL: {item['html_url']}")
    print("---")
