import os, json
import urllib.request

DB_ID = "2ea67749af5c804494fb000cff9c2455"       # secret_xxx
TOKEN = "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"        # 32位hex（去掉短横线）

req = urllib.request.Request(
    f"https://api.notion.com/v1/databases/{DB_ID}",
    headers={
        "Authorization": f"Bearer {TOKEN}",
        "Notion-Version": "2022-06-28",
    }
)
with urllib.request.urlopen(req) as r:
    data = json.loads(r.read())

print("=== 数据库名称 ===")
print(data.get("title", [{}])[0].get("plain_text", ""))

print("\n=== 所有字段名及类型 ===")
for name, prop in data["properties"].items():
    ptype = prop["type"]
    extra = ""
    if ptype == "relation":
        related_db = prop["relation"].get("database_id", "")
        extra = f"  → related_db_id: {related_db}"
    print(f"  [{ptype:20s}]  {name}{extra}")