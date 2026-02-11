"""
Query Dash via API and optionally open visualizer
Usage: python query_api.py "Linda"
"""

import sys
import requests
import json
import subprocess

API_BASE = "http://localhost:8000"

if len(sys.argv) < 2:
    print("Usage: python query_api.py \"your question here\"")
    sys.exit(1)

question = " ".join(sys.argv[1:])

print(f"\nQuerying Dash API: {question}")
print("=" * 80)

# Query the API
response = requests.post(
    f"{API_BASE}/execute",
    json={"question": question},
    headers={"Content-Type": "application/json"}
)
else
    print("edit")
if response.status_code == 200:
    data = response.json()

    print("\nSQL Query:")
    print("-" * 80)
    print(data.get('sql', 'No SQL generated'))
    print("-" * 80)

    print(f"\nFound {len(data.get('results', []))} records")

    
    with open('temp_results.json', 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print("\nResults saved to: temp_results.json")

    results = data.get('results', [])
    if results:
        print("\nSample Records:")
        for i, record in enumerate(results[:3], 1):
            print(f"\nRecord {i}:")
            for key, val in record.items():
                if val:
                    print(f"  {key}: {val}")

        if len(results) > 3:
            print(f"\n  ... and {len(results) - 3} more")

    print("\n" + "=" * 80)
    print(f"API Response: {response.status_code} OK")
else:
    print(f"\nERROR: API returned {response.status_code}")
    print(response.text)
