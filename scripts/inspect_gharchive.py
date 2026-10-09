import gzip
import json
from collections import Counter

FILE =  "data/gharchive_sample.json.gz"

seen_issues = set()
label_counts = Counter()

total_events = 0
issue_events = 0
unique_issues = 0
labelled_issues = 0
unlabelled_issues = 0

with gzip.open(FILE, "rt", encoding="utf-8") as f:

    for line in f:

        total_events += 1

        event = json.loads(line)

        if event.get("type") != "IssuesEvent":
            continue

        issue_events += 1

        issue = event.get("payload", {}).get("issue", {})

        issue_id = issue.get("id")

        if issue_id in seen_issues:
            continue

        seen_issues.add(issue_id)
        unique_issues += 1

        labels = issue.get("labels", [])

        if labels:
            labelled_issues += 1

        else:
            unlabelled_issues += 1

        for label in labels:
            name = label.get("name")

            if name:
                label_counts[name] += 1


print("\n===== GH ARCHIVE ANALYSIS =====")

print(f"Total events: {total_events}")
print(f"IssuesEvent: {issue_events}")
print(f"Unique issues: {unique_issues}")

print(f"\nLabelled issues: {labelled_issues}")
print(f"Unlabelled issues: {unlabelled_issues}")

print("\nTop 30 labels:")

for label, count in label_counts.most_common(30):
    print(f"{label}: {count}")