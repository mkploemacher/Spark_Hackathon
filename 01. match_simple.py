# --- Simple 1/0 Matching System ---
# This script loads startups and investors, then scores each startup for a given investor
# using a very simple rule-based system where each condition is either 1 (match) or 0 (no match)

import pandas as pd

# Load CSV files
startups = pd.read_csv("startups.csv", dtype=str).fillna("")
investors = pd.read_csv("investors.csv", dtype=str).fillna("")

# Helper function to turn semicolon-separated fields into a Python set
def to_set(s):
    return set([x.strip().lower() for x in s.split(';') if x.strip()])

# Main scoring function
def score_pair(startup, investor):
    tests = []

    # 1. Check if any sector overlaps
    tests.append(1 if to_set(startup['sectors']) & to_set(investor['preferred_sectors']) else 0)

    # 2. Check if investor supports this stage
    tests.append(1 if startup['stage'].strip().lower() in to_set(investor['preferred_stages']) else 0)

    # 3. Check if location matches investor preference
    tests.append(1 if startup['location'].strip().lower() in to_set(investor['preferred_locations']) else 0)

    # 4. Check if investor's cheque size fits startup need
    try:
        target = float(startup['raise_target_usd'])
        cmin = float(investor['check_min'])
        cmax = float(investor['check_max'])
        tests.append(1 if cmin <= target <= cmax else 0)
    except:
        # In case numbers are not properly formatted
        tests.append(0)

    # 5. Check if any thesis keyword appears in the startup description
    keywords = to_set(investor['thesis_keywords'])
    desc = startup['description'].lower()
    tests.append(1 if any(k in desc for k in keywords) else 0)

    # Calculate final score (0 to 1)
    raw = sum(tests)
    normalized = raw / len(tests)

    return normalized, tests

# Choose an investor to test matching
investor_id = '1'
inv = investors[investors['id'] == investor_id].iloc[0]

results = []

# Score every startup for this investor
for _, s in startups.iterrows():
    score, breakdown = score_pair(s, inv)
    results.append((s['id'], s['name'], score, breakdown))

# Sort by score high to low
results = sorted(results, key=lambda x: x[2], reverse=True)

# Print results
for r in results:
    print(r)
