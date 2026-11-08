import pandas as pd


KEY = ["Note ID", "Calculator ID"]
FIELDS = ["Patient Note", "Relevant Entities"]
extracted_rule_ids = [4, 8, 16, 18, 20, 33, 45, 48, 51]

curr_test = pd.read_csv("dataset/test_data_10_22.csv")
verified = pd.read_csv("dataset/test_verified.csv")
v1_0 = pd.read_csv("dataset/test_data.csv")
# same Note ID / Calculator ID pairs as v1.1 for these calculators
test_1_17 = pd.read_csv("dataset/test_data_1_17_2025.csv")

curr_test = curr_test[curr_test["Calculator ID"].isin(extracted_rule_ids)].copy()

pd.set_option("display.width", 200)


# Composition: which earlier versions each current row (Note ID + Calculator ID) comes from
sources = {"Verified": verified, "v1.0": v1_0, "1/17 (v1.1)": test_1_17}
for name, df in sources.items():
    keys = set(zip(df["Note ID"], df["Calculator ID"]))
    curr_test[name] = [k in keys for k in zip(curr_test["Note ID"], curr_test["Calculator ID"])]
curr_test["Found in"] = curr_test[list(sources)].apply(lambda r: " + ".join(n for n in sources if r[n]) or "none", axis=1)
curr_test["Calculator"] = curr_test["Calculator ID"].astype(str) + " " + curr_test["Calculator Name"].str[:25]

counts = curr_test.groupby("Calculator", sort=False)[list(sources)].sum()
counts.loc["Total"] = counts.sum()
print(f"Rows found in each version (of {len(curr_test)}):")
print(counts.to_string(), "\n")

composition = pd.crosstab(curr_test["Calculator"], curr_test["Found in"], margins=True, margins_name="Total")
composition = composition.loc[counts.index, curr_test["Found in"].value_counts().index.tolist() + ["Total"]]
print("Composition (each row counted once):")
print(composition.to_string(), "\n")


def compare(curr, other, name):
    # merging on Note ID + Calculator ID lines the rows up before the fields are compared
    merged = curr[KEY + FIELDS].merge(other[KEY + FIELDS], on=KEY, how="left", suffixes=("", f" ({name})"), indicator=True)
    found = merged[merged["_merge"] == "both"]
    print(f"In {name}: {len(found)} of {len(curr)}")

    for field in FIELDS:
        mismatched = found[found[field] != found[f"{field} ({name})"]]
        print(f"  '{field}' differs from {name} in {len(mismatched)} rows")
        for _, row in mismatched.iterrows():
            print(f"    Note ID {row['Note ID']}, Calculator ID {row['Calculator ID']}")

    # rows of curr that are not in other
    return merged[merged["_merge"] == "left_only"]


not_in_verified = compare(curr_test, verified, "Verified")

# items which are not in Verified need to be in v1.0
missing = compare(not_in_verified, v1_0, "v1.0")
print(f"Not in Verified or v1.0: {len(missing)}")
for _, row in missing.iterrows():
    print(f"  Note ID {row['Note ID']}, Calculator ID {row['Calculator ID']}")
