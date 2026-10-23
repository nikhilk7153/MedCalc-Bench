import pandas as pd


curr_test = pd.read_csv("dataset/test_data_10_22.csv")
verified = pd.read_csv("dataset/test_verified.csv")
v1_0 = pd.read_csv("dataset/test_data.csv")


extracted_rule_ids = ["4", "8", "16", "18", "20", "33", "45", "48", "51"]


# Report the number of items which are in verified and curr_test for those extracted_rule_ids


overlap_count = 0
not_in_verified_count = 0
missing_from_v1_0 = []

for rule_id in extracted_rule_ids:
    curr_test_rule = curr_test[curr_test['Calculator ID'] == int(rule_id)]
    verified_rule = verified[verified['Calculator ID'] == int(rule_id)]
    
    overlap = pd.merge(curr_test_rule, verified_rule, on=['Note ID', 'Calculator ID'])
    overlap_count += len(overlap)

    # check that those items match exactly which overlap 
    # check that the patient note and the relevant fields match exactly 

    key = ['Note ID', 'Calculator ID']

    curr_test_rule_overlap = curr_test_rule[curr_test_rule['Note ID'].isin(overlap['Note ID'])].set_index(key).sort_index()

    verified_rule_overlap = verified_rule[verified_rule['Note ID'].isin(overlap['Note ID'])].set_index(key).sort_index()

    # make sure the note id and calc id line up row-for-row in both dataframes before comparing fields
    assert curr_test_rule_overlap.index.equals(verified_rule_overlap.index), f"Note ID / Calculator ID mismatch for rule ID {rule_id}"

    # check that the relevant fields match exactly
    relevant_fields = ["Patient Note", "Relevant Entities"]

    for field in relevant_fields:
        mismatched = curr_test_rule_overlap[field] != verified_rule_overlap[field]

        if mismatched.any():
            print(f"Mismatch found in field '{field}' for rule ID {rule_id} ({mismatched.sum()} rows)")
            for note_id, calc_id in curr_test_rule_overlap.index[mismatched]:
                print(f"  Note ID {note_id}, Calculator ID {calc_id}")
                print("    Curr Test Data:", curr_test_rule_overlap.loc[(note_id, calc_id), field])
                print("    Verified Data: ", verified_rule_overlap.loc[(note_id, calc_id), field])
        else:
            print(f"Field '{field}' matches for rule ID {rule_id}")

    # items which are not in verified need to be in v1.0
    not_in_verified = curr_test_rule[~curr_test_rule['Note ID'].isin(overlap['Note ID'])].set_index(key).sort_index()
    not_in_verified_count += len(not_in_verified)

    v1_0_rule = v1_0[v1_0['Calculator ID'] == int(rule_id)].set_index(key)
    in_v1_0 = not_in_verified.index.isin(v1_0_rule.index)

    for note_id, calc_id in not_in_verified.index[~in_v1_0]:
        print(f"Not in Verified or v1.0 for rule ID {rule_id}: Note ID {note_id}")
        missing_from_v1_0.append((note_id, calc_id))

    # check that the relevant fields match v1.0 exactly
    not_in_verified_v1_0 = not_in_verified[in_v1_0]
    v1_0_rule_overlap = v1_0_rule.loc[not_in_verified_v1_0.index]

    for field in relevant_fields:
        mismatched = not_in_verified_v1_0[field] != v1_0_rule_overlap[field]

        if mismatched.any():
            print(f"Mismatch vs v1.0 in field '{field}' for rule ID {rule_id} ({mismatched.sum()} rows)")
            for note_id, calc_id in not_in_verified_v1_0.index[mismatched]:
                print(f"  Note ID {note_id}, Calculator ID {calc_id}")
                print("    Curr Test Data:", not_in_verified_v1_0.loc[(note_id, calc_id), field])
                print("    v1.0 Data:     ", v1_0_rule_overlap.loc[(note_id, calc_id), field])
        else:
            print(f"Field '{field}' matches v1.0 for rule ID {rule_id} ({len(not_in_verified_v1_0)} rows not in Verified)")


print(f"Total overlap count for extracted rule IDs: {overlap_count}")
print(f"Not in Verified: {not_in_verified_count}, of which missing from v1.0: {len(missing_from_v1_0)}")
