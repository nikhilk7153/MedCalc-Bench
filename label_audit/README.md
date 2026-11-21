# Individual patient-note label review

All 1,047 records across 55 calculators have a recorded review outcome. The original test_data.csv and calculator implementations were not edited.

| Outcome | Records |
|---|---:|
| no error identified | 665 |
| incorrect | 181 |
| ambiguous | 105 |
| insufficient information | 60 |
| out of domain | 30 |
| minor rounding difference | 3 |
| invalid numeric target | 3 |

## Files

- [All 1,047 row outcomes](individual_note_review.csv)
- [Incorrect labels, with corrections or qualified replacements](incorrect_labels.csv)
- [Invalid numeric targets](invalid_numeric_targets.csv)
- [Ambiguous, missing-information and out-of-domain cases](needs_adjudication.csv)

Use **Row Number**, the dataset column, to locate records. It is not a physical CSV line number: patient notes contain embedded newlines. Each report row contains the original label, reason, corrected value/range when defensible, question, narrative evidence, full note, original entities, original explanation and available rule reference.

## Method and limits

Each record was reviewed against its calculator-relevant narrative passages, with full-note follow-up for ambiguous risk factors and missing-input findings. This is not a claim that every unrelated sentence was individually clinically adjudicated. Arithmetic screening was checked against the notes rather than accepted as ground truth. The manual narrative decisions supersede the earlier entity-only screen in findings.csv. The exporter only combines recorded decisions; running it does not perform a fresh clinical review.

**Incorrect** means a demonstrable arithmetic, scoring or narrative-input error affects the target. Some have only a conservative bound, or require a repaired note before a unique replacement is possible. **Invalid numeric target** means a numeric total cannot be assigned, for example GCS with a non-testable component. **Ambiguous** includes competing encounters, source-unit conflicts, unspecified scoring conventions and explanations whose errors may cancel numerically. **Insufficient information** means the supplied note cannot establish required inputs. **Out of domain** flags inappropriate populations or presentations; it is not counted as a proven arithmetic error. **No error identified** is a review outcome, not certification of clinical correctness. Minor rounding differences within the benchmark tolerance are reported separately.

An incorrect target does not necessarily fail the benchmark’s supplied acceptance interval. Some systematic input errors are numerically small enough to fall inside it; the CSV reports this separately when an exact numeric replacement is available.

Conventions: the named legacy calculators are retained (including 2005 Caprini and legacy MELD-Na); newer calculator versions are not silently substituted. Current and prior events are separated. Source typos are not silently repaired. Conservative Caprini bounds omit uncertain or overlapping features and count at most one trauma and one thrombophilia item. Conditional values and ranges should not be copied into a corrected training target without resolving their stated conditions.

## Incorrect labels by calculator

| Calculator | Count | Row Number values |
|---|---:|---|
| Caprini Score for Venous Thromboembolism (2005) | 20 | 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927 |
| Sequential Organ Failure Assessment (SOFA) Score | 18 | 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907 |
| APACHE II Score | 15 | 871, 872, 873, 875, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887 |
| Estimated Due Date | 15 | 928, 929, 930, 931, 932, 935, 936, 937, 938, 940, 943, 944, 945, 946, 947 |
| Glasgow Coma Score (GCS) | 15 | 848, 849, 850, 851, 852, 853, 854, 855, 858, 859, 861, 864, 865, 866, 867 |
| Child-Pugh Score for Cirrhosis Mortality | 13 | 200, 201, 203, 204, 205, 208, 209, 210, 212, 213, 214, 216, 217 |
| Framingham Risk Score for Hard Coronary Heart Disease | 13 | 552, 553, 554, 555, 556, 557, 558, 559, 560, 563, 564, 566, 567 |
| Revised Cardiac Risk Index for Pre-Operative Risk | 11 | 240, 241, 242, 243, 245, 246, 250, 253, 255, 256, 259 |
| Charlson Comorbidity Index (CCI) | 9 | 428, 429, 430, 431, 435, 437, 439, 442, 445 |
| SIRS Criteria | 8 | 588, 591, 592, 593, 596, 598, 599, 606 |
| CHA2DS2-VASc Score for Atrial Fibrillation Stroke Risk | 6 | 41, 43, 45, 46, 47, 49 |
| FeverPAIN Score for Strep Pharyngitis | 5 | 453, 457, 460, 462, 463 |
| Serum Osmolality | 5 | 405, 408, 416, 417, 418 |
| MELD Na (UNOS/OPTN) | 4 | 340, 342, 343, 344 |
| Glasgow-Blatchford Bleeding Score (GBS) | 3 | 389, 390, 399 |
| HAS-BLED Score for Major Bleeding Risk | 3 | 348, 349, 361 |
| PSI Score: Pneumonia Severity Index for CAP | 3 | 829, 831, 833 |
| CKD-EPI Equations for Glomerular Filtration Rate | 2 | 23, 25 |
| Calcium Correction for Hypoalbuminemia | 2 | 114, 117 |
| HEART Score for Major Cardiac Events | 2 | 265, 271 |
| LDL Calculated | 2 | 513, 521 |
| Wells' Criteria for DVT | 2 | 223, 237 |
| Wells' Criteria for Pulmonary Embolism | 2 | 121, 123 |
| Albumin Corrected Anion Gap | 1 | 780 |
| Centor Score (Modified/McIsaac) for Strep Pharyngitis | 1 | 300 |
| Creatinine Clearance (Cockcroft-Gault Equation) | 1 | 3 |

## Selected examples

| Row Number | Calculator | Existing label | Reviewed result | Reason |
|---:|---|---|---|---|
| 47 | CHA2DS2-VASc Score for Atrial Fibrillation Stroke Risk | 2 | 1 | Portal hypertension is misclassified as systemic hypertension. Female sex contributes 1; no qualifying systemic hypertension or other factors are documented. |
| 152 | MDRD GFR Equation | 3.685 | Not determinable or not applicable from the supplied note | C-reactive protein 12.35 mg/dL is misclassified as creatinine; no serum creatinine is supplied. |
| 343 | MELD Na (UNOS/OPTN) | 20 | 23 | Legacy MELD-Na: clamp creatinine to 1–4 mg/dL, bilirubin/INR to at least 1 and sodium to 125–137; round the MELD score after multiplying by 10. Dataset rounds the unscaled score to an integer, yielding an incorrect multiple of 10. |
| 445 | Charlson Comorbidity Index (CCI) | 10 | 9 | Age 3 + MI 1 + diabetic end-organ damage 2 + CKD 2 + TIA 1 = 9. Severe liver disease and peripheral vascular disease are unsupported by the full note. |
| 772 | Albumin Corrected Anion Gap | -2.75 | Not determinable or not applicable from the supplied note | Albumin 17.1 is actually hemoglobin 17.1 g/dL. No albumin concentration is supplied. |
| 879 | APACHE II Score | 26 | 3 | Only age contributes 3; GCS 15 contributes zero. |
| 908 | Caprini Score for Venous Thromboembolism (2005) | 6 | At least 18 | Documented narrative risk factors alone exceed the label. This conservative Caprini lower bound counts at most one trauma and one thrombophilia category and omits uncertain mobility or overlapping surgery points. |
| 928 | Estimated Due Date | 09/23/2014 | 09/11/2014 | Shorter cycles require subtracting days; label adds days instead. |

## Rule references

- [CKD-EPI Equations for Glomerular Filtration Rate](https://www.kidney.org/ckd-epi-creatinine-equation-2021)
- [CHA2DS2-VASc Score for Atrial Fibrillation Stroke Risk](https://www.escardio.org/static-file/Escardio/Guidelines/Documents/ehaa612.pdf)
- [Child-Pugh Score for Cirrhosis Mortality](https://www.merckmanuals.com/professional/multimedia/table/child-turcotte-pugh-scoring-system)
- [Revised Cardiac Risk Index for Pre-Operative Risk](https://pubmed.ncbi.nlm.nih.gov/10477528/)
- [Centor Score (Modified/McIsaac) for Strep Pharyngitis](https://www.idsociety.org/practice-guideline/streptococcal-pharyngitis2/)
- [Glasgow Coma Score (GCS)](https://www.glasgowcomascale.org/downloads/GCS-Assessment-Aid-English.pdf)
- [MELD Na (UNOS/OPTN)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3305837/)
- [Glasgow-Blatchford Bleeding Score (GBS)](https://handbook.ggcmedicines.org.uk/guidelines/gastrointestinal-system/glasgow-blatchford-score/)
- [APACHE II Score](https://www.merckmanuals.com/professional/multimedia/table/acute-physiologic-assessment-and-chronic-health-evaluation-apache-ii-scoring-system)
- [Charlson Comorbidity Index (CCI)](https://ncbi.nlm.nih.gov/books/NBK587905/?report=reader)
- [FeverPAIN Score for Strep Pharyngitis](https://www.nice.org.uk/guidance/ng84/resources/sore-throat-acute-in-adults-antimicrobial-prescribing-visual-summary-pdf-11315864557)
- [Caprini Score for Venous Thromboembolism (2005)](https://www.ahrq.gov/patient-safety/settings/hospital/vtguide/appb2.html)
- [Sequential Organ Failure Assessment (SOFA) Score](https://pmc.ncbi.nlm.nih.gov/articles/PMC4968574/)
- [Framingham Risk Score for Hard Coronary Heart Disease](https://aclanthology.org/anthology-files/anthology-files/pdf/naacl/2025.naacl-long.263.pdf)
- [Morphine Milligram Equivalents (MME) Calculator](https://www.cdc.gov/mmwr/volumes/71/rr/rr7103a1.htm)
- [SIRS Criteria](https://pmc.ncbi.nlm.nih.gov/articles/PMC4968574/)
- [Estimated of Conception](https://www.merckmanuals.com/professional/gynecology-and-obstetrics/family-planning/fertility-awareness-based-methods-of-contraception)

Sources support scoring definitions; row-level findings are this review’s inferences from the supplied notes. For routine algebra, the equation specified in the dataset is retained and independently recalculated.

## Reproduction and provenance

`python3 label_audit/build_individual_report.py` regenerates these exports from the recorded review decisions and preliminary arithmetic findings. The clinical judgments themselves are manual and are not reproduced by executing a script.

Source CSV SHA-256: `c2d23097f8bdb7ae4a77dbf4a5a767335df7303998d7c7664a73a29fd8e2b31a`
