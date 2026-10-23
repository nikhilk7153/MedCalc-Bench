---
license: cc-by-sa-4.0
task_categories:
- question-answering
language:
- en
tags:
- medical
pretty_name: MedCalc-Bench Verified
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
  - split: one_shot
    path: data/one_shot-*
dataset_info:
  features:
  - name: Row Number
    dtype: int64
  - name: Calculator ID
    dtype: int64
  - name: Calculator Name
    dtype: string
  - name: Category
    dtype: string
  - name: Output Type
    dtype: string
  - name: Note ID
    dtype: string
  - name: Note Type
    dtype: string
  - name: Patient Note
    dtype: string
  - name: Question
    dtype: string
  - name: Relevant Entities
    dtype: string
  - name: Ground Truth Answer
    dtype: string
  - name: Lower Limit
    dtype: string
  - name: Upper Limit
    dtype: string
  - name: Ground Truth Explanation
    dtype: string
  splits:
  - name: train
    num_bytes: 51519033
    num_examples: 10538
  - name: test
    num_bytes: 5410894
    num_examples: 1100
  - name: one_shot
    num_bytes: 245999
    num_examples: 55
  download_size: 22992587
  dataset_size: 57175926
---


# Updates

Updates to MedCalc-Bench Verified will be made on this page going forward. 

Here is the github link for our repository: https://github.com/nikhilk7153/MedCalc-Bench-Verified

This is an updated version that is modified from MedCalc-Bench-v1.2. 
 
While we have audited MedCalc-Bench Verified on mulitple occasions, should there by any corrections or enhancements, we will update with a new release and specify any changes. 
The HuggingFace dataset and main branch will always match together with the latest release. You should specify which version (e.g. v.1.0.6) in your paper when using this dataset for reproducibility purposes.


# MedCalc-Bench Verified

MedCalc-Bench Verified is a re-verified version of MedCalc-Bench used to benchmark LLMs ability to serve as clinical calculators. Each instance in the dataset consists of a patient note, a question asking to compute a specific clinical value, a final answer value, and a step-by-step solution explaining how the final answer was obtained. Our dataset covers 55 different calculation tasks which are either rule-based calculations or are equation-based calculations. This dataset contains a training dataset of 10,538 instances and a testing dataset of 1,100 instances.
 <br>

In all, we hope that our dataset and benchmark serves as a call to improve the computational reasoning skills of LLMs in medical settings. 

## MedCalc-Bench Verified Dataset


In addition to the 1,100 evaluation instances, we also provide a training dataset of 10,538 instances which can be used for fine-tuning open-source LLMs. The training data can be found in the ```dataset/train_data.csv.zip``` file and can be unzipped to obtain ```train_data.csv```. This training dataset can also be found in the train split of the HuggingFace link. 

Each Instance in the dataset contains the following information: 

- **Row Number**: Specifies the index of the instance.
- **Calculator ID**: Specifies the integer ID of the calculator.
- **Calculator Name**: Specifies the name of the clinical calculation task.
- **Category**: Specifies the sub-category of the calculator. For equation-based calculators, the options are lab test, dosage, date, or physical and for rule-based calculators, the options are risk, severity, and diagnosis.
- **Output Type**: Specifies the format type that the calculator will return. The options are decimal, integer, date (MM/DD/YY), or time in terms of weeks and days (i.e. (17 weeks, 4 days)).
- **Note ID**: Specifies the ID of the patient note. The ID of the note will either be the ID given by Open-Patients or it will be an integer value if the patient note was handwritten by clinicians or synthesized by a template.
- **Note Type**: Specifies whether the patient note was LLM generated and then edited/approved by a clinician (Synthetic), produced from a python template (Template), or was extracted from PMC-Patients (Extracted).
- **Patient Note**: Specifies the patient note which provides the information needed to compute the final answer.
- **Question**: Specifies the question that is asked to the model to compute a specific medical value based on a particular calculator.
- **Relevant Entities**: Provides a dictionary of the parameters and their extracted values based on the patient note.
- **Ground Truth Answer**: Specifies the ground truth value without any units for the medical value that needs to be calculated.
- **Lower Limit**: For equation-based calculators whose output is a decimal, this value is 95% of the ground truth answer value. For all other cases, the lower limit is the same as the ground-truth value.
- **Upper Limit**: For equation-based calculators whose output is a decimal, this value is 105% of the ground truth answer value. For all other cases, the upper limit is the same as the ground-truth value.
- **Ground Truth Explanation**: The ground truth explanation for the data instance providing a step-by-step explanation for how the final answer was obtained.



## Maintenance and Responsibility 

For any changes to this dataset, (i.e. adding new notes, calculators, modifying existing ones), we will update the README instructions, test_set.csv, and train_set.csv. We will still keep older versions of these datasets as separate branches and update the versions on Github for new releases. We will also update train and test sets for HuggingFace as well. 

## License 

MedCalc-Bench Verified is released under the CC-BY-SA 4.0 license. 


