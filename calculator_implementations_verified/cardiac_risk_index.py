import json 
import unit_converter_new

def compute_cardiac_index_explanation(input_variables):
    # List of parameters and their default values
    parameters = {
        'elevated_risk_surgery': "an elevated risk surgery",
        'ischemic_heart_disease': "a history of ischemic heart disease",
        'congestive_heart_failure': "a history of congestive heart failure", 
        'cerebrovascular_disease': "a history of cerebrovascular disease",
        'pre_operative_insulin_treatment': "a pre-operative insulin treatment",
        'pre_operative_creatinine': "pre-operative creatinine" 
    }

    # v1.0 style: no criteria list before the step-by-step scoring

    output = ""

    # Initializing scores and output explanation
    cri = 0
    output += "The current cardiac risk index is 0.\n"

    for param_name, full_name in parameters.items():
        param_value = input_variables.get(param_name)

        # If parameter is missing, assume it as False
        if param_value is None:
            output += f"The patient note does not mention anything about {full_name} and is assumed to be absent. "
            input_variables[param_name] = False
            param_value = False
        elif param_name != 'pre_operative_creatinine':
            value = 'absent' if not param_value else 'present'
            output += f"Based on the patient note, {full_name} is determined to be {value} for the patient. "
        elif param_name == 'pre_operative_creatinine':
            explanation, param_value = unit_converter_new.conversion_explanation(param_value[0], "Pre-Operative Creatinine", 113.12, None, param_value[1], "mg/dL" )
            input_variables['pre_operative_creatinine'] = [param_value, "mg/dL"]
            output += explanation
          
        if param_name == 'pre_operative_creatinine':

            if param_value > 2: 
                output += f"The patient's pre-operative creatinine is greater than 2 mg/dL, so we increment the score by one and the current total will be {cri} + 1 = {cri + 1}.\n"
                cri += 1
            else:
                output += f"The patient's pre-operative creatinine is less than or equal to 2 mg/dL, so we keep the score the same at {cri}.\n"
            continue

        if param_value:
            output += f"This means that we increment the score by one and the current total will be {cri} + 1 = {cri + 1}.\n"
            cri += 1
        else:
            output += f"This means that the total score remains unchanged at {cri}.\n"


    output += f"\nThe cardiac risk index score is {cri}."

    return {"Explanation": output, "Answer": cri}



