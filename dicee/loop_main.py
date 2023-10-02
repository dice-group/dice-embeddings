import subprocess
import sys
from static_funcs_training import evaluate_lp
import json

parameter_values = [
    {"p": 1, "q": 0, "r": 0},
    {"p": 2, "q": 0, "r": 0},
    {"p": 3, "q": 0, "r": 0},
]

evaluate_lp_outputs = []
for i in range(5):
    subprocess_output = subprocess.run(["python", "/home/dice/Desktop/dice-embeddings/main.py", "--num_epochs",str(i)],stdout=subprocess.PIPE,universal_newlines=True,)
    #eval_output = subprocess_output.stdout.strip()
    #print(evaluate_lp(eval_output))

    try:
        output_dict = json.loads(subprocess_output.stdout)
        print(output_dict)
        evaluate_lp_result = evaluate_lp(output_dict)  # Call the imported evaluate_lp function


        evaluate_lp_outputs.append(evaluate_lp_result)
    except json.JSONDecodeError:
        # Handle JSON decoding errors if necessary
        print(f"Subprocess output for iteration {i}:")
        print(subprocess_output.stdout)

print(evaluate_lp_outputs)
    
    
