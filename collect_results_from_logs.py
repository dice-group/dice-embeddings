import argparse

# defined command line options
# this also generates --help and error handling
arg = argparse.ArgumentParser()
arg.add_argument(
    "--logs",  # name on the CLI - drop the `--` for positional/required parameters
    nargs="*",  # 0 or more values expected => creates a list
    type=str,
    default=['dummy.log', 'dummy.log', 'dummy.log'],  # default if nothing is provided
)

# parse the command line
args = arg.parse_args()
results = []
for i in args.logs:
    model_name = i[:-4]
    keyword_to_search    = f'Evaluate {model_name} on test set:'
    keyward_to_runtime   = f'Runtime of {model_name}:'
    keyward_to_num_param = f'NumParam of {model_name}:'
    keyward_to_estimated = f'Estimated of {model_name}:'

    with open(i, 'r') as reader:
        for x in reader:
            if keyward_to_runtime in x:
                runtime = eval(x[x.index(keyward_to_runtime) + len(keyward_to_runtime):])
            if keyward_to_num_param in x:
                num_param = eval(x[x.index(keyward_to_num_param) + len(keyward_to_num_param):])
            if keyward_to_estimated in x:
                size_ = eval(x[x.index(keyward_to_estimated) + len(keyward_to_estimated):])

            if keyword_to_search in x:
                res_dict_ = eval(x[x.index(keyword_to_search) + len(keyword_to_search):])
                s = f'{model_name}\t:\tMRR:{res_dict_["MRR"]:.3f}\tH@1:{res_dict_["H@1"]:.3f}\tH@3:{res_dict_["H@3"]:.3f}\tH@10:{res_dict_["H@10"]:.3f}'

        results.append((res_dict_["MRR"], s+f'\tRT: {runtime:.3f} seconds\tParam:{num_param}\tSize:{size_}'))
results.sort(key=lambda tup: tup[0], reverse=True)

for ith,(_, report) in enumerate(results):
    print(f'{ith+1}. {report}')
