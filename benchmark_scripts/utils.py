import json
import os
import numpy as np


def runtime_report(path):
    folders = os.listdir(path)
    runtimes = []
    for folder in folders:
        tmp_path = os.path.join(path, folder)
        files = os.listdir(tmp_path)
        for file in files:
            if file == "report.json":
                report_file = os.path.join(path, folder, file)
                with open(report_file, "r") as f:
                    report_dict = json.load(f)
                    runtimes.append(report_dict["Runtime"])

    runtimes_np = np.array(runtimes)
    runtimes_std = np.std(runtimes_np)
    runtimes_mean = np.mean(runtimes_np)

    print(f"mean: {runtimes_mean}, std: {runtimes_std}")

    return


def pykeen_runtime_report(path):
    folders = os.listdir(path)
    runtimes = []
    for folder in folders:
        tmp_path = os.path.join(path, folder)
        files = os.listdir(tmp_path)
        for file in files:

            if file == "results.json":
                report_file = os.path.join(path, folder, file)
                with open(report_file, "r") as f:
                    report_dict = json.load(f)
                    # print(report_dict['times']['training'])
                    runtimes.append(report_dict["times"]["training"])

    runtimes_np = np.array(runtimes)
    runtimes_std = np.std(runtimes_np)
    runtimes_mean = np.mean(runtimes_np)

    print(f"mean: {runtimes_mean}, std: {runtimes_std}")

    return


def evaluation_report(path):
    folders = os.listdir(path)

    hit_1_max = 0
    hit_3_max = 0
    hit_10_max = 0
    mrr_max = 0

    for folder in folders:
        tmp_path = os.path.join(path, folder)
        files = os.listdir(tmp_path)
        for file in files:
            if file == "eval_report.json":
                report_file = os.path.join(path, folder, file)
                with open(report_file, "r") as f:
                    report_dict = json.load(f)
                    hit_1_max = (
                        report_dict["Test"]["H@1"]
                        if report_dict["Test"]["H@1"] > hit_1_max
                        else hit_1_max
                    )
                    hit_3_max = (
                        report_dict["Test"]["H@3"]
                        if report_dict["Test"]["H@3"] > hit_3_max
                        else hit_3_max
                    )
                    hit_10_max = (
                        report_dict["Test"]["H@10"]
                        if report_dict["Test"]["H@10"] > hit_10_max
                        else hit_10_max
                    )
                    mrr_max = (
                        report_dict["Test"]["MRR"]
                        if report_dict["Test"]["MRR"] > mrr_max
                        else mrr_max
                    )

    print(
            f"hit_1_max:{hit_1_max}\nhit_3_max:{hit_3_max}\nhit_10_max:{hit_10_max}\nmrr_max:{mrr_max}"
        )

    # runtimes_np = np.array(runtimes)
    # runtimes_std = np.std(runtimes_np)
    # runtimes_mean = np.mean(runtimes_np)

    # print(f'mean: {runtimes_mean}, std: {runtimes_std}')

    return


def evaluation_pykeen_report(path):
    folders = os.listdir(path)
    hit_1_max = 0
    hit_3_max = 0
    hit_10_max = 0
    mrr_max = 0

    for folder in folders:
        tmp_path = os.path.join(path, folder)
        files = os.listdir(tmp_path)
        for file in files:
            if file == "results.json":
                report_file = os.path.join(path, folder, file)
                with open(report_file, "r") as f:
                    report_dict = json.load(f)
                    mrr_max = (
                        mrr_max
                        if report_dict["metrics"]["both"]["optimistic"][
                            "inverse_harmonic_mean_rank"
                        ]
                        < mrr_max
                        else report_dict["metrics"]["both"]["optimistic"][
                            "inverse_harmonic_mean_rank"
                        ]
                    )
                    hit_1_max = (
                        hit_1_max
                        if report_dict["metrics"]["both"]["optimistic"]["hits_at_1"]
                        < hit_1_max
                        else report_dict["metrics"]["both"]["optimistic"]["hits_at_1"]
                    )
                    hit_3_max = (
                        hit_3_max
                        if report_dict["metrics"]["both"]["optimistic"]["hits_at_3"]
                        < hit_3_max
                        else report_dict["metrics"]["both"]["optimistic"]["hits_at_3"]
                    )
                    hit_10_max = (
                        hit_10_max
                        if report_dict["metrics"]["both"]["optimistic"]["hits_at_10"]
                        < hit_10_max
                        else report_dict["metrics"]["both"]["optimistic"]["hits_at_10"]
                    )

    print(
        f"hit_1_max:{hit_1_max}\nhit_3_max:{hit_3_max}\nhit_10_max:{hit_10_max}\nmrr_max:{mrr_max}"
    )

def pykeen_runtime_report1(path):
  
  files = os.listdir(path)
  runtimes = []
  for file in files:
    tmp_path = os.path.join(path,file)
    with open(tmp_path, "r") as f:
      report_dict = json.load(f)
      
      runtimes.append(report_dict['report_time'])
  
  runtimes_np = np.array(runtimes)
  runtimes_std = np.std(runtimes_np)
  runtimes_mean = np.mean(runtimes_np)

  print(f"mean: {runtimes_mean}, std: {runtimes_std}")
  
  
def pykeen_eval(path):
  max_mrr = 0
  max_hit1=0
  max_hit3=0
  max_hit10=0
  
  for i in range(5):
    
    new_path = path[:-1]+str(i)
    
    with open(new_path, "r") as f:
        for line in f:
          metric = line.rstrip()
          if "('inverse_harmonic_mean_rank', 'both', 'optimistic')" in metric:
            value = metric[metric.index(':')+1:]
            if float(value) > max_mrr:
              max_mrr = float(value)
            
          if "('hits_at_1', 'both', 'optimistic')" in metric:
            value = metric[metric.index(':')+1:]
            if float(value) > max_hit1:
              max_hit1 = float(value)
           
          if "('hits_at_3', 'both', 'optimistic')" in metric:
            value = metric[metric.index(':')+1:]
            if float(value) > max_hit3:
              max_hit3 = float(value)
           
          if "('hits_at_10', 'both', 'optimistic')" in metric:
            value = metric[metric.index(':')+1:]
            if float(value) > max_hit10:
              max_hit10 = float(value)
        
  print(f'MRR:{max_mrr}')
  print(f'HIT1:{max_hit1}')
  print(f'HIT3:{max_hit3}')
  print(f'HIT10:{max_hit10}')
  
# pykeen_eval('pykeen_small/eval/gpu_umls_distmult_eval0')
# print('----------------------')
# pykeen_eval('pykeen_small/eval/gpu_kinship_distmult_eval0')
# print('----------------------')
# pykeen_eval('pykeen_small/eval/gpu_umls_complex_eval0')
# print('----------------------')
# pykeen_eval('pykeen_small/eval/gpu_kinship_complex_eval0')



# pykeen_runtime_report1('pykeen_small/slcwa/slcwa1_gpu_umls_distmult')
# pykeen_runtime_report1('pykeen_small/slcwa/slcwa1_gpu_kinship_distmult')
# pykeen_runtime_report1('pykeen_small/slcwa/slcwa1_gpu_umls_complex')
# pykeen_runtime_report1('pykeen_small/slcwa/slcwa1_gpu_kinship_complex')

# evaluation_report('complex_kinships_cpu')




# runtime_report('slcwa16_pykeen_distmult_umls_gpu_1/')
# runtime_report('slcwa16_pykeen_distmult_umls_gpu_2/')
# runtime_report('slcwa16_pykeen_distmult_umls_cpu/')
# runtime_report('slcwa16_pykeen_complex_umls_gpu_1/')
# runtime_report('slcwa16_pykeen_complex_umls_gpu_2/')
# runtime_report('slcwa16_pykeen_complex_umls_cpu/')

# runtime_report('slcwa_pykeen_distmult_umls_cpu')
# runtime_report('slcwa_pykeen_distmult_umls_gpu_1')
# runtime_report('slcwa_pykeen_distmult_umls_gpu_2')
# runtime_report('slcwa_pykeen_distmult_kinship_cpu')
# runtime_report('slcwa_pykeen_distmult_kinship_gpu_1')
# runtime_report('slcwa_pykeen_distmult_kinship_gpu_2')
# runtime_report('slcwa_pykeen_complex_umls_cpu')
# runtime_report('slcwa_pykeen_complex_umls_gpu_1')
# runtime_report('slcwa_pykeen_complex_umls_gpu_2')
# runtime_report('slcwa_pykeen_complex_kinship_cpu')
# runtime_report('slcwa_pykeen_complex_kinship_gpu_1')
# runtime_report('slcwa_pykeen_complex_kinship_gpu_2')



# runtime_report('complex_kinships_gpu_1/')
# runtime_report('complex_kinships_gpu_2/')
# runtime_report('complex_kinships_gpu_3/')
# runtime_report('complex_kinships_cpu/')

# runtime_report('integrated_pykeen\slcwa\half_neg\pykeen_distmult_umls_gpu_1')
# runtime_report('integrated_pykeen\slcwa\half_neg\pykeen_distmult_umls_gpu_2')
# runtime_report('integrated_pykeen\slcwa\half_neg\pykeen_distmult_umls_cpu')


# runtime_report('integrated_pykeen\slcwa\half_neg\pykeen_distmult_kinship_gpu_1')
# runtime_report('integrated_pykeen\slcwa\half_neg\pykeen_distmult_kinship_gpu_2')
# runtime_report('integrated_pykeen\slcwa\half_neg\pykeen_distmult_kinship_cpu')


# runtime_report('integrated_pykeen\slcwa\half_neg\pykeen_complex_umls_gpu_1')
# runtime_report('integrated_pykeen\slcwa\half_neg\pykeen_complex_umls_gpu_2')
# runtime_report('integrated_pykeen\slcwa\half_neg\pykeen_complex_umls_cpu')

# runtime_report('integrated_pykeen\slcwa\half_neg\pykeen_complex_kinship_gpu_1')
# runtime_report('integrated_pykeen\slcwa\half_neg\pykeen_complex_kinship_gpu_2')
# runtime_report('integrated_pykeen\slcwa\half_neg\pykeen_complex_kinship_cpu')


# runtime_report('pykeen_complex_umls_gpu_1')
# runtime_report('pykeen_complex_umls_gpu_2')
# runtime_report('pykeen_complex_umls_cpu')

# runtime_report('pykeen_complex_kinship_gpu_1')
# runtime_report('pykeen_complex_kinship_gpu_2')
# runtime_report('pykeen_complex_kinship_cpu')


# runtime_report('kvsall/distmult_umls_cpu')
# runtime_report('kvsall/distmult_umls_gpu_1')
# runtime_report('kvsall/distmult_umls_gpu_2')


# runtime_report('pykeen_complex_umls_cpu/')
# runtime_report('pykeen_complex_umls_gpu_1/')
# runtime_report('dice_small/complex_umls_gpu_2/')
# runtime_report('dice_small/complex_kinships_gpu_2/')

# runtime_report('pykeen_distmult_kinship_cpu')
# runtime_report('pykeen_distmult_kinship_gpu_1')
# runtime_report('pykeen_distmult_kinship_gpu_2')


# runtime_report('pykeen_complex_kinship_cpu')
# runtime_report('pykeen_complex_kinship_gpu_1')
# runtime_report('pykeen_complex_kinship_gpu_2')


# runtime_report('integrated_pykeen\slcwa\pykeen_complex_kinship_cpu')
# runtime_report('integrated_pykeen\slcwa\pykeen_complex_kinship_gpu_1')
# runtime_report('integrated_pykeen\slcwa\pykeen_complex_kinship_gpu_2')


# evaluation_pykeen_report('pykeen_benchmarks\lcwa\cpu\pykeen_distmultumls')
# evaluation_pykeen_report('pykeen_benchmarks\lcwa\cpu\pykeen_Distmult_kinships')
# evaluation_pykeen_report('pykeen_benchmarks\lcwa\cpu\pykeen_ComplEx_umls')
# evaluation_pykeen_report('pykeen_benchmarks\lcwa\cpu\pykeen_ComplEx_kinships')


# evaluation_report('dice_small/slcwa/slcwa_pykeen_distmult_umls_cpu')
# evaluation_report('dice_small/slcwa/slcwa_pykeen_distmult_umls_gpu_1')
# evaluation_report('dice_small/slcwa/slcwa_pykeen_distmult_umls_gpu_2')
# # evaluation_report('dice_small/slcwa/slcwa_pykeen_complex_umls_gpu_2')
# evaluation_report('dice_small/slcwa/slcwa16_pykeen_distmult_umls_cpu')
# evaluation_report('dice_small/slcwa/slcwa16_pykeen_distmult_umls_gpu_1')
# evaluation_report('dice_small/slcwa/slcwa16_pykeen_distmult_umls_gpu_2')
# evaluation_report('dice_small/slcwa/slcwa32_pykeen_distmult_umls_cpu')
# evaluation_report('dice_small/slcwa/slcwa32_pykeen_distmult_umls_gpu_1')
# evaluation_report('dice_small/slcwa/slcwa32_pykeen_distmult_umls_gpu_2')


# evaluation_report('dice_small/slcwa/slcwa_pykeen_distmult_kinship_cpu')
# evaluation_report('dice_small/slcwa/slcwa_pykeen_distmult_kinship_gpu_1')
# evaluation_report('dice_small/slcwa/slcwa_pykeen_distmult_kinship_gpu_2')
# # evaluation_report('dice_small/slcwa/slcwa_pykeen_complex_kinship_gpu_2')
# evaluation_report('dice_small/slcwa/slcwa16_pykeen_distmult_kinship_cpu')
# evaluation_report('dice_small/slcwa/slcwa16_pykeen_distmult_kinship_gpu_1')
# evaluation_report('dice_small/slcwa/slcwa16_pykeen_distmult_kinship_gpu_2')
# evaluation_report('dice_small/slcwa/slcwa32_pykeen_distmult_kinship_cpu')
# evaluation_report('dice_small/slcwa/slcwa32_pykeen_distmult_kinship_gpu_1')
# evaluation_report('dice_small/slcwa/slcwa32_pykeen_distmult_kinship_gpu_2')



# evaluation_report('dice_small/slcwa/slcwa_pykeen_complex_umls_cpu')
# evaluation_report('dice_small/slcwa/slcwa_pykeen_complex_umls_gpu_1')
# evaluation_report('dice_small/slcwa/slcwa_pykeen_complex_umls_gpu_2')
# # evaluation_report('dice_small/slcwa/slcwa_pykeen_complex_umls_gpu_2')
# evaluation_report('dice_small/slcwa/slcwa16_pykeen_complex_umls_cpu')
# evaluation_report('dice_small/slcwa/slcwa16_pykeen_complex_umls_gpu_1')
# evaluation_report('dice_small/slcwa/slcwa16_pykeen_complex_umls_gpu_2')
# evaluation_report('dice_small/slcwa/slcwa32_pykeen_complex_umls_cpu')
# evaluation_report('dice_small/slcwa/slcwa32_pykeen_complex_umls_gpu_1')
# evaluation_report('dice_small/slcwa/slcwa32_pykeen_complex_umls_gpu_2')


# evaluation_report('dice_small/slcwa/slcwa_pykeen_complex_kinship_cpu')
# evaluation_report('dice_small/slcwa/slcwa_pykeen_complex_kinship_gpu_1')
# evaluation_report('dice_small/slcwa/slcwa_pykeen_complex_kinship_gpu_2')
# evaluation_report('dice_small/slcwa/slcwa16_pykeen_complex_kinship_cpu')
# evaluation_report('dice_small/slcwa/slcwa16_pykeen_complex_kinship_gpu_1')
# evaluation_report('dice_small/slcwa/slcwa16_pykeen_complex_kinship_gpu_2')
# evaluation_report('dice_small/slcwa/slcwa32_pykeen_complex_kinship_cpu')
# evaluation_report('dice_small/slcwa/slcwa32_pykeen_complex_kinship_gpu_1')
# evaluation_report('dice_small/slcwa/slcwa32_pykeen_complex_kinship_gpu_2')



# evaluation_report('dice_small/complex_umls_cpu')
# evaluation_report('dice_small/complex_umls_gpu_1')
# evaluation_report('dice_small/complex_umls_gpu_2')
# evaluation_report('dice_small/complex_umls_gpu_3')


# evaluation_report('dice_small/complex_kinships_cpu')
# evaluation_report('dice_small/complex_kinships_gpu_1')
# evaluation_report('dice_small/complex_kinships_gpu_2')
# evaluation_report('dice_small/complex_kinships_gpu_3')

# evaluation_report('dice_small/pykeen_distmult_umls_cpu')
# evaluation_report('dice_small/pykeen_distmult_umls_gpu_1')
# evaluation_report('dice_small/pykeen_distmult_umls_gpu_2')
# evaluation_report('dice_small/pykeen_distmult_umls_gpu_3')

# evaluation_report('dice_small/pykeen_complex_kinship_cpu')
# evaluation_report('dice_small/pykeen_complex_kinship_gpu_1')
# evaluation_report('dice_small/pykeen_complex_kinship_gpu_2')
# evaluation_report('dice_small/pykeen_complex_kinship_gpu_3')

# evaluation_report('dice_small/kvsall_complex_umls_cpu')
# evaluation_report('dice_small/kvsall_complex_umls_gpu_1')
# evaluation_report('dice_small/kvsall_complex_umls_gpu_2')
# evaluation_report('dice_small/kvsall_complex_umls_gpu_3')

# evaluation_report('dice_small/kvsall_distmult_kinships_cpu')
# evaluation_report('dice_small/kvsall_distmult_kinships_gpu_1')
# evaluation_report('dice_small/kvsall_distmult_kinships_gpu_2')
# evaluation_report('dice_small/kvsall_distmult_kinships_gpu_3')


# evaluation_report('dice_small/pykeen_complex_umls_cpu')
# evaluation_report('dice_small/pykeen_complex_umls_gpu_1')
# evaluation_report('dice_small/pykeen_complex_umls_gpu_2')
# evaluation_report('dice_small/pykeen_complex_umls_gpu_3')

# evaluation_report('dice_small/pykeen_complex_kinship_cpu')
# evaluation_report('dice_small/pykeen_complex_kinship_gpu_1')
# evaluation_report('dice_small/pykeen_complex_kinship_gpu_2')
# evaluation_report('dice_small/pykeen_complex_kinship_gpu_3')


# runtime_report('dice_small/slcwa/slcwa16_pykeen_complex_kinship_cpu')
# runtime_report('dice_small/slcwa/slcwa16_pykeen_complex_kinship_gpu_1')
# runtime_report('kvsall_distmult_kinships_gpu_3/')
# runtime_report('kvsall_distmult_kinships_cpu/')


# runtime_report('slcwa_pykeen_distmult_umls_gpu_1/')
# runtime_report('slcwa_pykeen_distmult_umls_gpu_2/')
# runtime_report('slcwa_pykeen_distmult_umls_cpu/')

# runtime_report('slcwa_pykeen_distmult_kinship_gpu_1/')
# runtime_report('slcwa_pykeen_distmult_kinship_gpu_2/')
# runtime_report('slcwa_pykeen_distmult_kinship_cpu/')

# runtime_report('slcwa_pykeen_complex_umls_cpu/')
# runtime_report('slcwa_pykeen_complex_umls_gpu_1/')
# runtime_report('slcwa_pykeen_complex_umls_gpu_2/')

# runtime_report('slcwa_pykeen_complex_kinship_cpu/')
# runtime_report('slcwa_pykeen_complex_kinship_gpu_1/')
# runtime_report('slcwa_pykeen_complex_kinship_gpu_2/')

# pykeen_runtime_report1('pykeen_small/cpu_umls_distmult/')
# pykeen_runtime_report1('pykeen_small/gpu_umls_distmult/')
# pykeen_runtime_report1('pykeen_small/cpu_kinship_distmult/')
# pykeen_runtime_report1('pykeen_small/gpu_kinship_distmult/')

# pykeen_runtime_report1('pykeen_small/cpu_umls_complex/')
# pykeen_runtime_report1('pykeen_small/gpu_umls_complex/')
# pykeen_runtime_report1('pykeen_small/cpu_kinship_complex/')
# pykeen_runtime_report1('pykeen_small/gpu_kinship_complex/')