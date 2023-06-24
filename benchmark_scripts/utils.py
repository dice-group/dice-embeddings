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


# evaluation_report('complex_kinships_cpu')



# runtime_report('distmult_umls_cpu/')
# runtime_report('distmult_umls_gpu_1/')
# runtime_report('distmult_umls_gpu_2/')
# runtime_report('distmult_umls_gpu_3/')


# runtime_report('complex_umls_cpu/')
# runtime_report('complex_umls_gpu_1/')
# runtime_report('complex_umls_gpu_2/')
# runtime_report('complex_umls_gpu_3/')

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
# runtime_report('pykeen_complex_umls_gpu_2/')


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


# evaluation_report('integrated_pykeen\pykeen_distmult_umls_cpu')
# evaluation_report('integrated_pykeen\pykeen_distmult_kinship_cpu')
# evaluation_report('integrated_pykeen\pykeen_complex_umls_cpu')
# evaluation_report('integrated_pykeen\pykeen_complex_kinship_cpu')


# evaluation_report('dice_benchmark\distmult_umls_cpu')
# evaluation_report('dice_benchmark\distmult_kinships_cpu')
# evaluation_report('dice_benchmark\complex_umls_cpu')
# evaluation_report('dice_benchmark\complex_kinships_cpu')


# pykeen_runtime_report('pykeen_benchmarks\slcwa\half_neg_sampl\gpu\pykeen_distmultumls')
# pykeen_runtime_report('pykeen_benchmarks\slcwa\half_neg_sampl\gpu\pykeen_Distmult_kinships')
# pykeen_runtime_report('pykeen_benchmarks\slcwa\half_neg_sampl\gpu\pykeen_ComplEx_umls')
# pykeen_runtime_report('pykeen_benchmarks\slcwa\half_neg_sampl\gpu\pykeen_ComplEx_kinships')

# pykeen_runtime_report('pykeen_benchmarks\slcwa\half_neg_sampl\gpu\pykeen_distmultumls')
# pykeen_runtime_report('pykeen_benchmarks\slcwa\half_neg_sampl\gpu\pykeen_Distmult_kinships')
# pykeen_runtime_report('pykeen_benchmarks\slcwa\half_neg_sampl\gpu\pykeen_ComplEx_umls')
# pykeen_runtime_report('pykeen_benchmarks\slcwa\half_neg_sampl\gpu\pykeen_ComplEx_kinships')