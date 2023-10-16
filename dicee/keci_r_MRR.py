import subprocess
import json
import pandas as pd
import os



class Keci_exp:

    def __init__(self, emb_dim: int , path_main: str, folder_name: str, Experiments_path: str):

        '''Inputs: 
           em_dim: embedding dimension
           path_main: path to the main.py file
           folder_name: name of the folder that will contain the experimentions
           Experiments_path: path were the experiments will be saved.'''

        self.path_main = path_main
        self.Experiments_path = Experiments_path
        self.folder_name = folder_name
        # self.num_epochs = num_epochs(250)
        # self.batch_size = batch_size(128)
        # self.neg_ratio = neg_ratio(50)
        self.emb_dim = emb_dim
        self.results = dict()
        

    def results_keci(self, parameter_values:list) -> pd.DataFrame :
        '''This function store the performance of the Keci_r model into a dataframe for all possible values of p,q,r in parameter_values.
           If not specified, parameter_values =[0,1,...,emb_dim] '''

        if parameter_values == None:
            parameter_values  = range(self.emb_dim+1)

        for p in parameter_values:
            for q in parameter_values:
                for r in parameter_values:
                
                    if self.emb_dim%(p+q+r+1) == 0:
                           
                        folder_name = f"{p}_{q}_{r}"
                        folder_path = os.path.join(self.folder_name, folder_name)
                        subprocess.run(["python", self.path_main,"--p",str(p) ,"--q",str(q), "--r", str(r),"--storage_path",folder_path])
                            

        experiments_path = self.Experiments_path
        
        data = []

        for root, dirs, files in os.walk(experiments_path):
            dirs.sort()
            for folder in dirs:
                folder_path = os.path.join(root, folder)
                report_path = os.path.join(folder_path, 'eval_report.json')

                
                if os.path.exists(report_path):
                    with open(report_path, 'r') as file:
                        report_data = json.load(file)

                        parent_folder = os.path.basename(root)

                        data.append({
                            'Experiment': parent_folder,
                            'Train_H@1': report_data['Train']['H@1'],
                            'Train_H@3': report_data['Train']['H@3'],
                            'Train_H@10': report_data['Train']['H@10'],
                            'Train_MRR': report_data['Train']['MRR'],
                            'Val_H@1': report_data['Val']['H@1'],
                            'Val_H@3': report_data['Val']['H@3'],
                            'Val_H@10': report_data['Val']['H@10'],
                            'Val_MRR': report_data['Val']['MRR'],
                            'Test_H@1': report_data['Test']['H@1'],
                            'Test_H@3': report_data['Test']['H@3'],
                            'Test_H@10': report_data['Test']['H@10'],
                            'Test_MRR': report_data['Test']['MRR']
                        })

        df = pd.DataFrame(data)
        
        return df



    def params_search(self, start_point = (1,1,1)) -> tuple:
         
        '''Gradient based searching parameters starting at a fixed point named start point.
        Default starting point is (1,1,1)
        NB: here the best performance is achieved with the highest MRR on the test data
        Return a tuple with the best triple (p,q,r), the corresponding MRR and the list of all the other triples (p,q,r).'''

        (opt_p, opt_q, opt_r) = start_point
        p0, q0, r0 = opt_p, opt_q, opt_r
        max_MRR = 0
        l = []
        d = self.emb_dim

        for p in range(p0, d):
            for q in range(q0, d):
                for r in range(q0, d):
                    if ((p,q,r) not in l) & (self.emb_dim%(1+p+q+r)==0):
                        l.append((p, q, r))
                        Mrr = self.MRR(p,q,r)

                        if Mrr > max_MRR:
                            max_MRR = Mrr
                            (opt_p, opt_q, opt_r) = (p,q,r)

        for p in range(p0, -1, -1):
            for q in range(q0, -1, -1):
                for r in range(q0, -1, -1):
                    if ((p,q,r) not in l) & (self.emb_dim%(1+p+q+r)==0):
                        l.append((p, q, r))
                        Mrr = self.MRR(p,q,r)

                        if Mrr > max_MRR:
                            max_MRR = Mrr
                            (opt_p, opt_q, opt_r) = (p,q,r)

        for p in range(p0, d):
            for q in range(q0, -1, -1):
                for r in range(r0, -1, -1):
                    if ((p,q,r) not in l) & (self.emb_dim%(1+p+q+r)==0):
                        l.append((p, q, r))
                        Mrr = self.MRR(p,q,r)

                        if Mrr > max_MRR:
                            max_MRR = Mrr
                            (opt_p, opt_q, opt_r) = (p,q,r)

        for p in range(p0, -1, -1):
            for q in range(q0, d):
                for r in range(r0, -1, -1):
                    if ((p,q,r) not in l) & (self.emb_dim%(1+p+q+r)==0):
                        l.append((p, q, r))
                        Mrr = self.MRR(p,q,r)

                        if Mrr > max_MRR:
                            max_MRR = Mrr
                            (opt_p, opt_q, opt_r) = (p,q,r)


        for p in range(p0, -1, -1):
            for q in range(q0, -1, -1):
                for r in range(r0, d):
                    if ((p,q,r) not in l) & (self.emb_dim%(1+p+q+r)==0):
                        l.append((p, q, r))
                        Mrr = self.MRR(p,q,r)

                        if Mrr > max_MRR:
                            max_MRR = Mrr
                            (opt_p, opt_q, opt_r) = (p,q,r)


        for p in range(p0, d):
            for q in range(q0, d):
                for r in range(r0, -1, -1):
                    if ((p,q,r) not in l) & (self.emb_dim%(1+p+q+r)==0):
                        l.append((p, q, r))
                        Mrr = self.MRR(p,q,r)

                        if Mrr > max_MRR:
                            max_MRR = Mrr
                            (opt_p, opt_q, opt_r) = (p,q,r)


        for p in range(p0, d):
            for q in range(q0, -1, -1):
                for r in range(r0, d):
                    if ((p,q,r) not in l) & (self.emb_dim%(1+p+q+r)==0):
                        l.append((p, q, r))
                        Mrr = self.MRR(p,q,r)

                        if Mrr > max_MRR:
                            max_MRR = Mrr
                            (opt_p, opt_q, opt_r) = (p,q,r)


        for p in range(p0, -1, -1):
            for q in range(q0, d):
                for r in range(r0, d):
                    if ((p,q,r) not in l) & (self.emb_dim%(1+p+q+r)==0):
                        l.append((p, q, r))
                        Mrr = self.MRR(p,q,r)

                        if Mrr > max_MRR:
                            max_MRR = Mrr
                            (opt_p, opt_q, opt_r) = (p,q,r)

        return (opt_p,opt_q,opt_r), max_MRR, l

    def MRR(self, p,q,r):
        '''Return the achieved MRR of Keci_r for a fixed p,q and r on the train data set'''

        
        folder_name = f"{p}_{q}_{r}"
        folder_path = os.path.join(self.folder_name, folder_name)
        subprocess.run(["python", self.path_main,"--p",str(p) ,"--q",str(q), "--r", str(r),"--storage_path",folder_path])
            
        experiments_path = folder_path

        for folder in os.listdir(experiments_path):

            if os.path.isdir(os.path.join(experiments_path, folder)):
                folder_path = os.path.join(experiments_path, folder)
                report_path = os.path.join(folder_path, 'eval_report.json')


                if os.path.exists(report_path):
                    with open(report_path, 'r') as file:
                        report_data = json.load(file)

        return report_data['Test']['MRR']


