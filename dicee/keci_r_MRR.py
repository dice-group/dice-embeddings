import subprocess
import json
import pandas as pd
import os



class Keci_exp:

    def __init__(self, emb_dim : int , path_main : str):

        self.parameter_values = range(emb_dim+1)
        self.path_main = path_main
        # self.num_epochs = num_epochs
        # self.batch_size = batch_size
        # self.neg_ratio = neg_ratio
        self.emb_dim = emb_dim
        self.results = dict()
        

    def results_keci(self) -> pd.DataFrame :
        '''This function store the performance of the Keci_r model into a dataframe for all possible values of p,q,r in [0, emb_dim]'''

        for p in self.parameter_values:
            for q in self.parameter_values:
                for r in self.parameter_values:
                
                    if 16%(p+q+r+1) == 0:
                           
                        folder_name = f"keci_{p}{q}{r}"
                        folder_path = os.path.join("Experiments", folder_name)
                        subprocess.run(["python", self.path_main,"--p",str(p) ,"--q",str(q), "--r", str(r),"--storage_path",folder_path,"--num_epochs",str(1)])
                            

        experiments_path = '/home/dice/Desktop/dice-embeddings/Experiments'
        
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



    def params_search(self, start_point = (1,1,1)):
         
        '''Gradient based searching parameters starting at a fixed point named start point.
        Default starting point is (1,1,1)
        NB: here the best performance is achieved with the highest MRR
        Return a tuple with the best triple (p,q,r), the corresponding MRR and the list of all the other configurations.'''

        (opt_p, opt_q, opt_r) = start_point
        p0, q0, r0 = opt_p, opt_q, opt_r
        max_MRR = 0
        l = []
        d = self.emb_dim

        for p in range(p0, d):
            for q in range(q0, d):
                for r in range(q0, d):
                    if ((p,q,r) not in l) & (16%(1+p+q+r)==0):
                        l.append((p, q, r))
                        Mrr = self.MRR(p,q,r)

                        if Mrr > max_MRR:
                            max_MRR = Mrr
                            (opt_p, opt_q, opt_r) = (p,q,r)

        for p in range(p0, -1, -1):
            for q in range(q0, -1, -1):
                for r in range(q0, -1, -1):
                    if ((p,q,r) not in l) & (16%(1+p+q+r)==0):
                        l.append((p, q, r))
                        Mrr = self.MRR(p,q,r)

                        if Mrr > max_MRR:
                            max_MRR = Mrr
                            (opt_p, opt_q, opt_r) = (p,q,r)

        for p in range(p0, d):
            for q in range(q0, -1, -1):
                for r in range(r0, -1, -1):
                    if ((p,q,r) not in l) & (16%(1+p+q+r)==0):
                        l.append((p, q, r))
                        Mrr = self.MRR(p,q,r)

                        if Mrr > max_MRR:
                            max_MRR = Mrr
                            (opt_p, opt_q, opt_r) = (p,q,r)

        for p in range(p0, -1, -1):
            for q in range(q0, d):
                for r in range(r0, -1, -1):
                    if ((p,q,r) not in l) & (16%(1+p+q+r)==0):
                        l.append((p, q, r))
                        Mrr = self.MRR(p,q,r)

                        if Mrr > max_MRR:
                            max_MRR = Mrr
                            (opt_p, opt_q, opt_r) = (p,q,r)


        for p in range(p0, -1, -1):
            for q in range(q0, -1, -1):
                for r in range(r0, d):
                    if ((p,q,r) not in l) & (16%(1+p+q+r)==0):
                        l.append((p, q, r))
                        Mrr = self.MRR(p,q,r)

                        if Mrr > max_MRR:
                            max_MRR = Mrr
                            (opt_p, opt_q, opt_r) = (p,q,r)


        for p in range(p0, d):
            for q in range(q0, d):
                for r in range(r0, -1, -1):
                    if ((p,q,r) not in l) & (16%(1+p+q+r)==0):
                        l.append((p, q, r))
                        Mrr = self.MRR(p,q,r)

                        if Mrr > max_MRR:
                            max_MRR = Mrr
                            (opt_p, opt_q, opt_r) = (p,q,r)


        for p in range(p0, d):
            for q in range(q0, -1, -1):
                for r in range(r0, d):
                    if ((p,q,r) not in l) & (16%(1+p+q+r)==0):
                        l.append((p, q, r))
                        Mrr = self.MRR(p,q,r)

                        if Mrr > max_MRR:
                            max_MRR = Mrr
                            (opt_p, opt_q, opt_r) = (p,q,r)


        for p in range(p0, -1, -1):
            for q in range(q0, d):
                for r in range(r0, d):
                    if ((p,q,r) not in l) & (16%(1+p+q+r)==0):
                        l.append((p, q, r))
                        Mrr = self.MRR(p,q,r)

                        if Mrr > max_MRR:
                            max_MRR = Mrr
                            (opt_p, opt_q, opt_r) = (p,q,r)

        return (opt_p,opt_q,opt_r), max_MRR, l

    def MRR(self, p,q,r):
        '''Return the achieved MRR of Keci_r for a fixed p,q and r on the train data set'''

        
        folder_name = f"keci_{p}{q}{r}"
        folder_path = os.path.join("Experiments", folder_name)
        subprocess.run(["python", self.path_main,"--p",str(p) ,"--q",str(q), "--r", str(r),"--storage_path",folder_path,"--num_epochs",str(1)])

        experiments_path = folder_path

        for folder in os.listdir(experiments_path):

            if os.path.isdir(os.path.join(experiments_path, folder)):
                folder_path = os.path.join(experiments_path, folder)
                report_path = os.path.join(folder_path, 'eval_report.json')

                if os.path.exists(report_path):
                    with open(report_path, 'r') as file:
                        report_data = json.load(file)

        return report_data['Train']['MRR']


