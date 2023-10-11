import subprocess
import json
import pandas as pd



class Keci_exp:

    def __init__(self, emb_dim : int , path_main : str, num_epochs : int ,batch_size : int, neg_ratio : int):

        self.parameter_values = range(emb_dim+1)
        self.path_main = path_main
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.emb_dim = emb_dim
        self.results = dict()
        

    def results_keci(self) -> pd.DataFrame :
        '''This function store the performance of the Keci_r model into a dataframe for all possible values of p,q,r in [0, emb_dim]'''

        df = pd.DataFrame()

        i = 0 #for indexing
        for p in self.parameter_values:
            for q in self.parameter_values:
                for r in self.parameter_values:
                
                    if 16%(p+q+r+1) == 0:
                            subprocess_output = subprocess.run(["python", self.path_main,"--p",str(p) ,"--q",str(q), "--r", str(r), "--neg_ratio",str(self.neg_ratio)\
                                            ,"--batch_size",str(self.batch_size),"--num_epochs",str(self.num_epochs)],stdout=subprocess.PIPE,universal_newlines=True,)
                            eval_output = subprocess_output.stdout.strip()

                            result_output = eval_output[-147:-30]
                            #print(result_output)

                            time_output = eval_output[-29:-1]
                            #print(time_output)
                            parts = time_output.split()
                            run_time = float(parts[-2])

                            start_index = result_output.find("{")

                            data_str = result_output[start_index:]

                            data_str = data_str.replace("'", "\"")

                            data_dict = json.loads(data_str)

                            self.results.update({'p':p,'q':q,'r':r})
                            self.results.update(data_dict)
                            self.results.update({'time(s)': run_time})
                            
                            dfi = pd.DataFrame(self.results, index = [i])

                            df = pd.concat([df,dfi])

                            i += 1
        # df.to_csv('keci_exp_UMLS.csv') #save as csv.
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

        # print(l)
        # print(max_MRR)
        # print((opt_p,opt_q,opt_r))

        return (opt_p,opt_q,opt_r), max_MRR, l

    def MRR(self, p,q,r):
        '''Return the achieved MRR of Keci_r for a fixed p,q and r'''

        subprocess_output = subprocess.run(["python", self.path_main,"--p",str(p) ,"--q",str(q), "--r", str(r), "--neg_ratio",str(self.neg_ratio)\
                                            ,"--batch_size",str(self.batch_size),"--num_epochs",str(self.num_epochs)],stdout=subprocess.PIPE,universal_newlines=True,)
        eval_output = subprocess_output.stdout.strip()
        
        result_output = eval_output[-147:-30]

        start_index = result_output.find("{")

        data_str = result_output[start_index:]

        data_str = data_str.replace("'", "\"")

        data_dict = json.loads(data_str)

        return data_dict['MRR']


