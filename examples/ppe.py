from dicee.executer import Execute
from dicee.config import Namespace

args = Namespace()
args.model = 'Keci'
args.p = 0
args.q = 1
args.scoring_technique = "KvsAll"
args.dataset_dir = "KGs/UMLS"
args.num_epochs = 200
args.lr = 0.1
args.embedding_dim = 32
args.batch_size = 1024
reports = Execute(args).start()
"""
Evaluate Keci on Train set: Evaluate Keci on Train set
{'H@1': 0.9966449386503068, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 0.9983064928425357}
Evaluate Keci on Validation set: Evaluate Keci on Validation set
{'H@1': 0.6134969325153374, 'H@3': 0.8098159509202454, 'H@10': 0.9424846625766872, 'MRR': 0.7293869361804316}
Evaluate Keci on Test set: Evaluate Keci on Test set
{'H@1': 0.6437216338880484, 'H@3': 0.8275340393343419, 'H@10': 0.959909228441755, 'MRR': 0.751216359363361}
Total Runtime: 13.259 seconds
"""
args = Namespace()
args.model = 'Keci'
args.p = 0
args.q = 1
args.scoring_technique = "KvsAll"
args.dataset_dir = "KGs/UMLS"
args.num_epochs = 200
args.lr = 0.1
args.embedding_dim = 32
args.batch_size = 1024
args.callbacks = {"PPE": {"epoch_to_start": 100}}
reports = Execute(args).start()
"""
Evaluate Keci on Train set: Evaluate Keci on Train set
{'H@1': 0.9934815950920245, 'H@3': 1.0, 'H@10': 1.0, 'MRR': 0.9966609151329243}
Evaluate Keci on Validation set: Evaluate Keci on Validation set
{'H@1': 0.7001533742331288, 'H@3': 0.8696319018404908, 'H@10': 0.9585889570552147, 'MRR': 0.7946759330503159}
Evaluate Keci on Test set: Evaluate Keci on Test set
{'H@1': 0.710287443267776, 'H@3': 0.8789712556732224, 'H@10': 0.9780635400907716, 'MRR': 0.8082179592109334}
Total Runtime: 12.497 seconds
"""