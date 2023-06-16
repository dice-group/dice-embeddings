from pykeen.hpo.hpo import hpo_pipeline_from_path,hpo_pipeline_from_config
import os
import json



def read_config():
    for filename in os.listdir('./hpo_config/'):
        dot_index = filename.find('.')
        dir_name = filename[:dot_index]
        hpo_pipeline_result = hpo_pipeline_from_path('./hpo_config/' + filename)
        hpo_pipeline_result.save_to_directory(os.path.join('./hpo_results/',dir_name))
        

        


if __name__ == '__main__':
    read_config()




