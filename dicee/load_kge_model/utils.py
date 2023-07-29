import json
import os



class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


def load_args_from_file(file_folder):
  # check if configuration.json in the folder of the given file_path
  # folder_path = os.path.dirname(file_path)
  config_file_path = os.path.join(file_folder, "configuration.json")
  config_file_exist = os.path.isfile(config_file_path)
  
  if not config_file_exist:
    raise FileNotFoundError(f"The configuration.json file for args does not exist.")
  
  
  # Load the JSON file
  with open(config_file_path, "r") as file:
      args = json.load(file)

 
  return args
  
  




if __name__ == '__main__':
  # file_path = "../../Experiments/2023-07-29 22-42-44.871546/configuration.json"
  file_path = "E:\\DICEE\\dice-embeddings\\Experiments\\2023-07-29 22-42-44.871546\\configuration.json"
  # load_args_from_file(file_path)

