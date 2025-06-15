import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

def load_dataset_content(dataset_paths):
    """
    Load content from datasets for tokenizer training.
    
    Args:
        dataset_paths: List of paths to datasets
        
    Returns:
        List of text lines for training
    """
    all_lines = []
    
    for dataset_path in dataset_paths:
        for filename in ["train.txt", "valid.txt", "test.txt"]:
            file_path = os.path.join(dataset_path, filename)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if not first_line.isdigit():
                        all_lines.append(first_line)
                    
                    for line in f:
                        line = line.strip()
                        if line:
                            all_lines.append(line)
    
    return all_lines

def main():
    umls_path = "KGs/UMLS"  
    countries_s1_path = "KGs/Countries-S1"
    countries_s2_path = "KGs/Countries-S2"
    countries_s3_path = "KGs/Countries-S3"
    kinship_path = "KGs/KINSHIP"
    nell_h100 = "KGs/NELL-995-h100"
    nell_h75 = "KGs/NELL-995-h75"
    nell_h25 = "KGs/NELL-995-h25"
    fb_15k_237 = "KGs/FB15k-237"


    output_dir = "C:\\Users\\Harshit Purohit\\Tokenizer"
    
    corpus_lines = load_dataset_content([countries_s1_path])
    # corpus_lines = load_dataset_content([countries_s1_path])
    # corpus_lines = load_dataset_content([countries_s2_path])
    # corpus_lines = load_dataset_content([countries_s3_path])
    
    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))

    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        vocab_size=1000000,  
        min_frequency=2,
        max_token_length=100
    )
    
    tokenizer.pre_tokenizer = WhitespaceSplit()
    
    
    tokenizer.train_from_iterator(corpus_lines, trainer)
    
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    
    pretrained_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    new_tokens = [" "]
    added_count = pretrained_tokenizer.add_tokens(new_tokens)
    print(f"Added {added_count} new token(s): {new_tokens}")

    pretrained_tokenizer.save_pretrained(output_dir)
    
    print(f"Tokenizer training completed and saved to {output_dir}")
    
    test_text = "ent_1 rel_1 ent_2"
    encoded = pretrained_tokenizer.encode(test_text)
    print(f"Example encoding: {encoded}")

if __name__ == "__main__":
    main()
