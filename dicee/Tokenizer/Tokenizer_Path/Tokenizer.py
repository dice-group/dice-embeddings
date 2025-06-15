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
                    # Skip header line if it's just a count (common in knowledge graph datasets)
                    first_line = f.readline().strip()
                    if not first_line.isdigit():
                        all_lines.append(first_line)
                    
                    # Process rest of the file
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            all_lines.append(line)
    
    return all_lines

def main():
    # Define paths to datasets
    umls_path = "C:\\Users\\Harshit Purohit\\KGDatasets\\UMLS"  # Replace with actual path
    kinship_path = "C:\\Users\\Harshit Purohit\\KGDatasets\\Kinship"  # Replace with actual path
    output_dir = "C:\\Users\\Harshit Purohit\\Tokenizer"
    
    # Load content from both datasets
    corpus_lines = load_dataset_content([umls_path, kinship_path])
    
    # Initialize tokenizer with BPE model
    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))

    # Configure the trainer
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        vocab_size=10000,  # Adjust as needed
        min_frequency=2    # Minimum frequency for a token to be included
    )
    
    # Set pre-tokenizer to split on whitespace
    tokenizer.pre_tokenizer = WhitespaceSplit()
    
    # Train the tokenizer using the iterator of lines
    # This completes the missing part in the original code snippet
    tokenizer.train_from_iterator(corpus_lines, trainer)
    
    # Save the raw tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    
    # Convert to PreTrainedTokenizerFast as in the original code
    pretrained_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    new_tokens = [" "]
    added_count = pretrained_tokenizer.add_tokens(new_tokens)
    print(f"Added {added_count} new token(s): {new_tokens}")

    pretrained_tokenizer.save_pretrained(output_dir)
    
    print(f"Tokenizer training completed and saved to {output_dir}")
    
    # Example usage
    test_text = "ent_1 rel_1 ent_2"
    encoded = pretrained_tokenizer.encode(test_text)
    print(f"Example encoding: {encoded}")
    print(f"Decoded tokens: {pretrained_tokenizer.convert_ids_to_tokens(encoded.ids)}")

if __name__ == "__main__":
    main()
