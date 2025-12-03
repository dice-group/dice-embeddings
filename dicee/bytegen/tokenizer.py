from typing import List, Union, Tuple
import torch
import os

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

class ByteTokenizer:
    """
    Tokenizer for converting Knowledge Graph components (entities, relations) 
    into byte sequences with special structural tokens.
    """
    def __init__(self, pad_token_id: int = 256, sep_hr_token_id: int = 257, sep_rt_token_id: int = 258, eos_token_id: int = 259, vocab_size: int = 260):
        self.pad_token_id = pad_token_id
        self.sep_hr_token_id = sep_hr_token_id
        self.sep_rt_token_id = sep_rt_token_id
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        """Converts a string into a list of utf-8 byte integers."""
        return list(text.encode('utf-8'))

    def decode(self, tokens: Union[List[int], torch.Tensor], remove_special_tokens: bool = False) -> str:
        """
        Converts a list of token IDs (bytes + special tokens) back into a string.
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
            
        decoded = ""
        buffer = bytearray()
        
        for token in tokens:
            if token < 256:
                buffer.append(token)
            else:
                # Flush buffer when we hit a special token
                if buffer:
                    try:
                        decoded += buffer.decode('utf-8')
                    except UnicodeDecodeError:
                        # Fallback for partial/invalid bytes
                        decoded += str(buffer)
                    buffer = bytearray()
                
                if not remove_special_tokens:
                    if token == self.sep_hr_token_id:
                        decoded += " <SEP_HR> "
                    elif token == self.sep_rt_token_id:
                        decoded += " <SEP_RT> "
                    elif token == self.eos_token_id:
                        decoded += " <EOS> "
                    
                    elif token == self.pad_token_id:
                        decoded += " <PAD> "
        
        # Flush remaining buffer
        if buffer:
            try:
                decoded += buffer.decode('utf-8')
            except UnicodeDecodeError:
                decoded += str(buffer)
                
        return decoded.strip()

    def triple_to_ids(self, h: str, r: str, t: str) -> List[int]:
        """
        Converts a triple (head, relation, tail) into a token sequence:
        [h_bytes, SEP_HR, r_bytes, SEP_RT, t_bytes]
        """
        return (
            self.encode(h) + 
            [self.sep_hr_token_id] + 
            self.encode(r) + 
            [self.sep_rt_token_id] + 
            self.encode(t)
        )
        
    def batch_decode(self, batch_tokens: Union[List[List[int]], torch.Tensor]) -> List[str]:
        """Decodes a batch of token sequences."""
        if isinstance(batch_tokens, torch.Tensor):
            batch_tokens = batch_tokens.tolist()
        return [self.decode(seq) for seq in batch_tokens]


class BPETokenizer:
    """
    BPE Tokenizer using HuggingFace tokenizers library.
    """
    def __init__(self, vocab_size: int = 30000, path: str = None):
        self.vocab_size = vocab_size
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.tokenizer = Tokenizer(models.BPE())
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self.tokenizer.decoder = decoders.ByteLevel()
            
            # Placeholder IDs until trained/loaded
            self.pad_token_id = 0
            self.sep_hr_token_id = 1
            self.sep_rt_token_id = 2
            self.eos_token_id = 3

    def train(self, files: List[str]):
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size, 
            special_tokens=["<PAD>", "<SEP_HR>", "<SEP_RT>", "<EOS>", "<UNK>"]
        )
        self.tokenizer.train(files, trainer)
        self._update_ids()

    def _update_ids(self):
        self.pad_token_id = self.tokenizer.token_to_id("<PAD>")
        self.sep_hr_token_id = self.tokenizer.token_to_id("<SEP_HR>")
        self.sep_rt_token_id = self.tokenizer.token_to_id("<SEP_RT>")
        self.eos_token_id = self.tokenizer.token_to_id("<EOS>")
        self.vocab_size = self.tokenizer.get_vocab_size()

    def save(self, path: str):
        self.tokenizer.save(path)

    def load(self, path: str):
        self.tokenizer = Tokenizer.from_file(path)
        self._update_ids()

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, tokens: Union[List[int], torch.Tensor], remove_special_tokens: bool = False) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens, skip_special_tokens=remove_special_tokens)

    def triple_to_ids(self, h: str, r: str, t: str) -> List[int]:
        return (
            self.encode(h) + 
            [self.sep_hr_token_id] + 
            self.encode(r) + 
            [self.sep_rt_token_id] + 
            self.encode(t)
        )
    
    def batch_decode(self, batch_tokens: Union[List[List[int]], torch.Tensor]) -> List[str]:
        if isinstance(batch_tokens, torch.Tensor):
            batch_tokens = batch_tokens.tolist()
        return self.tokenizer.decode_batch(batch_tokens)

def train_bpe_tokenizer(folder_path: str, save_path: str, vocab_size: int = 30000, inverse: bool = False):
    """
    Helper function to train a BPE tokenizer on the dataset files.
    
    Args:
        folder_path: Path to the dataset folder
        save_path: Path to save the trained tokenizer
        vocab_size: Vocabulary size for BPE
        inverse: If True, also include inverse relations (INV_<relation>) in training data
    """
    import tempfile
    
    if not inverse:
        # Original behavior: train on raw files
        files = []
        for split in ['train', 'valid', 'test']:
            p = os.path.join(folder_path, f"{split}.txt")
            if os.path.exists(p):
                files.append(p)
                
        if not files:
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
            
        if not files:
            raise ValueError(f"No .txt files found in {folder_path}")

        print(f"Training BPE tokenizer on {files}...")
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        tokenizer.train(files)
    else:
        # Include inverse relations in training data
        all_text = []
        for split in ['train', 'valid', 'test']:
            p = os.path.join(folder_path, f"{split}.txt")
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) < 3:
                            parts = line.strip().split()
                        if len(parts) >= 3:
                            h, r, t = parts[0], parts[1], parts[2]
                            all_text.append(f"{h}\t{r}\t{t}")
                            all_text.append(f"{t}\tINV_{r}\t{h}")  # Add inverse
        
        if not all_text:
            raise ValueError(f"No triples found in {folder_path}")
        
        # Write to temp file and train
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write('\n'.join(all_text))
            temp_path = f.name
        
        print(f"Training BPE tokenizer with inverse relations ({len(all_text)} lines)...")
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        tokenizer.train([temp_path])
        os.unlink(temp_path)  # Clean up temp file
    
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")
    return tokenizer