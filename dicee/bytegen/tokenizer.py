from typing import List, Union, Tuple
import torch

class ByteTokenizer:
    """
    Tokenizer for converting Knowledge Graph components (entities, relations) 
    into byte sequences with special structural tokens.
    """
    def __init__(self, pad_token_id: int = 256, sep_hr_token_id: int = 257, sep_rt_token_id: int = 258, vocab_size: int = 259):
        self.pad_token_id = pad_token_id
        self.sep_hr_token_id = sep_hr_token_id
        self.sep_rt_token_id = sep_rt_token_id
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

