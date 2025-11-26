from typing import List, Union, Tuple
import torch

class GraphTokenizer:
    """
    Tokenizer for converting Knowledge Graph components (entities, relations) 
    into byte sequences with special structural tokens.
    """
    PAD_TOKEN_ID = 256
    SEP_HR_TOKEN_ID = 257
    SEP_RT_TOKEN_ID = 258
    VOCAB_SIZE = 259

    def __init__(self):
        pass

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
                    if token == self.SEP_HR_TOKEN_ID:
                        decoded += " <SEP_HR> "
                    elif token == self.SEP_RT_TOKEN_ID:
                        decoded += " <SEP_RT> "
                    # PAD is usually ignored or handled by strip() if at ends, 
                    # but here we just skip or add placeholder?
                    # Usually decode() is for human readability, so we might omit PAD or show it.
                    # Let's omit PAD by default in output string or represent it?
                    # The reference implementation didn't handle PAD explicitly in the loop shown,
                    # assuming it breaks or filters before. 
                    # Let's add a placeholder if not removing.
                    elif token == self.PAD_TOKEN_ID:
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
            [self.SEP_HR_TOKEN_ID] + 
            self.encode(r) + 
            [self.SEP_RT_TOKEN_ID] + 
            self.encode(t)
        )
        
    def batch_decode(self, batch_tokens: Union[List[List[int]], torch.Tensor]) -> List[str]:
        """Decodes a batch of token sequences."""
        if isinstance(batch_tokens, torch.Tensor):
            batch_tokens = batch_tokens.tolist()
        return [self.decode(seq) for seq in batch_tokens]

