import pytest
import torch
from dicee.bytegen.tokenizer import ByteTokenizer

@pytest.fixture
def tokenizer():
    return ByteTokenizer()

def test_constants(tokenizer):
    assert tokenizer.pad_token_id == 256
    assert tokenizer.sep_hr_token_id == 257
    assert tokenizer.sep_rt_token_id == 258
    assert tokenizer.vocab_size == 259

def test_encode_basic(tokenizer):
    text = "hello"
    encoded = tokenizer.encode(text)
    expected = list(b"hello")
    assert encoded == expected

def test_encode_special_chars(tokenizer):
    text = "Café"
    encoded = tokenizer.encode(text)
    expected = list("Café".encode('utf-8'))
    assert encoded == expected

def test_triple_to_ids(tokenizer):
    h, r, t = "Head", "Relation", "Tail"
    ids = tokenizer.triple_to_ids(h, r, t)
    
    expected = (
        list(b"Head") + 
        [tokenizer.sep_hr_token_id] + 
        list(b"Relation") + 
        [tokenizer.sep_rt_token_id] + 
        list(b"Tail")
    )
    assert ids == expected

def test_decode_basic(tokenizer):
    tokens = list(b"hello")
    decoded = tokenizer.decode(tokens)
    assert decoded == "hello"

def test_decode_with_special_tokens(tokenizer):
    # "Head" <SEP_HR> "Rel" <SEP_RT> "Tail"
    tokens = (
        list(b"Head") + 
        [tokenizer.sep_hr_token_id] + 
        list(b"Rel") + 
        [tokenizer.sep_rt_token_id] + 
        list(b"Tail")
    )
    decoded = tokenizer.decode(tokens)
    assert decoded == "Head <SEP_HR> Rel <SEP_RT> Tail"

def test_decode_remove_special_tokens(tokenizer):
    tokens = (
        list(b"Head") + 
        [tokenizer.sep_hr_token_id] + 
        list(b"Rel") + 
        [tokenizer.sep_rt_token_id] + 
        list(b"Tail")
    )
    decoded = tokenizer.decode(tokens, remove_special_tokens=True)
    assert decoded == "HeadRelTail"

def test_batch_decode(tokenizer):
    batch = [
        list(b"hi"),
        list(b"yo")
    ]
    decoded = tokenizer.batch_decode(batch)
    assert decoded == ["hi", "yo"]

def test_torch_tensor_input(tokenizer):
    tensor = torch.tensor(list(b"test"), dtype=torch.long)
    decoded = tokenizer.decode(tensor)
    assert decoded == "test"
