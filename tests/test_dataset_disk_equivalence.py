import os
import shutil
import pytest
import torch
from dicee.bytegen.dataset import ByteGenDataset
from dicee.bytegen.dataset_disk import DiskByteGenDataset, preprocess_dataset
from dicee.bytegen.tokenizer import train_bpe_tokenizer

@pytest.fixture
def umls_data_path(tmp_path):
    """
    Creates a temporary copy of the UMLS train.txt for testing.
    """
    # Locate the repository root relative to this test file
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    umls_source = os.path.join(repo_root, "KGs", "UMLS")
    
    if not os.path.exists(umls_source):
        pytest.skip("UMLS dataset not found in KGs/UMLS")
    
    # Create a temporary directory structure
    dest_dir = tmp_path / "UMLS"
    dest_dir.mkdir()
    
    # Copy train.txt to the temp directory
    shutil.copy(os.path.join(umls_source, "train.txt"), dest_dir / "train.txt")
    
    return str(dest_dir)

def test_dataset_equivalence(umls_data_path):
    """
    Verifies that ByteGenDataset (memory) and DiskByteGenDataset (disk)
    produce the same set of triples and have the same length when using UMLS.
    """
    print(f"\nTesting with data at: {umls_data_path}")

    # 1. Setup Tokenizer
    # We train a small BPE tokenizer on the fly for this test
    tokenizer_path = os.path.join(umls_data_path, "tokenizer.json")
    tokenizer = train_bpe_tokenizer(
        umls_data_path, 
        tokenizer_path, 
        vocab_size=1000, 
        inverse=True
    )
    
    # 2. Preprocess for Disk Dataset
    # This generates the .npy and .pkl files in umls_data_path/processed_mmap
    preprocess_dataset(umls_data_path, "train", tokenizer, inverse=True)
    
    # 3. Initialize Datasets
    block_size = 256
    
    # Disk-based
    processed_dir = os.path.join(umls_data_path, "processed_mmap")
    disk_ds = DiskByteGenDataset(processed_dir, tokenizer, block_size=block_size)
    
    # Memory-based
    mem_ds = ByteGenDataset(umls_data_path, tokenizer, split='train', block_size=block_size, inverse=True)
    
    # 4. Compare Lengths
    print(f"Disk Dataset Length: {len(disk_ds)}")
    print(f"Memory Dataset Length: {len(mem_ds)}")
    assert len(disk_ds) == len(mem_ds), "Datasets should have the same number of triples"

    # 5. Compare Content
    # The Disk dataset sorts edges by Head ID during preprocessing, while the Memory dataset
    # keeps the file order. Therefore, we cannot compare ds[i] == mem[i].
    # Instead, we extract the start triple (H, R, T) from every sequence and compare the SETS of triples.
    
    def extract_start_triple(seq_tensor, tokenizer):
        """Extracts (h_tokens, r_tokens, t_tokens) tuple from a token sequence."""
        # Sequence format: [EOS] H [SEP_HR] R [SEP_RT] T ...
        seq = seq_tensor.tolist()
        
        # Remove EOS at start if present
        start_idx = 0
        if seq[0] == tokenizer.eos_token_id:
            start_idx = 1
            
        try:
            # Find separators
            idx_sep_hr = seq.index(tokenizer.sep_hr_token_id, start_idx)
            idx_sep_rt = seq.index(tokenizer.sep_rt_token_id, idx_sep_hr)
            
            # Find end of T (next separator, EOS, or PAD)
            # T ends before the first special token found after SEP_RT
            idx_end_t = len(seq)
            for special in [tokenizer.sep_hr_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
                if special in seq[idx_sep_rt+1:]:
                    found = seq.index(special, idx_sep_rt+1)
                    idx_end_t = min(idx_end_t, found)
            
            h = tuple(seq[start_idx : idx_sep_hr])
            r = tuple(seq[idx_sep_hr+1 : idx_sep_rt])
            t = tuple(seq[idx_sep_rt+1 : idx_end_t])
            
            return (h, r, t)
        except ValueError:
            # Malformed sequence or padding only
            return None

    # Collect triples from Disk Dataset
    print("Collecting triples from Disk Dataset...")
    disk_triples = set()
    for i in range(len(disk_ds)):
        triple = extract_start_triple(disk_ds[i], tokenizer)
        if triple:
            disk_triples.add(triple)

    # Collect triples from Memory Dataset
    print("Collecting triples from Memory Dataset...")
    mem_triples = set()
    for i in range(len(mem_ds)):
        triple = extract_start_triple(mem_ds[i], tokenizer)
        if triple:
            mem_triples.add(triple)

    # Verify equivalence
    assert len(disk_triples) > 0, "Disk dataset yielded no valid triples"
    assert len(mem_triples) > 0, "Memory dataset yielded no valid triples"
    
    # Check for mismatches
    diff_disk_mem = disk_triples - mem_triples
    diff_mem_disk = mem_triples - disk_triples
    
    if diff_disk_mem or diff_mem_disk:
        print(f"Triples in Disk but not Mem: {len(diff_disk_mem)}")
        print(f"Triples in Mem but not Disk: {len(diff_mem_disk)}")
        
    assert disk_triples == mem_triples, "Sets of start triples do not match!"
    print("Test Passed: Datasets are equivalent.")