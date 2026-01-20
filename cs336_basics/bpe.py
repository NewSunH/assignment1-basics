from dataclasses import dataclass
from re import split
from typing import BinaryIO
import regex as re
from dataclasses import dataclass, field
from typing import ClassVar
import os
from collections import Counter
import multiprocessing as mp

def _compile_special_tokens(
        special_tokens: list[str]
) -> re.Pattern[str]:
    toks_sorted = sorted(special_tokens, key = len, reverse = True)
    toks = '(' + '|'.join(re.escape(t) for t in toks_sorted) + ')'
    return re.compile(toks)

def _split_special_tokens(
        corpus: str,
        special_tokens: list[str]
) -> list[str] :
    '''
    输入：
    - corpus字符串
    - special tokens 字符串列表
    输出：
    - 分割special tokens 的 corpus
    '''
    re_compiled_toks = _compile_special_tokens(special_tokens)
    splited_corpus = re.split(re_compiled_toks, corpus)
    return splited_corpus

def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

@dataclass
class BpeTokenizer:
    special_tokens: list[str]
    vocab_size: int

    PAT: ClassVar[str] = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokenize_jobs: int = 4
    SPLIT_SPECIAL_TOKEN: ClassVar[bytes] = b"<|endoftext|>"

    vocab: dict[int, bytes] = field(default_factory=dict)
    merges: list[tuple[bytes, bytes]] = field(default_factory=list)
    bytes_pair_count: dict[tuple[bytes, bytes], int] = field(default_factory=dict)

    def __init_vocab__(self):
        self.vocab = {}
        for i in range(256):
            self.vocab[i] = bytes([i])
        for tok in self.special_tokens:
            self.vocab[len(self.vocab)] = tok.encode("utf-8")
    
    def __post_init__(self):
        self.__init_vocab__()

    def pretokenizer(
            self,
            corpus: str
    ) -> list[str]:
        return re.findall(self.PAT, corpus)
    
    def counting_pairs(
            self,
            corpus_byte_dict: dict[tuple[bytes,...], int]
    ) -> None:
        """
        Optimizing the merging step The naïve implementation of BPE training in the stylized example above is slow because for every merge, it iterates over all byte pairs to identify the most frequent pair. However, the only pair counts that change after each merge are those that overlap with the merged pair. Thus, BPE training speed can be improved by indexing the counts of all pairs and incrementally updating these counts, rather than explicitly iterating over each pair of bytes to count pair frequencies. You can get significant speedups with this caching procedure, though we note that the merging part of BPE training is not parallelizable in Python.
        """
        pair_counter = Counter()
        for seq, freq in corpus_byte_dict.items():
        # 长度 < 2，不可能产生 pair
            if len(seq) < 2:
                continue
            for a, b in zip(seq, seq[1:]):
                pair_counter[(a, b)] += freq
        self.bytes_pair_count = dict(pair_counter)

    def merging_pairs(
            self,
            corpus_byte_dict: dict[tuple[bytes,...], int]
    ) -> dict[tuple[bytes,...], int]:
        
        best_pair = max(
                self.bytes_pair_count.items(),
                key=lambda item: (item[1], item[0])
        )[0]
        a, b = best_pair
        new_tok = a + b
        new_corpus = {}
        
        for tok_seq in corpus_byte_dict:
            new_tok_seq = []
            i = 0
            while i < len(tok_seq):
                if i + 1 < len(tok_seq) and tok_seq[i] == a and tok_seq[i+1] == b:
                    new_tok_seq.append(new_tok)
                    i += 2
                else:
                    new_tok_seq.append(tok_seq[i])
                    i += 1
            new_seq = tuple(new_tok_seq)
            new_corpus[new_seq] = new_corpus.get(new_seq, 0) + corpus_byte_dict[tok_seq]
        
        self.merges.append(best_pair)
        self.vocab[len(self.vocab)] = new_tok
        return new_corpus

    def train_from_file(
            self,
            file: BinaryIO
    ) -> None:
        file.seek(0)
        raw = file.read()
        text = raw.decode("utf-8", errors="ignore")
        corpus_split_seq = _split_special_tokens(text, self.special_tokens)
        corpus_byte_dict: dict[tuple[bytes,...], int] = {}

        for part in corpus_split_seq:
            if not part:
                continue
            if part in self.special_tokens:
                continue
            for tok in self.pretokenizer(part):
                bs = tok.encode("utf-8",errors="ignore")
                seq = tuple(bytes([b]) for b in bs)
                corpus_byte_dict[seq] = corpus_byte_dict.get(seq, 0) + 1

        while len(self.vocab) < self.vocab_size:
            self.counting_pairs(corpus_byte_dict)
            if not self.bytes_pair_count:
                break
            best_freq = max(self.bytes_pair_count.values())
            if best_freq <= 1:
                break
            corpus_byte_dict = self.merging_pairs(corpus_byte_dict)
               
        

