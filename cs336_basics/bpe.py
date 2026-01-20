from dataclasses import dataclass
from re import split
from typing import BinaryIO
import regex as re
from dataclasses import dataclass, field
from typing import ClassVar
import os

def _compile_special_tokens(
        special_tokens: list[str]
) -> re.Pattern[str]:
    special_tokens.sort(key = len, reverse = True)
    toks = '(' + '|'.join(re.escape(t) for t in special_tokens) + ')'
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

    PAT: ClassVar[str] = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokenize_jobs: int = 4

    vocab: dict[int, bytes] = field(default_factory=dict)
    merges: list[tuple[bytes, bytes]] = field(default_factory=list)

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
    
    # def train_bpe(
    #         self,
    #         corpus: BinaryIO,

    # )