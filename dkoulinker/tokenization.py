

from flair.tokenization import SentenceSplitter, SegtokTokenizer,Tokenizer,Sentence,split_multi,stagger
from typing import List,Optional

#The same as in flair.tokenization, just modified to return plan sentences.
class SegtokSentenceSplitter(SentenceSplitter):
    """
        Implementation of :class:`SentenceSplitter` using the SegTok library.

        For further details see: https://github.com/fnl/segtok
    """

    def __init__(self, tokenizer: Tokenizer = SegtokTokenizer()):
        super(SegtokSentenceSplitter, self).__init__()
        self._tokenizer = tokenizer

    def split(self, text: str) -> List[Sentence]:
        plain_sentences: List[str] = list(split_multi(text))

        try:
            sentence_offset: Optional[int] = text.index(plain_sentences[0])
        except ValueError as error:
            raise AssertionError(f"Can't find the sentence offset for sentence {repr(plain_sentences[0])} "
                                 f"from the text's starting position") from error

        sentences: List[Sentence] = []
        for sentence, next_sentence in stagger(plain_sentences, offsets=(0, 1), longest=True):

            sentences.append(
                Sentence(
                    text=sentence,
                    use_tokenizer=self._tokenizer,
                    start_position=sentence_offset
                )
            )

            offset: int = sentence_offset + len(sentence)
            try:
                sentence_offset = text.index(
                    next_sentence, offset) if next_sentence is not None else None
            except ValueError as error:
                raise AssertionError(f"Can't find the sentence offset for sentence {repr(sentence)} "
                                     f"starting from position {repr(offset)}") from error

        return plain_sentences, sentences
