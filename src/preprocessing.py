from dataclasses import dataclass
from datasets import load_dataset
from pathlib import Path

@dataclass(frozen=True)
class LabeledSentence:
    tokens: tuple[str, ...]
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, tuple):
            raise ValueError("Tokens are not a tuple")
        if not isinstance(self.labels, tuple):
            raise ValueError("Labels are not a tuple")
    
    def to_dict(self):
        return {"tokens": list(self.tokens), "labels": list(self.labels)}

def read_conll_file(path: str | Path, delimiter: str, langugage='en') -> list[LabeledSentence]:
    # create list labeled sentences
    sentences: list[LabeledSentence] = []

    # open file
    with open(path, "r", encoding="utf-8") as f:
        # create empty tokens, labels
        tokens = ()
        labels = ()

        # read line
        for line in f:
            # if text line
            if line != "\n" and not "-DOCSTART-" in line:

                items = line.strip('\n').split(sep=delimiter)

                # add 0th elem (word) to tokens
                tokens += (items[0],)

                if langugage=='en':
                    # English: add 4th elem (BIO tag) to labels
                    labels += (items[3],)
                    # German: add 5th elem (BIO tag) to labels
                elif langugage=='de':
                    labels += (items[4],)
                else:
                    raise AttributeError(f"language={langugage} is invalid."
                                         f"Choose 'en' or 'de'!")

            # if blank line
            else:
                # add non-empty sentence to list
                if tokens:
                    sentences.append(LabeledSentence(tokens, labels))
                    tokens = ()
                    labels = ()

    # if no newline at end of doc, append tokens and labels of final sentence
    if tokens:
        sentences.append(LabeledSentence(tokens, labels))
        tokens = ()
        labels = ()

    # return labeled sentence list
    return sentences



