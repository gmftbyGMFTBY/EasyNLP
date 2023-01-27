import json
import nltk
from nltk.stem import WordNetLemmatizer
import re
import argparse
from os import path
import pdb

_lemmatizer = WordNetLemmatizer()


def tokenize(example, ppln):
    for fn in ppln:
        example = fn(example)
    return example


def lower(tokens):
    if not isinstance(tokens, str):
        return [lower(token) for token in tokens]
    return tokens.lower()


def pos_tag(tokens):
    return nltk.pos_tag(tokens)


def to_basic_form(tokens):
    if not isinstance(tokens, tuple):
        return [to_basic_form(token) for token in tokens]
    word, tag = tokens
    if tag.startswith('NN'):
        pos = 'n'
    elif tag.startswith('VB'):
        pos = 'v'
    elif tag.startswith('JJ'):
        pos = 'a'
    else:
        return word
    return _lemmatizer.lemmatize(word, pos)


def kw_tokenize(string):
    return tokenize(string, [nltk_tokenize, lower, pos_tag, to_basic_form])


def nltk_tokenize(string):
    return nltk.word_tokenize(string)


def is_reach_goal(context, goal):
    context = kw_tokenize(context)
    context = [re.sub(r"[^a-zA-Z0-9]", "", c) for c in context]  # e.g., 'word -> word

    if goal in context:
       return True

    return False


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=path.abspath, help='Path to result file')
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    successes, success_turns = [], []

    with open(args.input) as f:
        counter = 0
        for line in f:
            data = json.loads(line.strip())
            utters = data["conversation_plan"].split('[SEP]')
            is_success = False
            success_turn = 0

            for i, utter in enumerate(utters, start=1):
                if is_reach_goal(utter, data["target_word"]):
                    is_success = True
                    success_turn = (i + 1) // 2
                    success_turns.append(success_turn)
                    break

            result_str = "SUCCESS" if is_success else "FAILED"
            successes.append(1 if result_str == "SUCCESS" else 0)

            if counter > 20000:
                break
            counter += 1

    print("-----------------")
    print("SUCCESS RATE: {} ({} / {})".format(sum(successes) / len(successes) if successes else 0., sum(successes), len(successes)))
    print("AVERAGE TURNS: {}".format(sum(success_turns) / len(success_turns) if success_turns else 0.))


if __name__ == "__main__":
    main()
