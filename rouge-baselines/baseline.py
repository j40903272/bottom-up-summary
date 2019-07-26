#!/usr/bin/env python

from __future__ import print_function, division

import argparse, os, re, time
from g_rouge import rouge
from util import has_repeat, n_grams
from functools import reduce


def split_sentences(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    bare_sents = re.findall(r'%s (.+?) %s' % (sentence_start_tag, sentence_end_tag), article)
    return bare_sents

# convenient decorator
def register_to_registry(registry):
    def _register(func):
        registry[func.__name__] = func
        return func
    return _register

baseline_registry = {}
register = register_to_registry(baseline_registry)

# baseline methods
@register
def first_sentence(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    ''' use sentence tags to output the first sentence of an article as its summary. '''
    sents = split_sentences(article, sentence_start_tag, sentence_end_tag)
    return sents[:1]

@register
def first_three_sentences(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    sents = split_sentences(article, sentence_start_tag, sentence_end_tag)
    return sents[:3]

@register
def first_two_sentences(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    sents = split_sentences(article, sentence_start_tag, sentence_end_tag)
    return sents[:2]

@register
def verbatim(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    sents = split_sentences(article, sentence_start_tag, sentence_end_tag)
    return sents

@register
def pre_sent_tag_verbatim(article):
    sents = article.split('<t>')
    good_sents = []
    for sent in sents:
        sent = sent.strip()
        if len(sent.split()) > 0:
            good_sents.append(sent)
    # print(good_sents)
    return good_sents

@register
def sent_tag_verbatim(article):
    sents = split_sentences(article, '<t>', '</t>')
    # print(sents)
    return sents

@register
def sent_tag_p_verbatim(article):
    bare_article = article.strip()
    bare_article += ' </t>'
    sents = split_sentences(bare_article, '<t>', '</t>')
    # print(sents)
    return sents

@register
def adhoc_old0(article):
    sents = split_sentences(article, '<t>', '</t>')
    good_sents = []
    for sent in sents:
        # Remove <unk>
        tokens = [x for x in sent.split() if x != '<unk>']
        # Ignore length 1 sententces
        if len(tokens) > 1:
            good_sents.append(' '.join(tokens))
    return good_sents

@register
def full(article):
    return [article]

@register
def adhoc_base(article):
    article += ' </t> </t>'
    first_end = article.index(' </t> </t>')
    article = article[:first_end] + ' </t>'
    sents = split_sentences(article)
    good_sents = []
    for sent in sents:
        # Remove <unk>
        tokens = [x for x in sent.split() if x != '<unk>']
        # Ignore length 1 sententces
        if len(tokens) > 1:
            good_sents.append(' '.join(tokens))
    return good_sents

@register
def no_sent_tag(article):
    article = article.strip()
    try:
        if article[-1] != '.':
            article += ' .'
    except:
        article += ' .'
    good_sents = list(re.findall(r'.+?\.', article))
    return good_sents

@register
def second_sentence(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    sents = split_sentences(article, sentence_start_tag, sentence_end_tag)
    return sents[1:2]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', required=True, help='Path to the tokenized source file. One sample per line with sentence tags.')
    parser.add_argument('-t', '--target', required=False, help='Path to the tokenized target file. One sample per line with sentence tags.')
    parser.add_argument('-m', '--method', default='first_sentence', choices=baseline_registry.keys(), help='Baseline method to use.')
    parser.add_argument('-d', '--delete', action='store_true', help='Delete the temporary files created during evaluation.')
    parser.add_argument('-g', '--google', dest='run_google_rouge', action='store_true', help='Evaluate with the ROUGE implementation from google/seq2seq.')
    parser.add_argument('--no-rouge', dest='run_rouge', action='store_false', help='Skip ROUGE evaluation.')
    parser.add_argument('-r', '--check-repeats', action='store_true', help='Evaluate self repeats.')
    # ROUGE arguments
    parser.add_argument('--no-stemming', dest='stemming', action='store_false', help='Turn off stemming in ROUGE.')
    parser.add_argument('--n-bootstrap', type=int, default=1000, help='The number of bootstrap samples used in ROUGE.')

    args = parser.parse_args()

    # Check the presence of target file
    if args.run_rouge or args.run_google_rouge:
        assert args.target is not None, 'Need the path to target file `--target` for ROUGE evaluations.'

    process = baseline_registry[args.method]

    # Read and preprocess generated summary
    n_source = 0
    references = []
    summaries = []
    with open(args.source, 'r', encoding="utf-8") as f:
        for i, article in enumerate(f):
            summary = process(article)
            summaries.append(summary)
            n_source += 1

    # Read and preprocess a single candidate reference summary for each example
    if args.run_rouge or args.run_google_rouge:
        n_target = 0
        with open(args.target, 'r', encoding="utf-8") as f:
            for i, article in enumerate(f):
                if args.method == "full":
                    candidate = [article]
                else:
                    candidate = split_sentences(article)
                references.append([candidate])
                n_target += 1

        assert n_source == n_target, 'Source and target must have the same number of samples.'

    # Run official ROUGE evaluation
    if args.run_rouge:
        from util import evaluate_rouge

        rouge_args = rouge_args = [
            '-c', 95, # 95% confidence intervals, necessary for the dictionary conversion routine
            '-n', 2, # up to bigram
            '-a',
            '-r', args.n_bootstrap, # the number of bootstrap samples for confidence bounds
        ]

        if args.stemming:
            # add the stemming flag
            rouge_args += ['-m']

        t0 = time.time()
        # evaluate with official ROUGE script v1.5.5
        scores = evaluate_rouge(summaries, references, remove_temp=args.delete, rouge_args=rouge_args)
        dt = time.time() - t0

        print('* method', args.method)
        headers = ['rouge_1_recall', 'rouge_1_precision', 'rouge_1_f_score', 'rouge_2_recall', 'rouge_2_precision', 'rouge_2_f_score', 'rouge_l_recall', 'rouge_l_precision', 'rouge_l_f_score']

        print(headers)
        for header in headers:
            print("{:.2f}".format(scores[header]*100), end='\t')
        print()

        print('* evaluated {} samples, took {:.3f}s, averaging {:.3f}s/sample'.format(n_target, dt, dt / n_target))

    # Run Google's ROUGE evaluation
    if args.run_google_rouge:
        # Based on https://github.com/google/seq2seq, modified to support multi-sentence summaries
        t0 = time.time()
        g_scores = rouge(summaries, [candidates[0] for candidates in references])
        dt = time.time() - t0

        g_headers = ['rouge_1/r_score', 'rouge_1/p_score', 'rouge_1/f_score', 'rouge_2/r_score', 'rouge_2/p_score', 'rouge_2/f_score', 'rouge_l/r_score', 'rouge_l/p_score', 'rouge_l/f_score']

        print(g_headers)
        for header in g_headers:
            print("{:.2f}".format(g_scores[header]*100), end='\t')
        print()

        print('* evaluated {} samples, took {:.3f}s, averaging {:.3f}s/sample'.format(n_target, dt, dt / n_target))

    # Evaluate self-repetitions
    if args.check_repeats:
        t0 = time.time()
        # Counts
        n_sent_repeats = 0
        ngram_repeats = {2: 0, 4: 0, 8: 0, 16: 0, 32: 0}
        for summary in summaries:
            # Sentence-level repeats
            # Count of samples containing self-repetitions of a full sentence
            n_sent_repeats += has_repeat(summary)

            # N-gram repeats
            for n in ngram_repeats.keys():
                # Respect sentence boundary
                grams = reduce(lambda x, y: x+y, [n_grams(sent.split(), n) for sent in summary], [])
                ngram_repeats[n] += has_repeat(grams)

        dt = time.time() - t0

        print('* portion of samples that contains self-repetitions')

        # Sort the statistics by importance
        str_keys = ['full-sent'] + list(map(lambda n: '%d-gram' % n, sorted(ngram_repeats.keys(), reverse=True)))
        print(','.join(str_keys))
        print("{:.2f}%".format(n_sent_repeats / n_source*100), end=',\t')
        for n in sorted(ngram_repeats.keys(), reverse=True):
            print("{:.2f}%".format(ngram_repeats[n] / n_source*100), end=',\t')
        print()

        print('* evaluated {} samples, took {:.3f}s, averaging {:.3f}s/sample'.format(n_source, dt, dt / n_source))
