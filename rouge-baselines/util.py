from __future__ import print_function
from six.moves import xrange
from pyrouge import Rouge155
import tempfile, os, glob, shutil
import numpy as np


def evaluate_rouge(summaries, references, remove_temp=False, rouge_args=[]):
    '''
    Args:
        summaries: [[sentence]]. Each summary is a list of strings (sentences)
        references: [[[sentence]]]. Each reference is a list of candidate summaries.
        remove_temp: bool. Whether to remove the temporary files created during evaluation.
        rouge_args: [string]. A list of arguments to pass to the ROUGE CLI.
    '''
    temp_dir = tempfile.mkdtemp()
    system_dir = os.path.join(temp_dir, 'system')
    model_dir = os.path.join(temp_dir, 'model')
    # directory for generated summaries
    os.makedirs(system_dir)
    # directory for reference summaries
    os.makedirs(model_dir)
    print(temp_dir, system_dir, model_dir)

    assert len(summaries) == len(references)
    for i, (summary, candidates) in enumerate(zip(summaries, references)):
        summary_fn = '%i.txt' % i
        for j, candidate in enumerate(candidates):
            candidate_fn = '%i.%i.txt' % (i, j)
            with open(os.path.join(model_dir, candidate_fn), 'w', encoding="utf-8") as f:
                f.write('\n'.join(candidate))

        with open(os.path.join(system_dir, summary_fn), 'w', encoding="utf-8") as f:
            f.write('\n'.join(summary))

    args_str = ' '.join(map(str, rouge_args))
    rouge = Rouge155(rouge_args=args_str)
    rouge.system_dir = system_dir
    rouge.model_dir = model_dir
    rouge.system_filename_pattern = '(\d+).txt'
    rouge.model_filename_pattern = '#ID#.\d+.txt'
    output = rouge.convert_and_evaluate()

    r = rouge.output_to_dict(output)
    print(output)

    # remove the created temporary files
    if remove_temp:
        shutil.rmtree(temp_dir)
    return r

def n_grams(tokens, n):
    l = len(tokens)
    return [tuple(tokens[i:i + n]) for i in xrange(l) if i + n < l]

def has_repeat(elements):
    d = set(elements)
    return len(d) < len(elements)


if __name__ == '__main__':
    article = [
        u"marseille prosecutor says `` so far no videos were used in the crash investigation '' despite media reports .",
        u"journalists at bild and paris match are `` very confident '' the video clip is real , an editor says .",
        u'andreas lubitz had informed his lufthansa training school of an episode of severe depression , airline says .',
    ]

    candidates = [article]
    references = [candidates]
    summaries = [article]

    rouge_args = [
        '-c', 95,
        '-U',
        '-r', 1,
        '-n', 2,
        '-a',
    ]
    print(evaluate_rouge(summaries, references, True, rouge_args))
