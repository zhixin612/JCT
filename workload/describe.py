import nltk
import loader
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Tuple
from collections import Counter

# nltk.download('punkt')
matplotlib.use('Agg')


def describe_dataset():
    datasets = ['mmlu', 'dolly', 'alpaca', 'alpaca_python', 'dialogsum', 'openorca']
    for dataset in datasets:
        def count_tokens(text):
            return len(nltk.word_tokenize(text))

        ds = loader.load_input(dataset)
        # ds = ds.select(range(10000))
        # print(ds, ds[0])
        ds = ds.map(lambda it: {
            'input_count': count_tokens(it['input']),
            'output_count': count_tokens(it['output'])
        })

        for t in ['input', 'output']:
            token_counts = ds['{}_count'.format(t)]

            hist, bins = np.histogram(token_counts, bins=100, density=True)

            plt.figure(figsize=(4, 2.5))
            plt.hist(bins[:-1], bins=bins, weights=hist, edgecolor='black')
            plt.title('{} distribution ({})'.format(t, dataset), fontsize=12)
            plt.xlabel('token count')
            plt.ylabel('probability density')

            plt.tight_layout()
            plt.savefig('fig/token_count_{}_{}.png'.format(dataset, t))
            plt.savefig('fig/token_count_{}_{}.pdf'.format(dataset, t))

            plt.yscale('log')
            plt.savefig('fig/token_count_{}_{}_log.png'.format(dataset, t))
            plt.savefig('fig/token_count_{}_{}_log.pdf'.format(dataset, t))

            print('Token count distribution for {} ({}) saved'.format(t, dataset))


def describe_trace():
    trace_names = ['tweet', 'wiki', 'burstgpt_0', 'burstgpt_1']
    for name in trace_names:
        ts = loader.load_trace(name)
        stamps = [0]
        for gap in ts:
            stamps.append(stamps[-1] + gap / 1000)
        counts = Counter([int(ts) for ts in stamps])
        counts = sorted(counts.items())
        counts = [_[1] for _ in counts]
        print(counts)

        plt.figure(figsize=(4, 2.5))
        plt.plot(counts)
        plt.title('{} trace'.format(name), fontsize=12)
        plt.xlabel('time (second)')
        plt.ylabel('request rate (req/s)')

        plt.tight_layout()
        plt.savefig('fig/trace_{}.png'.format(name))
        plt.savefig('fig/trace_{}.pdf'.format(name))

        plt.figure(figsize=(4, 2.5))
        plt.plot(1000/ts)
        plt.plot(np.convolve(1000/ts, np.ones(1000) / 1000, mode='same'))
        plt.title('{} trace'.format(name), fontsize=12)
        plt.xlabel('request id')
        plt.ylabel('request rate (req/s)')

        plt.tight_layout()
        plt.savefig('fig/trace_{}_id.png'.format(name))
        plt.savefig('fig/trace_{}_id.pdf'.format(name))

        print('Trace distribution for {} saved'.format(name))


if __name__ == '__main__':
    # describe_dataset()
    describe_trace()


