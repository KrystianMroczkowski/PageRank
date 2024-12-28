import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])

    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    print('Sum of page ranks: ', round(sum(ranks.values()), 4), "\n")

    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    print('Sum of page ranks: ', round(sum(ranks.values()), 4))


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob_distribution = dict()
    links = corpus[page]
    num_links = len(links)

    if num_links != 0:
        probability_1 = damping_factor / num_links
        probability_2 = (1 - damping_factor) / len(corpus)

        for key in corpus:
            prob_distribution[key] = probability_2
        for link in links:
            prob_distribution[link] += probability_1
            prob_distribution[link] = round(prob_distribution[link], 3)
    else:
        probability = round(1 / len(corpus), 3)
        for key in corpus:
            prob_distribution[key] = probability

    return prob_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    data = {key: 0 for key in corpus.keys()}
    next_page = random.choice(list(corpus.keys()))

    for _ in range(n):
        prob_distribution = transition_model(corpus, next_page, damping_factor)
        next_page = random.choices(
            list(prob_distribution.keys()),
            weights=list(prob_distribution.values()),
            k=1
        )[0]
        data[next_page] += 1

    page_ranks = {key: round(value / n, 4) for key, value in data.items()}
    return page_ranks

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_ranks = {key: 1 / len(corpus) for key in corpus}
    random_choice_prob = (1 - damping_factor) / len(corpus)
    threshold = 0.001

    while True:
        convergence = True
        new_page_ranks = {}

        for page in page_ranks:
            surf_prob = 0
            for corpus_key, corpus_value in corpus.items():
                if len(corpus_value) == 0 or page in corpus_value:
                    surf_prob += page_ranks[corpus_key] / (len(corpus) if len(corpus_value) == 0 else len(corpus_value))

            new_rank = random_choice_prob + (damping_factor * surf_prob)
            new_page_ranks[page] = new_rank

            if abs(page_ranks[page] - new_rank) > threshold:
                convergence = False

        page_ranks = new_page_ranks
        if convergence:
            break

    return page_ranks


if __name__ == "__main__":
    main()
