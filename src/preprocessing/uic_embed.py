import numpy as np
import pandas as pd
import os
import glob
import json 
import argparse
from pathlib import Path
from joblib import Parallel, delayed
import spacy
import lftk
import re
import requests
import stanza
from collections import defaultdict
from functools import reduce, lru_cache
import string
import nltk


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')
from nltk.corpus import words

from sklearn.feature_extraction.text import TfidfVectorizer


# from util.decorators import cache_to_file_decorator
# from preprocessing.preprocessor import Preprocessor
# from dataloader.dataset import TextDataset, TabularDataset
# from util.helpers import safe_divide

def safe_divide(a, b):
    return a / b if b != 0 else 0

class LinguisticFeatureCalculator():

    def __init__(self, doc, constants=None):
        self.CONSTANTS = constants
        self.doc = doc
        self.words_enriched = [word for sent in doc.sentences for word in sent.words]
        self.words_raw = [word.text for sent in doc.sentences for word in sent.words]
        self.sentences_enriched = doc.sentences
        self.sentences_raw = [[w.text for w in sentence.words] for sentence in doc.sentences]


    def n_words(self):
        """ Number of words """
        return len(self.words_raw)

    def n_unique(self):
        """ Number of unique words / types """
        return len(set(self.words_raw))

    def word_length(self):
        """ Average length of words in letters """
        return safe_divide(sum([len(word) for word in self.words_raw]), self.n_words())

    def sentence_length(self):
        """ Average length of sentences in words """
        return safe_divide(sum([len(sentence) for sentence in self.sentences_raw]), len(self.sentences_raw))

    def pos_counts(self):
        """
        Count number of POS tags in doc
        """
        # universal POS (UPOS) tags (https://universaldependencies.org/u/pos/)
        count_pos = defaultdict(lambda: 0)
        # treebank-specific POS (XPOS) tags (https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)
        count_xpos = defaultdict(lambda: 0)
        for word in self.words_enriched:
            count_pos[word.pos] += 1
            count_xpos[word.xpos] += 1

        return count_pos, count_xpos

    def constituency_rules_count(self):
        """
        Returns counts of all CFG production rules from the constituency parsing of the doc
        Check http://surdeanu.cs.arizona.edu/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html
        for available tags
        """
        def constituency_rules(doc):
            def get_constituency_rule_recursive(tree):
                label = tree.label
                children = tree.children
                if len(children) == 0:  # this is a leaf
                    return []

                # find children label, if child is not a leaf. also ignore full stops
                children_label = "_".join([c.label for c in children if len(c.children) > 0 and c.label != '.'])
                if len(children_label) > 0:
                    rule = f"{label} -> {children_label}"
                    # print(f"{rule}     ({tree})")
                else:
                    return []

                children_rules = [get_constituency_rule_recursive(c) for c in children]
                children_rules_flat = [ll for l in children_rules for ll in l]

                return children_rules_flat + [rule]

            all_rules = []
            for sentence in doc.sentences:
                tree = sentence.constituency
                rules = get_constituency_rule_recursive(tree)
                all_rules += rules

            return all_rules

        cfg_rules = constituency_rules(self.doc)
        count_rules = defaultdict(lambda: 0)
        for rule in cfg_rules:
            count_rules[rule] += 1

        return count_rules

    def get_constituents(self):
        """
        Get constituents of the document, with corresponding text (concatenation of the leaf words)
        Check http://surdeanu.cs.arizona.edu/mihai/teaching/ista555-fall13/readings/PennTreebankConstituents.html
        for available tags
        """
        def get_leafs_text(tree):
            if len(tree.children) == 0:
                return tree.label
            return " ".join([get_leafs_text(c) for c in tree.children])

        def get_constituents_recursive(tree):
            label = tree.label
            if len(tree.children) == 0:
                return []

            children_constituents = [get_constituents_recursive(c) for c in tree.children]
            children_constituents_flat = [ll for l in children_constituents for ll in l]

            text = get_leafs_text(tree)
            return children_constituents_flat + [{'label': label, 'text': text}]

        all_constituents = []
        for sentence in self.sentences_enriched:
            tree = sentence.constituency
            rules = get_constituents_recursive(tree)
            all_constituents += rules
        return all_constituents

    def mattr(self, window_length=20):
        """
        Moving-Average Type-Token Ratio (MATTR)
        Adapted from https://github.com/kristopherkyle/lexical_diversity/blob/master/lexical_diversity/lex_div.py
        """
        if len(self.words_raw) == 0:
            return 0
        elif len(self.words_raw) < (window_length + 1):
            ma_ttr = len(set(self.words_raw)) / len(self.words_raw)
        else:
            sum_ttr = 0
            for x in range(len(self.words_raw) - window_length + 1):
                window = self.words_raw[x:(x + window_length)]
                sum_ttr += len(set(window)) / float(window_length)
            ma_ttr = sum_ttr / len(self.words_raw)
        return ma_ttr

    def ttr(self):
        """
        Type-token ratio
        Adapted from https://github.com/kristopherkyle/lexical_diversity/blob/master/lexical_diversity/lex_div.py
        """
        ntokens = len(self.words_raw)
        ntypes = len(set(self.words_raw))
        return safe_divide(ntypes, ntokens)

    def brunets_indes(self):
        """
        Brunét's index, based on definition in https://aclanthology.org/2020.lrec-1.176.pdf
        """
        n_tokens = len(self.words_raw)
        n_types = len(set(self.words_raw))
        if n_tokens > 0:
            return n_tokens ** n_types ** (-0.165)
        else:
            return 0

    def honores_statistic(self):
        """
        Honoré's statistic, based on definition in https://aclanthology.org/2020.lrec-1.176.pdf
        """
        n_words_with_one_occurence = len(list(filter(lambda w: self.words_raw.count(w) == 1, self.words_raw)))
        n_words = len(self.words_raw)
        n_types = len(set(self.words_raw))
        if n_words == 0:
            return 0
        elif (1 - n_words_with_one_occurence / n_types) == 0:
            return 0
        else:
            return (100 * np.log(n_words)) / (1 - n_words_with_one_occurence / n_types)


    def _count_syllables(self, word):
        """
        Simple syllable counting, from on  https://github.com/cdimascio/py-readability-metrics/blob/master/readability/text/syllables.py
        """
        word = word if type(word) is str else str(word)
        word = word.lower()
        if len(word) <= 3:
            return 1
        word = re.sub('(?:[^laeiouy]es|[^laeiouy]e)$', '', word)  # removed ed|
        word = re.sub('^y', '', word)
        matches = re.findall('[aeiouy]{1,2}', word)
        return len(matches)

    def flesch_kincaid(self):
        """
        Flesch–Kincaid grade level
        Based on https://github.com/cdimascio/py-readability-metrics/blob/master/readability/scorers/flesch_kincaid.py
        Formula adapted according to https://en.wikipedia.org/wiki/Flesch–Kincaid_readability_tests
        """
        words = [word for sent in self.sentences_raw for word in sent]

        syllables_per_word = [self._count_syllables(w) for w in words]
        avg_syllables_per_word = safe_divide(sum(syllables_per_word), len(words))

        avg_words_per_sentence = safe_divide(len(words), len(self.sentences_raw))

        return (0.39 * avg_words_per_sentence + 11.8 * avg_syllables_per_word) - 15.59

    def _cos_dist_between_sentences(self, sentences_raw):
        """
        Average cosine distance between utterances, and proportion of sentence pairs whose cosine distance is less than or equal to 0.5
        Based on and adapted from https://github.com/vmasrani/dementia_classifier/blob/1f48dc89da968a6c9a4545e27b162c603eb9a310/dementia_classifier/feature_extraction/feature_sets/psycholinguistic.py#L686
        """
        stop = nltk.corpus.stopwords.words('english')
        stemmer = nltk.PorterStemmer()
        def not_only_stopwords(text):
            unstopped = [w for w in text.lower() if w not in stop]
            return len(unstopped) != 0
        def stem_tokens(tokens):
            return [stemmer.stem(item) for item in tokens]
        def normalize(text):
            text = str(text).lower()
            return stem_tokens(nltk.word_tokenize(text))

        def cosine_sim(text1, text2):
            # input: list of raw utterances
            # returns: list of cosine similarity between all pairs
            if not_only_stopwords(text1) and not_only_stopwords(text2):
                # Tfid raises error if text contain only stopwords. Their stopword set is different
                # than ours so add try/catch block for strange cases
                try:
                    vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')  # Punctuation remover
                    tfidf = vectorizer.fit_transform([text1, text2])
                    return ((tfidf * tfidf.T).A)[0, 1]
                except ValueError as e:
                    print("Error:", e)
                    print('Returning 0 for cos_sim between: "', text1, '" and: "', text2, '"')
                    return 0
            else:
                return 0

        def compare_all_utterances(uttrs):
            # input: list of raw utterances
            # returns: (float)average similarity over all similarities
            similarities = []
            for i in range(len(uttrs)):
                for j in range(i + 1, len(uttrs)):
                    similarities.append(cosine_sim(uttrs[i], uttrs[j]))
            return similarities

        def avg_cos_dist(similarities):
            # returns:(float) Minimum similarity over all similarities
            return safe_divide(reduce(lambda x, y: x + y, similarities), len(similarities))

        def proportion_below_threshold(similarities, thresh):
            # returns: proportion of sentence pairs whose cosine distance is less than or equal to a threshold
            valid = [s for s in similarities if s <= thresh]
            return safe_divide(len(valid), float(len(similarities)))

        if len(sentences_raw) < 2:
            # only one sentence -> return 0
            return 0, 0
        sentences_as_text = [" ".join(sentence) for sentence in sentences_raw]
        similarities = compare_all_utterances(sentences_as_text)
        return avg_cos_dist(similarities), proportion_below_threshold(similarities, 0.5)

    def cos_dist_between_sentences(self):
        return self._cos_dist_between_sentences(self.sentences_raw)


    def get_english_dictionary(self):
        # word list from spell checker library
        # https://github.com/barrust/pyspellchecker/blob/master/spellchecker/resources/en.json.gz


        with open('/sc/arion/projects/EASI-COG/jt_workspace/spellcheck_corpus/en.json') as f:
            english_dict = json.load(f)
        
        english_words = [word for word in english_dict.keys()]
        english_words += ["n't", "'re", "'ll"]

        return english_words

    def not_in_dictionary(self):
        words_greater_than_two = [w for w in self.words_raw if len(w) > 2]  # only consider words greater than length 2
        #not_found = [w for w in words_greater_than_two if w.lower() not in nltk.corpus.words.words()]
        english_words = self.get_english_dictionary()
        not_found = [w for w in words_greater_than_two if w.lower() not in english_words]
        print("Words not in dictionary: ", not_found)
        return safe_divide(len(not_found), len(words_greater_than_two))


class LinguisticFeaturesLiterature():
    """
    Linguistic features based on previous literature
    [1] Fraser, Kathleen C., Jed A. Meltzer, and Frank Rudzicz. "Linguistic features identify Alzheimer’s disease in narrative speech." Journal of Alzheimer's Disease 49.2 (2016): 407-422.
    [2] Balagopalan, Aparna, et al. "To BERT or not to BERT: comparing speech and language-based approaches for Alzheimer's disease detection." arXiv preprint arXiv:2008.01551 (2020).
    [3] Parsapoor, Mahboobeh, Muhammad Raisul Alam, and Alex Mihailidis. "Performance of machine learning algorithms for dementia assessment: impacts of language tasks, recording media, and modalities." BMC Medical Informatics and Decision Making 23.1 (2023): 45.
    [4] Liu, Ziming, et al. "Automatic diagnosis and prediction of cognitive decline associated with alzheimer’s dementia through spontaneous speech." 2021 ieee international conference on signal and image processing applications (icsipa). IEEE, 2021.
    [5] Syed, Zafi Sherhan, et al. "Automated recognition of Alzheimer’s dementia using bag-of-deep-features and model ensembling." IEEE Access 9 (2021): 88377-88390.
    [6] Priyadarshinee, Prachee, et al. "Alzheimer’s Dementia Speech (Audio vs. Text): Multi-Modal Machine Learning at High vs. Low Resolution." Applied Sciences 13.7 (2023): 4244.
    [7] Eyigoz, Elif, et al. "Linguistic markers predict onset of Alzheimer's disease." EClinicalMedicine 28 (2020).
    [8] Diaz-Asper, Catherine, et al. "Increasing access to cognitive screening in the elderly: Applying natural language processing methods to speech collected over the telephone." Cortex 156 (2022): 26-38.
    [9] Tang, Lijuan, et al. "Explainable Alzheimer's Disease Detection Using linguistic features From Automatic Speech Recognition." Dementia and Geriatric Cognitive Disorders (2023).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Linguistic Features (Literature)"
        print(f"Initializing {self.name}")

        # stanza nlp pipeline
        self.nlp_pipeline = stanza.Pipeline('en')

    def _calculate_for_text(self, text):
        doc = self.nlp_pipeline(text.lower())

        feature_calculator = LinguisticFeatureCalculator(doc, constants=None)
        count_pos, count_xpos = feature_calculator.pos_counts()
        n_words = feature_calculator.n_words()

        # todo: Use sigmoid_fraction for e.g. pronoun-noun-ratio, since it's symmetrical and actually makes the
        # features more expressive. we don't do it right now to stay consistent with prior literature.
        def sigmoid_fraction(a, b):
            assert a >= 0 and b >= 0
            a = a+0.001 if a == 0 else a
            b = b + 0.001 if b == 0 else b
            return 1/(1+np.exp(-np.log(a/b)))


        features_pos = {
            'pronoun_noun_ratio': safe_divide(count_pos['PRON'], count_pos['NOUN']),  # [1], [2], [4]
            'verb_noun_ratio': safe_divide(count_pos['VERB'], count_pos['NOUN']),  # [4]
            'subordinate_coordinate_conjunction_ratio': safe_divide(count_pos['SCONJ'], count_pos['CCONJ']), # [3]
            'adverb_ratio': safe_divide(count_pos['ADV'], n_words),  # [1], [2], [9]
            'noun_ratio': safe_divide(count_pos['NOUN'], n_words),  # [1], [8], [9]
            'verb_ratio': safe_divide(count_pos['VERB'], n_words),  # [1], [9]
            'pronoun_ratio': safe_divide(count_pos['PRON'], n_words),  # [2], [9]
            'personal_pronoun_ratio': safe_divide(count_xpos['PRP'], n_words),  # [2]
            'determiner_ratio': safe_divide(count_pos['DET'], n_words),  # [8]
            'preposition_ratio': safe_divide(count_xpos['IN'], n_words),  # [9]
            'verb_present_participle_ratio': safe_divide(count_xpos['VBG'], n_words),  # [2, 8]
            'verb_modal_ratio': safe_divide(count_xpos['MD'], n_words),  # [8]
            'verb_third_person_singular_ratio': safe_divide(count_xpos['VBZ'], n_words), # [1] (I suppose by inflected verbs they mean 3. person)
        }

        constituency_rules_count = feature_calculator.constituency_rules_count()

        constituents = feature_calculator.get_constituents()
        NP = [c for c in constituents if c['label'] == 'NP']
        PP = [c for c in constituents if c['label'] == 'PP']
        VP = [c for c in constituents if c['label'] == 'VP']
        PRP = [c for c in constituents if c['label'] == 'PRP']

        features_constituency = {
            # NP -> PRP means "count the number of noun phrases (NP) that consist of a pronoun (PRP)"
            'NP -> PRP': constituency_rules_count['NP -> PRP'],  # [1]
            'ADVP -> RB': constituency_rules_count['ADVP -> RB'],  # [1], [2]
            'NP -> DT_NN': constituency_rules_count['NP -> DT_NN'],  # [1]
            'ROOT -> FRAG': constituency_rules_count['ROOT -> FRAG'],  # [1]
            'VP -> AUX_VP': constituency_rules_count['VP -> AUX_VP'],  # [1]
            'VP -> VBG': constituency_rules_count['VP -> VBG'],  # [1]
            'VP -> VBG_PP': constituency_rules_count['VP -> VBG_PP'],  # [1]
            'VP -> IN_S': constituency_rules_count['VP -> IN_S'],  # [1]
            'VP -> AUX_ADJP': constituency_rules_count['VP -> AUX_ADJP'],  # [1]
            'VP -> AUX': constituency_rules_count['VP -> AUX'],  # [1]
            'VP -> VBD_NP': constituency_rules_count['VP -> VBD_NP'],  # [1]
            'INTJ -> UH': constituency_rules_count['INTJ -> UH'],  # [1]
            'NP_ratio': safe_divide(len(NP), len(constituents)),  # [9]
            'PRP_ratio': safe_divide(len(PRP), len(constituents)),  # [9]
            'PP_ratio': safe_divide(len(PP), len(constituents)),  # [1]
            'VP_ratio': safe_divide(len(VP), len(constituents)),  # [1]
            'avg_n_words_in_NP': safe_divide(sum([len(c['text'].split(" ")) for c in NP]), len(NP)) ,  # [9]
        }

        simple_features = {
            'n_words': n_words,  # [9], [4], [6], [8]
            'n_unique_words': feature_calculator.n_unique(),  # [6], [8]
            'avg_word_length': feature_calculator.word_length(),  # [1], [2]
            'avg_sentence_length': feature_calculator.sentence_length(),  # [4]
            'words_not_in_dict_ratio': feature_calculator.not_in_dictionary(),  # [1], [2]
        }

        vocabulary_richness_features = {
            'brunets_index': feature_calculator.brunets_indes(),  # [3], [8]
            'honores_statistic': feature_calculator.honores_statistic(),  # [1], [9], [3], [8]
            'ttr': feature_calculator.ttr(),  # [4], [8]
            'mattr': feature_calculator.mattr(),  # [8]
        }

        readability_features = {
            'flesch_kincaid': feature_calculator.flesch_kincaid() # [3]
        }

        avg_distance_between_utterances, prop_dist_thresh_05 = feature_calculator.cos_dist_between_sentences()
        repetitiveness_features = {
            'avg_distance_between_utterances': avg_distance_between_utterances,  # [1], [2]
            'prop_utterance_dist_below_05': prop_dist_thresh_05  # [1], [2]
        }

        density_features = {
            'propositional_density': safe_divide(count_pos['VERB'] + count_pos['ADJ'] + count_pos['ADV'] + count_pos['ADP'] +
                                      count_pos['CCONJ'] + count_pos['SCONJ'], n_words),  # [3], [7]
            'content_density': safe_divide(count_pos['NOUN'] + count_pos['VERB'] + count_pos['ADJ'] + count_pos['ADV'], n_words),
            # [3], [8], [9]
        }

        all_features = {**features_pos, **features_constituency, **simple_features, **vocabulary_richness_features,
                        **readability_features, **repetitiveness_features, **density_features}

        # sorted by gini feature importance
        #sorted_feature_names = ['word_length', 'adverb', 'PRP_freq', 'honores_statistic', 'personal_pronoun', 'flesch_kincaid', 'pronoun_noun_ratio', 'pronoun', 'noun', 'NP -> DT_NN', 'n_unique', 'mattr', 'VP_freq', 'avg_distance_between_utterances', 'content_density', 'verb_present_participle', 'determiners', 'n_words', 'verb_third_person_singular', 'preposition', 'propositional_density', 'PP_freq', 'prop_dist_thresh_05', 'verb', 'VP -> VBG_PP', 'NP_freq', 'brunets_index', 'avg_n_words_in_NP', 'verb_noun_ratio', 'sentence_length', 'ttr', 'subordinate_coordinate_conjunction_ratio', 'NP -> PRP', 'verb_modal', 'ADVP -> RB', 'VP -> VBG', 'ROOT -> FRAG', 'INTJ -> UH', 'VP -> VBD_NP', 'words_not_in_dict', 'VP -> AUX_VP', 'VP -> IN_S', 'VP -> AUX_ADJP', 'VP -> AUX']
        #n = 20
        #top_n_features = {key: all_features[key] for key in sorted_feature_names[:n]}

        # let's drop the null feature (0 in every sample in ADReSS):
        all_features = {f: all_features[f] for f in all_features if f not in ['VP -> AUX_VP', 'VP -> IN_S', 'VP -> AUX_ADJP', 'VP -> AUX']}

        return all_features


    def _calculate_features(self, dataset):
        features = [self._calculate_for_text(text) for text in dataset.data]

        return features



    def _load_features(self, dataset):
        features = self._calculate_features(dataset)
        features_df = pd.DataFrame(features)

        return features_df



    def preprocess_dataset(self, dataset):
        print(f"Calculating linguistic features for dataset {dataset}")
        return self._load_features(dataset)

def embed_file(filepath, ling_feats):
    data = json.load(open(filepath))
    text = data['text']

    features = ling_feats._calculate_for_text(text)

    sequence_embedding = pd.DataFrame(features, index=[0]).values

    return sequence_embedding

def process_directory_bert(transcript_path, output_path):

    ling_feats = LinguisticFeaturesLiterature()

    lftk_embedded_sequences = {}
    filepaths = glob.glob(os.path.join(transcript_path,'*.json'))
    filepaths.sort()

    for filepath in filepaths:
      participant = filepath.split('/')[-1][:-5]
      print(participant)
      sequence_embedding = embed_file(filepath, ling_feats)
      np.save(os.path.join(output_path,participant+"_0000.npy"),sequence_embedding)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--dataset_config", help="Path to the JSON configuration file.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for transcription")
    args = parser.parse_args()
    with open(args.dataset_config) as f:
        config = json.load(f)

    for dataset_config in config["datasets"]:
        print(dataset_config['setname'])
        transcript_path = dataset_config['fixed_timestamped_transcription_outdir']
        output_path = dataset_config['significant_nltk_outdir']
        os.makedirs(output_path, exist_ok=True)
        process_directory_bert(
            transcript_path=transcript_path, 
            output_path=output_path)

