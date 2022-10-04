## Referensi:
# https://www.analyticssteps.com/blogs/nltk-python-tutorial-beginners
# https://www.nltk.org/howto/stem.html

import os
import pickle
import contextlib
import heapq
import time
import math

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import *
from nltk import word_tokenize
from string import punctuation

punctuation = list(punctuation)

nltk.download('stopwords')
nltk.download('punkt')


class Cleaner:
    # stemmer = PorterStemmer()
    stemmer = SnowballStemmer("english")
    stop_words = stopwords.words('english')

    @staticmethod
    def clean_and_tokenize(uncleaned_sentence):
        tokenized_words = word_tokenize(uncleaned_sentence)
        stemmed = [Cleaner.stemmer.stem(word) for word in tokenized_words]
        cleaned_tokens = []
        for token in stemmed:
            if (token not in Cleaner.stop_words) and (token not in punctuation):
                cleaned_tokens.append(token)
        return cleaned_tokens


class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """

    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.doc_length = dict()
        self.average_doc_length = -1
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.loaded = False

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_length, doc_id_map, and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)
        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as merged_index:
            self.doc_length = merged_index.doc_length
            self.average_doc_length = merged_index.average_doc_length

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """

        block_path = os.path.join(self.data_dir, block_dir_relative)

        td_pairs = []
        # print(f"Currently processing... {block_path}")
        for doc_file_name in next(os.walk(block_path))[2]:
            doc_path = f'./{os.path.join(block_path, doc_file_name)}'
            current_doc_id = self.doc_id_map[doc_path]
            with open(doc_path, "r") as f:
                tokenized_words = Cleaner.clean_and_tokenize(f.read())
                for token in tokenized_words:
                    current_term_id = self.term_id_map[token]
                    td_pairs.append((current_term_id, current_doc_id))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = dict()
        for term_id, doc_id in td_pairs:
            term_dict.setdefault(term_id, dict())
            term_dict[term_id].setdefault(doc_id, 0)
            term_dict[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            doc_fq_pairs = sorted(term_dict[term_id].items())
            unzipped = list(zip(*doc_fq_pairs))
            index.append(term_id, list(unzipped[0]), list(unzipped[1]))

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)  # first item
        for t, postings_, tf_list_ in merged_iter:  # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve(self, query, debug=False):
        if not self.loaded:
            self.load()
            self.loaded = 1
        tokenized_query = Cleaner.clean_and_tokenize(query)
        if (debug):
            print("tokenized into: ")
            print(tokenized_query)
        lists_of_query = []

        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as merged_index:
            for token in tqdm(tokenized_query):
                if token not in self.term_id_map: continue
                lists_of_query.append(merged_index.get_postings_list(self.term_id_map[token]))

        lists_of_query.sort(key=lambda x: len(x[0]))

        return lists_of_query

    def TaaT(self, lists_of_query, scoreFunction):
        n = len(self.doc_length)
        result = []
        len_postings = len(lists_of_query)
        for i in range(len_postings):
            df = len(lists_of_query[i][0])
            idf = math.log(n / df)
            current_pairs = []
            for j in range(df):
                # count score in this doc iff lists_of_query_tf[i][j] > 0
                assert len(lists_of_query[i][0]) == len(lists_of_query[i][1])
                # print(lists_of_query[0][i][j], lists_of_query_tf[i][j])
                current_score = scoreFunction(
                    doc_id=lists_of_query[i][0][j],
                    tf=lists_of_query[i][1][j],
                    idf=idf
                )
                current_pairs.append((lists_of_query[i][0][j], current_score))
            result = sorted_merge_posts_and_tfs(result, current_pairs)
        return result

    def sort_and_cut(self, result, k):
        result = sorted(result, key=lambda x: x[1], reverse=True)
        if len(result) > k:
            result = result[:k + 1]
        for i in range(len(result)):
            # print(result[i][0], self.doc_id_map[result[i][0]], result[i][1])
            result[i] = (result[i][1], self.doc_id_map[result[i][0]])
        return result

    def retrieve_tfidf(self, query, k=10, debug=False):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan:
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        lists_of_query = self.retrieve(query, debug)

        def tfidf(doc_id, tf, idf):
            return (1 + math.log(tf)) * idf

        result = self.TaaT(lists_of_query, tfidf)
        return self.sort_and_cut(result, k)

    def retrieve_bm25(self, query, k=10, k1=1.6, b=0.75, debug=False):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = ((k1 + 1) * tf(t, D)) / (k1 * ((1 - b) + b * docsLength(D) / averageDocsLength) + tf(t, D)s)

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan:
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        lists_of_query = self.retrieve(query, debug)
        def bm25(doc_id, tf, idf):
            numerator = (k1 + 1) * tf
            denominator = k1
            denominator *= (1 - b) + b * self.doc_length[doc_id] / self.average_doc_length
            denominator += tf
            return (numerator / denominator) * idf
        result = self.TaaT(lists_of_query, bm25)
        return self.sort_and_cut(result, k)

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_' + block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(
                    InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                    for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":

    try:
        os.mkdir("index")
    except:
        pass
    BSBI_instance = BSBIIndex(data_dir='collection',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    BSBI_instance.index()  # memulai indexing!
