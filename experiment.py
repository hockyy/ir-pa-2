import math
import re
from bsbi import BSBIIndex
from compression import VBEPostings


######## >>>>> 3 IR metrics: RBP p = 0.8, DCG, dan AP

def rbp(ranking, p=0.8):
    """ menghitung search effectiveness metric score dengan
        Rank Biased Precision (RBP)

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score RBP
    """
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] * (p ** (i - 1))
    return (1 - p) * score


def dcg(ranking):
    """ menghitung search effectiveness metric score dengan
        Discounted Cumulative Gain

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score DCG
    """

    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] / math.log2(i + 1)
    return score


def prec(ranking):
    return sum(ranking) / len(ranking)


def ap(ranking):
    """ menghitung search effectiveness metric score dengan
        Average Precision

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score AP
    """
    score = 0.
    r = sum(ranking)
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += (prec(ranking[:i]) / r) * ranking[pos]
    return score


######## >>>>> memuat qrels

def load_qrels(qrel_file="qrels.txt", max_q_id=30, max_doc_id=1033):
    """ memuat query relevance judgment (qrels)
        dalam format dictionary of dictionary
        qrels[query id][document id]

        dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
        relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
        Doc 10 tidak relevan dengan Q3.

    """
    qrels = {"Q" + str(i): {i: 0 for i in range(1, max_doc_id + 1)} \
             for i in range(1, max_q_id + 1)}
    with open(qrel_file) as file:
        for line in file:
            parts = line.strip().split()
            qid = parts[0]
            did = int(parts[1])
            qrels[qid][did] = 1
    return qrels


######## >>>>> EVALUASI !

def eval(qrels, scoring_name, retrieve_function, query_file="queries.txt", k=1000, should_print=True):
    """
      loop ke semua 30 query, hitung score di setiap query,
      lalu hitung MEAN SCORE over those 30 queries.
      untuk setiap query, kembalikan top-1000 documents
    """

    with open(query_file) as file:
        rbp_scores = []
        dcg_scores = []
        ap_scores = []
        for qline in file:
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])

            # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
            # yang tertera di qrels
            ranking = []
            for (score, doc) in retrieve_function(query, k=k):
                did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                ranking.append(qrels[qid][did])
            rbp_scores.append(rbp(ranking))
            dcg_scores.append(dcg(ranking))
            ap_scores.append(ap(ranking))

    if should_print:
        print(f"Hasil evaluasi {scoring_name} terhadap 30 queries")
        print("RBP score =", sum(rbp_scores) / len(rbp_scores))
        print("DCG score =", sum(dcg_scores) / len(dcg_scores))
        print("AP score  =", sum(ap_scores) / len(ap_scores))

    return (sum(rbp_scores) / len(rbp_scores)), (sum(dcg_scores) / len(dcg_scores)), (sum(ap_scores) / len(ap_scores))


if __name__ == '__main__':
    qrels = load_qrels()

    BSBI_instance = BSBIIndex(data_dir='collection',
                              postings_encoding=VBEPostings,
                              output_dir='index')

    assert qrels["Q1"][166] == 1, "qrels salah"
    assert qrels["Q1"][300] == 0, "qrels salah"


    def bm25_generator(k1=1.6, b=0.75, optimize=True):
        def bm25(query, k=10):
            return BSBI_instance.retrieve_bm25(query, k, optimize, k1, b)

        return bm25


    def bm25_alt2_generator(k1=1.6, b=0.75, optimize=True):
        def bm25(query, k=10):
            return BSBI_instance.retrieve_alt2_bm25(query, k, optimize, k1, b)

        return bm25


    eval(qrels, "TF Doang", BSBI_instance.retrieve_tf)
    eval(qrels, "TF-IDF", BSBI_instance.retrieve_tfidf)
    import numpy as np
    from tqdm import tqdm
    # 1.2 <= k1 <= 2 dan b = 0.75
    max_rbp = (-1, (0, 0))
    max_dcg = (-1, (0, 0))
    max_ap = (-1, (0, 0))
    # print("Grid searching ...")
    # k1 = 2.2, b = 0.8 is good in:
    # for i in tqdm(np.arange(1.2, 2.2, 0.1)):
    #     for j in np.arange(0.5, 1, 0.05):

    # k1 = 2.65, b = 0.76 is good in:
    # for i in tqdm(np.arange(2.0, 2.7, 0.05)):
    #     for j in np.arange(0.70, 0.85, 0.03):

    # k1 = 2.75 to 2.9, b = 0.74
    # for i in tqdm(np.arange(2.15, 2.8, 0.02)):
    #     for j in np.arange(0.70, 0.8, 0.02):

    # Max grid search didapat saat konstanta bernilai sekitar
    # k1 = 2.745
    # b = 0.75
    # for i in tqdm(np.arange(2.7, 2.9, 0.005)):
    #     for j in np.arange(0.73, 0.75, 0.005):
    #         rbp_val, dcg_val, ap_val = eval(qrels, f"BM25, k1 = {i:.3f}, b = {j:.3f}", bm25_generator(i, j), should_print=False)
    #         if rbp_val > max_rbp[0]:
    #             max_rbp = (rbp_val, (i, j))
    #         if dcg_val > max_dcg[0]:
    #             max_dcg = (dcg_val, (i, j))
    #         if ap_val > max_ap[0]:
    #             max_ap = (ap_val, (i, j))
    # print("Melalui grid search: didapat ")
    # eval(qrels, f"BM25, k1 = {max_rbp[1][0]:.3f}, b = {max_rbp[1][1]:.3f}", bm25_generator(*(max_rbp[1])))
    # eval(qrels, f"BM25, k1 = {max_dcg[1][0]:.3f}, b = {max_dcg[1][1]:.3f}", bm25_generator(*(max_dcg[1])))
    # eval(qrels, f"BM25, k1 = {max_ap[1][0]:.3f}, b = {max_ap[1][1]:.3f}", bm25_generator(*(max_ap[1])))
    eval(qrels, "BM25, k1 = 1.6, b = 0.8", bm25_generator(1.6, 0.8))
    eval(qrels, "BM25, k1 = 2.745, b = 0.750", bm25_generator(2.745, 0.750))

    # Max grid search didapat saat konstanta bernilai sekitar
    # k1 = 2.715
    # b = 0.74
    # for i in tqdm(np.arange(2.7, 2.9, 0.005)):
    #     for j in np.arange(0.73, 0.75, 0.005):
    #         rbp_val, dcg_val, ap_val = eval(qrels, f"BM25 Alternative 2, k1 = {i:.3f}, b = {j:.3f}", bm25_alt2_generator(i, j), should_print=False)
    #         if rbp_val > max_rbp[0]:
    #             max_rbp = (rbp_val, (i, j))
    #         if dcg_val > max_dcg[0]:
    #             max_dcg = (dcg_val, (i, j))
    #         if ap_val > max_ap[0]:
    #             max_ap = (ap_val, (i, j))
    # print("Melalui grid search: didapat ")
    # eval(qrels, f"BM25 Alternative 2, k1 = {max_rbp[1][0]:.3f}, b = {max_rbp[1][1]:.3f}", bm25_alt2_generator(*(max_rbp[1])))
    # eval(qrels, f"BM25 Alternative 2, k1 = {max_dcg[1][0]:.3f}, b = {max_dcg[1][1]:.3f}", bm25_alt2_generator(*(max_dcg[1])))
    # eval(qrels, f"BM25 Alternative 2, k1 = {max_ap[1][0]:.3f}, b = {max_ap[1][1]:.3f}", bm25_alt2_generator(*(max_ap[1])))

    eval(qrels, "BM25 alternative 2, k1 = 1.6, b = 0.8", bm25_alt2_generator(1.6, 0.8))
    eval(qrels, "BM25 alternative 2, k1 = 2.715, b = 0.74", bm25_alt2_generator(2.715, 0.74))