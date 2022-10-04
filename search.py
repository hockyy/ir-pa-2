from bsbi import BSBIIndex
from compression import VBEPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir='collection',
                          postings_encoding=VBEPostings,
                          output_dir='index')

queries = ["alkylated with radioactive iodoacetate",
           "psychodrama for disturbed children",
           "lipid metabolism in toxemia and normal pregnancy"]

for query in queries:
    print("Query  : ", query)
    result = BSBI_instance.retrieve_bm25(query, k=10)
    # result = BSBI_instance.retrieve_tfidf(query, k=10)
    print("Results:")
    print(f"{'Document Name':30} Score")
    for (score, doc) in result:
        print(f"{doc:30} {score:>.3f}")
    print()
