## Cara Menjalankan

Buat folder tmp, dan virtual environment, dan install requirements untuk mac dan gnu/linux:

```bash
mkdir tmp
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Untuk windows:

```bash
mkdir tmp
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

atau:

```bash
mkdir tmp
virtualenv venv
venv\Scripts\activate
pip install -r requirements.txt
```


Kalau yang windows venvnya salah commandnya, bisa di browsing kak hehe saya ga pake windows soalnya maaf kak :pray: 

Selanjutnya jalankan search.py.

Perhatikan bahwa signature fungsi retrieve:

`def retrieve_[method scoring](self, query, k=10, optimize=True, debug=False):`

## WARNING!

**`experiment.py` bisa berbeda saat dijalankan di Operating System atau komputer lain jika menggunakan index postings list yang berbeda.**

Hal ini karena pada saat reindexing (menjalankan `bsbi.py`) os.walk() yang urutannya bisa berbeda (tidak ada komparator khusus), karena python `sorted()` dan `sort()` stable, hal ini bisa menyebabkan dokumen releval/tidak relevan yang memiliki score sama bisa masuk ke hasil evaluasi, dan menyebabkan sedikit perbedaan pada document name `search.py` dan hasil `experiment.py`

## Hasil Run

**Flag optimize digunakan untuk menggunakan Top-K Wand Algorithm.**

Hasil menggunakan `retrieve_tfidf`

```
Query  :  alkylated with radioactive iodoacetate
Results:
Document Name                  Score
./collection/6/507.txt         24.036
./collection/6/554.txt         11.771
./collection/11/1003.txt       9.404
./collection/4/388.txt         8.386
./collection/4/387.txt         8.386
./collection/11/1018.txt       6.765
./collection/2/144.txt         6.765
./collection/2/119.txt         6.765
./collection/3/247.txt         6.765
./collection/6/512.txt         6.765

Query  :  psychodrama for disturbed children
Results:
Document Name                  Score
./collection/9/820.txt         20.276
./collection/10/918.txt        12.965
./collection/9/821.txt         12.948
./collection/1/36.txt          11.402
./collection/9/817.txt         11.243
./collection/10/926.txt        10.335
./collection/8/799.txt         10.335
./collection/10/927.txt        10.072
./collection/1/99.txt          9.199
./collection/2/100.txt         9.199

Query  :  lipid metabolism in toxemia and normal pregnancy
Results:
Document Name                  Score
./collection/1/7.txt           28.687
./collection/2/159.txt         18.274
./collection/4/328.txt         17.877
./collection/4/306.txt         17.659
./collection/1/81.txt          16.615
./collection/4/329.txt         12.836
./collection/3/288.txt         12.630
./collection/4/371.txt         11.406
./collection/9/881.txt         10.723
./collection/3/201.txt         10.566
```



Hasil menggunakan `retrieve_bm25`

```
[nltk_data] Downloading package stopwords to /home/hocky/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
[nltk_data] Downloading package punkt to /home/hocky/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
Query  :  alkylated with radioactive iodoacetate
Results:
Document Name                  Score
./collection/6/507.txt         21.465
./collection/11/1003.txt       9.548
./collection/6/554.txt         8.549
./collection/6/506.txt         7.108
./collection/8/793.txt         7.082
./collection/2/119.txt         6.724
./collection/4/387.txt         6.574
./collection/3/247.txt         6.178
./collection/8/745.txt         6.015
./collection/11/1018.txt       5.882

Query  :  psychodrama for disturbed children
Results:
Document Name                  Score
./collection/9/820.txt         19.885
./collection/9/821.txt         10.632
./collection/1/36.txt          9.860
./collection/10/918.txt        9.477
./collection/9/817.txt         8.179
./collection/10/926.txt        8.087
./collection/9/814.txt         7.642
./collection/1/99.txt          7.552
./collection/9/894.txt         7.358
./collection/5/493.txt         7.355

Query  :  lipid metabolism in toxemia and normal pregnancy
Results:
Document Name                  Score
./collection/1/7.txt           26.316
./collection/4/328.txt         17.556
./collection/4/306.txt         14.148
./collection/2/159.txt         13.300
./collection/1/81.txt          10.982
./collection/7/602.txt         10.343
./collection/4/371.txt         9.948
./collection/4/329.txt         9.542
./collection/3/288.txt         8.923
./collection/1/12.txt          8.415
```

