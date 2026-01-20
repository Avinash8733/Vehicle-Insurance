import sklearn
with open('version.txt', 'w', encoding='utf-8') as f:
    f.write(sklearn.__version__)
