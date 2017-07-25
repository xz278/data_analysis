# coding=utf-8
import pandas as pd
import numpy as np
import utils
import joblib
from sklearn.decomposition import PCA
import math


def compute_stats():
    """
    Compute IV and PCA variance raito of
    the keywords.
    """
    print('Loading data ...')
    print()
    data = pd.read_csv('./data/model_data.csv', index_col='贷款编号', encoding='gb18030')
    kw = data.columns.values[:-1]
    data['res'] = [x == 1 for x in data['label']]

    # compute information value
    print('Computing IV ...')
    ivs = []
    # pb = []
    # p = ''
    # for i in range(101):
    #     p += '#'
    #     pb.append(p)

    # c = 0
    # l = len(kw)
    for w in kw:
        # c += 1
        # pct = math.floor((c / l) * 100)
        # a, b = divmod(pct, 10)
        # if b == 0:
        #     print(' -->' + pb[pct] + '\r')
        iv = utils.compute_woe(df=data[[w, 'res']], bin_col=w, res_col='res')
        ivs.append(iv)
        iv = None

    print()

    # create a dataframe

    df = pd.DataFrame()
    df['关键词'] = kw
    df = df.set_index('关键词')

    # add iv
    ivs_2 = [x[0] for x in ivs]
    df['IV'] = ivs_2

    print('Computing PCA ...')
    # compute pca explained variance ratio
    pca = PCA(n_components=500)
    pca = pca.fit(X=data[kw], y=data['label'])

    # save model
    joblib.dump(pca, './data/pca_keywords.dat')

    # add pca var ratio
    df['pca'] = pca.explained_variance_ratio_

    # save stats
    print('Save data in ./data/key_word_stats.csv')
    df.to_csv('./data/key_word_stats.csv', encoding='gb18030')


def main():
    """
    Handle command line options.
    """
    compute_stats()


if __name__ == '__main__':
    main()