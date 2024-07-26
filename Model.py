import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import normalize
import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.utils.validation import check_array
from sklearn.utils.extmath import safe_sparse_dot
warnings.filterwarnings("ignore")
books = pd.read_csv(r"S:\File\School\NMTTNT\BTL\Data\Books.csv")
users = pd.read_csv(r"S:\File\School\NMTTNT\BTL\Data\Users.csv")
ratings = pd.read_csv(r"S:\File\School\NMTTNT\BTL\Data\Ratings.csv")
books.isnull().sum()
users.isnull().sum()
ratings.isnull().sum()
ratings_with_book_titles = ratings.merge(books,on='ISBN')
ratings_with_book_titles.drop(columns=["ISBN","Image-URL-S","Image-URL-M"],axis=1,inplace=True)
complete_df = ratings_with_book_titles.merge(users.drop("Age", axis=1), on="User-ID")
complete_df['Location'] = complete_df['Location'].str.split(',').str[-1].str.strip()
min_ratings_threshold = 200
num_ratings_per_user = complete_df.groupby('User-ID')['Book-Rating'].count()
knowledgeable_user_ids = num_ratings_per_user[num_ratings_per_user > min_ratings_threshold].index
knowledgeable_user_ratings = complete_df[complete_df['User-ID'].isin(knowledgeable_user_ids)]
min_ratings_count_threshold=50
rating_counts= knowledgeable_user_ratings.groupby('Book-Title').count()['Book-Rating']
popular_books = rating_counts[rating_counts >= min_ratings_count_threshold].index
final_ratings =  knowledgeable_user_ratings[knowledgeable_user_ratings['Book-Title'].isin(popular_books)]
pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID'
                          ,values='Book-Rating')                     
pt.fillna(0,inplace=True)


def _return_float_dtype(X, Y):
    if not issparse(X) and not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if Y is None:
        Y_dtype = X.dtype
    elif not issparse(Y) and not isinstance(Y, np.ndarray):
        Y = np.asarray(Y)
        Y_dtype = Y.dtype
    else:
        Y_dtype = Y.dtype

    if X.dtype == Y_dtype == np.float32:
        dtype = np.float32
    else:
        dtype = float

    return X, Y, dtype


def check_pairwise_arrays(
    X,
    Y,
    *,
    precomputed=False,
    dtype=None,
    accept_sparse="csr",
    force_all_finite=True,
    copy=False,
):
    X, Y, dtype_float = _return_float_dtype(X, Y)

    estimator = "check_pairwise_arrays"
    if dtype is None:
        dtype = dtype_float

    if Y is X or Y is None:
        X = Y = check_array(
            X,
            accept_sparse=accept_sparse,
            dtype=dtype,
            copy=copy,
            force_all_finite=force_all_finite,
            estimator=estimator,
        )
    else:
        X = check_array(
            X,
            accept_sparse=accept_sparse,
            dtype=dtype,
            copy=copy,
            force_all_finite=force_all_finite,
            estimator=estimator,
        )
        Y = check_array(
            Y,
            accept_sparse=accept_sparse,
            dtype=dtype,
            copy=copy,
            force_all_finite=force_all_finite,
            estimator=estimator,
        )

    if precomputed:
        if X.shape[1] != Y.shape[0]:
            raise ValueError(
                "Precomputed metric requires shape "
                "(n_queries, n_indexed). Got (%d, %d) "
                "for %d indexed." % (X.shape[0], X.shape[1], Y.shape[0])
            )
    elif X.shape[1] != Y.shape[1]:
        raise ValueError(
            "Incompatible dimension for X and Y matrices: "
            "X.shape[1] == %d while Y.shape[1] == %d" % (X.shape[1], Y.shape[1])
        )

    return X, Y
    
def cosine_similarity(X, Y=None, dense_output=True):
    X, Y = check_pairwise_arrays(X, Y)

    X_normalized = normalize(X, copy=True)
    if X is Y:
        Y_normalized = X_normalized
    else:
        Y_normalized = normalize(Y, copy=True)

    K = safe_sparse_dot(X_normalized, Y_normalized.T, dense_output=dense_output)

    return K


similarity_score = cosine_similarity(pt)

def recommend(book_name):
    index = np.where(pt.index==book_name)[0][0]
    similar_books = sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1], reverse=True)[1:6]
    
    data = []
    
    for i in similar_books:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(item)
    return data


                          
                          
                          