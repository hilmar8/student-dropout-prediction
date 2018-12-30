import re
import json
import numpy as np

from difflib import SequenceMatcher
from geopy.distance import geodesic

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

from sklearn.preprocessing import FunctionTransformer

# Reduce the names of courses of study to match those provided by namskra.is
COURSE_OF_STUDY_MAP = {
    'félagsfræðabraut': [
        'tónlistarsvið, með tungumála- og félagsgreinaáherlsu',
        'félagsvísindabraut',
        'stúdentsbraut-félagsgreinalína',
        'félags- og hugvísindabraut',
        'félags- og hugvísindabraut - afrekssvið',
    ],
    'opin braut': ['opin braut ný námskrá',
                   'opin braut ný námskrá hraðferð',
                   'opin stúdentsbraut',
                   'stúdentsbraut - opin lína',
                   'opin stúdentsbraut - opið svið',
                   'opin stúdentsbraut - tungumálasvið',
                   'opin stúdentsbraut - viðskipta- og hagfræðisvið',
                   'opin stúdentsbraut, almennt kjörsvið',
                  ],
    'náttúrufræðibraut': [
        'náttúrufræðabraut - afrekssvið',
        'náttúruvísindabraut',
        'stúdentsbraut-náttúrufræðilína',
        'náttúrufræðabraut',
    ],
    'nýsköpunarbraut': [
        'nýsköpunarbraut til stúdentsprófs',
        'nýsköpunar- og tæknibraut',
        'nýsköpunar- og tæknibraut - kvikmyndagerð',
    ],
    'fjölgreinabraut': ['fjölgreinabraut 2015'],
    'málabraut': ['tungumála og félagsgreinasvið'],
    'íþróttabraut': ['stúdentsbraut-íþróttalína', 'íþróttabraut til stúdentsprófs'],
    'alþjóðabraut': ['alþjóðabraut - viðskiptasvið',
                     'alþjóðabraut - alþjóðasamskiptasvið',
                     'alþjóðabraut - menningarsvið',
                     'stúdentsbraut-alþjóðalína'],
    'listnámsbraut': [
        'listnámsbraut til stúdentsprófs',
        'listdansbraut ný námskrá',
        'stúdentsbraut-listalína',
        'tónlistarsvið með raungreinaáherslu',
        'a - klassísk tónlistarbraut ( stúdent)',
        'a - rytmísk tónlistarbraut( stúdent)',

        'hönnunarbraut',
        'nýsköpunarbraut',
        'hönnunar- og markaðsbraut',
        'listabraut',
    ],
    'viðskiptabraut': [
        'viðskipta- og hagfræðibraut',
        'stúdentsbraut-viðskipta- og hagfræðilína',
        'viðskipta-og hagfræðibraut - afrekssvið',
    ],
    'starfsbraut': [
        'viðbótarnám til stúdentsprófs af starfsnámsbrautum',
    ],
    'hestabraut': ['stúdentsbraut-hestalína', 'hestabraut-2015'],
    'raungreinabraut': ['raungreinasvið'],
    'upplýsinga- og tæknibraut': ['tölvufræðibraut'],
}

# Helper function.
def course_of_study_name(x):
    name = str(x).lower().strip().replace(' - ný námskrá 2011', '')
    return next(iter([s for s in COURSE_OF_STUDY_MAP if name in COURSE_OF_STUDY_MAP[s]]), name)

# A Scikit transformer to reduce the names of courses of study to match those provided by namskra.is
class CourseOfStudyNamer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['COURSE_OF_STUDY'] = X['COURSE_OF_STUDY'].apply(course_of_study_name)

        return X


# Transformer to fix elementary school names using fuzzy text matching.
class ElementaryNameFixer(BaseEstimator, TransformerMixin):
    def __init__(self):
        with open('elementary_geo.json', encoding='utf8') as f:
            self.elementary_gps = json.load(f)

    def fuzzy_elementary_name(self, x):
        """
            Given an elementary school name x.
            Returns the name most similar to x
            from known names along with a score.
        """
        if not x or x in self.elementary_gps.keys():
            return x
        
        name_scores = []

        for g in self.elementary_gps:
            name_scores.append({'name': g, 'score': SequenceMatcher(None, x, g).ratio()})

        return sorted(name_scores, key=lambda g: g['score'], reverse=True)[0]

    def match_elementary_name(self, x, fuzzy_elementary_name_map, threshold=0.845):
        """
            Return the correct elementary school
            name if the score is above a certain
            threshold.
        """
        if not x or x in self.elementary_gps.keys():
            return x

        name_map_entry = fuzzy_elementary_name_map.get(x, None)
        
        if not name_map_entry:
            return x
        
        if name_map_entry['score'] > threshold:
            return name_map_entry['name']
        else:
            return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Trim, lower, remove unknown values.
        X['ELEMENTARY_SCHOOL'] = X['ELEMENTARY_SCHOOL'].apply(lambda x: re.sub(' +', ' ', str(x).lower().strip()))
        X['ELEMENTARY_SCHOOL'] = X['ELEMENTARY_SCHOOL'].apply(lambda x: None if x == 'óþekkt' or x == 'nan' else x)

        # Map potential spelling mistakes for quicker lookup.
        fuzzy_elementary_name_map = dict()
        for elementary_school in X['ELEMENTARY_SCHOOL'].unique():
            if elementary_school:
                fuzzy_elementary_name_map[elementary_school] = self.fuzzy_elementary_name(elementary_school)

        # Fuzzy match names.
        X['ELEMENTARY_SCHOOL'] = X['ELEMENTARY_SCHOOL'].apply(lambda x: self.match_elementary_name(x, fuzzy_elementary_name_map))

        return X


# Transformer to calculate the distance from the elementary school to the upper secondary school.
class ElementarySchoolDistance(BaseEstimator, TransformerMixin):
    def __init__(self):
        with open('elementary_geo.json', encoding='utf8') as f:
            self.elementary_gps = json.load(f)

        with open('school_geo.json', encoding='utf8') as f:
            self.school_gps = json.load(f)

    def elementary_school_distance(self, x):
        if x.SCHOOL not in self.school_gps:
            return None
        if x.ELEMENTARY_SCHOOL not in self.elementary_gps:
            return None
        
        return geodesic(self.school_gps[x.SCHOOL], self.elementary_gps[x.ELEMENTARY_SCHOOL]).km

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['ELEMENTARY_SCHOOL_DISTANCE'] = X.apply(lambda x: self.elementary_school_distance(x), axis=1)
        
        return X.drop('ELEMENTARY_SCHOOL', axis=1)


# Transformer to reduce the nationality to "IS" or "Other"
class NationalitySelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def nationality_name(self, x):
        if not x:
            return x

        return 'IS' if x == 'IS' else 'OT'

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['NATIONALITY'] = X['NATIONALITY'].apply(lambda x: self.nationality_name(x))

        return X



# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# A copy of a class from a upcoming version of Scikit-learn for converting category features to numbers.
# Definition of the CategoricalEncoder class, copied from PR #9151.

# https://github.com/scikit-learn/scikit-learn/blob/d929fb3/sklearn/preprocessing/data.py#L2889
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array.
    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features can be encoded using a one-hot (aka one-of-K or dummy)
    encoding scheme (``encoding='onehot'``, the default) or converted
    to ordinal integers (``encoding='ordinal'``).
    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    Parameters
    ----------
    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
        The type of encoding to use (default is 'onehot'):
        - 'onehot': encode the features using a one-hot aka one-of-K scheme
          (or also called 'dummy' encoding). This creates a binary column for
          each category and returns a sparse matrix.
        - 'onehot-dense': the same as 'onehot' but returns a dense array
          instead of a sparse matrix.
        - 'ordinal': encode the features as ordinal integers. This results in
          a single column of integers (0 to n_categories - 1) per feature.
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories must be sorted and should not mix
          strings and numeric values.
        The used categories can be found in the ``categories_`` attribute.
    dtype : number type, default np.float64
        Desired dtype of output.
    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros. In the inverse transform, an unknown category
        will be denoted as None.
        Ignoring unknown categories is not supported for
        ``encoding='ordinal'``.
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order corresponding with output of ``transform``).
    Examples
    --------
    Given a dataset with two features, we let the encoder find the unique
    values per feature and transform the data to a binary one-hot encoding.
    >>> from sklearn.preprocessing import CategoricalEncoder
    >>> enc = CategoricalEncoder(handle_unknown='ignore')
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >>> enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> enc.transform([['Female', 1], ['Male', 4]]).toarray()
    array([[1., 0., 1., 0., 0.],
           [0., 1., 0., 0., 0.]])
    >>> enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
    array([['Male', 1],
           [None, 2]], dtype=object)
    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      integer ordinal features. The ``OneHotEncoder assumes`` that input
      features take on values in the range ``[0, max(feature)]`` instead of
      using the unique values.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """
        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        if self.categories != 'auto':
            for cats in self.categories:
                if not np.all(np.sort(cats) == np.array(cats)):
                    raise ValueError("Unsorted categories are not yet "
                                     "supported")

        X_temp = check_array(X, dtype=None)
        if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=np.object)
        else:
            X = X_temp

        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                if self.handle_unknown == 'error':
                    valid_mask = np.in1d(Xi, self.categories[i])
                    if not np.all(valid_mask):
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(self.categories[i])

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using specified encoding scheme.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X_temp = check_array(X, dtype=None)
        if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=np.object)
        else:
            X = X_temp

        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            Xi = X[:, i]
            valid_mask = np.in1d(Xi, self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    Xi = Xi.copy()
                    Xi[~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(Xi)

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        feature_indices = np.cumsum(n_values)

        indices = (X_int + feature_indices[:-1]).ravel()[mask]
        indptr = X_mask.sum(axis=1).cumsum()
        indptr = np.insert(indptr, 0, 0)
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csr_matrix((data, indices, indptr),
                                shape=(n_samples, feature_indices[-1]),
                                dtype=self.dtype)
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out

    def inverse_transform(self, X):
        """Convert back the data to the original representation.
        In case unknown categories are encountered (all zero's in the
        one-hot encoding), ``None`` is used to represent this category.
        Parameters
        ----------
        X : array-like or sparse matrix, shape [n_samples, n_encoded_features]
            The transformed data.
        Returns
        -------
        X_tr : array-like, shape [n_samples, n_features]
            Inverse transformed array.
        """
        check_is_fitted(self, 'categories_')
        X = check_array(X, accept_sparse='csr')

        n_samples, _ = X.shape
        n_features = len(self.categories_)
        n_transformed_features = sum([len(cats) for cats in self.categories_])

        # validate shape of passed X
        msg = ("Shape of the passed X data is not correct. Expected {0} "
               "columns, got {1}.")
        if self.encoding == 'ordinal' and X.shape[1] != n_features:
            raise ValueError(msg.format(n_features, X.shape[1]))
        elif (self.encoding.startswith('onehot')
                and X.shape[1] != n_transformed_features):
            raise ValueError(msg.format(n_transformed_features, X.shape[1]))

        # create resulting array of appropriate dtype
        dt = np.find_common_type([cat.dtype for cat in self.categories_], [])
        X_tr = np.empty((n_samples, n_features), dtype=dt)

        if self.encoding == 'ordinal':
            for i in range(n_features):
                labels = X[:, i].astype('int64')
                X_tr[:, i] = self.categories_[i][labels]

        else:  # encoding == 'onehot' / 'onehot-dense'
            j = 0
            found_unknown = {}

            for i in range(n_features):
                n_categories = len(self.categories_[i])
                sub = X[:, j:j + n_categories]

                # for sparse X argmax returns 2D matrix, ensure 1D array
                labels = np.asarray(_argmax(sub, axis=1)).flatten()
                X_tr[:, i] = self.categories_[i][labels]

                if self.handle_unknown == 'ignore':
                    # ignored unknown categories: we have a row of all zero's
                    unknown = np.asarray(sub.sum(axis=1) == 0).flatten()
                    if unknown.any():
                        found_unknown[i] = unknown

                j += n_categories

            # if ignored are found: potentially need to upcast result to
            # insert None values
            if found_unknown:
                if X_tr.dtype != object:
                    X_tr = X_tr.astype(object)

                for idx, mask in found_unknown.items():
                    X_tr[mask, idx] = None

        return X_tr
