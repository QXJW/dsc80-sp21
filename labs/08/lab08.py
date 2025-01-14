import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import Binarizer, QuantileTransformer, FunctionTransformer

# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------


def best_transformation():
    """
    Returns an integer corresponding to the correct option.

    :Example:
    >>> best_transformation() in [1,2,3,4]
    True
    """

    # take log and square root of the dataset
    # look at the fit of the regression line (and R^2)

    return 1

# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def ord_helper(col,order):
    enc = {y:x for (x,y) in enumerate(order)}
    return col.replace(enc)

def create_ordinal(df):
    """
    create_ordinal takes in diamonds and returns a dataframe of ordinal
    features with names ordinal_<col> where <col> is the original
    categorical column name.

    :Example:
    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_ordinal(diamonds)
    >>> set(out.columns) == {'ordinal_cut', 'ordinal_clarity', 'ordinal_color'}
    True
    >>> np.unique(out['ordinal_cut']).tolist() == [0, 1, 2, 3, 4]
    True
    """
    cut_vals = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    cut = ord_helper(df['cut'],cut_vals)
    
    clarity_vals = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    clarity = ord_helper(df['clarity'],clarity_vals)
    
    color_vals = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
    color = ord_helper(df['color'],color_vals)
    
    return pd.DataFrame({'ordinal_cut':cut, 'ordinal_clarity':clarity, 'ordinal_color':color})

# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------



def create_one_hot(df):
    """
    create_one_hot takes in diamonds and returns a dataframe of one-hot 
    encoded features with names one_hot_<col>_<val> where <col> is the 
    original categorical column name, and <val> is the value found in 
    the categorical column <col>.

    :Example:
    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_one_hot(diamonds)
    >>> out.shape == (53940, 20)
    True
    >>> out.columns.str.startswith('one_hot').all()
    True
    >>> out.isin([0,1]).all().all()
    True
    """
    def hot_helper(col):
        cn = str(col)
        cols = df[cn].unique()
        one_hot = df[cn].apply(lambda x: pd.Series(x == cols, index = cols, dtype = int))
        one_hot = one_hot.rename(lambda x: "one_hot_"+cn+"_"+str(x), axis = 1)
        return one_hot
    cols = ['cut', 'color', 'clarity']
    new = pd.DataFrame()
    for col in cols:
        new = pd.concat([new, hot_helper(col)], axis = 1)
    return new


def create_proportions(df):
    """
    create_proportions takes in diamonds and returns a 
    dataframe of proportion-encoded features with names 
    proportion_<col> where <col> is the original 
    categorical column name.

    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_proportions(diamonds)
    >>> out.shape[1] == 3
    True
    >>> out.columns.str.startswith('proportion_').all()
    True
    >>> ((out >= 0) & (out <= 1)).all().all()
    True
    """
    noms = df[['cut','clarity','color']]
    df = pd.DataFrame()
    for col in noms.columns:
        colname = 'proportion_{}'.format(col)
        vc = (noms[col].value_counts() / noms[col].shape[0]).to_dict()
        df[colname] = noms[col].replace(vc)
    return df

# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------


def create_quadratics(df):
    """
    create_quadratics that takes in diamonds and returns a dataframe 
    of quadratic-encoded features <col1> * <col2> where <col1> and <col2> 
    are the original quantitative columns 
    (col1 and col2 should be distinct columns).

    :Example:
    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_quadratics(diamonds)
    >>> out.columns.str.contains(' * ').all()
    True
    >>> ('x * z' in out.columns) or ('z * x' in out.columns)
    True
    >>> out.shape[1] == 15
    True
    """
    
    quants = df[['carat','x','y','z','depth','table']]
    final_df = pd.DataFrame()
    for c1 in quants.columns:
        for c2 in quants.columns:
            if (c1==c2):
                continue
            name1 = '{} * {}'.format(c1,c2)
            name2 = '{} * {}'.format(c2,c1)
            if (name1 in final_df.columns) or (name2 in final_df.columns):
                continue
            final_df[name1] = quants[c1] * quants[c2]
    return final_df


# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------

def comparing_performance():
    """
    Hard coded answers to comparing_performance.

    :Example:
    >>> out = comparing_performance()
    >>> len(out) == 6
    True
    >>> import numbers
    >>> isinstance(out[0], numbers.Real)
    True
    >>> all(isinstance(x, str) for x in out[2:-1])
    True
    >>> 0 <= out[-1] <= 1
    True
    """

    # create a model per variable => (variable, R^2, RMSE) table

    return [0.849, 1548.53, 'x', 'carat * x', 'ordinal_color', 0.041431655535624445]

# ---------------------------------------------------------------------
# Question # 6, 7, 8
# ---------------------------------------------------------------------


class TransformDiamonds(object):
    
    def __init__(self, diamonds):
        self.data = diamonds
        
    def transformCarat(self, data):
        """
        transformCarat takes in a dataframe like diamonds 
        and returns a binarized carat column (an np.ndarray).

        :Example:
        >>> diamonds = sns.load_dataset('diamonds')
        >>> out = TransformDiamonds(diamonds)
        >>> transformed = out.transformCarat(diamonds)
        >>> isinstance(transformed, np.ndarray)
        True
        >>> transformed[172, 0] == 1
        True
        >>> transformed[0, 0] == 0
        True
        """
        bi = Binarizer(threshold = 1)
        binarized = bi.transform(data['carat'].values.reshape(-1, 1))
        return binarized
    
    def transform_to_quantile(self, data):
        """
        transform_to_quantiles takes in a dataframe like diamonds 
        and returns an np.ndarray of quantiles of the weight 
        (i.e. carats) of each diamond.

        :Example:
        >>> diamonds = sns.load_dataset('diamonds')
        >>> out = TransformDiamonds(diamonds.head(10))
        >>> transformed = out.transform_to_quantile(diamonds)
        >>> isinstance(transformed, np.ndarray)
        True
        >>> 0.2 <= transformed[0,0] <= 0.5
        True
        >>> np.isclose(transformed[1,0], 0, atol=1e-06)
        True
        """
        qt = QuantileTransformer()
        qt.fit(self.data['carat'].values.reshape(-1, 1))
        return qt.transform(data['carat'].values.reshape(-1,1))
    
    
    def transform_to_depth_pct(self, data):
        """
        transform_to_volume takes in a dataframe like diamonds 
        and returns an np.ndarray consisting of the approximate 
        depth percentage of each diamond.

        :Example:
        >>> diamonds = sns.load_dataset('diamonds').drop(columns='depth')
        >>> out = TransformDiamonds(diamonds)
        >>> transformed = out.transform_to_depth_pct(diamonds)
        >>> len(transformed.shape) == 1
        True
        >>> np.isclose(transformed[0], 61.286, atol=0.0001)
        True
        """
        def depth_calc(x):
            return (x[2] / ((x[0] + x[1]) / 2)) * 100
        ft = FunctionTransformer(depth_calc)
        x = np.array(data[['x', 'y', 'z']])
        return ft.transform(x.T)


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['best_transformation'],
    'q02': ['create_ordinal'],
    'q03': ['create_one_hot', 'create_proportions'],
    'q04': ['create_quadratics'],
    'q05': ['comparing_performance'],
    'q06,7,8': ['TransformDiamonds']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
