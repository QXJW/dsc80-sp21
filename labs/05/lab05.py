import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def first_round():
    """
    :return: list with two values
    >>> out = first_round()
    >>> isinstance(out, list)
    True
    >>> out[0] < 1
    True
    >>> out[1] is "NR" or out[1] is "R"
    True
    """
    
    return [.159, 'NR']


def second_round():
    """
    :return: list with three values
    >>> out = second_round()
    >>> isinstance(out, list)
    True
    >>> out[0] < 1
    True
    >>> out[1] is "NR" or out[1] is "R"
    True
    >>> out[2] is "ND" or out[2] is "D"
    True
    """
    return [.026, 'R', 'D']


# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def verify_child(heights):
    """
    Returns a series of p-values assessing the missingness
    of child-height columns on father height.

    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> heights = pd.read_csv(fp)
    >>> out = verify_child(heights)
    >>> out['child_50'] < out['child_95']
    True
    >>> out['child_5'] > out['child_50']
    True
    """
    
    n_repetitions = 100
    heights = heights.copy()
    columns = list(filter(lambda x: 'child' in x and '_' in x, heights.columns))
    
    pval_list = []
    for child in columns:
        ks_list = []

        loop_df = heights[["father", child]]
        loop_df = loop_df.assign(is_null = loop_df[child].isnull())
        
        gpA = loop_df.loc[loop_df['is_null'] == True, 'father']
        gpB = loop_df.loc[loop_df['is_null'] == False, 'father']
        obs = ks_2samp(gpA, gpB).statistic

        for _ in range(n_repetitions):
            # shuffle the weights
            shuffled_father = (
                loop_df['father']
                .sample(replace=False, frac=1)
                .reset_index(drop=True)
            )

            # put them in a table
            shuffled = (loop_df
                       .assign(**{
                           'father': shuffled_father,
                       }))

            # compute the group differences (test statistic!)
            grps = shuffled.groupby('is_null')['father']
            ks = ks_2samp(grps.get_group(True), grps.get_group(False)).statistic
            
            # add it to the list of results
            ks_list.append(ks)

        pval = np.mean(np.array(ks_list) > obs)
        pval_list.append(pval)
        
    return pd.Series(pval_list,index=columns)


def missing_data_amounts():
    """
    Returns a list of multiple choice answers.

    :Example:
    >>> set(missing_data_amounts()) <= set(range(1,6))
    True
    """

    return [1,2,5]


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def cond_single_imputation(new_heights):
    """
    cond_single_imputation takes in a dataframe with columns 
    father and child (with missing values in child) and imputes 
    single-valued mean imputation of the child column, 
    conditional on father. Your function should return a Series.

    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> df['child'] = df['child_50']
    >>> out = cond_single_imputation(df)
    >>> out.isnull().sum() == 0
    True
    >>> (df.child.std() - out.std()) > 0.5
    True
    """
    new_heights['quartile'] = pd.qcut(new_heights['father'],4, labels=['Q1','Q2','Q3','Q4'])
    filled = new_heights.groupby('quartile')['child'].transform(lambda x: x.fillna(x.mean()))
    return filled

# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------


def quantitative_distribution(child, N):
    """
    quantitative_distribution that takes in a Series and an integer 
    N > 0, and returns an array of N samples from the distribution of 
    values of the Series as described in the question.
    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> child = df['child_50']
    >>> out = quantitative_distribution(child, 100)
    >>> out.min() >= 56
    True
    >>> out.max() <= 79
    True
    >>> np.isclose(out.mean(), child.mean(), atol=1)
    True
    """
    rand_list = []
    
    no_na = child.dropna()
    counts,bins = np.histogram(no_na,bins=10)
    density = counts / counts.sum()
    
    for _ in range(N):
        rand_bin = np.random.choice(range(0,10),p=density)
        rand_fill = np.random.uniform(bins[rand_bin],bins[rand_bin+1])
        rand_list.append(rand_fill)
    return pd.Series(rand_list)


def impute_height_quant(child):
    """
    impute_height_quant takes in a Series of child heights 
    with missing values and imputes them using the scheme in
    the question.

    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> child = df['child_50']
    >>> out = impute_height_quant(child)
    >>> out.isnull().sum() == 0
    True
    >>> np.isclose(out.mean(), child.mean(), atol=0.5)
    True
    """
    
    num_null = child.isnull().sum() 
    fill_values = quantitative_distribution(child, num_null)
    fill_values.index = child.loc[child.isnull()].index
    filled = child.fillna(fill_values)
    
    return filled


# ---------------------------------------------------------------------
# Question # X
# ---------------------------------------------------------------------

def answers():
    """
    Returns two lists with your answers
    :return: Two lists: one with your answers to multiple choice questions
    and the second list has 6 websites that satisfy given requirements.
    >>> list1, list2 = answers()
    >>> len(list1)
    4
    >>> len(list2)
    6
    """
    return [1, 2, 1, 1],['soundcloud.com','qq.com','fc2.com','linkedin.com','facebook.com','instagram.com']




# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['first_round', 'second_round'],
    'q02': ['verify_child', 'missing_data_amounts'],
    'q03': ['cond_single_imputation'],
    'q04': ['quantitative_distribution', 'impute_height_quant'],
    'q05': ['answers']
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
                In %s, part %s is missing" % (q, elt)
                raise Exception(stmt)

    return True
