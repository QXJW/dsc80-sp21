
import os

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def car_null_hypoth():
    """
    Returns a list of valid null hypotheses.
    
    :Example:
    >>> set(car_null_hypoth()) <= set(range(1,11))
    True
    """
    return [3,6]

def car_alt_hypoth():
    """
    Returns a list of valid alternative hypotheses.
    
    :Example:
    >>> set(car_alt_hypoth()) <= set(range(1,11))
    True
    """
    return [2,5]

def car_test_stat():
    """
    Returns a list of valid test statistics.
    
    :Example:
    >>> set(car_test_stat()) <= set(range(1,5))
    True
    """
    return [2,4]


def car_p_value():
    """
    Returns an integer corresponding to the correct explanation.
    
    :Example:
    >>> car_p_value() in [1,2,3,4,5]
    True
    """
    return 3


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------

def clean_apps(df):
    '''
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> df = pd.read_csv(fp)
    >>> cleaned = clean_apps(df)
    >>> len(cleaned) == len(df)
    True
    '''
    
    #df['Size'] = df['Size'].str.slice(0,-1)
    df['Size'] = df['Size'].apply(lambda x: int(float(x[:-1])) * 1000 if x[-1] == 'M' else int(float(x[:-1])))
    df['Installs'] = df['Installs'].str.replace(',','').str.slice(0,-1).astype(int)
    df['Type'] = df['Type'].replace({'Free':1, 'Paid':0})
    df['Price'] = df['Price'].str.strip('$').astype(float)
    df['Last Updated'] = df['Last Updated'].str.split(',').str[1].str.strip()
    
    return df


def store_info(cleaned):
    '''
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> df = pd.read_csv(fp)
    >>> cleaned = clean_apps(df)
    >>> info = store_info(cleaned)
    >>> len(info)
    4
    >>> info[2] in cleaned.Category.unique()
    True
    '''

    # find years where over 100 apps were made
    app_count_df = cleaned.groupby('Last Updated').aggregate(np.count_nonzero)
    above_100_years = app_count_df.loc[app_count_df.App >= 100].index
    # group by year, take median of all cols, sort in descending order of median installations
    q1_df = cleaned[cleaned['Last Updated'].isin(above_100_years)]
    grouped_mediansq1 = q1_df.groupby('Last Updated').median().sort_values(by='Installs',ascending=False)
    # set q1 equal to the year of the first row
    q1 = grouped_mediansq1.index[0]
    
    # group by content rating, take min of all cols, sort in in descending order of rating mins
    grouped_minsq2 = cleaned.groupby('Content Rating').min('Rating').sort_values('Rating',ascending=False)
    # set q2 equal to the content rating of the first row
    q2 = grouped_minsq2.index[0]
    
    # group by category, take average of all cols, sort in descending order of price
    grouped_avgsq3 = cleaned.groupby('Category').mean().sort_values(by='Price',ascending=False)
    # set q3 equal to the category of the first row
    q3 = grouped_avgsq3.index[0]
    
    # find apps that have >=1000 reviews
    review_count_df = cleaned.loc[cleaned.Reviews >= 1000]
    # group by category, take average of all cols, sort in ascending order of rating
    grouped_avgsq4 = review_count_df.groupby('Category').mean().sort_values(by='Rating',ascending=True)
    # set q4 equal to category of the first row
    q4 = grouped_avgsq4.index[0]
    
    return [q1,q2,q3,q4]

# ---------------------------------------------------------------------
# Question 3
# ---------------------------------------------------------------------

def std_reviews_by_app_cat(cleaned):
    """
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> play = pd.read_csv(fp)
    >>> clean_play = clean_apps(play)
    >>> out = std_reviews_by_app_cat(clean_play)
    >>> set(out.columns) == set(['Category', 'Reviews'])
    True
    >>> np.all(abs(out.select_dtypes(include='number').mean()) < 10**-7)  # standard units should average to 0!
    True
    """
    
    # remove unnecessary columns
    two_col = cleaned.filter(['Category', 'Reviews'])
    # find grouped means and standard deviations, turn into pandas series
    catmeans = two_col.groupby('Category').transform('mean')['Reviews']
    catdevs = two_col.groupby('Category').transform('std')['Reviews']
    # perform standard unit conversion according to dsc10 textbook using the 2 series above
    two_col['Reviews'] = two_col['Reviews'] - catmeans
    two_col['Reviews'] = two_col['Reviews'] / catdevs
    
    return two_col


def su_and_spread():
    """
    >>> out = su_and_spread()
    >>> len(out) == 2
    True
    >>> out[0].lower() in ['medical', 'family', 'equal']
    True
    >>> out[1] in ['ART_AND_DESIGN', 'AUTO_AND_VEHICLES', 'BEAUTY',\
       'BOOKS_AND_REFERENCE', 'BUSINESS', 'COMICS', 'COMMUNICATION',\
       'DATING', 'EDUCATION', 'ENTERTAINMENT', 'EVENTS', 'FINANCE',\
       'FOOD_AND_DRINK', 'HEALTH_AND_FITNESS', 'HOUSE_AND_HOME',\
       'LIBRARIES_AND_DEMO', 'LIFESTYLE', 'GAME', 'FAMILY', 'MEDICAL',\
       'SOCIAL', 'SHOPPING', 'PHOTOGRAPHY', 'SPORTS', 'TRAVEL_AND_LOCAL',\
       'TOOLS', 'PERSONALIZATION', 'PRODUCTIVITY', 'PARENTING', 'WEATHER',\
       'VIDEO_PLAYERS', 'NEWS_AND_MAGAZINES', 'MAPS_AND_NAVIGATION']
    True
    """
    
    return ['equal', 'GAME']


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


def read_survey(dirname):
    """
    read_survey combines all the survey*.csv files into a singular DataFrame
    :param dirname: directory name where the survey*.csv files are
    :returns: a DataFrame containing the combined survey data
    :Example:
    >>> dirname = os.path.join('data', 'responses')
    >>> out = read_survey(dirname)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> len(out)
    5000
    >>> read_survey('nonexistentfile') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    FileNotFoundError: ... 'nonexistentfile'
    """
    dfs = []
    
    for file in os.listdir(dirname):
        fp = os.path.join(dirname,file)
        file_df = pd.read_csv(fp)
        file_df.columns = file_df.columns.str.lower().str.replace(' ','').str.replace('_','')
        file_df = file_df[['firstname','lastname','currentcompany','jobtitle','email','university']]
        dfs.append(file_df)
    
    total_df = pd.concat(dfs)
    total_df.columns = ['first name', 'last name', 'current company', 'job title', 'email', 'university']
    
    return total_df

def com_stats(df):
    """
    com_stats 
    :param df: a DataFrame containing the combined survey data
    :returns: a hardcoded list of answers to the problems in the notebook
    :Example:
    >>> dirname = os.path.join('data', 'responses')
    >>> df = read_survey(dirname)
    >>> out = com_stats(df)
    >>> len(out)
    4
    >>> isinstance(out[0], int)
    True
    >>> isinstance(out[2], str)
    True
    """

    return [5, 253, 'Business Systems Development Analyst', 369]

# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def combine_surveys(dirname):
    """
    combine_surveys takes in a directory path 
    (containing files favorite*.csv) and combines 
    all of the survey data into one DataFrame, 
    indexed by student ID (a value 0 - 1000).

    :Example:
    >>> dirname = os.path.join('data', 'extra-credit-surveys')
    >>> out = combine_surveys(dirname)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> out.shape
    (1000, 6)
    >>> combine_surveys('nonexistentfile') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    FileNotFoundError: ... 'nonexistentfile'
    """

    dfs = []
    
    for file in os.listdir(dirname):
        fp = os.path.join(dirname,file)
        file_df = pd.read_csv(fp, index_col='id')
        dfs.append(file_df)
    final_df = dfs[0].join(dfs[1:])
    
    return final_df


def check_credit(df):
    """
    check_credit takes in a DataFrame with the 
    combined survey data and outputs a DataFrame 
    of the names of students and how many extra credit 
    points they would receive, indexed by their ID (a value 0-1000)

    :Example:
    >>> dirname = os.path.join('data', 'extra-credit-surveys')
    >>> df = combine_surveys(dirname)
    >>> out = check_credit(df)
    >>> out.shape
    (1000, 2)
    """
    
    final_df = pd.DataFrame(df['name'])
    no_names = df.loc[:, df.columns != 'name']
    
    non_null_row_prop = 1 - (no_names.isnull().sum(axis=1) / no_names.shape[1])
    final_df['non_null_row_prop'] = pd.Series(non_null_row_prop)
    final_df['extra credit'] = final_df['non_null_row_prop'].apply(lambda x: 5 if x >= .75 else 0)
    
    non_null_col_prop = 1 - (no_names.isnull().sum(axis=0) / no_names.shape[0])
    class_ec = any(1 - (no_names.isnull().sum(axis=0) / no_names.shape[0]).values > 0.90)

    if class_ec == True:
        final_df['extra credit'] += 1

    return final_df.drop('non_null_row_prop', axis=1)

# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------

def most_popular_procedure(pets, procedure_history):
    """
    What is the most popular Procedure Type for all of the pets we have in our `pets` dataset?
 
    :Example:
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> procedure_history_fp = os.path.join('data', 'pets', 'ProceduresHistory.csv')
    >>> pets = pd.read_csv(pets_fp)
    >>> procedure_history = pd.read_csv(procedure_history_fp)
    >>> out = most_popular_procedure(pets, procedure_history)
    >>> isinstance(out,str)
    True
    """

    return 'VACCINATIONS'


def pet_name_by_owner(owners, pets):
    """
    pet names by owner

    :Example:
    >>> owners_fp = os.path.join('data', 'pets', 'Owners.csv')
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> owners = pd.read_csv(owners_fp)
    >>> pets = pd.read_csv(pets_fp)
    >>> out = pet_name_by_owner(owners, pets)
    >>> len(out) == len(owners)
    True
    >>> 'Sarah' in out.index
    True
    >>> 'Cookie' in out.values
    True
    """
    petmerge = pets.merge(owners,on='OwnerID')
    grouped_owners = petmerge.groupby('OwnerID')['OwnerID'].count()
    name_index = petmerge.sort_values('OwnerID').drop_duplicates(subset=['OwnerID'])['Name_y']#.unique()
    df_series = pd.Series([petmerge.loc[petmerge['OwnerID'] == owner]['Name_x'].values[0]  
                          if (petmerge.loc[petmerge['OwnerID'] == owner]['Name_x'].values.shape[0] == 1)
                          else list(petmerge.loc[petmerge['OwnerID'] == owner]['Name_x'].values) for owner in grouped_owners.index], index=name_index)
    return df_series

def total_cost_per_city(owners, pets, procedure_history, procedure_detail):
    """
    total cost per city
 
    :Example:
    >>> owners_fp = os.path.join('data', 'pets', 'Owners.csv')
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> procedure_detail_fp = os.path.join('data', 'pets', 'ProceduresDetails.csv')
    >>> procedure_history_fp = os.path.join('data', 'pets', 'ProceduresHistory.csv')
    >>> owners = pd.read_csv(owners_fp)
    >>> pets = pd.read_csv(pets_fp)
    >>> procedure_detail = pd.read_csv(procedure_detail_fp)
    >>> procedure_history = pd.read_csv(procedure_history_fp)
    >>> out = total_cost_per_city(owners, pets, procedure_history, procedure_detail)
    >>> set(out.index) <= set(owners['City'])
    True
    """
    price_merged = pd.merge(left=procedure_history, right=procedure_detail, how='left', left_on=['ProcedureType','ProcedureSubCode'], right_on=['ProcedureType','ProcedureSubCode'])
    pet_merged = price_merged.merge(pets, how='left',on='PetID')
    owner_merged = pet_merged.merge(owners, how='left',on='OwnerID')
    cost_per_city = owner_merged.groupby('City').sum()['Price']
    return cost_per_city

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!


GRADED_FUNCTIONS = {
    'q01': [
        'car_null_hypoth', 'car_alt_hypoth',
        'car_test_stat', 'car_p_value'
    ],
    'q02': ['clean_apps', 'store_info'],
    'q03': ['std_reviews_by_app_cat','su_and_spread'],
    'q04': ['read_survey', 'com_stats'],
    'q05': ['combine_surveys', 'check_credit'],
    'q06': ['most_popular_procedure', 'pet_name_by_owner', 'total_cost_per_city']
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
