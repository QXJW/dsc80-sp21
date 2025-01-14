import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Question # 0
# ---------------------------------------------------------------------

def consecutive_ints(ints):
    """
    consecutive_ints tests whether a list contains two 
    adjacent elements that are consecutive integers.
    :param ints: a list of integers
    :returns: a boolean value if ints contains two 
    adjacent elements that are consecutive integers.
    :Example:
    >>> consecutive_ints([5,3,6,4,9,8])
    True
    >>> consecutive_ints([1,3,5,7,9])
    False
    """

    if len(ints) == 0:
        return False

    for k in range(len(ints) - 1):
        diff = abs(ints[k] - ints[k+1])
        if diff == 1:
            return True

    return False

# ---------------------------------------------------------------------
# Question # 1 
# ---------------------------------------------------------------------

def median_vs_average(nums):
    """
    median takes a non-empty list of numbers,
    returning a boolean of whether the median is
    greater or equal than the average
    If the list has even length, it should return
    the mean of the two elements in the middle.
    :param nums: a non-empty list of numbers.
    :returns: bool, whether median is greater or equal than average.
    
    :Example:
    >>> median_vs_average([6, 5, 4, 3, 2])
    True
    >>> median_vs_average([50, 20, 15, 40])
    False
    >>> median_vs_average([1, 2, 3, 4])
    True
    """
    nums.sort()
    mean = sum(nums) / len(nums)
    half = len(nums) // 2

    if len(nums) % 2 == 1:
        median = nums[half]
    else:
        median = (nums[half-1] + nums[half]) / 2
        
    return median >= mean

# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------
def same_diff_ints(ints):
    """
    same_diff_ints tests whether a list contains
    two list elements i places apart, whose distance 
    as integers is also i.
    :param ints: a list of integers
    :returns: a boolean value if ints contains two 
    elements as described above.
    :Example:
    >>> same_diff_ints([5,3,1,5,9,8])
    True
    >>> same_diff_ints([1,3,5,7,9])
    False
    """
    if len(ints) == 0:
        return False
    
    for i in range(1, len(ints)):
        for j in range(len(ints)):
            if i+j >= len(ints):
                break
            else:
                if abs(i-j) == abs(ints[i]-ints[j]):
                    return True
   
    return False
# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def n_prefixes(s, n):
    """
    n_prefixes returns a string of n
    consecutive prefix of the input string.

    :param s: a string.
    :param n: an integer

    :returns: a string of n consecutive prefixes of s backwards.
    :Example:
    >>> n_prefixes('Data!', 3)
    'DatDaD'
    >>> n_prefixes('Marina', 4)
    'MariMarMaM'
    >>> n_prefixes('aaron', 2)
    'aaa'
    """
    return_str = ''
    
    if len(s) == 0:
        return
    
    while n > 0:
        return_str += s[:n]
        n -= 1
    return return_str
# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------
def exploded_numbers(ints, n):
    """
    exploded_numbers returns a list of strings of numbers from the
    input array each exploded by n.
    Each integer is zero padded.

    :param ints: a list of integers.
    :param n: a non-negative integer.

    :returns: a list of strings of exploded numbers. 
    :Example:
    >>> exploded_numbers([3, 4], 2) 
    ['1 2 3 4 5', '2 3 4 5 6']
    >>> exploded_numbers([3, 8, 15], 2)
    ['01 02 03 04 05', '06 07 08 09 10', '13 14 15 16 17']
    """
    new_list = [x + n for x in ints]  # create list where n is added to all elements of ints
    fillcount = len(str(max(new_list)))  # find maximum digit count of list for zfill

    explode_lst = []
    for x in ints:
        seq_str = ''
        for y in range(x - n, x + n + 1):
            seq_str += str(y).zfill(fillcount) + ' '
        explode_lst.append(seq_str[:-1])

    return explode_lst
# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------

def last_chars(fh):
    """
    last_chars takes a file object and returns a 
    string consisting of the last character of the line.
    :param fh: a file object to read from.
    :returns: a string of last characters from fh
    :Example:
    >>> fp = os.path.join('data', 'chars.txt')
    >>> last_chars(open(fp))
    'hrg'
    """
    read_data = fh.readlines()
    return_str = ''
    for line in read_data:
        if len(line.strip('\n')) > 0:
            return_str += line.strip('\n')[-1]
    fh.close()
    return return_str

# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------

def arr_1(A):
    """
    arr_1 takes in a numpy array and
    adds to each element the square-root of
    the index of each element.
    :param A: a 1d numpy array.
    :returns: a 1d numpy array.
    :Example:
    >>> A = np.array([2, 4, 6, 7])
    >>> out = arr_1(A)
    >>> isinstance(out, np.ndarray)
    True
    >>> np.all(out >= A)
    True
    """
    sqrt_indices = np.sqrt(np.arange(len(A)))
    return A + sqrt_indices

def arr_2(A):
    """
    arr_2 takes in a numpy array of integers
    and returns a boolean array (i.e. an array of booleans)
    whose ith element is True if and only if the ith element
    of the input array is a perfect square.
    :param A: a 1d numpy array.
    :returns: a 1d numpy boolean array.
    :Example:
    >>> out = arr_2(np.array([1, 2, 16, 17, 32, 49]))
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('bool')
    True
    """
    sqrts = np.sqrt(A)
    return sqrts % 1 == 0

def arr_3(A):
    """
    arr_3 takes in a numpy array of stock
    prices per share on successive days in
    USD and returns an array of growth rates.
    :param A: a 1d numpy array.
    :returns: a 1d numpy array.
    :Example:
    >>> fp = os.path.join('data', 'stocks.csv')
    >>> stocks = np.array([float(x) for x in open(fp)])
    >>> out = arr_3(stocks)
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('float')
    True
    >>> out.max() == 0.03
    True
    """

    return np.round(np.diff(A)/np.delete(A, A.size-1),2)

def arr_4(A):
    """
    Create a function arr_4 that takes in A and 
    returns the day on which you can buy at least 
    one share from 'left-over' money. If this never 
    happens, return -1. The first stock purchase occurs on day 0
    :param A: a 1d numpy array of stock prices.
    :returns: the day on which you can buy at least one share from 'left-over' money
    :Example:
    >>> import numbers
    >>> stocks = np.array([3, 3, 3, 3])
    >>> out = arr_4(stocks)
    >>> isinstance(out, numbers.Integral)
    True
    >>> out == 1
    True
    """

    daily_rem = 20 % A
    cumsum = np.cumsum(daily_rem)
    bool_list = list(cumsum > A)
    return bool_list.index(next(filter(lambda i: i != 0, bool_list)))

# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def salary_stats(salary):
    """
    salary_stats returns a series as specified in the notebook.
    :param salary: a dataframe of NBA salaries as found in `salary.csv`
    :return: a series with index specified in the notebook.
    :Example:
    >>> salary_fp = os.path.join('data', 'salary.csv')
    >>> salary = pd.read_csv(salary_fp)
    >>> out = salary_stats(salary)
    >>> isinstance(out, pd.Series)
    True
    >>> 'total_highest' in out.index
    True
    >>> isinstance(out.loc['duplicates'], bool)
    True
    """
    try:
        NumPlayers = len(salary.index)
    except:
        print('an exception has occurred with NumPlayers')
    try:
        NumTeams = len(pd.unique(salary['Team']))
    except:
        print('an exception has occurred with NumTeams')
        
    try:
        TotalSalary = salary['Salary'].sum()
    except:
        print('an exception has occurred with TotalSalary')
    
    try:
        HighestSalaryAmount = salary['Salary'].max()
        HighestSalary = salary.loc[(salary.Salary == HighestSalaryAmount)]['Player'].item()
    except:
        print('an exception has occurred with HighestSalary')
    
    try:
        AvgBos = round(salary.loc[(salary.Team == "BOS")].Salary.mean(), 2)
    except:
        print('an exception has occurred with AvgBos')
        
    try:
        ThirdLowest = salary.sort_values(ascending=True,by='Salary').iloc[2]['Player'] + ', ' + salary.sort_values(ascending=True,by='Salary').iloc[2]['Team']
    except:
        print('an exception has occurred with ThirdLowest')
        
    try:
        last_names = []
        for i in range(len(salary['Player'])):
            last_names.append(salary['Player'].iloc[i].split(' ')[1])
        Duplicates = len(salary['Player']) == len(last_names)
    except:
        print('an exception has occurred with Duplicates')
    
    try:
        highest_team = salary.sort_values(ascending=False,by='Salary').loc[salary.Player =='Stephen Curry'].iloc[0]['Team']
        TotalHighest = salary.loc[(salary.Team == "GSW")]['Salary'].sum()
    except:
        print('an exception has occurred with TotalHighest')
    
    output = {'num_players':NumPlayers, 'num_teams':NumTeams, 'total_salary':TotalSalary, 'highest_salary':HighestSalary, 'avg_bos':AvgBos, 'third_lowest':ThirdLowest, 'duplicates':Duplicates, 'total_highest':TotalHighest}
    return pd.Series(output)
    

# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def parse_malformed(fp):
    """
    Parses and loads the malformed csv data into a 
    properly formatted dataframe (as described in 
    the question).
    :param fh: file handle for the malformed csv-file.
    :returns: a Pandas DataFrame of the data, 
    as specificed in the question statement.
    :Example:
    >>> fp = os.path.join('data', 'malformed.csv')
    >>> df = parse_malformed(fp)
    >>> cols = ['first', 'last', 'weight', 'height', 'geo']
    >>> list(df.columns) == cols
    True
    >>> df['last'].dtype == np.dtype('O')
    True
    >>> df['height'].dtype == np.dtype('float64')
    True
    >>> df['geo'].str.contains(',').all()
    True
    >>> len(df) == 100
    True
    >>> dg = pd.read_csv(fp, nrows=4, skiprows=10, names=cols)
    >>> dg.index = range(9, 13)
    >>> (dg == df.iloc[9:13]).all().all()
    True
    """
    data = []
    with open(fp, 'r') as file:
        variables = file.readline().strip().split(',')
        
        for line in file:
            messy = line.replace(',', ' ').split()
            messy[0] = str(messy[0].replace('\"', ''))
            messy[1] = str(messy[1].replace('\"', ''))
            messy[2] = float(messy[2].replace('\"', ''))
            messy[3] = float(messy[3].replace('\"', ''))
            messy[4] = str(messy[4].replace('\"', '') + ',' + messy[5].replace('\"',''))
            
            messy.remove(messy[5])
            data.append(messy)
    return pd.DataFrame(data, columns = variables)


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------

# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q00': ['consecutive_ints'],
    'q01': ['median_vs_average'],
    'q02': ['same_diff_ints'],
    'q03': ['n_prefixes'],
    'q04': ['exploded_numbers'],
    'q05': ['last_chars'],
    'q06': ['arr_%d' % d for d in range(1, 5)],
    'q07': ['salary_stats'],
    'q08': ['parse_malformed']
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