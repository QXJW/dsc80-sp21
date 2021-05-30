import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------

def get_san(infp, outfp):
    """
    get_san takes in a filepath containing all flights and a filepath where
    filtered dataset #1 is written (that is, all flights arriving or departing
    from San Diego International Airport in 2015).
    The function should return None.

    :Example:
    >>> infp = os.path.join('data', 'flights.test')
    >>> outfp = os.path.join('data', 'santest.tmp')
    >>> get_san(infp, outfp)
    >>> df = pd.read_csv(outfp)
    >>> df.shape
    (53, 31)
    >>> os.remove(outfp)
    """
    dfs = []
    
    chunk = pd.read_csv(infp, chunksize=1000)
    
    for df in chunk:
        chunkdf = df.loc[(df['ORIGIN_AIRPORT'] == 'SAN') | (df['DESTINATION_AIRPORT'] == 'SAN')]
        dfs.append(chunkdf)
        
    concatted = pd.concat(dfs)
    formatted = concatted.reset_index(drop=True)
    formatted.to_csv(outfp,index=False)
    
    return None


def get_sw_jb(infp, outfp):
    """
    get_sw_jb takes in a filepath containing all flights and a filepath where
    filtered dataset #2 is written (that is, all flights flown by either
    JetBlue or Southwest Airline in 2015).
    The function should return None.

    :Example:
    >>> infp = os.path.join('data', 'flights.test')
    >>> outfp = os.path.join('data', 'jbswtest.tmp')
    >>> get_sw_jb(infp, outfp)
    >>> df = pd.read_csv(outfp)
    >>> df.shape
    (73, 31)
    >>> os.remove(outfp)
    """
    
    airlines = ['JB','SW']
    dfs = []
    
    chunk = pd.read_csv(infp, chunksize=1000)
    
    for df in chunk:
        chunkdf = df.loc[(df['YEAR'] == 2015)]
        chunkdf = df.loc[(chunkdf['AIRLINE'] == 'B6') | (chunkdf['AIRLINE'] == 'WN')]
        dfs.append(chunkdf)
    
    sw_jb = pd.concat(dfs)
    sw_jb = sw_jb.reset_index(drop = True)
    
    chunkdf.to_csv(outfp,index=False)
    
    return None


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------

def data_kinds():
    """
    data_kinds outputs a (hard-coded) dictionary of data kinds, keyed by column
    name, with values Q, O, N (for 'Quantitative', 'Ordinal', or 'Nominal').

    :Example:
    >>> out = data_kinds()
    >>> isinstance(out, dict)
    True
    >>> set(out.values()) == {'O', 'N', 'Q'}
    True
    """
    return {'YEAR':'O', 'MONTH':'O', 'DAY':'O', 'DAY_OF_WEEK':'O', 'AIRLINE':'N', 'FLIGHT_NUMBER':'Q',
       'TAIL_NUMBER':'N', 'ORIGIN_AIRPORT':'N', 'DESTINATION_AIRPORT':'N',
       'SCHEDULED_DEPARTURE':'Q', 'DEPARTURE_TIME':'Q', 'DEPARTURE_DELAY':'Q', 'TAXI_OUT':'Q',
       'WHEELS_OFF':'Q', 'SCHEDULED_TIME':'Q', 'ELAPSED_TIME':'Q', 'AIR_TIME':'Q', 'DISTANCE':'Q',
       'WHEELS_ON':'Q', 'TAXI_IN':'Q', 'SCHEDULED_ARRIVAL':'Q', 'ARRIVAL_TIME':'Q',
       'ARRIVAL_DELAY':'Q', 'DIVERTED':'Q', 'CANCELLED':'Q', 'CANCELLATION_REASON':'N',
       'AIR_SYSTEM_DELAY':'Q', 'SECURITY_DELAY':'Q', 'AIRLINE_DELAY':'Q',
       'LATE_AIRCRAFT_DELAY':'Q', 'WEATHER_DELAY':'Q'}


def data_types():
    """
    data_types outputs a (hard-coded) dictionary of data types, keyed by column
    name, with values str, int, float.

    :Example:
    >>> out = data_types()
    >>> isinstance(out, dict)
    True
    >>> set(out.values()) == {'int', 'str', 'float', 'bool'}
    True
    """

    return {'YEAR':'int', 'MONTH':'int', 'DAY':'int', 'DAY_OF_WEEK':'int', 'AIRLINE':'str', 'FLIGHT_NUMBER':'int',
       'TAIL_NUMBER':'str', 'ORIGIN_AIRPORT':'str', 'DESTINATION_AIRPORT':'str',
       'SCHEDULED_DEPARTURE':'int', 'DEPARTURE_TIME':'float', 'DEPARTURE_DELAY':'float', 'TAXI_OUT':'float',
       'WHEELS_OFF':'float', 'SCHEDULED_TIME':'int', 'ELAPSED_TIME':'float', 'AIR_TIME':'float', 'DISTANCE':'int',
       'WHEELS_ON':'float', 'TAXI_IN':'float', 'SCHEDULED_ARRIVAL':'int', 'ARRIVAL_TIME':'float',
       'ARRIVAL_DELAY':'float', 'DIVERTED':'bool', 'CANCELLED':'bool', 'CANCELLATION_REASON':'str',
       'AIR_SYSTEM_DELAY':'float', 'SECURITY_DELAY':'float', 'AIRLINE_DELAY':'float',
       'LATE_AIRCRAFT_DELAY':'float', 'WEATHER_DELAY':'float'}


# ---------------------------------------------------------------------
# Question #3
# ---------------------------------------------------------------------

def basic_stats(flights):
    """
    basic_stats takes flights and outputs a dataframe that contains statistics
    for flights arriving/departing for SAN.
    That is, the output should have have two rows, indexed by ARRIVING and
    DEPARTING, and have the following columns:

    * number of arriving/departing flights to/from SAN (count).
    * mean flight (arrival) delay of arriving/departing flights to/from SAN
      (mean_delay).
    * median flight (arrival) delay of arriving/departing flights to/from SAN
      (median_delay).
    * the airline code of the airline with the longest flight (arrival) delay
      among all flights arriving/departing to/from SAN (airline).
    * a list of the three months with the greatest number of arriving/departing
      flights to/from SAN, sorted from greatest to least (top_months).

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> dtypes = data_types()
    >>> flights = pd.read_csv(fp, dtype=dtypes)
    >>> out = basic_stats(flights)
    >>> out.index.tolist() == ['ARRIVING', 'DEPARTING']
    True
    >>> cols = ['count', 'mean_delay', 'median_delay', 'airline', 'top_months']
    >>> out.columns.tolist() == cols
    True
    """
    from_sd = flights.loc[flights['ORIGIN_AIRPORT'] == 'SAN']
    to_sd = flights.loc[flights['DESTINATION_AIRPORT'] == 'SAN']
    
    q1_dep = from_sd.shape[0]
    q1_arr = to_sd.shape[0]
    
    q2_dep = from_sd['ARRIVAL_DELAY'].mean()
    q2_arr = to_sd['ARRIVAL_DELAY'].mean()
    
    q3_dep = from_sd['ARRIVAL_DELAY'].median()
    q3_arr = to_sd['ARRIVAL_DELAY'].median()
    
    q4_arr = from_sd.loc[from_sd['ARRIVAL_DELAY'] == from_sd['ARRIVAL_DELAY'].max()]['AIRLINE'].values[0]
    q4_dep = to_sd.loc[to_sd['ARRIVAL_DELAY'] == to_sd['ARRIVAL_DELAY'].max()]['AIRLINE'].values[0]
    
    q5_dep = from_sd.groupby('MONTH').count().sort_values('DAY',ascending=False).index.tolist()[:3]
    q5_arr = to_sd.groupby('MONTH').count().sort_values('DAY',ascending=False).index.tolist()[:3]
    
    df = pd.DataFrame({'count':[q1_arr,q1_dep], 'mean_delay':[q2_arr,q2_dep], 'median_delay':[q3_arr,q3_dep], 'airline':[q4_arr,q4_dep], 'top_months':[q5_dep,q5_arr]},index=['ARRIVING','DEPARTING'])
    
    return df


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


def depart_arrive_stats(flights):
    """
    depart_arrive_stats takes in a dataframe like flights and calculates the
    following quantities in a series (with the index in parentheses):
    - The proportion of flights from/to SAN that
      leave late, but arrive early or on-time (late1).
    - The proportion of flights from/to SAN that
      leaves early, or on-time, but arrives late (late2).
    - The proportion of flights from/to SAN that
      both left late and arrived late (late3).

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> dtypes = data_types()
    >>> flights = pd.read_csv(fp, dtype=dtypes)
    >>> out = depart_arrive_stats(flights)
    >>> out.index.tolist() == ['late1', 'late2', 'late3']
    True
    >>> isinstance(out, pd.Series)
    True
    >>> out.max() < 0.30
    True
    """
    
    late1 = flights.loc[(flights['DEPARTURE_DELAY'] > 0) & (flights['ARRIVAL_DELAY'] <= 0)].shape[0] / flights.shape[0]
    
    late2 = flights.loc[(flights['DEPARTURE_DELAY'] <= 0) & (flights['ARRIVAL_DELAY'] > 0)].shape[0] / flights.shape[0]
    
    late3 = flights.loc[(flights['DEPARTURE_DELAY'] > 0) & (flights['ARRIVAL_DELAY'] > 0)].shape[0] / flights.shape[0]
    
    df = pd.Series([late1, late2, late3],index=['late1','late2','late3'])
    return df


def depart_arrive_stats_by_month(flights):
    """
    depart_arrive_stats_by_month takes in a dataframe like flights and
    calculates the quantities in depart_arrive_stats, broken down by month

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> dtypes = data_types()
    >>> flights = pd.read_csv(fp, dtype=dtypes)
    >>> out = depart_arrive_stats_by_month(flights)
    >>> out.columns.tolist() == ['late1', 'late2', 'late3']
    True
    >>> set(out.index) <= set(range(1, 13))
    True
    """

    return flights.groupby('MONTH').apply(depart_arrive_stats)


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def cnts_by_airline_dow(flights):
    """
    mean_by_airline_dow takes in a dataframe like flights and outputs a
    dataframe that answers the question:
    Given any AIRLINE and DAY_OF_WEEK, how many flights were there (in 2015)?

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = cnts_by_airline_dow(flights)
    >>> set(out.columns) == set(flights['AIRLINE'].unique())
    True
    >>> set(out.index) == set(flights['DAY_OF_WEEK'].unique())
    True
    >>> (out >= 0).all().all()
    True
    """
    df = flights.pivot_table(values='DAY',index='DAY_OF_WEEK',columns='AIRLINE',aggfunc='count')
    return df


def mean_by_airline_dow(flights):
    """
    mean_by_airline_dow takes in a dataframe like flights and outputs a
    dataframe that answers the question:
    Given any AIRLINE and DAY_OF_WEEK, what is the average ARRIVAL_DELAY (in
    2015)?

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = mean_by_airline_dow(flights)
    >>> set(out.columns) == set(flights['AIRLINE'].unique())
    True
    >>> set(out.index) == set(flights['DAY_OF_WEEK'].unique())
    True
    """
    df = flights.pivot_table(values='ARRIVAL_DELAY',index='DAY_OF_WEEK',columns='AIRLINE',aggfunc='mean')
    return df


# ---------------------------------------------------------------------
# Question #6
# ---------------------------------------------------------------------

def predict_null_arrival_delay(row):
    """
    predict_null takes in a row of the flights data (that is, a Series) and
    returns True if the ARRIVAL_DELAY is null and otherwise False.

    :param row: a Series that represents a row of `flights`
    :returns: a boolean representing when `ARRIVAL_DELAY` is null.

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = flights.drop('ARRIVAL_DELAY', axis=1).apply(predict_null_arrival_delay, axis=1)
    >>> set(out.unique()) - set([True, False]) == set()
    True
    """
    if row['CANCELLED'] == True | row['DIVERTED'] == True:
        return True
    else:
        return False


def predict_null_airline_delay(row):
    """
    predict_null takes in a row of the flights data (that is, a Series) and
    returns True if the AIRLINE_DELAY is null and otherwise False. Since the
    function doesn't depend on AIRLINE_DELAY, it should work a row even if that
    index is dropped.

    :param row: a Series that represents a row of `flights`
    :returns: a boolean representing when `AIRLINE_DELAY` is null.

    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = flights.drop('AIRLINE_DELAY', axis=1).apply(predict_null_airline_delay, axis=1)
    >>> set(out.unique()) - set([True, False]) == set()
    True
    """

    if row['ARRIVAL_DELAY'] < 15 or row['CANCELLED'] == True or row['DIVERTED'] == True:
        return True
    else:
        return False


# ---------------------------------------------------------------------
# Question #7
# ---------------------------------------------------------------------
def ts_calc(data, col):
    df = data[['DEPARTURE_DELAY',col,'SHUFFLED']]
    #pivotted = pd.DataFrame(df.pivot_table(index='DEPARTURE_DELAY',columns=col,aggfunc='size'))
    pivotted = pd.DataFrame(df.pivot_table(columns=col,aggfunc='size'))
    distr = pivotted / pivotted.sum(axis=0)
    ts = distr.diff(axis=0).iloc[-1].abs().sum() / 2
    return ts
def sim_null(data, col):
    shuffled = np.random.permutation(data[col].values)
    data['SHUFFLED'] = shuffled
    test_stat = ts_calc(data,col)
    return test_stat
def perm4missing(flights, col, N):
    """
    perm4missing takes in flights, a column col, and a number N and returns the
    p-value of the test (using N simulations) that determines if
    DEPARTURE_DELAY is MAR dependent on col.
    
    :Example:
    >>> fp = os.path.join('data', 'to_from_san.csv')
    >>> flights = pd.read_csv(fp, nrows=100)
    >>> out = perm4missing(flights, 'AIRLINE', 100)
    >>> 0 <= out <= 1
    True
    """
  
    pivotted = pd.DataFrame(flights[['DEPARTURE_DELAY',col]].pivot_table(columns=col,aggfunc='size'))
    distr = pivotted / pivotted.sum(axis=0)
    obs = distr.diff(axis=0).iloc[:,-1].abs().sum() / 2
    differences = []
    
    for trial in range(N):
        trial_null = sim_null(flights,col)
        differences.append(trial_null)
    
    return np.count_nonzero(differences >= obs) / N


def dependent_cols():
    """
    dependent_cols gives a list of columns on which DEPARTURE_DELAY is MAR
    dependent on.

    :Example:
    >>> out = dependent_cols()
    >>> isinstance(out, list)
    True
    >>> cols = 'YEAR DAY_OF_WEEK AIRLINE DIVERTED CANCELLATION_REASON'.split()
    >>> set(out) <= set(cols)
    True
    """

    return ['AIRLINE','DAY_OF_WEEK']


def missing_types():
    """
    missing_types returns a Series
    - indexed by the following columns of flights:
    CANCELLED, CANCELLATION_REASON, TAIL_NUMBER, ARRIVAL_TIME.
    - The values contain the most-likely missingness type of each column.
    - The unique values of this Series should be MD, MCAR, MAR, MNAR, NaN.

    :param:
    :returns: A series with index and values as described above.

    :Example:
    >>> out = missing_types()
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) - set(['MD', 'MCAR', 'MAR', 'NMAR', np.NaN]) == set()
    True
    """
    indexes = 'CANCELLED CANCELLATION_REASON TAIL_NUMBER ARRIVAL_TIME'.split()
    return pd.Series([np.NaN, 'MD', 'MCAR', 'MD'],index=indexes)


# ---------------------------------------------------------------------
# Question #8
# ---------------------------------------------------------------------

def prop_delayed_by_airline(jb_sw):
    """
    prop_delayed_by_airline takes in a dataframe like jb_sw and returns a
    DataFrame indexed by airline that contains the proportion of each airline's
    flights that are delayed.

    :param jb_sw: a dataframe similar to jb_sw
    :returns: a dataframe as above

    :Example:
    >>> fp = os.path.join('data', 'jetblue_or_sw.csv')
    >>> jb_sw = pd.read_csv(fp, nrows=100)
    >>> out = prop_delayed_by_airline(jb_sw)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> (out >= 0).all().all() and (out <= 1).all().all()
    True
    >>> len(out.columns) == 1
    True
    """
    pd.options.mode.chained_assignment = None  # default='warn'
    
    codes = 'ABQ BDL BUR DCA MSY PBI PHX RNO SJC SLC'.split()
    
    jb_sw = jb_sw[['ORIGIN_AIRPORT','AIRLINE','DEPARTURE_DELAY','CANCELLED']]
    jb_sw = jb_sw.query("""ORIGIN_AIRPORT in @codes""")
    
    jb_sw['DELAYED'] = (jb_sw['DEPARTURE_DELAY'] > 0) & (jb_sw['CANCELLED'] == False)
    jb_sw = jb_sw.drop(['DEPARTURE_DELAY','CANCELLED'],axis=1)
    
    result = jb_sw.groupby("AIRLINE").mean()['DELAYED'].to_frame()
    
    return result


def prop_delayed_by_airline_airport(jb_sw):
    """
    prop_delayed_by_airline_airport that takes in a dataframe like jb_sw and
    returns a DataFrame, with columns given by airports, indexed by airline,
    that contains the proportion of each airline's flights that are delayed at
    each airport.

    :param jb_sw: a dataframe similar to jb_sw
    :returns: a dataframe as above.

    :Example:
    >>> fp = os.path.join('data', 'jetblue_or_sw.csv')
    >>> jb_sw = pd.read_csv(fp, nrows=100)
    >>> out = prop_delayed_by_airline_airport(jb_sw)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> ((out >= 0) | (out <= 1) | (out.isnull())).all().all()
    True
    >>> len(out.columns) == 6
    True
    """
    pd.options.mode.chained_assignment = None  # default='warn'
    
    codes = 'ABQ BDL BUR DCA MSY PBI PHX RNO SJC SLC'.split()
    
    jb_sw = jb_sw[['ORIGIN_AIRPORT','AIRLINE','DEPARTURE_DELAY','CANCELLED']]
    jb_sw = jb_sw.query("""ORIGIN_AIRPORT in @codes""")
    
    jb_sw['DELAYED'] = (jb_sw['DEPARTURE_DELAY'] > 0) #& (jb_sw['CANCELLED'] == False)
    jb_sw = jb_sw.drop(['DEPARTURE_DELAY','CANCELLED'],axis=1)

    result = jb_sw.pivot_table(index='AIRLINE',columns='ORIGIN_AIRPORT', values='DELAYED',aggfunc='mean')
    #result = mean#.dropna(axis=1)
    
    return result#.dropna()#[result['DELAY_PROP'] > 0]

# ---------------------------------------------------------------------
# Question #9
# ---------------------------------------------------------------------

def verify_simpson(df, group1, group2, occur):
    """
    verify_simpson verifies whether a dataset displays Simpson's Paradox.

    :param df: a dataframe
    :param group1: the first group being aggregated
    :param group2: the second group being aggregated
    :param occur: a column of df with values {0,1}, denoting
    if an event occurred.
    :returns: a boolean. True if simpson's paradox is present,
    otherwise False.

    :Example:
    >>> df = pd.DataFrame([[4,2,1], [1,2,0], [1,4,0], [4,4,1]], columns=[1,2,3])
    >>> verify_simpson(df, 1, 2, 3) in [True, False]
    True
    >>> verify_simpson(df, 1, 2, 3)
    False
    """

    group1_values = df.groupby(group1).mean()[occur].diff().iloc[-1] < 0
    
    group2 = df.pivot_table(index=group1, columns=group2, aggfunc='mean',values=occur)
    group2_values = group2.diff().iloc[-1] < 0
    
    if all(group2_values):
        return group1_values != all(group2_values)
    elif all(group2_values == False):
        return group1_values != all(group2_values)
    else:
        return False


# ---------------------------------------------------------------------
# Question #10
# ---------------------------------------------------------------------

def search_simpsons(jb_sw, N):
    """
    search_simpsons takes in the jb_sw dataset and a number N, and returns a
    list of N airports for which the proportion of flight delays between
    JetBlue and Southwest satisfies Simpson's Paradox.

    Only consider airports that have '3 letter codes',
    Only consider airports that have at least one JetBlue and Southwest flight.

    :Example:
    >>> fp = os.path.join('data', 'jetblue_or_sw.csv')
    >>> jb_sw = pd.read_csv(fp, nrows=1000)
    >>> pair = search_simpsons(jb_sw, 2)
    >>> len(pair) == 2
    True
    """

    return ...


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_san', 'get_sw_jb'],
    'q02': ['data_kinds', 'data_types'],
    'q03': ['basic_stats'],
    'q04': ['depart_arrive_stats', 'depart_arrive_stats_by_month'],
    'q05': ['cnts_by_airline_dow', 'mean_by_airline_dow'],
    'q06': ['predict_null_arrival_delay', 'predict_null_airline_delay'],
    'q07': ['perm4missing', 'dependent_cols', 'missing_types'],
    'q08': ['prop_delayed_by_airline', 'prop_delayed_by_airline_airport'],
    'q09': ['verify_simpson'],
    'q10': ['search_simpsons']
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
