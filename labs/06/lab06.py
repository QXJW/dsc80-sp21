import os
import pandas as pd
import numpy as np
import requests
import bs4
import json
from bs4 import BeautifulSoup
import datetime


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------


def question1():
    """
    NOTE: You do NOT need to do anything with this function.

    The function for this question makes sure you
    have a correctly named HTML file in the right
    place. Note: This does NOT check if the supplementary files
    needed for your page are there!

    >>> os.path.exists('lab06_1.html')
    True
    """

    # Don't change this function body!
    # No python required; create the HTML file.

    return


# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def extract_book_links(text):
    """
    :Example:
    >>> fp = os.path.join('data', 'products.html')
    >>> out = extract_book_links(open(fp, encoding='utf-8').read())
    >>> url = 'scarlet-the-lunar-chronicles-2_218/index.html'
    >>> out[1] == url
    True
    """
    ratings = ['four','five']
    urls = []
    books = bs4.BeautifulSoup(text,features="lxml").find_all('article',attrs={'class':'product_pod'})
    for book in books:
        if (book.find('p').attrs['class'][1].lower() in ratings) and (float(book.find('p',attrs={'class':'price_color'}).text.strip('£').strip('Â').strip('£')) < 50):
            urls.append(book.find('a').attrs['href'].replace('catalogue/',''))
    return urls


def get_product_info(text, categories):
    """
    :Example:
    >>> fp = os.path.join('data', 'Frankenstein.html')
    >>> out = get_product_info(open(fp, encoding='utf-8').read(), ['Default'])
    >>> isinstance(out, dict)
    True
    >>> 'Category' in out.keys()
    True
    >>> out['Rating']
    'Two'
    """
    
    book = bs4.BeautifulSoup(text,features="lxml")
    cat = book.find('ul',attrs={'class':'breadcrumb'}).find_all('a')[2].text
    
    if cat not in categories:
        return None
    else:
        table_entries = book.find('table',attrs={'class':'table table-striped'}).find_all('tr')
        entries = []
        for x in table_entries:
            cut1 = str(x).find('<td>') + 4
            cut2= str(x).find('</td>')
            entries.append(str(x)[cut1:cut2])
        
        description = book.find('article', attrs={'class':'product_page'}).find_all('p')[3].text

        main = book.find('div', attrs={'class':'col-sm-6 product_main'})
        title = main.find('h1').text
        rating = main.find_all('p')[2].attrs['class'][1]
        
        df_dict = {'Availability':entries[5], 'Category':cat, 'Description':description, 
                   'Number of reviews':entries[6], 'Price (excl. tax)':entries[2], 'Price (incl. tax)':entries[3],
                   'Product Type':entries[1], 'Rating':rating, 'Tax':entries[4], 'Title':title, 'UPC':entries[0]}
    return df_dict


def scrape_books(k, categories):
    """
    :param k: number of book-listing pages to scrape.
    :returns: a dataframe of information on (certain) books
    on the k pages (as described in the question).

    :Example:
    >>> out = scrape_books(1, ['Mystery'])
    >>> out.shape
    (1, 11)
    >>> out['Rating'][0] == 'Four'
    True
    >>> out['Title'][0] == 'Sharp Objects'
    True
    """
    pages = []
    df = pd.DataFrame()
    for i in range (1,k+1):
        pages.append('http://books.toscrape.com/catalogue/page-{}.html'.format(i))
    for fp in pages:
        page = requests.get(fp).text
        books = extract_book_links(page)
        for book in books:
            link = 'http://books.toscrape.com/catalogue/' + book
            book_page = requests.get(link).text
            book_dict = get_product_info(book_page,categories)
            if book_dict is not None:
                book_df = pd.DataFrame([book_dict])
                df = df.append(book_df)
    return df.reindex(sorted(df.columns),axis=1)


# ---------------------------------------------------------------------
# Question 3
# ---------------------------------------------------------------------

def stock_history(ticker, year, month):
    """
    Given a stock code and month, return the stock price details for that month
    as a dataframe

    >>> history = stock_history('BYND', 2019, 6)
    >>> history.shape == (20, 13)
    True
    >>> history.label.iloc[-1]
    'June 03, 19'
    """
    date_range = pd.date_range(start = f'{str(year)}-{str(month)}', end = f'{str(year)}-{str(int(month) + 1)}')[:-1]
    key = 'fe8f70fbc0359ff10974537662eb687f'
    stock_endpoint = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={date_range[0].strftime("%Y-%m-%d")}-1&to={date_range[-1].strftime("%Y-%m-%d")}&apikey={key}'
    response = requests.get(stock_endpoint).json()
    stock_info = response['historical']
    return pd.DataFrame(stock_info)


def stock_stats(history):
    """
    Given a stock's trade history, return the percent change and transactions
    in billion dollars.

    >>> history = stock_history('BYND', 2019, 6)
    >>> stats = stock_stats(history)
    >>> len(stats[0]), len(stats[1])
    (7, 6)
    >>> float(stats[0][1:-1]) > 30
    True
    >>> float(stats[1][:-1]) > 1
    True
    """
    df = history.sort_values('date')
    
    pc = (df.iloc[-1]['close'] - df.iloc[0]['open']) / df.iloc[0]['open'] * 100
    if pc > 0:
        pc = '+' + str(f"{pc:.2f}") + '%'
    else:
        pc = str(f"{percent:.2f}") + '%'
    
    ttv_series = df.apply(lambda row : (row.low + row.high) / 2 * row.volume, axis=1)
    ttv = ttv_series.sum() / 1000000000
    ttv = str(f"{ttv:.2f}") + 'B'
    return pc,ttv

# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------
def kids_dfs(comment_id,cols):
    link = "https://hacker-news.firebaseio.com/v0/item/{}.json".format(comment_id)
    load = requests.get(link).json()
    link_series = pd.Series(load)
    
    if 'kids' in link_series.index:
        kids = list(link_series['kids'])
        #recurse = [kids_dfs(kid,cols) for kid in kids]
        if 'dead' in link_series.index:
            return pd.concat([kids_dfs(kid,cols) for kid in kids], ignore_index=True)
        else:
            link_df = [pd.DataFrame([link_series[cols]])] + [kids_dfs(kid,cols) for kid in kids]
            return pd.concat(link_df, ignore_index=True)
    else:
        if 'dead' in link_series.index:
            return pd.DataFrame(columns=cols)
        else:
            return pd.DataFrame([link_series[cols]])
        
def get_comments(storyid):
    """
    Returns a dataframe of all the comments below a news story
    >>> out = get_comments(18344932)
    >>> out.shape
    (18, 5)
    >>> out.loc[5, 'by']
    'RobAtticus'
    >>> out.loc[5, 'time'].day
    31
    """
    story_endpoint = "https://hacker-news.firebaseio.com/v0/item/{}.json".format(storyid)
    load = requests.get(story_endpoint).json()
    story_df = pd.DataFrame(load)
    
    cols = ['id','by','parent','text','time']
    comment_df = pd.DataFrame(columns=cols)

    for comment_id in story_df['kids']:
        #comment_endpoint = "https://hacker-news.firebaseio.com/v0/item/{}.json".format(comment_id)
        comments = kids_dfs(comment_id,cols)
        comment_df = pd.concat([comment_df,comments],ignore_index=True)
    
    comment_df['time'] = pd.to_datetime(comment_df['time'], unit='s')
    comment_df = comment_df.astype({'id':'int','parent':'int'})
    
    return comment_df


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['question1'],
    'q02': ['extract_book_links', 'get_product_info', 'scrape_books'],
    'q03': ['stock_history', 'stock_stats'],
    'q04': ['get_comments']
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
