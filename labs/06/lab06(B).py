import os
import pandas as pd
import numpy as np
import requests
import bs4
import json


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

    soup = bs4.BeautifulSoup(text, 'html.parser')
    book_divs = soup.find_all('li', attrs = {'class':'col-xs-6 col-sm-4 col-md-3 col-lg-3'})
    book_lists = []

    def find_link(book_div):
        '''
        Scrapes and finds the link for our book
        '''
        return book_div.find('a', href = True)['href']

    def find_rating(book_div):
        '''
        Scrapes and filters down our rating into a number
        '''
        rating = book_div.find('p').attrs['class'][1]
        ratings = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        return ratings[rating.lower()]

    def find_price(book_div):
        '''
        Scrapes and turns our price into a comparable float.
        '''
        price = book_div.find('div', attrs = {'class':"product_price"}).find('p', attrs = {'class':"price_color"}).text.strip('Â').strip('£')
        stripped_price = ""
        for char in price:
            if char.isdigit() or char == '.':
                stripped_price += char
        return float(stripped_price)

    for book_div in book_divs:
        if find_price(book_div) < 50 and find_rating(book_div) >= 4:
            # http://books.toscrape.com/catalogue/ full link in case we need it
            book_lists.append(find_link(book_div))

    return book_lists


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

    soup = bs4.BeautifulSoup(text, 'html.parser')

    def get_category(soup):
        '''
        Take in a soup object and return the category of our book
        '''
        page_divs = soup.find("div", attrs = {"class": "container-fluid page"})
        #find our div containing our category
        page_divs = page_divs.find("div")
        #find our inner div
        page_divs = page_divs.find("ul")
        #find the list of tabs
        page_divs = page_divs.find_all('a')[2]
        #grab our mystery category tab
        return page_divs.text


    def get_info(soup):
        '''
        Take in a soup object and return the information about the book
        '''

        def get_category(soup):
            '''
            Take in a soup object and return the category of our book
            '''
            page_divs = soup.find("div", attrs = {"class": "container-fluid page"})
            #find our div containing our category
            page_divs = page_divs.find("div")
            #find our inner div
            page_divs = page_divs.find("ul")
            #find the list of tabs
            page_divs = page_divs.find_all('a')[2]
            #grab our mystery category tab
            return page_divs.text


        product_info = soup.find("div", attrs = {'id': 'content_inner'})
        #find inner content div
        product_info = product_info.find('article', attrs = {'class': 'product_page'})
        product_info = product_info.find('table', attrs = {'class': 'table table-striped'})
        #find table of products
        product_keys = [product.text for product in product_info.find_all('th')]
        #get the keys for the products on one side of the table
        product_values = [product.text for product in product_info.find_all('td')]
        #get the values for the products on the other side of the table

        product_keys.append("Category")
        product_values.append(get_category(soup))
        #append our category

        product_info2 = soup.find("div", attrs = {'id': 'content_inner'})
        product_info2 = product_info2.find('article', attrs = {'class': 'product_page'})
        product_info2 = product_info2.find('div', attrs = {'class': 'row'})
        product_info2 = product_info2.find('div', attrs = {'class':'col-sm-6 product_main'})

        product_title = product_info2.find('h1').text
        product_keys.append("Title")
        product_values.append(product_title)

        product_rating = product_info2.find_all('p')[2].attrs['class'][1]
        product_keys.append("Rating")
        product_values.append(product_rating)

        product_info3 = soup.find("div", attrs = {'id': 'content_inner'})
        product_info3 = product_info3.find('article', attrs = {'class': 'product_page'})

        product_desc = product_info3.find_all('p')[3].text
        product_keys.append('Description')
        product_values.append(product_desc)


        return dict(zip(product_keys, product_values)) #zip the two lists, turn into a dictionary



    if get_category(soup) in categories:
        return get_info(soup)

    return None


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

    def download_page(n):
        '''
        This function downloads an html page.
        '''
        url = f'http://books.toscrape.com/catalogue/page-{n}.html'
        response = requests.get(url)
        return response.text

    def download_details(link):
        '''
        Creates the full link for the books detail page
        '''
        url = 'http://books.toscrape.com/catalogue/' + link
        response = requests.get(url)
        return response.text

    def scrape_page(n):
        '''
        Scrapes our page :)!!
        '''
        html = download_page(n)
        books = extract_book_links(html)
        dataframe_rows = []
        for book in books:
            dataframe_rows.append(get_product_info(download_details(book), categories))

        return dataframe_rows

    df = []
    for page in range(1, k+1):
        df.extend(scrape_page(page))

    df = [book for book in df if book is not None]
    #filter out our None values

    frames = pd.DataFrame(df)

    return frames.reindex(sorted(frames.columns), axis = 1)


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

    historical_daily_prices = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={date_range[0].strftime("%Y-%m-%d")}-1&to={date_range[-1].strftime("%Y-%m-%d")}&apikey=a74e7dd7a8b20ae6b65ceb1c6413ead4'

    response = requests.get(historical_daily_prices).json()

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

    history.sort_values(by = "date", inplace = True)

    history = history.assign(**{"transaction volume": ((history["high"] + history["low"])/2)*history["volume"]})

    volume = tuple(history[["transaction volume", "changePercent"]].sum().values)[0]/1000000000
    volume = str(f"{volume:.2f}") + 'B'

    open_price = history.iloc[0]["open"]
    close_price = history.iloc[-1]["close"]

    percent = ((close_price - open_price) / open_price) * 100

    if percent < 0:
        percent = str(f"{percent:.2f}") + '%'
    else:
        percent = "+" + str(f"{percent:.2f}") + '%'
    return (percent, volume)



# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

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

    def single_comment(comment_id):
        comment_endpoint = f'https://hacker-news.firebaseio.com/v0/item/{comment_id}.json?print=pretty'
        comment_response = requests.get(comment_endpoint).json()
        if comment_response.get('dead', False):
            #check for dead comment
            return None, None
        return comment_response.get('kids', None), {i:comment_response.get(i, None) for i in ['id', 'by', 'parent', 'text', 'time']}

    def dfs(storyid):
        stack = []
        stack.append(storyid)
        df = pd.DataFrame(columns = ['id', 'by', 'parent', 'text', 'time'])

        while stack:
            cur_node = stack.pop()
            nodes, values = single_comment(cur_node)

            if values is not None:
                df = df.append(values, ignore_index = True)

            if nodes is not None:
                stack.extend(nodes[::-1])
                #add the nodes backwards so we have it in the right order

        df["time"] = pd.to_datetime(df["time"], unit = 's')
        return df.iloc[1:].reset_index(drop = True)

    return dfs(storyid)


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
