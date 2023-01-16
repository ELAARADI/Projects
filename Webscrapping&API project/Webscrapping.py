"""
This is the first part of the Python project, related to webscrapping data from
Boursorama website.
"""

import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

def data_collection(nb_pages):
    """ This is the webscrapping function that will allow us to retrieve fund
    data from Boursorama website"""
    # data storage
    response = []
    soup = []
    data = []
    rows = []

    # relevant info storage
    fund_name = []
    fund_last_close = []
    fund_var = []
    fund_yoy = []
    fund_risk = []

    for i in tqdm(range(1, nb_pages + 1), desc='Webscrapping pages from the website'):
        # here we get the data for each page, and append it and its equivalent
        # BeautifulSoup result and the tables found in it
        response.append(requests.get("https://www.boursorama.com/bourse/opcvm/" + \
                                     f"recherche/page-{i}?beginnerFundSearch%5Bsaving%5D=1",
                                     headers={'User-Agent': 'My User Agent 1.0'}))
        soup.append(BeautifulSoup(response[-1].text, 'lxml'))
        data.append(soup[-1].find_all('table', {'class': \
                'c-table c-table--generic c-table--generic c-shadow-overflow__table' + \
                    '-fixed-column c-shadow-overflow__table-fixed-column'})[0])

        for j in data[-1]:
            rows.append(j.find_all('tr'))
            for k in range(1, len(rows[-1])):
                # fund name
                if k == 1:
                    fund_name.append(rows[-1][0].find_all('a')[0].text)
                    info = rows[-1][0].find_all('td')
                    fund_last_close.append(info[2].text)
                    fund_var.append(info[3].text)
                    fund_yoy.append(info[4].text)
                    fund_risk.append(info[5].div['data-gauge-current-step'])

                name_cell = rows[-1][k].find_all('a')
                fund_name.append(name_cell[0].text)
                # other info on the fund
                info = rows[-1][k].find_all('td')
                fund_last_close.append(info[2].text)
                fund_var.append(info[3].text)
                fund_yoy.append(info[4].text)
                fund_risk.append(info[5].div['data-gauge-current-step'])

    # create a dictionary with the relevant information to then transform it into a dataframe
    dict_info = {
        'fund_name': fund_name,
        'last_close': fund_last_close,
        'fund_var': fund_var,
        'fund_YoY': fund_yoy,
        'risk_level': fund_risk
    }
    data_df = pd.DataFrame(data=dict_info).dropna().drop_duplicates().reset_index(drop=True)
    data_df.fund_name = data_df.fund_name.str.capitalize()

    # transformation of the number column strings into the correct type (float or integer)

    for i in range(len(data_df)):
        data_df.last_close.iloc[i] = float(data_df.last_close.iloc[i].split( \
            '\n')[-1].replace(" ", ""))
        data_df.fund_var.iloc[i] = float(data_df.fund_var.iloc[i].split('\n')[-1] \
                                         .replace(" ", "").strip('%'))
        if data_df.fund_YoY.iloc[i].split('\n')[-1]. \
                replace(" ", "").strip('%') == '-':  # Last funds' YoY perf data are missing
            data_df.drop(data_df.tail(len(data_df) - i).index, inplace=True)
            break
        else:
            data_df.fund_YoY.iloc[i] = float(data_df.fund_YoY.iloc[i].split('\n')[-1]. \
                                             replace(" ", "").strip('%'))

    data_df = data_df.astype({'last_close': 'float',
                              'fund_var': 'float',
                              'fund_YoY': 'float',
                              'risk_level': 'int32'})
    return data_df

df_funds = data_collection(31)

df_funds.to_csv("data_WS.csv")
