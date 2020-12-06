import io
import os
import matplotlib.pyplot as plt
import seaborn
from functools import partial
from time import sleep
import pandas as pd
from dateutil.relativedelta import relativedelta
from pytrends.exceptions import ResponseError
from pytrends.request import TrendReq
from datetime import datetime, date
import requests

list_topics = {
    'Mal de gorge': '/m/0b76bty',
    'Agueusie': '/m/05sfr2',
    'Anosmie': '/m/0m7pl',
    'Symptôme': '/m/01b_06',
    'Réaction en chaîne par polymérase': '/m/05w_j'
}


def _fetch_data(pytrends, build_payload, timeframe: str) -> pd.DataFrame:
    """
    Attempts to fecth data and retries in case of a ResponseError.
    :param pytrends: object used for starting the TrendRequest on a localisation
    :param build_payload: object used for initializing a payload containing a particular word
    :param timeframe: string representing the timeframe
    :return a dataframe containing an interest over time for a particular topic
    """
    attempts, fetched = 0, False
    while not fetched:
        try:
            build_payload(timeframe=timeframe)
        except ResponseError as err:
            print(err)
            print(f'Trying again in {60 + 5 * attempts} seconds.')
            sleep(60 + 5 * attempts)
            attempts += 1
            if attempts > 3:
                print('Failed after 3 attemps, abort fetching.')
                break
        else:
            fetched = True
    return pytrends.interest_over_time()


def convert_dates_to_timeframe(start, stop) -> str:
    """
    Given two dates, returns a stringified version of the interval between
    the two dates which is used to retrieve data for a specific time frame
    from Google Trends.
    :param start: start date
    :param stop: stop date
    """
    return f"{start} {stop}"


def get_trends(term, start_date, stop_date):
    pytrends = TrendReq(hl='fr-BE')
    if type(term) is list:
        build_payload = partial(pytrends.build_payload, kw_list=term, cat=0, geo='BE', gprop='')
    else:
        build_payload = partial(pytrends.build_payload, kw_list=[term], cat=0, geo='BE', gprop='')

    data = _fetch_data(pytrends, build_payload, convert_dates_to_timeframe(start_date, stop_date))
    data = data.drop(columns=['isPartial'])
    return data


def load_term(termname, term, begin_date, end_date,  dir="../data/", geo="BE-WAL", rolling=True):
    """
    Loads the results of the trends request in a CSV file for each topic.
    :param termname: name of the topic we want to evaluate with Google trends
    :param term: mid of the topic we want to evaluate with Google trends
    :param dir: directory where we will load the CSV files
    :param geo: geo localisation of the trends request
    :param start_year: year of the start date
    :param start_mon: month of the start date
    :param stop_year: year of the stop date
    :param stop_mon: month of the stop date
    :return: a dataframe containing the evaluation of the trends request for a particular term
    """
    path = f"{dir}{geo}-{termname}.csv"

    print(f"DL {geo} {termname}")
    content = get_trends(term, begin_date, end_date)
    if content.empty:
        return content
    else:
        content.to_csv(path)

    content = pd.read_csv(path)
    content = content.rename(columns={term: termname})
    content = content.set_index("date")
    if rolling:
        content = content.rolling(7, center=True).mean()
    return content


def show_trends(df_trends):
    # Drawing of the interest of the keywords over time
    seaborn.set(color_codes=True)
    dx = df_trends.plot.line(figsize=(9, 6), title="Interest over time in Belgium")
    dx.set_xlabel('Date')
    dx.set_ylabel('Trends index')
    dx.tick_params(axis='both', which='major', labelsize=13)
    plt.show()


def show_full(begin_date, end_date, normalization=False):
    if not normalization:
        total_df = pd.DataFrame()
        for key, val in list_topics.items():
            total_df = pd.concat([total_df, load_term(key, val, begin_date, end_date, rolling=True)], axis=1)
        total_df.to_csv('../data/BE-WAL-Full_Dataframe_Rolling.csv')
    else:
        total_df = get_trends(list(list_topics.values()), begin_date, end_date)
        for termname, term in list_topics.items():
            total_df = total_df.rename(columns={term: termname})
        total_df = total_df.rolling(7, center=True).mean()
        total_df.to_csv('../data/BE-WAL-Full_Dataframe_Normalization.csv')

    return total_df


if __name__ == '__main__':
    end_date = datetime.today().strftime('%Y-%m-%d')
    begin_date = date.today() + relativedelta(months=-3)
    begin_date = begin_date.strftime('%Y-%m-%d')
    df = show_full(begin_date, end_date, normalization=True)
    show_trends(df)
