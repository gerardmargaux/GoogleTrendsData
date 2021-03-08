import io
import math
import os
from copy import deepcopy

import pandas as pd
import numpy as np
import requests
import scipy
from dateutil.relativedelta import relativedelta
from pytrends.exceptions import ResponseError
from pytrends.request import TrendReq
from datetime import datetime, date
from time import sleep
import scipy
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from functools import partial

dict_population = {
    'austria': 8901064,
    'belgium': 11549888,
    'croatia': 4058165,
    'denmark': 5822763,
    'estonia': 1328976,
    'finland': 5525292,
    'france': 67098824,
    'ireland': 4963839,
    'italy': 60244639,
    'lithuania': 2794090,
    'luxembourg': 626108,
    'norway': 5367580,
    'spain': 47329981,
    'portugal': 10295909,
    'slovenia': 2095861,
    'hungary': 9769526,
    'poland': 37958138,
    'slovakia': 5457873,
    'bulgaria': 6951482}

dict_countries = {'austria': 'AT',
                 'belgium': 'BE',
                 'croatia': 'HR',
                 'czech republic': 'CZ',
                 'denmark': 'DK',
                 'estonia': 'EE',
                 'finland': 'FI',
                 'france': 'FR',
                 'germany': 'DE',
                 'greece': 'EL',
                 'ireland': 'IE',
                 'italy': 'IT',
                 'lithuania': 'LT',
                 'luxembourg': 'LU',
                 'netherlands': 'NL',
                 'norway': 'NO',
                 'romania' : 'RO',
                 'serbia': 'RS',
                 'spain': 'ES',
                 'sweden': 'SE',
                 'switzerland': 'CH',
                 'united kingdom': 'UK',
                 'portugal': 'PT',
                 'slovenia': 'SI',
                 'hungary': 'HU',
                 'poland': 'PL',
                 'bosnia and herzegovina': 'BA',
                 'slovakia': 'SK',
                 'bulgaria': 'BG'}

list_topics = {
        'Fièvre': '/m/0cjf0',
        'Toux': '/m/01b_21',
        'Coronavirus': '/m/01cpyy',
        'Symptôme': '/m/01b_06',
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


def get_trends(term, start_date, stop_date, geo='BE'):
    pytrends = TrendReq(hl='fr-BE')
    if type(term) is list:
        build_payload = partial(pytrends.build_payload, kw_list=term, cat=0, geo=geo, gprop='')
    else:
        build_payload = partial(pytrends.build_payload, kw_list=[term], cat=0, geo=geo, gprop='')

    data = _fetch_data(pytrends, build_payload, convert_dates_to_timeframe(start_date, stop_date))
    if 'isPartial' in data.columns:
        data = data.drop(columns=['isPartial'])
    return data


def load_term(termname, term, begin_date, end_date,  dir="../data/correlation/", country='belgium', rolling=True):
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
    path = f"{dir}{country}-{termname}.csv"
    #if not os.path.exists(path):
    print(f"DL {country} {termname}")
    content = get_trends(term, begin_date, end_date, geo=dict_countries[country])
    if content.empty:
        return content
    else:
        content.to_csv(path)

    content = pd.read_csv(path)
    content = content.rename(columns={term: termname})
    content = content.set_index("date")
    if rolling:
        content = content.rolling(7, center=True).mean()
        content = content.dropna(axis=0)
        content = content.apply(lambda x: 100*x/max(content[termname]))
    return content


def collect_data(begin_date, end_date):
    full_df = pd.DataFrame()
    for name, geo in dict_countries.items():
        if name in list(dict_population.keys()) and geo == 'BE':
            total_df = pd.DataFrame()
            for key, val in list_topics.items():
                total_df = pd.concat([total_df, load_term(key, val, begin_date, end_date, country=name, rolling=True)],
                                     axis=1)
            total_df['country'] = geo
            full_df = pd.concat([full_df, total_df], axis=0)
    full_df.to_csv('../data/correlation/Full_Dataframe.csv')
    full_df = full_df.reset_index().set_index('date')
    return full_df


def check_data():
    europe_df = pd.read_csv('./europe_data.csv')
    list_country = europe_df['country'].apply(lambda x: x.lower()).tolist()
    europe_df['country'] = list_country
    with_data = deepcopy(list(dict_countries.keys()))
    for country in dict_countries.keys():
        final_df = europe_df.loc[(europe_df['country'] == country) & (europe_df['indicator'] == 'Daily hospital occupancy')]
        if final_df.empty:
            with_data.remove(country)
    return with_data


def get_hospitalizations_total(start_date, stop_date, country):
    url_hospi = "https://raw.githubusercontent.com/pschaus/covidbe-opendata/master/static/csv/international_hospi.csv"
    s = requests.get(url_hospi).content
    europe_df = pd.read_csv(io.StringIO(s.decode('utf-8')))
    list_country = europe_df['country'].apply(lambda x: x.lower()).tolist()
    europe_df['country'] = list_country
    final_df = europe_df.loc[(europe_df['country'] == country) & (europe_df['indicator'] == 'Daily hospital occupancy')]
    final_df = final_df.loc[(final_df['date'] >= start_date) & (final_df['date'] <= stop_date)]
    final_df = final_df.set_index('date')
    final_df = final_df.drop(['country', 'indicator', 'year_week', 'source', 'url'], axis=1)
    list_hospi = final_df['value'].tolist()
    list_hospi = [100000 * x / dict_population[country] for x in list_hospi]
    total_hospi = pd.DataFrame()
    total_hospi['value'] = list_hospi
    total_hospi.index = final_df.index
    total_hospi = total_hospi.rolling(7, center=True).mean()
    total_hospi = total_hospi.dropna(axis=0)
    total_hospi = total_hospi.apply(lambda x: 100 * x / max(list_hospi))
    return total_hospi


def crosscorr(datax, datay, lag=0, wrap=False):
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))


def time_lagged_cross_correlation(start_date, stop_date, ref_country, plot_lag=True):
    reference_df = pd.DataFrame()
    for col in list_topics.keys():
        content = pd.read_csv(f"../data/correlation/{ref_country}-{col}.csv")
        content = content.loc[(content['date'] >= start_date) & (content['date'] <= end_date)]
        content = content.rename(columns={list_topics[col]: col})
        content = content.set_index("date")
        content = content.rolling(7, center=True).mean()
        content = content.dropna(axis=0)
        content = content.apply(lambda x: 100 * x / max(content[col]))
        reference_df = pd.concat([reference_df, content], axis=1)

    hospi_df = get_hospitalizations_total(start_date, stop_date, ref_country)
    reference_df = reference_df.loc[(reference_df.index >= max(min(reference_df.index), min(hospi_df.index))) & (
            reference_df.index <= min(max(reference_df.index), max(hospi_df.index)))]
    hospi_df = hospi_df.loc[(hospi_df.index >= max(min(reference_df.index), min(hospi_df.index))) & (
            hospi_df.index <= min(max(reference_df.index), max(hospi_df.index)))]

    if len(hospi_df) > len(reference_df):
        to_drop = hospi_df[(~hospi_df.index.isin(reference_df.index))]
        to_drop = to_drop.reset_index()['date'].tolist()
        hospi_df = hospi_df.drop(index=to_drop, axis=0)
    elif len(hospi_df) < len(reference_df):
        to_drop = reference_df[(~reference_df.index.isin(hospi_df.index))]
        to_drop = to_drop.reset_index()['date'].tolist()
        reference_df = reference_df.drop(index=to_drop, axis=0)

    dates = hospi_df.reset_index()['date'].tolist()
    dates = [datetime.strptime(date, '%Y-%m-%d').date() for date in dates]

    hospi_col = hospi_df['value']
    hospi_col = hospi_col.dropna(axis=0)

    col_names = reference_df.columns
    for col in col_names:
        ref_col = reference_df[col]
        correlation = [abs(crosscorr(ref_col, hospi_col, lag)) for lag in range(0, 26)]

        if plot_lag:
            plt.plot(range(0, 26), correlation, 'o', label=col)
            plt.xlabel("Days of lag")
            plt.ylabel("Correlation with the number of hospitalizations")
            plt.title(f"Correlation between trends and hospitalizations in {ref_country}")
        else:
            plt.plot_date(dates, ref_col, '-', label=col)
            plt.ylabel("Index of trends")
            plt.xlabel("Dates")
            plt.title(f"Correlation between trends and hospitalizations in {ref_country}")
            plt.legend()

    if not plot_lag:
        plt.plot_date(dates, hospi_col, '-', label='Hospitalizations')
        plt.legend()
        plt.show()
        #plt.savefig(f"../data/plot_correlation/correlation_{ref_country}.png")
    else:
        plt.legend()
        plt.show()
        #plt.savefig(f"../data/plot_correlation/correlation_lag_{ref_country}.png")

    return


def total_europe_hospi(start_date, end_date):
    normalized_hospi = pd.DataFrame()
    for country in dict_population.keys():
        total_hospi = get_hospitalizations_total(start_date, end_date, country)
        normalized_hospi = pd.concat([normalized_hospi, total_hospi], axis=1)
    normalized_hospi['hospi'] = normalized_hospi.mean(axis=1, skipna=True).tolist()
    return normalized_hospi


def total_europe_trends(start_date, end_date):
    normalized_trends = pd.DataFrame()
    topics = []
    index = None
    for col in list_topics.keys():
        current_df = pd.DataFrame()
        for country in dict_population.keys():
            content = pd.read_csv(f"../data/correlation/{country}-{col}.csv")
            content = content.loc[(content['date'] >= start_date) & (content['date'] <= end_date)]
            content = content.rename(columns={list_topics[col]: col})
            content = content.set_index("date")
            list_trends = content[col].tolist()
            list_trends = [100000*x/dict_population[country] for x in list_trends]
            content[col] = list_trends
            content = content.rolling(7, center=True).mean()
            content = content.dropna(axis=0)
            content = content.apply(lambda x: 100 * x / max(content[col]))
            current_df = pd.concat([current_df, content], axis=1)
            index = content.index
        current_df = current_df.mean(axis=1, skipna=True)
        normalized_trends = pd.concat([normalized_trends, current_df], axis=1)
        topics.append(col)
    normalized_trends.index = index
    normalized_trends.columns = topics
    return normalized_trends


def plot_correlation_europe(start_date, stop_date, plot_lag=True):
    reference_df = total_europe_trends(start_date, stop_date)
    hospi_df = total_europe_hospi(start_date, stop_date)

    reference_df = reference_df.loc[(reference_df.index >= max(min(reference_df.index), min(hospi_df.index))) & (
                reference_df.index <= min(max(reference_df.index), max(hospi_df.index)))]
    hospi_df = hospi_df.loc[(hospi_df.index >= max(min(reference_df.index), min(hospi_df.index))) & (
                hospi_df.index <= min(max(reference_df.index), max(hospi_df.index)))]

    if len(hospi_df) > len(reference_df):
        to_drop = hospi_df[(~hospi_df.index.isin(reference_df.index))]
        to_drop = to_drop.reset_index()['date'].tolist()
        hospi_df = hospi_df.drop(index=to_drop, axis=0)
    elif len(hospi_df) < len(reference_df):
        to_drop = reference_df[(~reference_df.index.isin(hospi_df.index))]
        to_drop = to_drop.reset_index()['date'].tolist()
        reference_df = reference_df.drop(index=to_drop, axis=0)

    dates = hospi_df.reset_index()['index'].tolist()
    dates = [datetime.strptime(date, '%Y-%m-%d').date() for date in dates]
    hospi_col = hospi_df['hospi']
    col_names = reference_df.columns
    for col in col_names:
        ref_col = reference_df[col]
        correlation = [abs(crosscorr(ref_col, hospi_col, lag)) for lag in range(0, 26)]

        if plot_lag:
            plt.plot(range(0, 26), correlation, 'o', label=col)
            plt.xlabel("Days of lag")
            plt.ylabel("Correlation with the number of hospitalizations")
            plt.title("Correlation between trends and hospitalizations in Europe")

        else:
            plt.plot_date(dates, ref_col, '-', label=col)
            plt.ylabel("Index of trends")
            plt.xlabel("Dates")
            plt.title(f"Correlation between trends and hospitalizations in Europe")
            plt.legend()

    if not plot_lag:
        plt.plot_date(dates, list(hospi_col), '-', label='Hospitalizations')
    plt.legend()
    plt.show()
    return


if __name__ == '__main__':
    end_date = datetime.today().strftime('%Y-%m-%d')
    end_week = datetime.strptime(end_date, '%Y-%m-%d').strftime("%V")
    begin_date = date.today() + relativedelta(months=-4)
    begin_date = begin_date.strftime('%Y-%m-%d')
    begin_week = datetime.strptime(begin_date, '%Y-%m-%d').strftime("%V")
    with_data = check_data()
    #full_df = collect_data(begin_date, end_date)
    time_lagged_cross_correlation(begin_date, end_date, 'belgium', plot_lag=False)
    #plot_correlation_europe(begin_date, end_date, plot_lag=False)
    #time_lagged_cross_correlation(begin_date, end_date, 'austria', plot_lag=True)

