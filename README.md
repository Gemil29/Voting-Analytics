# Voting-Analytics
#В аналитике используются SQL запросы к базе данных MySQL, CRM системе 1C.
#Анализ рассчитывает результаты голосований среди ТОП-менеджеров и исследует самые рисковые сделки финансов



import pandas as pd
import numpy as np
import statistics
from datetime import datetime, date, timedelta
import plotly.graph_objs as go
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected = True)
from matplotlib import colors
import math
from dateutil.relativedelta import relativedelta
import re
import warnings
warnings.filterwarnings('ignore')
import requests
import json
from pandas.io.json import json_normalize
import base64
import os
from tqdm import tqdm_notebook
from plotly import tools
import calendar
from IPython.display import display
import sys
#sys.path.append('Desktop/Данные/Для_КПЗ/Статичные_данные/')
from matplotlib import pyplot as plt
from RisksSQLConnector import RisksSQLConnector
import colorlover as cl
from IPython.display import HTML
import ast
from datetime import datetime
from dateutil.relativedelta import relativedelta
import datetime
import time
from datetime import timedelta
reds = cl.scales['9']['seq']['Reds'][::-1]
reds10 = cl.interp( reds, 13)
reds20 = cl.interp( reds, 25)
connect = RisksSQLConnector()
cur = connect.get_cursor()

# Настройки отображения 
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows', None)
#pd.set_option('display.float_format', '{:20,.4f}'.format)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 2000)

# Выгружаем реестр СЭБ
db_conf = {
    "host": "10.212",
    "database": "ess_scoring",
    "user": "",
    "password": "",
    "port":   
}
db_conn = psycopg2.connect(**db_conf)
cur = db_conn.cursor()
sql = f"""
SELECT 
    uid, ess_date, contractor_uid, status, calculated_signs,  
    calculated_signs->'selection_clients_for_modeling'->'customer_category' as customer_category, 
    scoring, result, result_only_for_risk_carrier, request 
FROM 
    worker_scoring 
WHERE 
    (ess_date BETWEEN '2022-12-20 00:00:00' AND '2023-01-16 23:59:59') AND 
    status IN ('ERROR', 'COMPLETED') 
ORDER BY 
    ess_date
"""

tasks_df = pd.read_sql_query(sql, cur.connection)
tasks_df

def get_contractor_inn_by_uid(uid):
    sql = f"""
    SELECT 
        response 
    FROM 
        contractors_history 
    WHERE 
        request LIKE '%{uid}%' 
    ORDER BY 
        created_at DESC
    LIMIT 
        1
    """
    cur.execute(sql)
    data = cur.fetchone()
    max_number_of_payments = 0
    if data:
        history = ast.literal_eval(data[0])
        return history.get('inn', '-')
    return '-'
    
    def get_kf_data_by_inn(inn):
    sql = f"""
    SELECT 
        response 
    FROM 
        kontur_fokus_history 
    WHERE 
        request = '{{"inn": "{inn}", "method": "analytics"}}'
    ORDER BY 
        created_at DESC
    LIMIT 
        1
    """
    cur.execute(sql)
    data = cur.fetchone()
    result = ast.literal_eval(data[0])
    return result[0]
    
    
max_count = tasks_df.shape[0]
pbar = IntProgress(min=0, max=max_count) # instantiate the bar
display(pbar) # display the bar

tasks_df['crm_guid'] = ''
tasks_df['contractor_ИНН'] = ''
tasks_df['Иски ФССП'] = ''
tasks_df['Арбитражи за последние 3 года'] = ''
tasks_df['Арбитражи за последний год'] = ''
tasks_df['Выручка'] = ''
# tasks_df['Проигранные дела за последний год'] = ''
for i, row in tasks_df.iterrows():
    pbar.value += 1
    request = json.loads(row['request'])
    tasks_df.at[i, 'crm_guid'] = request['project']['crm_guid']
    
    try:
        inn = get_contractor_inn_by_uid(row['contractor_uid'])
    except Exception as ex:
        inn = '-'
    tasks_df.at[i, 'contractor_ИНН'] = inn

    fssp = ab3 = ab1 = turnover = 0
    
    if inn != '-':
        try:
            kf_data = get_kf_data_by_inn(inn)
            fssp = kf_data['analytics'].get('s1002', 0)
            ab3 = kf_data['analytics'].get('s2002', 0)
            ab1 = kf_data['analytics'].get('s2001', 0)
            ab4 = kf_data['analytics'].get('s2016', 0)
            turnover = kf_data['analytics'].get('s6004', 0)
        except Exception as ex:
            fssp = ab3 = ab1 = ab4 = turnover =  0
    tasks_df.at[i, 'Иски ФССП'] = fssp
    tasks_df.at[i, 'Арбитражи за последние 3 года'] = ab3
    tasks_df.at[i, 'Арбитражи за последний год'] = ab1
    tasks_df.at[i, 'Проигранные дела за последний год'] = ab4
    tasks_df.at[i, 'Выручка'] = turnover
        
    
    
tasks_df = tasks_df.drop('request', axis=1)
tasks_df['ess_date'] = tasks_df['ess_date'].dt.tz_localize(None)
print(tasks_df.shape)
tasks_df.head()

# data = pd.read_excel('C:\\Users\\eakalimullin\\Реестр_Задач_СЭБ_10_20__30_10_2022_25.xlsx')
# data = pd.read_csv('C:\\Users\\eakalimullin\\Реестр задач СЭБ\\Реестр_Задач_СЭБ_10_20__30_10_2022_25.csv', 
#                   dtype = {'contractor_ИНН' : object})

data = tasks_df.copy()
print(data.shape)
data['ess_date_format'] = data['ess_date'].apply(lambda x: pd.to_datetime( str(x)[:10] , format = '%Y-%m-%d' ) )
data = data[data['status'] != 'ERROR']
print(data.shape)
data.head(1)
data.info()

# Текущая доля автоматов
print( " Тек доля автоматов:",
round(data.result.sum() / data.uid.count() ,4)*100 , '%' )  

print(data.uid.nunique())
print(data.contractor_uid.nunique())
print(data['ess_date_format'].min())
print(data['ess_date_format'].max())
print(data.ess_date.min())
print(data.ess_date.max())

# Вытаскиваем из CRM проекты, прошедшие по заявкам
scoring_crm = {"Текст":"""

ВЫБРАТЬ
 лкЗаключениеСЭБ.Ссылка КАК ЗаключениеСЭБ,
 лкЗаключениеСЭБ.Основание КАК Проект,
 блПроектДоговораЛизинга.Ссылка КАК ДоговорЛизинга,
 лкЗаключениеСЭБ.ДатаРешения КАК ДатаРешения,
 лкЗаключениеСЭБ.Ess_scoring_uid КАК Ess_scoring_uid,
 лкЗаключениеСЭБ.РешениеПоАвтоматическойПроверке КАК РешениеПоАвтоматическойПроверке,
 лкЗаключениеСЭБ.АвтоматическоеЗаключениеПоНосителюРиска КАК АвтоматическоеЗаключениеПоНосителюРиска,
 лкЗаключениеСЭБ.Основание.НомерПроекта КАК НомерПроекта,
 блПроектДоговораЛизинга.НомерДоговора КАК НомерДоговора
ИЗ
 Документ.лкЗаключениеСЭБ КАК лкЗаключениеСЭБ
  ЛЕВОЕ СОЕДИНЕНИЕ Документ.блПроектДоговораЛизинга КАК блПроектДоговораЛизинга
  ПО (лкЗаключениеСЭБ.Основание = блПроектДоговораЛизинга.Проект)
   И (НЕ блПроектДоговораЛизинга.ПометкаУдаления)
ГДЕ 
 лкЗаключениеСЭБ.ДатаРешения МЕЖДУ ДАТАВРЕМЯ(2022, 12, 20, 0, 0, 0) И ДАТАВРЕМЯ(2023, 01, 08, 23, 59, 59)
 И НЕ лкЗаключениеСЭБ.Ess_scoring_uid = ""
 И НЕ лкЗаключениеСЭБ.ПометкаУдаления

УПОРЯДОЧИТЬ ПО
 ДатаРешения
 
"""

}

variable = json.dumps(scoring_crm)
response = requests.post("http://COM1C:xCom78AL!",
data = variable, headers = {'Handler':u'Выполнение запроса HTTP сервиса'.encode('windows-1251'),
'Content-Type': 'application/json; charset=windows-1251'})
scoring_crm = json.loads(response.text)
scoring_crm = pd.DataFrame(scoring_crm)
scoring_crm.shape


print("Scorring CRM shape: ",scoring_crm.shape)
print("Count nunique UID in CRM: ",scoring_crm.Ess_scoring_uid.nunique())
cur_scoring_crm = scoring_crm[scoring_crm['Ess_scoring_uid'].isin(data.uid.unique() )  ]
print("Shape scoring_crm data UID: ", cur_scoring_crm.shape)
print("Count nunique UID in data: " ,data.uid.nunique())
print("Count nunique UID in CRM:",cur_scoring_crm.Ess_scoring_uid.nunique())

# Извлекаем uid + стоп факторы в словарь
def stop_fuctors(data):
#     data = pd.read_excel('C:\\Users\\eakalimullin\Реестр_Задач_СЭБ_10_20__30_10_2022_25.xlsx')
    data =  tasks_df.copy()
    data = data[data['calculated_signs'].notna()]
    data['calculated_signs'] = data['calculated_signs'].apply(lambda x: ast.literal_eval(str(x)) )
    stop_lts = []

    for i, row in data.iterrows():
        stop_factors_ = data['calculated_signs'][i]['stop_factors']
        stop_lts.append({
          'uid': row['uid'],
          'stop_factors': stop_factors_
                      })
    stop_lts
    st = pd.DataFrame(stop_lts)
    st['stop_factors'] = st['stop_factors'].apply(lambda x: ast.literal_eval(str(x)))
    lst_uid = []
    list_stop_f = []
    for i in range(len(stop_lts)):
        list_stop_f.append(stop_lts[i]['stop_factors'])
        lst_uid.append(stop_lts[i]['uid'])

    stop_f = json_normalize(list_stop_f)
    stop_f.head(1)

    lst_uid_data = pd.DataFrame(lst_uid, columns = ['uid'])
    data_stop_factors = pd.concat([lst_uid_data, stop_f], axis = 1)
    data_stop_factors = data_stop_factors.replace(True, 1)
    data_stop_factors = data_stop_factors.replace(False, 0).fillna(0)
    data_stop_factors = data_stop_factors.iloc[:,:-1]
    data_stop_factors = data_stop_factors[data_stop_factors['negative_in_related_projects'] == 1]
    print(data_stop_factors.shape)
    data_stop_factors['count_stop_factors'] = data_stop_factors.iloc[:, 1:].sum(axis = 1)
#     data_stop_factors = data_stop_factors[['uid', 'count_stop_factors']]
    
    data = data.merge(data_stop_factors, on = 'uid', how ='left')
    fin_data = data[['uid', 'ess_date', 'status', 'customer_category','scoring','result', 'contractor_ИНН', 'Иски ФССП',
     'Арбитражи за последний год','Проигранные дела за последний год','count_stop_factors','negative_in_related_projects']]
    fin_data = fin_data[fin_data['uid'].isin(data_stop_factors['uid'].unique())]
    return fin_data# fin_data  # data_stop_factors

df_data = stop_fuctors(data)
print(df_data.shape)
print(df_data.uid.nunique())

df_data = df_data[df_data['count_stop_factors'] == 1 ]
print(df_data.shape)
df_data.head(1)

df_data['len_INN'] = df_data['contractor_ИНН'].apply(lambda x: len(str(x)))
df_data['len_INN'].value_counts().head()                                                     

pr1 = {"Текст":"""
ВЫБРАТЬ
 Контрагенты.Представление КАК Контрагент,
 Контрагенты.ИНН КАК ИНН,
 Контрагенты.КПП КАК КПП,
 КонтрагентыКонтактнаяИнформация.Представление КАК КонтактныйТелефон,
 Контрагенты.ОсновноеКонтактноеЛицо КАК ОсновноеКонтактноеЛицо,
 КонтактныеЛицаКонтактнаяИнформацияМоб.Представление КАК МобильныйНомерТелефонаКонтактногоЛица,
 КонтактныеЛицаКонтактнаяИнформацияРаб.Представление КАК РабочийНомерТелефонаКонтактногоЛица
ИЗ
 Справочник.Контрагенты КАК Контрагенты
  ЛЕВОЕ СОЕДИНЕНИЕ Справочник.Контрагенты.КонтактнаяИнформация КАК КонтрагентыКонтактнаяИнформация
  ПО Контрагенты.Ссылка = КонтрагентыКонтактнаяИнформация.Ссылка
   И (КонтрагентыКонтактнаяИнформация.Вид = ЗНАЧЕНИЕ(Справочник.ВидыКонтактнойИнформации.ТелефонКонтрагента))
  ЛЕВОЕ СОЕДИНЕНИЕ Справочник.КонтактныеЛица.КонтактнаяИнформация КАК КонтактныеЛицаКонтактнаяИнформацияМоб
  ПО Контрагенты.ОсновноеКонтактноеЛицо = КонтактныеЛицаКонтактнаяИнформацияМоб.Ссылка
   И (КонтактныеЛицаКонтактнаяИнформацияМоб.Вид = ЗНАЧЕНИЕ(Справочник.ВидыКонтактнойИнформации.ТелефонМобильныйКонтактныеЛица))
  ЛЕВОЕ СОЕДИНЕНИЕ Справочник.КонтактныеЛица.КонтактнаяИнформация КАК КонтактныеЛицаКонтактнаяИнформацияРаб
  ПО Контрагенты.ОсновноеКонтактноеЛицо = КонтактныеЛицаКонтактнаяИнформацияРаб.Ссылка
   И (КонтактныеЛицаКонтактнаяИнформацияРаб.Вид = ЗНАЧЕНИЕ(Справочник.ВидыКонтактнойИнформации.ТелефонРабочийКонтактныеЛица))
ГДЕ
 НЕ Контрагенты.ПометкаУдаления
 
 """

}

variable = json.dumps(pr1)
response = requests.post("http://C*******",
data = variable, headers = {'Handler':u'Выполнение запроса HTTP сервиса'.encode('windows-1251'),
'Content-Type': 'application/json; charset=windows-1251'})
pr1 = json.loads(response.text)
pr1 = pd.DataFrame(pr1)

# КонтактныйТелефон #f['КонтактныйТелефон'] != ''
TelNumber = df.groupby(['КонтактныйТелефон'], as_index = False).agg({'ИНН': lambda x: len(set(x))}).sort_values(ascending = False, by = 'ИНН') .head(100)
TelNumber['share'] = TelNumber['ИНН']/TelNumber['ИНН'].sum()
TelNumber['cumsum'] = round(TelNumber['share'].cumsum()*100,4)
MostPopularNumber = TelNumber[TelNumber['cumsum'] < 90 ]
print(MostPopularNumber.shape)
MostPopularNumber.tail()

# МобильныйНомерТелефонаКонтактногоЛица [df['МобильныйНомерТелефонаКонтактногоЛица'] != '']
TelNumber1 = df.groupby(['МобильныйНомерТелефонаКонтактногоЛица'], as_index = False).agg({'ИНН': lambda x: len(set(x))}).sort_values(ascending = False, by = 'ИНН') .head(100)
TelNumber1['share'] = TelNumber1['ИНН']/TelNumber1['ИНН'].sum()
TelNumber1['cumsum'] = round(TelNumber1['share'].cumsum()*100,4)
MostPopularNumber1 = TelNumber1[TelNumber1['cumsum'] < 90 ]
print(MostPopularNumber1.shape)
MostPopularNumber1.head(1)

# РабочийНомерТелефонаКонтактногоЛица [df['РабочийНомерТелефонаКонтактногоЛица'] != ''].
TelNumber2 = df.groupby(['РабочийНомерТелефонаКонтактногоЛица'], as_index = False).agg({'ИНН': lambda x: len(set(x))}).sort_values(ascending = False, by = 'ИНН') .head(100)
TelNumber2['share'] = TelNumber2['ИНН']/TelNumber2['ИНН'].sum()
TelNumber2['cumsum'] = round(TelNumber2['share'].cumsum()*100,4)
MostPopularNumber2 = TelNumber2[TelNumber2['cumsum'] < 50 ]
print(MostPopularNumber2.shape)
MostPopularNumber2

t1 = MostPopularNumber.copy()
t2 = MostPopularNumber1.copy()
t3 = MostPopularNumber2.copy()
t1 = t1.rename(columns = {'КонтактныйТелефон': 'number'})
t2 = t2.rename(columns = {'МобильныйНомерТелефонаКонтактногоЛица': 'number'})
t3 = t3.rename(columns = {'РабочийНомерТелефонаКонтактногоЛица': 'number'})
t = pd.concat([t1,t2,t3], axis = 0)
# t.groupby(['number'], as_index = False).agg({'ИНН': sum}).sort_values(ascending = False, by = 'ИНН')\
#     .to_excel('C:\\Users\\eakalimullin\\popularNumber.xlsx')

# Искомые самые популярные qwerty телефоны
qwerty = list(MostPopularNumber['КонтактныйТелефон']) + list(MostPopularNumber1['МобильныйНомерТелефонаКонтактногоЛица']) + \
list( MostPopularNumber2['РабочийНомерТелефонаКонтактногоЛица'] )
qwerty = list(set(qwerty))
print(len(qwerty))
qwerty_df = pd.DataFrame(qwerty, columns= ['qwertyNumber'])

company_in_df_data = df[df['ИНН'].isin( df_data['contractor_ИНН'].unique() )  ]
print(company_in_df_data.shape)
company_in_df_data.head(2)

# Собираем инфо по номерам тел Контактов
p1 = company_in_df_data[['ИНН','КонтактныйТелефон']].rename(columns = {'КонтактныйТелефон': 'phoneNumber'})
p2 = company_in_df_data[['ИНН','МобильныйНомерТелефонаКонтактногоЛица']].rename(columns = {'МобильныйНомерТелефонаКонтактногоЛица': 'phoneNumber'})
p3 = company_in_df_data[['ИНН','РабочийНомерТелефонаКонтактногоЛица']].rename(columns = {'РабочийНомерТелефонаКонтактногоЛица': 'phoneNumber'})
phone_data = pd.concat([p1,p2,p3], axis = 0)
print(phone_data.shape)
# phone_data = phone_data[phone_data['phoneNumber'].notna() ]
print(phone_data.shape)
print(phone_data.nunique())

dp = phone_data.copy()
dp['uni']  = dp['ИНН'].astype(str) + dp['phoneNumber'].astype(str)
print(dp.shape)
dp = dp.drop_duplicates('uni')
print(dp.shape)
dp.head()

print(len(dp[dp['phoneNumber'].isna() ].ИНН.unique()))
dp[dp['phoneNumber'].isna() ].ИНН.unique()
### QWERTY телефоны в реестре задач СЭБ¶

print(len(dp[dp['phoneNumber'].isna()]['ИНН'].unique()))
dp[dp['phoneNumber'].isna()]['ИНН'].unique()

t1 = df[df.ИНН.isin( dp[dp['phoneNumber'].isna()]['ИНН'].unique() )  ]
t1[t1['РабочийНомерТелефонаКонтактногоЛица'].isna()]

print(df_data.shape)
df_data1 = df_data.drop_duplicates('contractor_ИНН')

print(df_data1.shape)
df_data1.to_excel('C:\\Users\\eakalimullin\\Стоп-факторы.xlsx')
