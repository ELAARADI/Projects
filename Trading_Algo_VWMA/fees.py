import pandas as pd
import numpy as np
import itertools
import math

fees = pd.read_csv('fees.csv')

fees = fees.drop(['bitmex', 'ftx'], axis=1)

def get_string(df, row, col):
    str = df.iloc[row,col]
    str = str.strip('][').split(', ')
    return str

def get_fee(notional, table):
    for i in range(len(table)-1):
        try:
            if notional<=table[i+1][0] or i+1==len(table)-1:
                taker=table[i+1][1]
                maker=table[i+1][2]
                break
        except:
            taker=table[i+1][1]
            maker=table[i+1][2]
            break
    return taker, maker

notionals = []
table = []
for i,col in enumerate(fees.columns[1:]):
    exchange=[col]
    str_taker = get_string(fees,0,i+1)
    str_maker = get_string(fees,1,i+1)
    str_notional = get_string(fees,2,i+1)
    notionals.append(str_notional)
    for j in range(len(str_notional)):
        row = [float(str_notional[j]),float(str_taker[j]),float(str_maker[j])]
        exchange.append(row)
    table.append(exchange)

notionals = np.sort(np.unique(np.array(list(itertools.chain(*notionals))).astype(int)))


best_fees = []
for notional in notionals:
    min_taker, min_taker_ex, min_maker, min_maker_ex= 999, 'None', 999, 'None'
    for i in range(len(table)):
        taker,maker = get_fee(notional, table[i])
        if taker < min_taker:
            min_taker = taker
            min_taker_ex = table[i][0]
        if maker < min_maker:
            min_maker = maker
            min_maker_ex = table[i][0]
    best_fees.append([notional,min_taker, min_taker_ex, min_maker, min_maker_ex])

fees_table = pd.DataFrame(best_fees, columns=['Notional', 'Taker Fees', 'Taker Exch', 'Maker Fees', 'Maker Exch'])
fees_table = fees_table.set_index('Notional')



# fees_table.to_csv('best_fees_table.csv')

print(fees_table.head())