import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt


# %matplotlib notebook

exchanges = {
    'binance':
        {
            'bitcoin':'BTC',
            'ethereum':'ETH',
            'litecoin':'LTC',
            'solana':'SOL',
            'ripple':'XRP',
            'dollar':'USDT',
        },
    'coinbase':
        {
            'bitcoin':'BTC',
            'ethereum':'ETH',
            'litecoin':'LTC',
            'solana':'SOL',
            #'ripple':'xrp',
            'dollar':'USD',
        }, 
    'ftx':
        {
            'bitcoin':'BTC',
            'ethereum':'ETH',
            'litecoin':'LTC',
            'solana':'SOL',
            'ripple':'XRP',
            'dollar':'USD',
        }, 
    'bitmex':
        {
            'bitcoin':'XBT',
            'ethereum':'ETH',
            'litecoin':'LTC',
            'solana':'SOL',
            'ripple':'XRP',
            'dollar':'USD',
        },
    'kraken':
        {
            'bitcoin':'XBT',
            'ethereum':'ETH',
            'litecoin':'LTC',
            'solana':'SOL',
            'ripple':'XRP',
            'dollar':'USD',
        },
    }
days = 6
date = datetime.datetime(2022, 2, 28)
dates = [(date + datetime.timedelta(days=x)).strftime('%Y-%m-%d') for x in range(days)]


exch = 'binance'
asset = 'ethereum'


# concatenate all date over the 6 day period into single dataframe resampled by frequency
ob_fq = pd.DataFrame()
for date_str in dates:
    try:
        path = 'tardis_raw.nosync/'+exch+'/'+exch+'_book_snapshot_5_'+date_str+'_'+exchanges[exch][asset]+exchanges[exch]['dollar']+'.csv.gz'
        ob = pd.read_csv(path, compression='gzip')
    except Exception as e:
        print(e)
        path = 'tardis_raw.nosync/'+exch+'/'+exch+'_book_snapshot_5_'+date_str+'_'+exchanges[exch][asset]+'USDT'+'.csv.gz'
        ob = pd.read_csv(path, compression='gzip')
        pass
    ob.sort_values(by='timestamp')
    ob['datetime'] = pd.to_datetime(ob['timestamp'], unit='us')
    ob_fq=pd.concat([ob_fq, ob], ignore_index=True)
ob_fq['mid_price'] = (ob_fq['asks[0].price'] + ob_fq['bids[0].price'])/2


# array of volumes
order_book = np.zeros((ob_fq.shape[0], 14))
order_book[:,5] = ob_fq['bids[0].amount']
order_book[:,8] = ob_fq['asks[0].amount']
for k in range(1,5):
    order_book[:,5-k] = ob_fq['bids['+str(k)+'].amount'] + order_book[:,5-k+1]
    order_book[:,8+k] = ob_fq['asks['+str(k)+'].amount'] + order_book[:,8+k-1]

order_book[:,6] = 0
order_book[:,7] = 0
order_book[:,0] = order_book[:,1]
order_book[:,-1] = order_book[:,-2]

# array of prices
order_book_price = np.zeros((ob_fq.shape[0], 14))
order_book_price[:,5] = ob_fq['bids[0].price']
order_book_price[:,8] = ob_fq['asks[0].price']
for k in range(1,5):
    order_book_price[:,5-k] = ob_fq['bids['+str(k)+'].price']
    order_book_price[:,8+k] = ob_fq['asks['+str(k)+'].price']

order_book_price[:,6] = (order_book_price[:,8]+order_book_price[:,5])/2
order_book_price[:,7] = order_book_price[:,8]
order_book_price[:,0] = order_book_price[:,1] * 0.999995
order_book_price[:,-1] = order_book_price[:,-2] * 1.00001

    
from matplotlib import animation

fig, ax = plt.subplots()

low = min(order_book_price[:,0])
high = max(order_book_price[:,0-1])
color = 'black'
def animate(i):
    global color
    i+=2918400
    mid_1 = (order_book_price[i-1,6]+order_book_price[i-1,5])/2
    mid = (order_book_price[i,6]+order_book_price[i,5])/2
    if mid > mid_1:
        color = 'green'
    elif mid<mid_1:
        color='red'
    else:
        color='black'
    ax.cla() # clear the previous image
    ax.step(order_book_price[i,:], order_book[i,:], color=color) # plot the line
    ax.set_xlim([order_book_price[i,0]*(0.9999), order_book_price[i,-1]*1.0001]) # fix the x axis
    ax.set_ylim([0, max(order_book[i,:])+5]) # fix the y axis

anim = animation.FuncAnimation(fig, animate, frames = len(order_book_price) + 1, interval = 1, blit = False)
plt.show()
