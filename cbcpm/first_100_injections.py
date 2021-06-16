from __future__ import division, print_function

import os
import sys
import pandas as pd
import numpy as np
import scipy
import deepdish
from itertools import islice

import tarfile


# tc = np.zeros(10)
# print(tc)
# nw = [8, 6, 2]
# count = 0
# for i in np.arange(10):
#     if i in nw:
#         continue
#
#     tc[count] = i
#     x = tc
#     print('x',x)
#     count += 1
#
# print('tc',tc)
#
# cnt = 0
# for i in np.arange(10):
#     if i in nw:
#         continue
#
#     x = 5 + tc[cnt]
#     print('x1',x)
#     cnt += 1



# tar = tarfile.open('data.tar.gz', mode = 'r')
# tar.extractall()
# print(tar)
# files = tar.getmembers()
# data = tar.extractall()
# data.readlines()
# print(data)
#
# injection_parameters =
# print(injection_parameters)


# n_inj = list(range(5,10))
# print(n_inj)
data00 =  pd.read_hdf('Injection_file/injections_10e6.hdf5')
data1 = deepdish.io.load('injections_10e6.hdf5')['injections']
# print('data1',data1)
# print(data1['redshift'])

data_short = data1.sort_values('redshift', ascending=True)
# print('data_short', data_short)
data_short_redshift = data_short['redshift']
print(data_short_redshift)

## Check for redshift. Selecting pandas DataFrame Rows Based On Conditions
redshift = data_short['redshift'] > 4
data2 = data_short[redshift]
# print('data2', data2)
# print('data2', data2['redshift'][:100])

## other way
cond1 = data_short['redshift'] > 4
cond2 = data_short['redshift'] < 10
data3 = data_short['redshift'][(cond1 & cond2)]
# print('data3', data3)
# print(data3['redshift'])


## Another way, Select DataFrame Rows Based on multiple conditions on columns
data_redshift = data_short[(data_short['redshift'] > 3) & (data_short['redshift'] < 6)]
# print('data_redshift', data_redshift)
# print(data_redshift['redshift'][:100])

## Selecting random 100 BBh signals
# data_redshift = data_redshift.sample(n=100)
# print('random 100 BBH signals', data_redshift)
# print('random 100 BBH signals redshift ', data_redshift['redshift'])

# data1 = data_short[10]
# for i in len(data1):
#     print('i',i)
#     x = data1[-i]
#     print('x',x)

# array = np.random.randint(3,9, 100) ## int array
# array = np.random.uniform(3,9,100)  ## float array
# print(array)
# array_choice = np.random.choice(array)
# print(array_choice)

# process = psutil.Process(os.getpid())
# process1 = process.memory_info().rss
# process2 = process1 / 10**6  ## In MB. divide by 10**9 for GB.
# print('memory',process2 )
# print('memory percent', process.memory_percent())
#
# t_c = np.zeros(100)
# cnt =0
# for i in range(100):
#     array = np.random.uniform(3, 9, 100)  ## float array
#     # print('array',array)
#     array_choice = np.random.choice(array)
#     print('array_choice',array_choice)
#
#     t_c[cnt] = array_choice
#     cnt +=1
#
# print('t_c',t_c)
# print('tc_10',t_c)
# print('tc ',t_c)