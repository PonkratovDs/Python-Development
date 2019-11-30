import sys
import os
import collections

m_br, m_mes = sys.argv[1], sys.argv[2]
len_m_br, len_m_mes = len(m_br), len(m_mes)
numb_m_br, numb_m_mes = 0, 0
open_br, close_br = 0, 0
dict_m_str = {}
n_str = 1
pos_m_br, pos_m_mes = 0, 0
count_b_br = 0
fd = os.open("log.txt", os.O_RDONLY)

def new_str():
    global open_br, close_br, n_str, numb_m_mes, numb_m_br, pos_m_br, pos_m_mes
    open_br    = 0
    close_br   = 0
    n_str     += 1
    numb_m_mes = 0
    numb_m_br  = 0
    pos_m_br   = 0
    pos_m_mes  = 0

def bracket(b_s):
    global open_br, close_br
    if b_s == '[':
        open_br  = 1
    elif b_s == ']':
        close_br = 1

def match_bracket(slice_str, i):
    global m_br, pos_m_br, numb_m_br, dict_m_str, len_m_br
    new_pos = pos_m_br
    for m_s, m_r in zip(slice_str[i:], m_br[pos_m_br:len_m_br]):
        if m_s == m_r:
            numb_m_br += 1
            new_pos += 1
            if numb_m_br == len_m_br:
                dict_m_str[n_str] = 0
                return
        else:
            new_pos = pos_m_br
            break
    pos_m_br = new_pos

def match_message(slice_str, i):
    global m_mes, pos_m_mes, numb_m_mes, dict_m_str, len_m_mes
    if dict_m_str.get(n_str) == 0:
        n_pos = pos_m_mes
        for m_s, m_mg in zip(slice_str[i:], m_mes[pos_m_mes:len_m_mes]):
            if m_s == m_mg:
                numb_m_mes += 1
                n_pos += 1
                if numb_m_mes == len_m_mes:
                    dict_m_str[n_str] = 1
                    return
            else:
                n_pos = pos_m_mes
                break
        pos_m_mes = n_pos

while True:
    slice_str = os.read(fd, len_m_mes).decode('utf-8')
    if len(slice_str) == 0:
        break
    for i in range(len_m_mes):
        try:
            b_s = slice_str[i]
        except IndexError:
            break
        if b_s == '\n':
            new_str()
        bracket(b_s)
        if b_s == m_br[pos_m_br] and open_br == 1 and close_br == 0:
            match_bracket(slice_str, i)
        if b_s == m_mes[pos_m_mes]:
            match_message(slice_str, i)

os.close(fd)

fd = open("log.txt", 'r')
for num, line in enumerate(fd, 1):
    if dict_m_str.get(num):
        print(line.rstrip())
fd.close()
