#!/bin/usr/env python3
#Author: Zex Li <top_zlynch@yahoo.com>
import re
import logging
import numpy as np
import time

def unify(text):
    text = text.strip()

    if not text:
        return text
    if re.match(r"(\d+)-(\d*)$", text): # perfect
        p1, p2 = text.split('-')
        if p1 and p2 and int(p2) < int(p1):
            text = "{}-{}{}".format(p1, p1[:len(p2)], p2)
        return text
    if re.match(r"(\d+)$", text): # fine
        return text
    if re.match(r"(\d+)-(\d*).(\d*)e+(\d*)", text): # ends with float
        return "{}{}".format(text[:text.index('-')+1], str(int(float(text.split('-')[1]))))
    if re.match(r"(\d+)/(\d+)-(\d+)", text): # discontinuous range
        return "{}-{}".format(text.split('/')[0], text.split('-')[1])
    return text # unhandled

def reuse_info(text):
    text = text.strip()

    if not text:
        return text
    
    if re.match("[^@]+@[^@]+\.[^@]+", text):
        # do someting
        return ''
    if re.match(r"(\+){0,1}\ *(\d)+", text):
        # do someting
        return ''

    return text

if __name__ == '__main__':
    cases = [
    "334-",
    "6651-52",
    "441-441.00e+3",
    "-",
    "32/37-99",
    "333"
    ]
    print(np.vectorize(unify)(cases))
    
    cases = [
        "24kffs@mdkd.com",
        "",
        "kf-fk.fs.dd@mdkd.net.cc.dd",
        "dd",
        "234",
        "2424",
        "98",
        "\"kffk.fs.dd\"@mdkd.net.cc.dd",
        "+ 334 1414",
        "hhh",
        "____@____.a",
        "889@[21.22.11.66]",
        "+++ 3 114",
        '"j\"s\""@cba.com',
        ]
    print(np.vectorize(reuse_info)(cases))
