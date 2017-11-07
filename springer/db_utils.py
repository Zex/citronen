#!/bin/use/env python3
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
    lgr.warn("[UNHANDLED] {}".format(text))
    return text # unhandled

cases = [
    "334-",
    "6651-52",
    "441-441.00e+3",
    "-",
    "32/37-99",
    "333"
    ]

print(np.vectorize(unify)(cases))