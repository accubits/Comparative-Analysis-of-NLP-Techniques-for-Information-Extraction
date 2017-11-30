# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:41:59 2017

@author: Vishnu
"""

import re

# =============================================================================
# module to clean pdf text in a page
# =============================================================================

def clean(raw_data):
    raw_data = raw_data.replace('€', '')
    raw_data = raw_data.replace('Œ', '')
    raw_data = re.sub(r'https?:\/\/.*[\r\n]*', '', raw_data, flags=re.MULTILINE)
    raw_data = re.sub(r'\<a href', ' ', raw_data)
    raw_data = re.sub(r'&amp;', '', raw_data) 
    raw_data = re.sub(r'[_"\-;%()|+&=*%!:#$@\[\]/]', ' ', raw_data)
    raw_data = re.sub(r'<br />', ' ', raw_data)
    raw_data = re.sub(r'\'', ' ', raw_data)
    raw_data = raw_data.replace("®", '')
    page_content = raw_data.replace(u"\u2122", '')
    return page_content