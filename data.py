# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:44:16 2017

@author: Vishnu
"""

from Cleaning import clean
import PyPDF2

# =============================================================================
# module to read pdf file and convert it to text page by page
# =============================================================================

def Data(path):
    full_data = []
    for i in path:
        pdf_file = open(i, 'rb')
        read_pdf = PyPDF2.PdfFileReader(pdf_file)
        number_of_pages = read_pdf.getNumPages()
        data = []
        for i in range(number_of_pages):
            page = read_pdf.getPage(i)
            page_content = page.extractText()
            page_content = clean(page_content)
            data.append(page_content)
        full_data.append(data)
    main_data = [j for i in full_data for j in i if j != '' and len(j.split()) > 10]
    return main_data