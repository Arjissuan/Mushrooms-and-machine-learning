
import pandas
import numpy
import os
try:
    dt = pandas.read_csv("files/SecondaryData1987DataMatching/1987_data_encoded_matched.csv")
    print(dt)
except pandas.errors.ParserError:
    print("Nie dziala")


