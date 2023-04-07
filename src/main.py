from data_access.data_source import DataSource
import pandas as pd
import numpy as np
import os



if __name__ =='__main__':
    ds = DataSource()
    print(ds.get_secondary_data_frame())