from pandas import read_csv

class participant:
    def __init__(self, assumptions, offers_csv, bids_csv):
        self.risk_level = # Risk level when choosing price bands for bids and offers. A risk level of 0 chooses first price band above/below forecase spot price [0,10]

        self.offers = read_csv(offers_csv) # Data frame of offer price bands over the simulation period 
        self.bids = read_csv(bids_csv) # Data frame of bid price bands over the simulation period