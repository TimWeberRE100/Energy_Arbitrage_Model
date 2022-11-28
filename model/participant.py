from pandas import read_csv

class participant:
    def __init__(self, assumptions):
        self.risk_level = int(assumptions["risk_level"]) # Risk level when choosing price bands for bids and offers. A risk level of 0 chooses first price band above/below forecase spot price [0,10]

        offer_df = read_csv("assumptions/" + assumptions["offers_csv"]) # Data frame of offer price bands over the simulation period 
        bid_df = read_csv("assumptions/" + assumptions["bids_csv"]) # Data frame of bid price bands over the simulation period

        self.offers = offer_df.values.tolist()
        self.bids = bid_df.values.tolist()