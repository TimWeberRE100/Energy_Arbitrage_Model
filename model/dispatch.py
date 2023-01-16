'''
Dispatch engine for a price-taker storage system.

Functions
---------
dispatchModel
'''

def dispatchModel(dispatch_bidsOffers, dispatch_prices,storage_system_inst):
    '''
    Generate the dispatch instructions for the trading day.

    Parameters
    ----------
    dispatch_bidsOffers : list
        Outputs from the scheduling model.
    dispatch_prices : list
        List of dispatch prices for the trading day.
    storage_system_inst : storage_system
        Object containing storage system attributes and current state.

    Returns
    -------
    dispatchInstructions
        List containing dispatch instructions for the trading day.

    '''
    # Define parameters
    dispatch_offers = dispatch_bidsOffers[2]
    dispatch_bids = dispatch_bidsOffers[3]
    dispatch_offer_cap = dispatch_bidsOffers[0]
    dispatch_bid_cap = dispatch_bidsOffers[1]
    ws = dispatch_bidsOffers[4]
    Total_dispatch_cap = dispatch_bidsOffers[5]
    
    # Define output
    dispatchInstructions = []
    
    # Determine if bids/offers are won for each dispatch interval. Define dispatch instructions accordingly
    for s in range(0,48):
        for t in range(0,6):
            if ws[s] == 0:
                if dispatch_offers[s] <= dispatch_prices[t+6*s]:
                    dispatchInstructions.append(dispatch_offer_cap[s])
                else:
                    if storage_system_inst.type == "PHS":
                        dispatchInstructions.append(storage_system_inst.h_range*[0])
                    else:
                        dispatchInstructions.append(0)
                
            else:
                if dispatch_bids[s] >= dispatch_prices[t+6*s]:
                    dispatchInstructions.append(dispatch_bid_cap[s])
                else:
                    if storage_system_inst.type == "PHS":
                        dispatchInstructions.append(storage_system_inst.g_range*[0])
                    else:
                        dispatchInstructions.append(0)
                    
    return [dispatchInstructions]