def min_greaterThan(load_PB,SP_s,risk_level):
    '''
    Calculates the minimum load price band that is greater than the specified spot price, accounting
    for the level of risk hedging.

    Parameters
    ----------
    load_PB : list
        The ordered list of up to 10 load price bands from smallest to largest [$/MWh].
    SP_s : float
        The forecast spot price for a trading interval [$/MWh].
    risk_level : integer
        Level of risk hedging (0 to 9), with lower lisk levels constituting price bands closer to
        the spot price.

    Returns
    -------
    float
        The price band used for the load bid in the trading interval.

    '''
    for b in range(0,len(load_PB)):
        if (load_PB[b] > SP_s) and (b + risk_level <= 9):
            return load_PB[b+risk_level]
    return load_PB[-1]

def max_lessThan(gen_PB_reverse,SP_s,risk_level):
    '''
    Calculates the maximum generator price band that is lower than the specified spot price, accounting
    for the level of risk hedging.

    Parameters
    ----------
    gen_PB_reverse : list
        The ordered list of up to 10 generator price bands [$/MWh] from largest to smallest.
    SP_s : float
        The forecast spot price for a trading interval [$/MWh].
    risk_level : integer
        Level of risk hedging (0 to 9), with lower lisk levels constituting price bands closer to
        the spot price.

    Returns
    -------
    float
        The price band used for the generator offer in the trading interval.

    '''
    for o in range(0,len(gen_PB_reverse)):
        if (gen_PB_reverse[o] < SP_s) and (o + risk_level <= 9):
            return gen_PB_reverse[o+risk_level]
    return gen_PB_reverse[-1]

def dispatchModel(dispatch_bidsOffers, dispatch_prices,spot_prices,system_assumptions):
    '''
    Generate the dispatch instructions for the trading day.

    Parameters
    ----------
    dispatch_bidsOffers : list
        Outputs from the scheduling model.
    dispatch_prices : list
        List of dispatch prices for the trading day.
    spot_prices : list
        List of historical spot prices for the trading day.
    system_assumptions : dictionary
        Dictionary of assumed parameters for the system.

    Returns
    -------
    list
        List containing dispatch instructions for the trading day.

    '''
    # Define parameters
    dispatch_offers = dispatch_bidsOffers[2]
    dispatch_bids = dispatch_bidsOffers[3]
    dispatch_offer_cap = dispatch_bidsOffers[0]
    dispatch_bid_cap = dispatch_bidsOffers[1]
    ws = dispatch_bidsOffers[4]
    Total_dispatch_cap = dispatch_bidsOffers[5]
    g_pumps = int(system_assumptions["g_index_range"])
    h_turbines = int(system_assumptions["h_index_range"])
    system_type = system_assumptions["system_type"]
    
    # Define output
    dispatchInstructions = []
    
    # Determine if bids/offers are won for each dispatch interval. Define dispatch instructions accordingly
    for s in range(0,48):
        for t in range(0,6):
            if ws[s] == 0:
                if dispatch_offers[s] <= dispatch_prices[t+6*s]:
                    dispatchInstructions.append(dispatch_offer_cap[s])
                else:
                    if system_type == "PHS":
                        dispatchInstructions.append(h_turbines*[0])
                    else:
                        dispatchInstructions.append(0)
                
            else:
                if dispatch_bids[s] >= dispatch_prices[t+6*s]:
                    dispatchInstructions.append(dispatch_bid_cap[s])
                else:
                    if system_type == "PHS":
                        dispatchInstructions.append(g_pumps*[0])
                    else:
                        dispatchInstructions.append(0)
                    
    return [dispatchInstructions]