Dependencies
------------------------
Python packages:
- Pandas
- Pyomo
- Concurrent.futures

Solver:
- CBC.exe

Configuration
------------------------
1. Edit the cbc_path.csv file to include the path to cbc.exe
2. If running embarassingly parallel simulations, edit the result_filenames.csv file to specify the names of the result files output by each job
3. Modify the Data files to include the dispatch prices (5min), spot prices (30min), and predispatch spot prices (30min) for your chosen electricity market
4. Edit the Assumptions files to define your chosen BESS or PHS system for analysis

Running the Model
-----------------------
1. Run the model/simulation.py file in the Python interpreter
