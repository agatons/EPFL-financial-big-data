# EPFL - _Financial Big Data_  - Project Repo
_________________
#### Authors
Erik Agaton Sjöberg  & Emil Immonen

#### Requirements
The project has a few libraries that are required in order for the run files to function properly. These libraries can be found in _requirements.txt_, just execute `pip install -r requirements.txt` in order to install the dependencies. The library _Ta-lib_ might be a bit tricky to install since it is a wrapper. Instructions on how to install it properly can be found [here](https://mrjbq7.github.io/ta-lib/install.html "Ta-lib install instructions"). 

#### Financial analysis metod
The folder _financials_method_ includes a framework used to find a strategy, with the help of machine learning, that trades the stock market using each stock's financials data.
In order to __run__ an example of trading on the Swedish stock market with the financials method, run financials_method/simple_trading.py file. This will show plots for training of the model, performance of strategy and print out values for strategy performance.

#### Technical analysis method
The folder _ta_method_ includes a framework that can be used to __build__ and __backtest__ a strategy that does rolling calculations on a stock's daily price data.
In order to __run__ an example of the bootstrap strategy test of the momentum strategy, run the ta_method/run_backtest.py file.
    
   
