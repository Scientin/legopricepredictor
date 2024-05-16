## Welcome to the Lego Price Predictor!

This projects consists of a linear regression neural network built using tensorflow's keras modules. 
The purpose of this model is to predict the price of a LEGO product based on a number of product specifications, namely IP (or "Theme"), piece count, minifigure count, and year of release.
This model was trained on product data scraped from the website Brickipedia (https://brickipedia.fandom.com/wiki/LEGO_Wiki) using the BeautifulSoup module.
At present, the model is trained on all LEGO products released between 1995 and 2023. 
We aim to update the preprocessing script to scrape data from previous years, accounting for discrepencies in HTML page design that exist between Brickipedia's pages.
The neural network currently reaches an absolute val_error of ~$10, which we hope to further minimize.

### Using the Predictor
To use the predictor, first run predictor.py within terminal, which should cause a GUI to open.
Input the number of pieces, number of minifigures, year of release, and theme of the set you want to predict, then press the predict button.
The predictor will reject any non-numerical inputs for pieces, figures, or year, and will also reject any incomplete inputs.
For theme, be sure to capitalize the theme properly and use spaces where appropriate 
(e.g. "BrickHeadz" instead of "brick headz", "Star Wars" instead of "starwars").

### Technical Details
Our model is a linear regression neural network built in keras, trained on batch sizes of 32. The model uses one dense hidden layer with 64 nodes and the relu activation, and is optimized with adam. The model has been pre-trained for the purposes of using the predictor, but is included for interested parties to experiment with.
