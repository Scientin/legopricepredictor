## Welcome to the Lego Price Predictor!

This projects consists of a linear regression neural network built using tensorflow's keras modules. 
The purpose of this model is to predict the price of a LEGO product based on a number of product specifications, namely IP (or "Theme"), piece count, minifigure count, and year of release.
This model was trained on product data scraped from the website Brickipedia (https://brickipedia.fandom.com/wiki/LEGO_Wiki) using the BeautifulSoup module.
At present, the model is trained on all LEGO products released between 2017 and 2023. 
We aim to update the preprocessing script to scrape data from previous years, accounting for discrepencies in HTML page design that exist between Brickipedia's pages.
The neural network currently reaches an absolute error of ~$12, which we hope to further minimize.
