# portfolio
A sampling of my coding and data science projects from coursework. 

These projects were completed by me, Jeff Orcutt, during coursework for my 
two Master's degree programs. For browsing ease, I've included a brief synopsis 
here of the project and the course for which it was completed. 

## R Projects

## Python Projects

## High Performance Computing Projects
* DS 730 - Final Project - Part 3. This project was an adaptation of the billion row
challenge project that compared the ability of Pig, Hive, and Scala through a Zeppelin Notebook to read a raw csv consisting of one billion rows and calculate the minimum, average, and maximum temperatures. This project involved spinning up an AWS EC2 instance, deploying a single hadoop node, and then comparing runtimes for various slicesof the dataset in a controlled environment. 

* DS 730 - Final Project - Part 1. This project entailed finding answers to various questions of a dataset regarding the temperatures collected at various times and dates
of two cities in Wisconsin. I leveraged Scala for Apache Spark in a Apache Zeppelin notebook. There are three files - an answers file providing the questions and answers, a stripped down code only .scala file, and a json which can be imported. There was some slight data cleaning that needed to occur for some erroneous or missing data. 
 
## Data Visualization Projects

## Other Products
* Traveling Salesman Projects - In a comparison of a heuristics versus brute force attempt at solving the traveling saleman problem. I created a threaded solution in Java that ensures every possible route is explored until the route time exceeds the current minimum time found. Runtimes are, understandably, less than desirable. In Python, I created a solution that randomly selects a city initially to explore then finds the nearest city to that city to the end of the route. I currently have the program randomly selecting the initial city 10000 times, just to ensure full coverage of the 12 cities. 

