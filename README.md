# portfolio
A sampling of my coding and data science projects from coursework. 

These projects were completed by me, Jeff Orcutt, during coursework for my 
two Master's degree programs. For browsing ease, I've included a brief synopsis 
here of the project and the course for which it was completed. All course questions 
are owned by the person who created them. ** All code is created by me and you do not have my permission to reuse my work. **



## Python Projects
Capstone Final Project - This project looked at predicting motorcycle helmet usage from hospitalization data and had insights on age, gender, provincial, and time-of-day usage patterns comparing a number of analysis methods and hyperparameter optimization methodologies. Additionally, all figures were made within Python's matplotlib, seaborn, and sklearn libraries. The aforementioned plots would normally include a figure title and description, but said features were removed due to the paper's formatting requirements. 
** Dataset and model joblibs were in excess of 100MB, and excluded due to Github free storage, contact for either or both. **

## High Performance Computing Projects
* DS 730 - Final Project - Part 3. This project was an adaptation of the billion row
challenge project that compared the ability of Pig, Hive, and Scala through a Zeppelin Notebook to read a raw csv consisting of one billion rows and calculate the minimum, average, and maximum temperatures. This project involved spinning up an AWS EC2 instance, deploying a single hadoop node, and then comparing runtimes for various slices of the dataset in a controlled environment. 

* DS 730 - Final Project - Part 1. This project entailed finding answers to various questions of a dataset regarding the temperatures collected at various times and dates
of two cities in Wisconsin. I leveraged Scala for Apache Spark in a Apache Zeppelin notebook. There are three files - an answers file providing the questions and answers, a stripped down code only .scala file, and a json which can be imported. There was some slight data cleaning that needed to occur for some erroneous or missing data. 
 
## Data Visualization Projects

## Other Products
* Traveling Salesman Projects - In a comparison of a heuristics versus brute force attempt at solving the traveling saleman problem. I created a threaded solution in Java that ensures every possible route is explored until the route time exceeds the current minimum time found. Runtimes are, understandably, less than desirable. In Python, I created a solution that randomly selects a city initially to explore then finds the nearest city to that city to the end of the route. I currently have the program randomly selecting the initial city 10000 times, just to ensure full coverage of the 12 cities. 

## R Projects
* DS 740 - Final Project. I was given free reign on this project to identify a dataset and apply appropriate techniques to clean, describe, and discover any predictive relationships and customer taste preference clusterings. The dataset deals with reported profiles of coffee beans by trained tasters at the Coffee Quality Institute. After cleaning the data, I used Caret's random forest with double cross-validation to assess and pick an appropriate model and a hierachial clustering approach with the Nbclust package to build an appropriate hierachial tree to better understand the clusters.
* Orcutt - Project 3 - This was a project where I employed multivariate linear models and R's ggplot2 package to understand what level of impact various features have upon the overall Ironman time of a participant. Includes a writeup and plots in a Word document. 

## SQL Queries
* DS 715 - Various queries collected throughout the semester. A handful of SQL examples of various major functions: aggregation, windows, subqueries, etc. Also includes some queries from OLAP cube queries using Microsoft SQL Server Analysis Services. 
