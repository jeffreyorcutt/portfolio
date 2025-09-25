/* Jeff Orcutt
 * 20 OCT 2023
 * Assignment 3d
 * DS 715
 */


/* 1 Product categories and number of products in each category? */

SELECT p.Product_Category, COUNT(p.Prod_ID) AS 'Num_Products'
	FROM Tb_Product AS p
	GROUP BY p.Product_Category

/* 2 Cities having at leASt 3 different consumers? */
SELECT c.City, COUNT(c.Con_ID) AS 'Num_Diff_Consumers'
	FROM Tb_Consumer AS c
	GROUP BY c.City
	HAVING COUNT(c.Con_ID) >= 3

/* 3) Cities and number of different products offered in city? */
SELECT s.City, COUNT(DISTINCT(p.Prod_ID)) AS 'Num_Diff_Prods'
	FROM Tb_Supplier AS s
	INNER JOIN Tb_Offers AS o
		ON s.Supp_ID = o.Supp_ID
	INNER JOIN Tb_Product AS p
		ON o.Prod_ID = p.Prod_ID
	GROUP BY s.City

/* 4) List of states and number of car suppliers in each state? */
SELECT s.State, COUNT(DISTINCT(o.Supp_ID)) AS 'Num_Car_Suppliers'
	FROM Tb_Supplier AS s
	INNER JOIN Tb_Offers AS o
		ON s.Supp_ID = o.Supp_ID
	INNER JOIN Tb_Product AS p
		ON o.Prod_ID = p.Prod_ID
	WHERE p.Name = 'Auto'
	GROUP BY s.State
/* 5) Product name and quantity offered in each city? */
SELECT s.City, p.Name, SUM(o.Quantity) AS 'Num_Offered'
	FROM Tb_Product AS p
	INNER JOIN Tb_Offers AS o
		ON p.Prod_ID = o.Prod_ID
	INNER JOIN Tb_Supplier AS s
		ON o.Supp_ID = s.Supp_ID
	GROUP BY p.Name, s.City

/* 6) Supplier name and quantity of TV sold in each consumer city? */
SELECT s.Name, s.City, SUM(t.Quantity) AS 'Num_Sold'
	FROM Tb_Supplier AS s
	INNER JOIN Tb_Transactions AS t
		ON s.Supp_ID = t.Supp_ID
	INNER JOIN Tb_Product AS p
		on t.Prod_ID = p.Prod_ID
	WHERE p.Name = 'TV'
	GROUP BY s.Name, s.City

/* 7) Supplier-consumer-product names such that supplier is selling product to
consumer with total value of transactions between supplier and consumer
for given product? */

SELECT s.Name AS 'Supplier' , c.Name AS 'Consumer', p.Name AS 'Product', SUM(t.Quantity*t.Price) AS 'Total Sales Value'
	FROM Tb_Supplier AS s
	INNER JOIN Tb_Transactions AS t
		ON s.Supp_ID = t.Supp_ID
	INNER JOIN Tb_Product AS p
		on t.Prod_ID = p.Prod_ID
	INNER JOIN Tb_Consumer AS c
		on t.Con_ID = c.Con_ID
	GROUP BY s.Name, c.Name, p.Name

/* 8) Monthly sales data (total transactions quantity, number of transactions,
total transactions value) by supplier, consumer, and product during the
year 2018? */
Select d.DateMonth AS 'Month', s.Name AS 'Supplier', c.Name AS 'Consumer', p.Name AS 'Product',
	COUNT(t.Tran_ID) AS 'Transaction Count', 
	SUM(t.Quantity) AS 'Total Units Sold', 
	SUM(t.Quantity * t.Price) AS 'Total Sales Value'
		FROM Tb_Supplier AS s
	INNER JOIN Tb_Transactions AS t
		ON s.Supp_ID = t.Supp_ID
	INNER JOIN Tb_Product AS p
		on t.Prod_ID = p.Prod_ID
	INNER JOIN Tb_Consumer AS c
		on t.Con_ID = c.Con_ID
	INNER JOIN Tb_Date AS d
		on t.DateKey = d.DateKey
	WHERE d.DateYear = 2018
	GROUP BY s.Name, c.Name, p.Name, d.DateMonth
	ORDER BY d.DateMonth

/* 9) States where the number of suppliers exceeds the number of consumers? */
SELECT s.State
	FROM Tb_Supplier AS s
	FULL JOIN Tb_Transactions AS t
		ON s.Supp_ID = t.Supp_ID
	FULL JOIN Tb_Consumer AS c
		on t.Con_ID = c.Con_ID
	GROUP BY s.State
	HAVING(COUNT(DISTINCT(s.Supp_ID)) > COUNT(DISTINCT(c.Con_ID)))



/* 10) Comparative list of supplier or consumer states and cities with
respective number of suppliers and consumers in each city (columns are
State, City, NumberOfSuppliers, NumberOfConsumers)? */

SELECT cs.State, cs.City, COUNT(DISTINCT(s.Supp_ID)) AS 'Supplier Count', COUNT(DISTINCT(c.CON_ID)) AS 'Consumer Count'
	FROM (SELECT State, City
			FROM Tb_Supplier
			UNION
			SELECT State, City
			FROM Tb_Consumer) AS cs
	FULL JOIN Tb_Supplier AS s
		ON cs.State = s.State  AND cs.City = s.City
	FULL JOIN Tb_Consumer AS c
		ON cs.State = c.State AND cs.City = c.City
	GROUP BY cs.State, cs.City
	ORDER BY cs.State, cs.City





/* 11) (Extra Credit 2%) For each product list the quantity sold by suppliers
in Madison to consumers in Chicago versus the quantity sold by suppliers
in Chicago to consumers in Madison (result columns: product name,
quantity Madison_Chicago, quantity Chicago_Madison)? */


SELECT p.Name AS 'Product', 
	SUM(CASE WHEN s.City = 'Madison' AND c.City = 'Chicago' THEN t.Quantity ELSE 0 end) AS 'Madison_Chicago',
	SUM(CASE WHEN s.City = 'Chicago' AND c.City = 'Madison' THEN t.Quantity ELSE 0 end) AS 'Chicago_Madison'
	FROM Tb_Supplier AS s
	INNER JOIN Tb_Transactions AS t
		ON s.Supp_ID = t.Supp_ID
	INNER JOIN Tb_Product AS p
		on t.Prod_ID = p.Prod_ID
	INNER JOIN Tb_Consumer AS c
		on t.Con_ID = c.Con_ID
	GROUP BY p.Name
	ORDER BY p.Name
