 SELECT DISTINCT S.Name [Supplier Name],
 S.City [Supplier City],
 S.State [Supplier State],
 C.Name [Consumer Name],
 C.City [Consumer City],
 C.State [Consumer State],
 P.Name [Product Name],
 SUM(Quantity) [Total Transactions Quantity],
 COUNT(Tran_ID) [Number of Transactions]
 INTO Tb_Transactions_Cube
 FROM Tb_Supplier S, Tb_Consumer C, Tb_Product P, Tb_Transactions T
 WHERE S.Supp_ID=T.Supp_ID AND
 C.Con_ID=T.Con_ID AND
 P.Prod_ID=T.Prod_ID
 GROUP BY CUBE((S.State, S.City, S.Name),
(C.State, C.City, C.Name),
P.Name),
 ROLLUP(S.State, S.City, S.Name),
 ROLLUP(C.State, C.City, C.Name) /* base */ SELECT *
FROM Tb_Transactions_Cube
WHERE [Supplier Name] IS NOT NULL
 AND [Supplier City] IS NOT NULL
 AND [Supplier State] IS NOT NULL
 AND [Consumer Name] IS NOT NULL
 AND [Consumer City] IS NOT NULL
 AND [Consumer State] IS NOT NULL
 AND [Product Name] IS NOT NULL 
 /* 1) Aggregates by combinations of supplier name and product name? */

SELECT *
FROM Tb_Transactions_Cube
WHERE [Supplier Name] IS NOT NULL
 AND [Supplier City] IS NOT NULL
 AND [Supplier State] IS NOT NULL
 AND [Consumer Name] IS NULL
 AND [Consumer City] IS NULL
 AND [Consumer State] IS NULL
 AND [Product Name] IS NOT NULL ORDER BY [Supplier Name] /* 2) Aggregates by supplier states? */SELECT *
FROM Tb_Transactions_Cube
WHERE [Supplier Name] IS NULL
 AND [Supplier City] IS NULL
 AND [Supplier State] IS NOT NULL
 AND [Consumer Name] IS NULL
 AND [Consumer City] IS NULL
 AND [Consumer State] IS NULL
 AND [Product Name] IS NULL /* 3) Number of transactions by supplier-city and consumer-city combinations? */  SELECT *
FROM Tb_Transactions_Cube
WHERE [Supplier Name] IS NULL
 AND [Supplier City] IS NOT NULL
 AND [Supplier State] IS NOT NULL
 AND [Consumer Name] IS NULL
 AND [Consumer City] IS NOT NULL
 AND [Consumer State] IS NOT NULL
 AND [Product Name] IS NULL /* 4) Name of each product and quantity Wisconsin suppliers sold of that product? */  SELECT [Product Name], [Supplier State], [Total Transactions Quantity],
[Number of Transactions]
FROM Tb_Transactions_Cube
WHERE [Supplier Name] IS NULL
 AND [Supplier City] IS NULL
 AND [Supplier State] = 'Wisconsin'
 AND [Consumer Name] IS NULL
 AND [Consumer City] IS NULL
 AND [Consumer State] IS NULL
 AND [Product Name] IS NOT NULL /* 5) Quantity of sales aggregated by product and supplier state? */  SELECT [Product Name], [Supplier State], [Total Transactions Quantity],
[Number of Transactions]
FROM Tb_Transactions_Cube
WHERE [Supplier Name] IS NULL
 AND [Supplier City] IS NULL
 AND [Supplier State] IS NOT NULL
 AND [Consumer Name] IS NULL
 AND [Consumer City] IS NULL
 AND [Consumer State] IS NULL
 AND [Product Name] IS NOT NULL ORDER BY [Supplier State] /* 6) Quantity of computer sales aggregated by suppliers in Wisconsin? */   SELECT [Supplier Name], [Product Name], [Total Transactions Quantity],
[Number of Transactions]
FROM Tb_Transactions_Cube
WHERE [Supplier Name] IS NOT NULL
 AND [Supplier City] IS NOT NULL
 AND [Supplier State] = 'Wisconsin'
 AND [Consumer Name] IS NULL
 AND [Consumer City] IS NULL
 AND [Consumer State] IS NULL
 AND [Product Name]  = 'Computer' ORDER BY [Supplier Name] /* 7) Quantity of auto sales by each supplier from Wisconsin to each auto
consumer in Illinois? */   SELECT [Supplier Name], [Product Name], [Total Transactions Quantity],
[Number of Transactions]
FROM Tb_Transactions_Cube
WHERE [Supplier Name] IS NOT NULL
 AND [Supplier City] IS NOT NULL
 AND [Supplier State] = 'Wisconsin'
 AND [Consumer Name] IS NULL
 AND [Consumer City] IS NULL
 AND [Consumer State] IS NULL
 AND [Product Name]  = 'Computer' ORDER BY [Supplier Name]/* 8) Quantity of each product sold by each supplier in Madison to each
consumer in Illinois? */SELECT [Supplier Name], [Consumer Name], [Product Name], [Total Transactions Quantity],
	[Number of Transactions]
FROM Tb_Transactions_Cube
WHERE [Supplier Name] IS NOT NULL
 AND [Supplier City] = 'Madison'
 AND [Supplier State] = 'Wisconsin'
 AND [Consumer Name] IS NOT NULL
 AND [Consumer City] IS NOT NULL
 AND [Consumer State] = 'Illinois'
 AND [Product Name]  IS NOT NULL ORDER BY [Supplier Name], [Consumer Name], [Product Name]/* 9) Quantity of each product sold by supplier Bernstein to consumers in
Chicago? */SELECT [Supplier Name], [Product Name], [Total Transactions Quantity],
	[Number of Transactions]
FROM Tb_Transactions_Cube
WHERE [Supplier Name] = 'Bernstein'
 AND [Supplier City] IS NOT NULL
 AND [Supplier State] IS NOT NULL
 AND [Consumer Name] IS NULL
 AND [Consumer City] = 'Chicago'
 AND [Consumer State] IS NOT NULL
 AND [Product Name]  IS NOT NULL ORDER BY [Supplier Name], [Consumer Name], [Product Name]/* 10) Quantity of milk sold by supplier Bernstein to each of his milk
consumers in Chicago? */SELECT [Supplier Name], [Consumer Name], [Product Name], [Total Transactions Quantity],
	[Number of Transactions]
FROM Tb_Transactions_Cube
WHERE [Supplier Name] = 'Bernstein'
 AND [Supplier City] IS NOT NULL
 AND [Supplier State] IS NOT NULL
 AND [Consumer Name] IS NOT NULL
 AND [Consumer City] = 'Chicago'
 AND [Consumer State] IS NOT NULL
 AND [Product Name]  = 'Milk' ORDER BY [Supplier Name], [Consumer Name], [Product Name]/* 11) (Extra Credit 2%) For each product list quantity sold by suppliers in
Madison to consumers in Chicago versus quantity sold by suppliers in
Chicago to consumers in Madison (result columns will be: product name,
quantity Madison_Chicago, quantity Chicago_Madison)? */

SELECT * FROM 
	(SELECT [Total Transactions Quantity]  AS Madison_Chicago
	FROM Tb_Transactions_Cube
	WHERE [Supplier Name] IS NULL
	 AND [Supplier City]= 'Madison'
	 AND [Supplier State] IS NOT NULL
	 AND [Consumer Name] IS NULL
	 AND [Consumer City]  = 'Chicago'
	 AND [Consumer State] IS NOT NULL
	 AND [Product Name] IS NULL) AS Madison_Chicago, 
	( SELECT [Total Transactions Quantity]  AS Chicago_Madison
		FROM Tb_Transactions_Cube
		WHERE [Supplier Name] IS NULL
		 AND [Supplier City]= 'Chicago'
		 AND [Supplier State] IS NOT NULL
		 AND [Consumer Name] IS NULL
		 AND [Consumer City]  = 'Madison'
		 AND [Consumer State] IS NOT NULL
		 AND [Product Name] IS NULL) AS Chicago_Madison