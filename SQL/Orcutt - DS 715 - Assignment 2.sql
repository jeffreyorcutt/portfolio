/*1) List full transaction data relating to suppliers from Madison and
consumers from Stevens Point where transaction value is higher than
$10,000 (show supplier, consumer and product names, quantity and price)?
*/

SELECT s.Name "Supplier", c.Name "Customer", p.Name "Product",
		t.Quantity, t.Price 
	FROM Tb_Transactions AS t
	LEFT JOIN Tb_Supplier AS s 
		ON s.Supp_ID =  t.Supp_ID
	LEFT JOIN Tb_Consumer as c
		ON c.Con_ID = t.Con_ID
	LEFT JOIN Tb_Product as p
		ON p.Prod_ID = t.Prod_ID
	WHERE (t.Quantity * t.Price > 10000) AND
		(s.City = 'Madison') AND
		(c.City = 'Stevens Point')

/* 2. Name of suppliers offering both computers and oranges? (do not use set
operations) */
SELECT s.Name 
	FROM Tb_Supplier AS s,
		Tb_Offers AS o1, Tb_Offers AS o2,
		Tb_Product as p1, Tb_Product AS p2
	WHERE s.Supp_ID = o1.Supp_ID
		AND s.Supp_ID = o2.Supp_ID
		AND o1.Prod_ID = p1.Prod_ID
		AND o2.Prod_ID = p2.Prod_ID
		AND p1.Name = 'Computer'
		AND p2.Name = 'Orange'

/* 3. Name of suppliers from Wausau or offering computers or offering
oranges? */
SELECT DISTINCT s.Name 
	FROM Tb_Supplier AS s
	WHERE s.City = 'Wausau'
	OR s.Supp_ID IN (SELECT o.Supp_ID
		FROM Tb_Offers AS o, Tb_Product AS p
		WHERE s.Supp_ID = o.Supp_ID
		AND o.Prod_ID = p.Prod_ID
		AND p.Name IN ('Computer','Orange'))


/* 4. Name of suppliers offering computer, auto and orange? */
SELECT s.Name 
	FROM Tb_Supplier AS s,
		Tb_Offers AS o1, Tb_Offers AS o2, Tb_Offers AS o3,
		Tb_Product as p1, Tb_Product AS p2, Tb_Product as p3
	WHERE s.Supp_ID = o1.Supp_ID
		AND s.Supp_ID = o2.Supp_ID
		AND s.Supp_ID = O3.Supp_ID
		AND o1.Prod_ID = p1.Prod_ID
		AND o2.Prod_ID = p2.Prod_ID
		AND o3.Prod_ID = p3.Prod_ID
		AND p1.Name = 'Computer'
		AND p2.Name = 'Orange'
		AND p3.Name = 'Auto'

/* 5. Name of products not offered in Chicago? */
SELECT p.Name
	FROM Tb_Product as p
	WHERE p.Prod_ID NOT IN 
		(SELECT o2.Prod_ID 
		FROM Tb_Offers as o2
		LEFT JOIN Tb_Supplier as s2
			ON s2.Supp_ID = o2.Supp_ID
		WHERE s2.City = 'Chicago')

/* 6. Name of consumers requesting only computers? */
SELECT c.Name 
	FROM Tb_Consumer as c, 
	Tb_Requests as r, 
	Tb_Product as p
	WHERE p.Name ='Computer'
		AND c.Con_ID = r.Con_ID
		AND r.Prod_ID = p.Prod_ID
		AND c.Name NOT IN (SELECT c2.Name 
			FROM Tb_Consumer as c2, 
			Tb_Requests as r2, 
			Tb_Product as p2
			WHERE p2.Name !='Computer'
				AND c2.Con_ID = r2.Con_ID
				AND r2.Prod_ID = p2.Prod_ID)
	ORDER BY c.Name

/* 7. Name of supplier cities where none of the suppliers has any offer? */
SELECT DISTINCT s.City 
	FROM Tb_Supplier as s
	WHERE s.City NOT IN (SELECT DISTINCT s2.City 
		FROM Tb_Supplier as s2
		LEFT JOIN Tb_Offers as o2
			ON o2.Supp_ID = s2.Supp_ID)


/* 8. Name of products requested by all consumers? */
SELECT p.Name 
	FROM Tb_Product AS p
	LEFT JOIN Tb_Requests as r
		ON (r.Prod_ID = p.Prod_ID)
	GROUP BY p.Name
	HAVING COUNT(r.Prod_ID) = 
		(SELECT COUNT(DISTINCT Tb_Consumer.Name) 
			FROM Tb_Consumer)
	
/* 9. Product name and supplier having the largest offer (as quantity) for that
product?*/
SELECT DISTINCT p.Name "Product", s.Name "Supplier", o.Quantity "Quantity"
	FROM Tb_Product AS p
	INNER JOIN Tb_Offers AS o
		ON o.Prod_ID = p.Prod_ID
	INNER JOIN Tb_Supplier AS s
		ON s.Supp_ID = o.Supp_ID
	WHERE o.Quantity >= (SELECT MAX(Quantity) 
		FROM Tb_Offers 
		WHERE Prod_ID = p.Prod_ID)
	ORDER BY p.Name, s.Name
	
/* 10. Product name and city where that product sold best, as in largest total
quantity? I did use the framework provided in your solution to help me with this. I 
had been learning Rank() the hard way trying to figure this one out. */
SELECT rt.Product, rt.City, rt.Total_Sold
FROM (SELECT p.Name AS Product, 
		c.City AS City, 
		SUM(t.Quantity) AS Total_Sold,
      RANK() OVER (
            PARTITION BY p.Name
            ORDER BY SUM(t.Quantity) DESC
            ) AS rn
      FROM Tb_Consumer AS c,
		Tb_Product AS p,
		Tb_Transactions AS t
      WHERE p.Prod_ID = t.Prod_ID 
		AND c.Con_ID  = t.Con_ID
      GROUP BY p.Name, c.City
      ) AS rt
WHERE rt.rn = 1

/* (Extra Credit 2%) Name of products requested in all consumer cities other
than Stevens Point? */
--SELECT p.Name, c.City FROM
--	Tb_Product AS p
--	LEFT JOIN Tb_Requests AS r
--		ON p.Prod_ID = r.Prod_ID
--	LEFT JOIN Tb_Consumer AS c
--		ON c.Con_ID = r.Con_ID
--	WHERE c.City <> 'Stevens Point'
--		GROUP BY c.City, p.Name


--	HAVING (COUNT(p.Name) = (COUNT(c.City)-1 ))

	

