A = LOAD 'measurements.txt' USING PigStorage() AS (city:chararray, temperature:float);



B = GROUP A BY city;

C = FOREACH B GENERATE group AS city, AVG(A.temperature) AS avg_temp, MAX(A.temperature) AS max_temp, MIN(A.temperature) AS min_temp;


STORE C INTO 'pig_output.csv' USING PigStorage(',');