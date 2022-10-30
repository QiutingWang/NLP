# Intermediate Cypher Queries (Part1)

## 1. Data Model

```
CALL db.schema.visualization()
```

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 11.15.26.png" alt="Graph1" style="zoom:30%;" />

- View the property types for nodes in the graph:

```
CALL db.schema.nodeTypeProperties()
```

- View the property types for relationships in the graph:

```
CALL db.schema.relTypeProperties()
```

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 11.28.44.png" alt="Graph2" style="zoom:40%;" />

- View the uniqueness constraint indexes 

  ```
  SHOW CONSTRAINTS
  ```

  <img src="/Users/macbookpro/Desktop/截屏2022-10-30 11.30.27.png" alt="Graph3" style="zoom:40%;" />

## 2. Main References

- [Neo4j Cypher Refcard](https://neo4j.com/docs/cypher-refcard/current/)
- [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/)

## 3. Basic Cypher Queries

### Testing Inequalities

- Test inequality of a property use `<>` predicate.

```
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WHERE p.name <> 'Tom Hanks'
AND m.title = 'Captain Phillips'
RETURN p.name
```

This query returns the names of all actors that acted in the movie Captain Phillips, where Tom Hanks is excluded.

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 11.40.51.png" alt="Graph4" style="zoom:50%;" />

### Testing less than or greater than

- Test both *numbers* and *strings* for values less than `<` or greater than `>` a value

```
MATCH (m:Movie) WHERE m.title = 'Toy Story'
RETURN
    m.year < 1995 AS lessThan,             //  Less than (false)
    m.year <= 1995 AS lessThanOrEqual,     // Less than or equal(true)
    m.year > 1995 AS moreThan,             // More than (false)
    m.year >= 1995 AS moreThanOrEqual      // More than or equal (true)
```

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 11.43.23.png" alt="Graph5" style="zoom:50%;" />

### Testing Ranges

- Use a combination of less than and greater than.

```
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WHERE p.name = 'Tom Hanks'
AND  2005 <= m.year <= 2010
RETURN m.title, m.released
```

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 11.44.42.png" alt="Graph6" style="zoom:50%;" />

Returns the four movies that Tom Hanks acted in between 2005 and 2010, inclusive.

### Testing `null` property values

- A property of a node or relationship is null **if it does not exist**.
  - Test the existence of a property for a node using the `IS NOT NULL` predicate.
  - Test if a property exists using the `IS NULL` predicate

```
MATCH (p:Person)
WHERE p.died IS NOT NULL
AND p.born.year >= 1985
RETURN p.name, p.born, p.died
```

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 11.48.34.png" alt="graph6" style="zoom:40%;" />

```
MATCH (p:Person)
WHERE p.died IS NULL
AND p.born.year <= 1922
RETURN p.name, p.born, p.died
```

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 11.50.46.png" alt="graph7" style="zoom:40%;" />

This query returns all people born before 1923 who do not have a died property value. 

### Testing labels or patterns

-  Test for a label’s existence on a node using the `{alias}:{label}`

```
MATCH (p:Person)
WHERE  p.born.year > 1960
AND p:Actor
AND p:Director
RETURN p.name, p.born, labels(p)
```

The `labels()` function returns the list of labels for a node.

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 11.55.12.png" alt="graph7" style="zoom:40%;" />

### Discovering relationship types

- use the `type()` function to return the type of the relationship, r.
- Check the uniqueness, use `WITH DISTINCT` to constrain.

## 4. Testing Strings

- Use `START WITH`, `END WITH`, `CONTAIN`  with `WHERE` clause to filter the characteristics in the string

- **Case sensitive**:  If string values could be mis-interpreted if the letters do not match in case, your queries may miss data in the graph. Then we use `toUpper`,`toLower` to unify the property while matching.

```
MATCH (p:Person)
WHERE toLower(p.name) ENDS WITH 'demille'
RETURN p.name
```

We do not know whether the data is 'DeMille', 'Demille', or 'deMlle' , to ensure matching all Person nodes that could be one of these, we transform the property value to *lower-case*.

```
MATCH (p:Person)
WHERE toUpper(p.name) CONTAINS ' DE '
RETURN p.name
```

To ensure matching all Person nodes that could be one of these, we transform the property value to *upper-case*.

- **Indexes for queries**: With any query,check if an index will be used by prefixing the query with `EXPLAIN`.

  ```
  EXPLAIN MATCH (m:Movie)
  WHERE  m.title STARTS WITH 'Toy Story'
  RETURN m.title, m.released
  ```

  <img src="/Users/macbookpro/Desktop/截屏2022-10-30 16.32.03.png" alt="graph8" style="zoom:40%;" />

  This query produces the execution plan where the first step is NodeIndexSeekByRange.

## 5. Query Patterns and Performance

- A pattern is a combination of *nodes and relationships* that is used to traverse the graph at runtime

Write queries that test whether a pattern exists in the graph, using `exists { }` test:

```
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WHERE  p.name = 'Tom Hanks'
AND exists {(p)-[:DIRECTED]->(m)}
RETURN p.name, labels(p), m.title
```

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 16.43.42.png" alt="Graph8" style="zoom:50%;" />

### Profiling queries

-  Use the `PROFILE` keyword to show the *total number of rows* retrieved from the graph in the query.

```
PROFILE MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WHERE  p.name = 'Tom Hanks'
AND exists {(p)-[:DIRECTED]->(m)}
RETURN m.title
```

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 16.47.48.png" alt="Graph9" style="zoom:50%;" />

 The graph shows initial row is retrieved, but then 38 rows are retrieved for each Movie that Tom Hanks acted in. 

There is a better way to do the same query:

```
PROFILE MATCH (p:Person)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(p)
WHERE  p.name = 'Tom Hanks'
RETURN  m.title
```

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 16.49.50.png" alt="Graph10" style="zoom:50%;" />

This traversal is very efficient because the graph engine can **take the [internal] relationship cardinalities into account.**  It retrieves one row then two rows; much less data than the first query

- Differences between `EXPLAIN` and `PROFILE`:

  - `EXPLAIN` provides estimates of the query steps 

  -  `PROFILE` provides the exact steps and number of rows retrieved for the query.

    - When query tuning,execute the query at least twice. The first `PROFILE` of a query will always be more *expensive* than subsequent queries.

      - First execution:  generation of the execution plan

      - Second execution: cached.

### Finding non-patterns

- Use `NOT exists { }` to exclude patterns

```
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WHERE  p.name = 'Tom Hanks'
AND NOT exists {(p)-[:DIRECTED]->(m)}
RETURN  m.title
```

Exclude the `:DIRECTED` relationships to movies for Tom Hank.

## 6. Multiple MATCH Clauses

### Using multiple patterns in the `MATCH` clause

```
MATCH (a:Person)-[:ACTED_IN]->(m:Movie),
      (m)<-[:DIRECTED]-(d:Person)
WHERE m.year > 2000
RETURN a.name, m.title, d.name
```

- In general, using a single `MATCH` clause will **perform better** than multiple `MATCH` clauses. Because *relationship* *uniquness* is enforced so there are fewer relationships traversed.

### Using a single pattern

```
MATCH (a:Person)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(d:Person)
WHERE m.year > 2000
RETURN a.name, m.title, d.name
```

### Optionally matching rows

```
MATCH (m:Movie) WHERE m.title = "Kiss Me Deadly"
MATCH (m)-[:IN_GENRE]->(g:Genre)<-[:IN_GENRE]-(rec:Movie)
OPTIONAL MATCH (m)<-[:ACTED_IN]-(a:Actor)-[:ACTED_IN]->(rec)
RETURN rec.title, a.name
```

-  Use **nulls** for *missing parts* of the pattern.
- Could be considered the Cypher equivalent of the **outer join** in SQL.

## 7. Ordering Returned Results

- `ORDER BY`  property `DESC`, the default ordering is ascending

- Specify `ORDER BY` in the `RETURN` clause
- Eliminating null values returned: use `IS NOT NULL`, `IS NULL` in `WHERE` clause to declare.
- No limit to the number of properties you can order by.
- Ordering multiple results separate them by comma. The precedures will first follow the first stated rule, then the others in queues.

## 8. Limiting or Counting Results Returned

- Use `LIMIT` at the end of queries
- Add a `SKIP` and `LIMIT` keyword to control what page of results are returned.

```
MATCH (p:Person)
WHERE p.born.year = 1980
RETURN  p.name as name,
p.born AS birthDate
ORDER BY p.born SKIP 40 LIMIT 10
```

In this query, we return 10 rows representing page 5, where each page contains 10 rows.

- Use `DISTINCT` eliminates duplicates for rows, property, nodes in `RETURN` clause. If we do not declare specificly, by default, it eliminates duplicated nodes.

## 9. Map Projections to Return Data

### Map Projections

```
MATCH (p:Person)
WHERE p.name CONTAINS "Thomas"
RETURN p { .* } AS person
ORDER BY p.name ASC
```

- Return data is without the internal node information(table, text,diagram...), that is, only property values with a JSON-style object for a node.

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 19.04.25.png" alt="Graph14" style="zoom:50%;" />

- Customize what properties you return in the objects by specifing the properties name in { }.

```RETURN p { .name, .born } AS person```

-  Adding information to the objects returned that are not part of the data in the graph.

  ```
  MATCH (m:Movie)<-[:DIRECTED]-(d:Director)
  WHERE d.name = 'Woody Allen'
  RETURN m {.*, favorite: true} AS movie
  ```

  <img src="/Users/macbookpro/Desktop/截屏2022-10-30 19.08.42.png" alt="graph15" style="zoom:50%;" />

## 10. Changing Result Returned

### Changing data returned

```
MATCH (m:Movie)<-[:ACTED_IN]-(p:Person)
WHERE m.title CONTAINS 'Toy Story' AND
p.died IS NULL
RETURN 'Movie: ' + m.title AS movie,
p.name AS actor,
p.born AS dob,
date().year - p.born.year AS ageThisYear
```

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 19.16.09.png" alt="Graph17" style="zoom:50%;" />

Performing string or numeric operations:

- Add data to each line by calculating the actor’s age.
- Concatenate string data returned.

### Conditionally changing data returned

Use `CASE` `WHEN` `ELSE` `END` clause that specify to compute the data returned which may be different from what is in the graph.

```
MATCH (m:Movie)<-[:ACTED_IN]-(p:Person)
WHERE p.name = 'Henry Fonda'
RETURN m.title AS movie,
CASE
WHEN m.year < 1940 THEN 'oldies'
WHEN 1940 <= m.year < 1950 THEN 'forties'
WHEN 1950 <= m.year < 1960 THEN 'fifties'
WHEN 1960 <= m.year < 1970 THEN 'sixties'
WHEN 1970 <= m.year < 1980 THEN 'seventies'
WHEN 1980 <= m.year < 1990 THEN 'eighties'
WHEN 1990 <= m.year < 2000 THEN 'nineties'
ELSE  'two-thousands'
END
AS timeFrame
```

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 19.18.26.png" alt="Graph18" style="zoom:40%;" />

## 11. Aggregating Data

### Counts

Use `count()` function  to  perform a count of nodes, relationships, paths, rows during query processing.

- Eager aggregation: In Cypher,  need not specify a grouping key. All non-aggregated result columns become grouping keys.
- If count(*) include null values.
- If count(n) return n non-null values.

```
MATCH (a:Person)-[:ACTED_IN]->(m:Movie)
WHERE a.name = 'Tom Hanks'
RETURN a.name AS actorName,
count(*) AS numMovies
```

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 19.26.01.png" alt="Graph19" style="zoom:50%;" />

### Lists

Return a list by specifying the square brackets

```
MATCH (p:Person)
RETURN p.name, [p.born, p.died] AS lifeTime
LIMIT 10
```

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 19.35.41.png" alt="Graph15" style="zoom:40%;" />



### Using `collect()` to create a list

`collect()` aggregate values into a list. The value can be any expression.

```
MATCH (a:Person)-[:ACTED_IN]->(m:Movie)
RETURN a.name AS actor,
count(*) AS total,
collect(m.title) AS movies
ORDER BY total DESC LIMIT 10
```

![Graph17](/Users/macbookpro/Desktop/截屏2022-10-30 19.37.40.png)

### Eliminating duplication in lists

Use `DISTINCT`

```
MATCH (a:Person)-[:ACTED_IN]->(m:Movie)
WHERE m.year = 1920
RETURN  collect( DISTINCT m.title) AS movies,
collect( a.name) AS actors
```

### Collecting nodes

```
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WHERE p.name ='Tom Cruise'
RETURN collect(m) AS tomCruiseMovies
```

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 19.58.47.png" alt="Graph19" style="zoom:40%;" />

This query returns a list of all Movie nodes for Tom Cruise.

### Accessing elements of a list

Access particular elements of the list using the `[index-value]` notation where a list *begins with index 0*. Return a slice of a collection

```
MATCH (a:Person)-[:ACTED_IN]->(m:Movie)
RETURN m.title AS movie,
collect(a.name)[2..] AS castMember,
size(collect(a.name)) as castSize
```

This query returns **the second to the end** of the list names of actors.

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 20.01.43.png" alt="Graph19" style="zoom:40%;" />



- Differences between `COUNT`() & `SIZE`()

  - `size()` function returns the number of **elements in a list**.

  -  use `count()` to count the number of rows, more efficient than `size()`

  - Use `profile()` to check

### List comprehension

```
MATCH (m:Movie)
RETURN m.title as movie,
[x IN m.countries WHERE x = 'USA' OR x = 'Germany']
AS country LIMIT 500
```

 Create a list by evaluating an expression that tests for list inclusion.

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 20.09.00.png" alt="Graph20" style="zoom:40%;" />

### Pattern comprehension

- Create lists without changing the cardinality of the query.

- ` [<pattern> | value]`
  - specify the list with the square braces to include the pattern followed by the pipe character to 
  - specify what value will be placed in the list from the pattern.

```
MATCH (m:Movie)
WHERE m.year = 2015
RETURN m.title,
[(dir:Person)-[:DIRECTED]->(m) | dir.name] AS directors,
[(actor:Person)-[:ACTED_IN]->(m) | actor.name] AS actors
```

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 20.12.38.png" alt="graph21" style="zoom:50%;" />

Create a list  *specify a filter* for the pattern:

```
MATCH (a:Person {name: 'Tom Hanks'})
RETURN [(a)-->(b:Movie)
WHERE b.title CONTAINS "Toy" | b.title + ": " + b.year]
AS movies
```

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 20.13.37.png" alt="Graph22" style="zoom:50%;" />

### Working with maps

- Maps: list of key/value pairs. A node or relationship can have a property that is a map.

- Return the value for one of its elements:

  -  `RETURN {maps}['key'] AS variableName` 

  -  `RETURN {maps}.key AS variableName`

-  Return a list of keys of a map :
  - `RETURN {maps} AS variableName`

- A node in the graph  returned in Neo4j Browser is a map.

## 12. Working with Dates and  Times

```
RETURN date(), datetime(), time()
```

 Create some date/time properties:

```
MERGE (x:Test {id: 1})
SET x.date = date(),
    x.datetime = datetime(),
    x.time = time()
RETURN x
```

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 20.26.58.png" alt="graph22" style="zoom:50%;" />

Show the types of the properties:

```
CALL apoc.meta.nodeTypeProperties()
```

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 20.28.39.png" alt="graph23" style="zoom:50%;" />

### Extracting components of a date or datetime

```
MATCH (x:Test {id: 1})
RETURN x.date.day, x.date.year,
x.datetime.year, x.datetime.hour,
x.datetime.minute
```

<img src="/Users/macbookpro/Desktop/截屏2022-10-30 20.33.29.png" alt="Graph23" style="zoom:50%;" />

### Setting date values

Use a string to set a value for a date

```MATCH (x:Test {id: 1})
SET x.date1 = date('2022-01-01'),
    x.date2 = date('2022-01-15')
RETURN x
```

<img src="/Users/macbookpro/Library/Application Support/typora-user-images/截屏2022-10-30 20.35.29.png" alt="截屏2022-10-30 20.35.29" style="zoom:50%;" />

### Setting datetime values

```
MATCH (x:Test {id: 1})
SET x.datetime1 = datetime('2022-01-04T10:05:20'),
    x.datetime2 = datetime('2022-04-09T18:33:05')
RETURN x
```

<img src="/Users/macbookpro/Library/Application Support/typora-user-images/截屏2022-10-30 20.37.43.png" alt="截屏2022-10-30 20.37.43" style="zoom:50%;" />

### Working with durations

Determine the difference between two date/datetime values or to add or subtract a duration to a value.

```
MATCH (x:Test {id: 1})
RETURN duration.between(x.date1,x.date2)
```

```
MATCH (x:Test {id: 1})
RETURN duration.inDays(x.datetime1,x.datetime2).days
```

```
MATCH (x:Test {id: 1})
RETURN x.date1 + duration({months: 6})
```

