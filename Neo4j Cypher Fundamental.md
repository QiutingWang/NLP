# Reading Data from Neo4j

## 1. Introduction to Cypher
------------------------------

```
MATCH
(p: Person{name:'Tom Hanks'})-[:ACTED_IN]->(m:Movie{title:'Cloud Atlas'})
RETURN p,m
```

- node: writen in the bracket() and colon:, with several properties in form of key value pairs
- :Person, :Movie the labels
- -->represent the relationship between two nodes with direction and type. Specifying the relationship type with square bracket[] with colon:
- MATCH: retreive the data in the graph
- use WHERE as a condition filter
- Return: request particular results

## 2. Finding Relationships
-----------------------------

```
MATCH
(variableA:labelA)-[:relationship_type]->(variableB:labelB)
WHERE condition
RETURN variableA, variableB
```

- Nodes written in parenthesis. 
- The dash arrow represents the directed relationship between nodes.
- variable names: used to represent the node references. And the specific value matches to variable

```
MATCH (m:Movie {title:'Cloud Atlas'})<-[:DIRECTED]-(p:Person) 
RETURN p.name,count(*)
```

- count(*) is a kind of [aggregate function](https://neo4j.com/docs/cypher-manual/current/functions/aggregating/)
  
## 3. Filtering Queries
------------------------

### Use `where` clause

```
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WHERE m.released = 2008 OR m.released = 2009
RETURN p, m
```

This query retrieves the Person nodes and Movie nodes where the person acted in a movie that was released in 2008 or 2009.
![graph1](https://ibb.co/ydTmbQ8)

### Filtering by node labels

```
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WHERE m.title='The Matrix'
RETURN p.name
```

OR

```
MATCH (p)-[:ACTED_IN]->(m)
WHERE p:Person AND m:Movie AND m.title='The Matrix'
RETURN p.name
```

It returns the names of all people who acted in the movie, The Matrix.

|"p.name"|  
|--|
|"Emil Eifrem"| 
|"Hugo Weaving"| 
|"Laurence Fishburne"| 
|"Carrie-Anne Moss"| 
|"Keanu Reeves"| 

### Filtering use ranges

```
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WHERE 2000 <= m.released <= 2003
RETURN p.name, m.title, m.released
```

Retrieve Person nodes of people who acted in movies released between 2000 and 2003.

![Graph2](https://ibb.co/z4ykKQC)

### Filtering by existence of a property

```
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WHERE p.name='Jack Nicholson' AND m.tagline IS NOT NULL
RETURN m.title, m.tagline
```

Only want to return Movie nodes where Jack Nicholson acted in the movie, and the movie has the tagline property.

![Graph3](https://ibb.co/fCRsWrV)

### Filtering by partial strings

```
MATCH (p:Person)-[:ACTED_IN]->()
WHERE p.name STARTS WITH 'Michael'
RETURN p.name
```

Use a set of string-related keywords that you can use in your `WHERE` clauses to test string property values. You can specify `STARTS WITH`, `ENDS WITH`, and `CONTAINS`.

![Graph4](https://ibb.co/bNDtdQc)

```
MATCH (p:Person)-[:ACTED_IN]->()
WHERE toLower(p.name) STARTS WITH 'michael'
RETURN p.name
```

Use the `toLower()` or `toUpper()` functions to ensure the test yields the correct results. 

### Filtering by patterns in the graph

```
MATCH (p:Person)-[:WROTE]->(m:Movie)
WHERE NOT exists( (p)-[:DIRECTED]->(m) )
RETURN p.name, m.title
```

Find all people who wrote a movie but did not direct that same movie.

![Graph5](https://ibb.co/yXrVXHj)

### Filtering use lists

```
MATCH (p:Person)
WHERE p.born IN [1965, 1970, 1975]
RETURN p.name, p.born
```

If you have a set of values you want to test with, you can place them in a list or you can test with an existing list in the graph. 

A Cypher list is a comma-separated set of values within square brackets. Either *numeric or string values* can be included in the list. But the elements of the list should be the same type.

Define the list in the `WHERE` clause. During the query, the graph engine will compare each property with the values `IN` the list. 

![Graph6](https://ibb.co/51SfWJx)

In this case, we want to retrieve Person nodes of people born in 1965, 1970, or 1975. Next, we compare a value to an *existing list* in the graph.

```
MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
WHERE  'Neo' IN r.roles AND m.title='The Matrix'
RETURN p.name, r.roles
```

![Graph7](https://ibb.co/8mKpspp)

### Check the property of a node/relationship

```
MATCH (p:Person)
RETURN p.name, keys(p)
```

Use the `keys()` function. Returns a list of all property keys for a node. In this case, discover the keys for the Person nodes in the graph.

### Check properties exist in the graph

```
CALL db.propertyKeys()
```

Return all the property keys defined in the graph.

![Graph8](https://ibb.co/3YcnRsq)





# Writing Data to Neo4j

## 1. Creating Nodes
------------------

```
MERGE (p:Person {name: 'Michael Cain'})
RETURN p
```

`MERGE` keyword to create a pattern in the database. A single node or a relationship between two nodes.

- Use MERGE in this training because it *eliminates duplication of nodes*. 
- First trying to find a pattern in the graph. 
    - If the pattern is found then the data already exists and is not created. 
    - If the pattern is not found, then the data can be created.

### Executing multiple Cypher clauses

```
MERGE (p:Person {name: 'Katie Holmes'})
MERGE (m:Movie {title: 'The Dark Knight'})
RETURN p, m
```

This code creates two nodes, each with a primary key property.

![Graph9](https://ibb.co/WvqTj3D)

### Using CREATE instead of MERGE to create nodes

It does not look up the *primary key* before adding the node.

Use `CREATE` if your data is clean and you want greater speed.

## 2. Creating Relationships
-----------------------------

- The realtionship of two nodes must contains: type, direction

```
MATCH (p:Person {name: 'Michael Cain'})
MATCH (m:Movie {title: 'The Dark Knight'})
MERGE (p)-[:ACTED_IN]->(m)
```
Use `MERGE` to create nodes in the graph. Then we use the reference to the found nodes to create the *ACTED_IN relationship*.

```
MATCH (p:Person {name: 'Michael Cain'})-[:ACTED_IN]-(m:Movie {title: 'The Dark Knight'})
RETURN p, m
```

You need not specify direction in the `MATCH` pattern since the query engine will look for all nodes that are connected.

### Creating nodes and relationships using multiple clauses

```
MERGE (p:Person {name: 'Chadwick Boseman'})
MERGE (m:Movie {title: 'Black Panther'})
MERGE (p)-[:ACTED_IN]-(m)
```

- Added 2 labels, created 2 nodes, set 2 properties, created 1 relationship
- By default, if you do not specify the direction when you create the relationship, it will always be assumed *left-to-right*.

Confirm that this relationship exist:
```
MATCH (p:Person {name: 'Chadwick Boseman'})-[:ACTED_IN]-(m:Movie {title: 'Black Panther'})
RETURN p, m
```
![Graph10](https://ibb.co/rt2PVxx)

### Using MERGE to create nodes and a relationship in single clause

```
MERGE (p:Person {name: 'Emily Blunt'})-[:ACTED_IN]->(m:Movie {title: 'A Quiet Place'})
RETURN p, m
```
![Graph11](https://ibb.co/ChFHY9B)

## 3. Updating Properties
------------

### Adding properties for a node or relationship

1. Inline as part of the `MERGE` clause,adding the property key/value pairs in braces `{ .. }`
   
   ```
   MERGE (p:Person {name: 'Michael Cain'})
   MERGE (m:Movie {title: 'Batman Begins'})
   MERGE (p)-[:ACTED_IN {roles: ['Alfred Penny']}]->(m)
   RETURN p,m 
   ```

2. Using the `SET` for a reference to a node or relationship
   
   ```
   MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
   WHERE p.name = 'Michael Cain' AND m.title = 'The Dark Knight'
   SET r.roles = ['Alfred Penny']
   RETURN p, r, m
   ```

   set multiple properties, separating by comma
  
   ```
   MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
   WHERE p.name = 'Michael Cain' AND m.title = 'The Dark Knight'
   SET r.roles = ['Alfred Penny'], r.year = 2008
   RETURN p, r, m
   ```

### Updating/Fix Properties
use `SET` to *modify* the property
```
MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
WHERE p.name = 'Michael Cain' AND m.title = 'The Dark Knight'
SET r.roles = ['Mr. Alfred Penny']
RETURN p, r, m
```


### Removing Properties

```
MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
WHERE p.name = 'Michael Cain' AND m.title = 'The Dark Knight'
REMOVE r.roles
RETURN p, r, m
```
OR

```
MATCH (p:Person)
WHERE p.name = 'Gene Hackman'
SET p.born = null
RETURN p
```
![Graph12](https://ibb.co/gP2wCWH)

Delete a property from a node or relationship by using the `REMOVE` keyword, or setting the property to `null`

## 4. Merge Processing
------------

### Customizing MERGE behavior

Use the `ON CREATE SET` or `ON MATCH SET` conditions, or the `SET` keywords to set any additional properties.

```
// Find or create a person with this name
MERGE (p:Person {name: 'McKenna Grace'})

// Only set the `createdAt` property if the node is created during this query
ON CREATE SET p.createdAt = datetime()

// Only set the `updatedAt` property if the node was created previously
ON MATCH SET p.updatedAt = datetime()

// Set the `born` property regardless
SET p.born = 2006

RETURN p
```
![Graph13](https://ibb.co/znxdqx0)

### Merging with relationships

```
MERGE (p:Person {name: 'Michael Cain'})-[:ACTED_IN]->(m:Movie {title: 'The Cider House Rules'})
RETURN p, m
```
![Graph14](https://ibb.co/YXxsF0Q)

## 5. Deleting Data
-----------
It can delete nodes, relationships, properties, labels

### Deleting node

```
MERGE (p:Person {name: 'Jane Doe'}) //we first create a new node

MATCH (p:Person) //first retrieve the node
WHERE p.name = 'Jane Doe'
DELETE p
```

### Deleting a relationship

First create the node and the relationship:
```
MATCH (m:Movie {title: 'The Matrix'})
MERGE (p:Person {name: 'Jane Doe'})
MERGE (p)-[:ACTED_IN]->(m)
RETURN p, m
```

To leave the Jane Doe node in the graph, but *remove the relationship*

```
MATCH (p:Person {name: 'Jane Doe'})-[r:ACTED_IN]->(m:Movie {title: 'The Matrix'})
DELETE r
RETURN p, m
```
![Graph15](https://ibb.co/vzBbF94)

If we just delete the node without delete the relationship first, Neo4j will return an error, preventing orphaned relationships in the graph.

### Deleting a node and its relationships use `DETACH DELETE`

```
MATCH (p:Person {name: 'Jane Doe'})
DETACH DELETE p
```

### Deleting labels
Create the person node
```
MERGE (p:Person {name: 'Jane Doe'})
RETURN p
```
Add the new label use `SET`
```
MATCH (p:Person {name: 'Jane Doe'})
SET p:Developer
RETURN p
```
Remove the new-added label use `REMOVE`
```
MATCH (p:Person {name: 'Jane Doe'})
REMOVE p:Developer
RETURN p
```

### Check the labels exist in the graph

`CALL db.labels()`