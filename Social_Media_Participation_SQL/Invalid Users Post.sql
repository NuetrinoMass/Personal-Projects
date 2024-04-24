SELECT * FROM Post
WHERE author_id NOT IN (SELECT id FROM User);

DELETE FROM Post
WHERE author_id NOT IN (SELECT id FROM User);