SELECT * FROM Comment c
WHERE author_id NOT IN (SELECT id FROM User);

DELETE FROM Comment c
WHERE author_id NOT IN (SELECT id FROM User);
