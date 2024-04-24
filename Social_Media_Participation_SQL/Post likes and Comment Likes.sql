SELECT 
	p.id as Post, 
	p.likes as Post_Likes,
	COUNT(c.id) as Number_of_Comments,
	SUM(c.likes) as total_comment_likes
FROM Post p, Comment c 
WHERE p.id = c.post_id
GROUP BY Post, Post_Likes
ORDER BY Post_Likes DESC
LIMIT 10;