SELECT id, name,
(SELECT SUM(likes) from Comment WHERE User.id = Comment.author_id) AS '#number of likes on comments',
(SELECT SUM(likes) from Post WHERE User.id = Post.author_id) AS '# of likes on Post',
((SELECT SUM(likes) from Comment WHERE User.id = Comment.author_id)+(SELECT SUM(likes) from Post WHERE User.id = Post.author_id)) AS tot_likes
From User
GROUP BY id, name
ORDER BY tot_likes DESC
LIMIT 10;