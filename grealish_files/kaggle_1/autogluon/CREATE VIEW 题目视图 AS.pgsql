CREATE VIEW 题目视图 AS
SELECT
    题目.题目ID,
    题目.名称 AS 题目名称,
    章节.名称 AS 章节名称,
    模块.名称 AS 模块名称
FROM
    题目
JOIN
    章节 ON 题目.章节ID = 章节.章节ID
JOIN
    模块 ON 题目.模块ID = 模块.模块ID;
