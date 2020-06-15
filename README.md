# amazon_review_recommendation

155K amazon reviews data (containing 9 columns: reviewerID, asin, reviewerName, helpful, reviewText, overall, summary, unixReviewTime, reviewTime)

Stopwords: remove those unhelpful words like i, me, etc. 

Index_dict: index dictionary to give unique index to each shingle

Main Idea: 
Using Min-hashing and locality sensitive hashing to find the approximate nearest neighbors 

Procedure:
1. keep reviewerID and reviewText columns of dataframe.
2. remove any punctuation marks, stop words(stop-words list), and converting the text to lower case.
3. remove the empty review or the review only containing stop word and punctuations.
4. pick 10,000 pairs of reviews at random and compute the average Jaccard distance and the lowest distance among all pairs. 
5. apply min-hashing and locality sensitive hashing to detect all pairs of reviews that are close to one another.
6. create a function that accepts a queried review and returns its approximate nearest neighbor (reviewerID). 
7. dump all such pairs to a CSV ﬁle and include it in the same zip folder.
