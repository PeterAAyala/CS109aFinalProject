---
title: Predicting Yelp Recommendations
---

# T-48 hours until doomsday



## Project Introduction

>It's so easy, my grandma could do it - Pavlos Protopapas
>
>> It's so easy, my grandpa could do it - Pavlos Protopapas

Our project revoles around everything Yelp. We were tasked with creating various recommendation systems for restaurants within the yelp academic dataset, found at https://www.yelp.com/dataset/challenge. Throughout this project, we created fairly simple baseline regressions, regularized regressions, matrix factorizations, KNN, and an ensemble method to predict a particular restaurant rating based on the user, the restaurant, and many of the traits these two items have. 

## Refined Problem Statement

Which factors are most influential in determining a given user’s rating for a given business, including factors relating to the business, factors relating to the user, and interactions among these factors? Can we use all these factors to try and predict a rating based on the user and the restaurant? 

## Data

We are using the Yelp academic dataset, which comes in a package including the following json files: ‘business,’ ‘checkin,’ ‘photos,’ ‘review,’ ‘tip,’ and ‘user.’

First, we eliminated the ‘checkin,’ ‘photos,’ and ‘tip’ datasets as the information in them were either not relevant to the task of creating a recommendation system or already incorporated within the other files. ‘checkin’ had information on numbers of check-ins at a given business throughout the week, but similar information was incorporated in the ‘business’ dataset under columns named something to the effect of “good_for_late_night” or “most_popular_day_Monday.” The “photos” dataset provided no useful information-- the most useful information was the labels of the type of  photos, which were limited to ‘food’, ‘inside’, and ‘outside’ and is captured in more detail by the ‘business’ dataset in columns such as ‘ambience’. The ‘tip’ dataset provided no information without text parsing, which would have been laborious for a large dataset and not as useful as review text.

In the relevant datasets, the data was a mixture of continuous and categorical variables. Within the ‘business’ dataset, we first removed all the rows that did not have ‘Restaurant’ as a category. Then we removed the ‘name,’ (captured by business id) ‘address,’ ‘latitude,’ ‘longitude,’ ‘neighborhood’, ‘isopen,’ and the ‘hours’ columns, as they provided no useful information to determining a rating (‘hours’ may have been useful but columns such as ‘latenight’ already existed). Furthermore, we removed irrelevant attributes that were specified towards other businesses such as hair salons, and removed all but the top 150 categories of business to remove any strange misclassifications such as Restaurant/Financial Services. We flattened any columns that had lists as cells.

In the ‘user’ dataset, we removed ‘name’. We converted the list of ‘friends’ to a number of friends (rather than a list of different user ids) and changed ‘elite’ to the number of years the particular user has been elite for. We changed the “yelp since” variable to just the year (not including the month or day) for simplification purposes.

In the ‘review’ dataset, we removed ‘text’, ‘useful’, ‘funny’, and ‘cool’, as ‘text’ would require text parsing for any useful information and the others were not information on the user’s decision making but other users’ reactions to the review.




​				
​			
​		
​	