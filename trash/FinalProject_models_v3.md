---
title: Creating the YUGE dataframe
notebook: FinalProject_models_v3.ipynb
nav_include: 2
---

## Contents
{:.no_toc}
*  
{: toc}

### Importing Data



```python

import numpy as np
import pandas as pd
import seaborn as sns
import json
import matplotlib
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from random import *
from math import log

from scipy.stats.stats import pearsonr   
%matplotlib inline
```




```python
import json
with open('dataset/business.json') as f:
    business_data = [json.loads(line) for line in f]
with open('dataset/user.json') as f:
    user_data = [json.loads(line) for line in f]

    
```




```python
import copy
with open('dataset/review.json') as f:
    review_data = [json.loads(line) for line in f]
```




```python
with open('dataset/restaurant_reviews_trimmed.json') as f:
    review_data = [json.loads(line) for line in f]
```




```python
len(user_data)
```





    1183362





```python
len(review_data[0])
```





    2927731





```python
restaurant_reviews = review_data[0]
```




```python
restaurant_reviews
```





    [{'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2017-06-03',
      'funny': 0,
      'review_id': 'ByRzJ8rF2KJWLr-cUNU6EA',
      'stars': 1,
      'text': 'This place is horrible, we were so excited to try it since I got a gift card for my birthday. We went in an ordered are whole meal and they did not except are gift card, because their system was down. Unacceptable, this would have been so helpful if we would have known this prior!!',
      'useful': 0,
      'user_id': 'kzyLOqiJvyw_FWFTw2rjiQ'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2015-03-26',
      'funny': 0,
      'review_id': 'i5UwUPlQFPLcE8p2gPFwBw',
      'stars': 4,
      'text': 'For being fairly "fast" food.. Pei Wei (pronounced pay way I confirmed haha) is pretty darn good. we got a few things to share. I had the Asian chicken salad and was impressed! There was a decent amount of chicken. Some more veggies would be nice, but overall pretty good. The steak teriyaki was great as well as the fried rice. Over all good was good! Nice, clean, and reasonable.',
      'useful': 1,
      'user_id': 'WZXp9-V2dqRRJqhGgRqueA'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 1,
      'date': '2012-12-30',
      'funny': 1,
      'review_id': 'EyQyvTTg2jX4or9bB8PC9g',
      'stars': 5,
      'text': "I decided to try it out, I'm celiac and therefore can only eat gluten free...\nThey have an easy to understand GF Menu with anything you can possibly want.\n\nI placed my order online and picked the exact pickup time. I cam and my food was ready for me. Driving home the smell in my car was so good I could barely wait to get home and try it - true umami!\n\nI got home and dug into the delicious spicy chicken and rice with sugar snapies and carrots. It was superb! For $9 I will definitely try this again! I did see a huge line at the store, so try doing an online order and pickup forsure!\n\np.s. they even include GF soy sauce!",
      'useful': 2,
      'user_id': 'XylT12exfdLiI_3uDLVIpw'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 1,
      'date': '2009-01-12',
      'funny': 1,
      'review_id': 'G-EFA005besj5uHsH0sQFA',
      'stars': 3,
      'text': 'I\'m not saying Pei Wei is the best asian food I\'ve ever tasted, far from it, it\'s a fairly large chain that puts on the appearance of something more refined, it\'s essentially to Asian food what Olive Garden is to Italian Food.\n\nWith that said I\'ve always had pretty good experiences with Pei Wei, the food although not spectacular is better than some of the overcooked chicken drowning in MSG offered by some of the local chinese restaurants.  the portions are good sized, the food is generally consistent, and the prices are really reasonable considering this is a corporate chain in some cases cheaper than the local establishments. or dare I say it\'s name "panda express" (which is overpriced crap)\n\nthe time before last that I went they forgot the tomato\'s and the dressing for an asian chopped chicken salad that my wife ordered, I didn\'t discover that the dressing was missing until I got home, I immediately called the restaurant and was speaking to a manager within 30 seconds. The manager apologized and asked me If I would like to come back to the restaurant or if he could have my address so he could send me a gift certificate. I decided to go back to the restaurant, when I got the restaurant I told the person at the counter my name and they already had a bag set aside for me, the manager came over and explained to me there was another full salad in the bag, and he put additional dressing for the salad we already had, and additionally he gave me a coupon for free lettuce wraps.(which mental note: I need to use)\n\nI must say I was impressed with this manager, and it was refreshing after being in situations where a manager has taken back the bag/plate and essentially "un-F$@k\'s" your food and returns it to you.\n\nOverall great customer service, consistent food, and a good option for takeout in surprise.',
      'useful': 1,
      'user_id': 'Ji9PeffxjwqPLO7pEfSpKQ'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2015-07-11',
      'funny': 0,
      'review_id': '6PcJSGUBSLjt4VLXos5C4A',
      'stars': 3,
      'text': 'Sometimes the food is spot on and delicious and other times it is quite salty at this location.  Very difficult to get a consistently good meal.  Menu items add up quickly.',
      'useful': 0,
      'user_id': 'TLIWzAJPrET0zX4_vgvLhg'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2015-05-27',
      'funny': 0,
      'review_id': 'PFJmyZD_lNBa_Y3kbX1VvA',
      'stars': 1,
      'text': 'Decent customer service but the food was awful. It was cold and had no sauce at all. I was expecting it to be good but this place really went down hill. I will never eat here again.',
      'useful': 0,
      'user_id': 'JZEiTNWBwmv6MOOXYCAaMQ'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 1,
      'date': '2015-02-28',
      'funny': 1,
      'review_id': '_Qv1FQUToLrKMuG6pV4Gzw',
      'stars': 5,
      'text': "Super clean restaurant and friendly staff. FRESH food. Hasn't been sitting under heat lamps. NO MSG, this is the good stuff. I have to have the Kung Pao Chicken weekly.",
      'useful': 1,
      'user_id': 'E56sVQT5-OWfSejJrma8_w'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2010-04-05',
      'funny': 0,
      'review_id': 's2mlqrFNaPEGtcnEu3EJ4Q',
      'stars': 4,
      'text': "Found this the other night.  It is the PF Chang fast food option and it worked perfectly for us.  Limited menu, but lower prices. Very basic decor, but clean and fast seating.  Lettuce Wraps just as good as Chang's.  Very busy, especially the take out.  Glad to have it close",
      'useful': 0,
      'user_id': '4WYICo4emecA9r7sPYQkBw'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2015-05-22',
      'funny': 0,
      'review_id': 'oiSzZRrbi3y01_wqU528ZQ',
      'stars': 1,
      'text': "The staff here is great and they're nice,  wonderful and quick. People were ranting in raving about pei wei, I had to try it.  Even good yelp reviews.  I'm highly dissatisfied with the flavor of the food. This  should be labeled Asian inspired and not Asian. I've tried a variety of Chinese restaurants, this doesn't taste close to anything I've had at other Asian restaurants. Their Mongolian beef  was 5 pieces of beef and large mushrooms cut into thirds in a thick sauce. You eat the rice to wash off the nasty flavor. My shrimp was thickly coated in an overpowering  sauce as well.  I only ate some of the veggies that take center stage on a meat dish.  The center of my pork egg roll was cold. The hot N sour soup was a much thicker consistency almost like that of a chili instead of being brothy. Worst of all was the price.  This was not worth it to us. Neither me or my husband enjoyed either of  our dishes.  We didn't even eat half of our plates.  We even refused to take it home with us.  If you like and enjoy what typical Asian food tastes like,  don't waste your time here.",
      'useful': 0,
      'user_id': 'P8mVj7AZwJTFFH5FXbbmUg'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2011-06-15',
      'funny': 0,
      'review_id': '4BPjRE9VI0HhyZzyyYv0BQ',
      'stars': 2,
      'text': 'I had the garlic ginger broccoli chicken and it was not very good. The broccoli was hardly cooked and the sauce was way to sweet. Everything else was great. I will give them a few more tries before I write them off as another crappy Asian restaurant in Surprise.',
      'useful': 1,
      'user_id': '7Y4NEBQqWg7j-TvrQi6UZQ'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2017-03-12',
      'funny': 0,
      'review_id': 'kznHtw1Qido_9GX6sDQPJw',
      'stars': 5,
      'text': "This review is based upon consistency of flavor and great customer service.  We came and there was an unknown issue that required a 25 minute wait for food.  The employee notified us, and although hesitant, we decided to stay.  We have been here numerous times before in the past years so we are familiar with this location.  The employee was apologetic and gave us a free drink.  That was a simple gesture but rarely do you see decent customer service anymore.  We received our food and had an issue with an incorrect order.  It was explained and the issue was resolved quickly.  They gave us a free appetizer.  We do not expect perfection, nor free food.  This restaurant cares for customers and works to provide a positive experience.  We would return again because they have good food and they care.  That is a rarity in today's restaurant culture.  Kudos to the manager for creating this culture.  Ordered- fried rive and Tofu, edamame, won ton soup, dynamite chx, and Thai curry chx",
      'useful': 0,
      'user_id': 'vgZqQqe8cj6SBMH0EqDliw'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 1,
      'date': '2016-12-19',
      'funny': 0,
      'review_id': 'HWRTVn3Lc-RwN6udv4WJzQ',
      'stars': 5,
      'text': "I love this place i'd recommend it to anyone ! We always order it togo and it never disappoints! The food always taste fresh and is always ready on time! Definitely our favorite lunch spot !",
      'useful': 1,
      'user_id': 'O7G_c6wFXSygr82qs0GAcA'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2011-08-10',
      'funny': 0,
      'review_id': 'GiEB_A-m9HuX521WQNbL8w',
      'stars': 4,
      'text': '1st! Place is not closed. There was an issue with legal workers in the franchise chain. Now they are back! Food is one of the best! Especially their chicken mango! Daughter is a huge PF Chang fan, so she loves this place also! Her favorite is their Mongolian beef, and noodles! They are always coming up with something new, so you may have to try them more then a few times.',
      'useful': 1,
      'user_id': 'UG4EKu13JRwzRix6ESINdg'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2013-06-17',
      'funny': 0,
      'review_id': 'GKi4i6qocIgaYcwv1_0zzQ',
      'stars': 2,
      'text': 'Definitely not a fan. Coming from Orange County, CA, I have high expectations when it comes to Asian food.  Suffice to say that Panda Express would have been a better choice, which is pretty sad.',
      'useful': 0,
      'user_id': 'ZZG6yR27lIy3xwUYVgHO7w'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2015-11-03',
      'funny': 0,
      'review_id': 'OrhWg2MmCznwfKfjHKvuhA',
      'stars': 3,
      'text': "Pretty good, not great. Definitely overpriced. It's barely a step up from Panda and not worth the price difference. Not saying it was bad, but $25 for two basic dishes is a bit much and they didn't make the order how I requested.",
      'useful': 0,
      'user_id': '1YorWW0Z-YDuYC5GplNabw'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2017-07-08',
      'funny': 0,
      'review_id': 'QXWku_OB3FCj9VCZfUZwwg',
      'stars': 1,
      'text': 'I wish I could give 1.5 stars. Nothing special. Lack of flavor. The entrees were either sweet or spicy. The crab Rangoon were.....different. The filling had a mealy consistency. Friend rice was bland. Plenty of other places to spend 50$ on takeout. Save your money.',
      'useful': 0,
      'user_id': 'ujOPJEz_KxzAyZDnji-2Ng'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2015-09-22',
      'funny': 0,
      'review_id': '5NtaW5EwXK595kP4Ynnisg',
      'stars': 2,
      'text': "Disappointed that on yelp their hours show them open at 1030am but when you arrive they don't open until 11am. Waiting in the car to pick up a large order makes me think twice about catering from here. Don't get me wrong the food is great as always but they need to fix this.",
      'useful': 1,
      'user_id': '6aEUn50d3Ts7MiGu6WdpKA'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2012-10-08',
      'funny': 0,
      'review_id': 'ai6O4UqqDqnjO7gfz6jBkA',
      'stars': 3,
      'text': '1st visit had the lo mein...delish!  \n2nd visit had the teriyaki...fine if you like salt.  Ick!',
      'useful': 0,
      'user_id': 'R6vb0FtmClhfwajs_AuusQ'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2015-12-28',
      'funny': 0,
      'review_id': 'ZrvsD7PSyPolII3gp4-uHw',
      'stars': 5,
      'text': "As a vegetarian, it can difficult to find places with good options, but I love getting a tofu bowl at Pei Wei. The first time I had it, I had to double check because it looks and has a similar texture to meat, but it's great. I love going to Pei Wei",
      'useful': 0,
      'user_id': 'CPuUagT2rfUJm6hRgxn3JQ'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2009-11-18',
      'funny': 0,
      'review_id': 'p7OqbXTjwmIN_XYohB6TFw',
      'stars': 2,
      'text': 'Typical big business chinese place. Slightly fancy but very average food. Kinda like a fast food place.... you walk up and order then get your own drinks etc. Portions are very small. Prices are higher than average.\n\nThey offer a few different vegetarian options but it all tastes pretty bad\n\ndo yourself a favor and also support a local business and eat at big buddha around the corner. Its much better food',
      'useful': 0,
      'user_id': 'OYRBjBWy1uOm12N3cokS_Q'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 2,
      'date': '2012-07-29',
      'funny': 2,
      'review_id': 'ukpjwnetF5wGhGrSXmPRwA',
      'stars': 4,
      'text': "I love Pei Wei since it's just a bit more affordable than PF Changs.\n\nThis is the first restaurant in the chain where I found patrons tipping the servers. Although I don't mind leaving a dollar when it's due I find it hard to do when they don't do very much, especially since it's more of a counter service experience.",
      'useful': 1,
      'user_id': 'PKZLwAGgBtQCjJtGhyPETA'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2016-06-19',
      'funny': 0,
      'review_id': 'mT6U5lujK_zIcIqux92seA',
      'stars': 4,
      'text': "Great fresh food and clean restaurant. Friendly and very accommodating staff. I'm a frequent visitor. This is a sister restaurant to PF Chang's, but without the formality or higher prices. Great place to eat if you want the same style foods as PF Chang's, but a more laid back environment without the need to dress up or spend more.",
      'useful': 1,
      'user_id': '9bJ6j0zrV1XSiSnzQWM5Tw'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2012-08-23',
      'funny': 0,
      'review_id': 'YxAxExTdWtdMhEb14RGFRg',
      'stars': 3,
      'text': "This is pretty good. My wife and I usually split lettuce wraps and an entree of some sort. It's convenient to be able to order take out online. Overall, if I'm really in the mood for chinese food I'm going to make the drive a little further out to Big Buddha on Greenway and the 303, but this is a good substitute as it's about 1 minute from my house.",
      'useful': 0,
      'user_id': '8nCmV4RMwf4GpaN-A_2Tfw'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 1,
      'date': '2009-06-25',
      'funny': 1,
      'review_id': 'ue6ts-gA9khywe76lEL1Kg',
      'stars': 4,
      'text': 'Food is good and consistent and the service is always great. I\'ve never dined in, but get take-out fairly often. I love being able to just say "same as last time" when I call it in. Not as good as PF Changs, but certainly better than most other Asian-ish places in Surprise. Be aware that they use shitake mushrooms is almost everything, but they are fine with custom orders if you don\'t want them. All in all, it\'s not my favorite place, but I always enjoy it every time.',
      'useful': 2,
      'user_id': 'tbAQMMVlhxvXhe6KifrZ-A'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2008-10-06',
      'funny': 0,
      'review_id': 'WsTYqsyNyUd7xpwFrgwI0g',
      'stars': 4,
      'text': 'The hubs and I dined at the Glendale location about two years ago. i guess we were hoping it was like PF Chang\'s....but unfortunately were dissapointed.\n\nWe discovered the Surprise location a couple weeks ago and decided that tonight we would grab a quick bite. The hubs ordered Mongolian Beef (his fav at PF Chang\'s) and I ordered my "tried and true" Ginger Broccoli and Chicken." Of course we started with the Lettuce Wraps because honestly...they are just the most delicious things ever!\n\nTonight we were pleasantly suprised that the food was quite delicious. Although the dishes are EXACTLY like what you would get at PF Chang\'s they are scrumptous in their own right. The only downfall is that the noise isn\'t conducive to conversation.  I haven\'t been to an establishment this noisy since Buca de Beppo\'s...which I love (don\'t get me wrong)!  But I will say (on a positive note) that the customer service is fabulous! Something that can be hard to come by at times...\n\nAs for price....$28 bucks for the two of us including an appetizer, two entrees and two drinks. (coke and iced tea...with free refills) Seems a bit pricy to me but honestly....EVERYTHING is expensive now days. \n\nOverall, try it out...they offer take out too so you can enjoy great food in the comfort of your home....without all the noise! (or they have a patio you can dine from)',
      'useful': 0,
      'user_id': '1s0Q1KwGpJIKvD-SRSpwjw'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 1,
      'date': '2015-03-17',
      'funny': 1,
      'review_id': 'g9yv-M3kvOQFPyJHCjUrdg',
      'stars': 3,
      'text': 'Very clean and staff is always friendly. I usually order the honey seared chicken but decided to switch it up. I ordered the Thai dynamite and it had a weird, almost chemical taste. I will go back but will stick to my usual order.',
      'useful': 2,
      'user_id': '-0kiduTUToVYFqN_NEqMSw'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2014-04-22',
      'funny': 0,
      'review_id': 'fk8OI26dAhQfot46T_SXWg',
      'stars': 1,
      'text': 'If I could give no stars I would. My family placed an order to be picked up and were told to be there at 7:20. We get there and I sit waiting for the food until 8! Nobody acted like it was a big deal when I told them and just said "oh.... Well we are running behind." Um okay?! We get home and they forgot to put the crab puffs in my bag. Didn\'t bother going back and moved on to our meals. My honey glazed chicken was average, edible. My husbands sweet and sour chicken was so soggy it was disgusting! Couldn\'t even eat it. My daughter loves fried rice and wouldn\'t even eat hers, it was so greasy. We all ended up eating PB&J\'s and it was 100x better than anything we brought home.',
      'useful': 0,
      'user_id': 'KrQ_dWOBn2voaQLNv7hj8A'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2016-05-07',
      'funny': 0,
      'review_id': 'ld06EwR5YyutWxud9ude0Q',
      'stars': 5,
      'text': 'I went to Surprise Pei Wei Restuarant tonight on Bell.  This was my first visit to a Pei Wei.  I was very impressed with the atmosphere, food and staff.  The chicken teriyaki with fried rice was DELICIOUS.  The spring rolls were delicious too!  Your Pei Wei location is my new fav restaurant..',
      'useful': 0,
      'user_id': '2L_1kyJDOkEaPDiBRohx9w'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 1,
      'date': '2014-05-03',
      'funny': 0,
      'review_id': 'CxVx2FD73kw--RuzXL_RVA',
      'stars': 4,
      'text': "I like Pei Wei overall and enjoy their varied menu selections.  The food is good and fresh.  Sometimes it's hard to find a place to sit though.",
      'useful': 1,
      'user_id': 'xlkjaJUu2fVojeaaVgQPOw'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 1,
      'date': '2014-05-29',
      'funny': 1,
      'review_id': 'jJaU6pbKkYgl15P-5zfg3A',
      'stars': 3,
      'text': "Everything was good except Thai chicken wraps. Too much cilantro, absolute overkill. I think long stems is just yucky, cut them off!! Korean beef was ok but just missed the flavor I'm used to in this dish. Service is always spot on.",
      'useful': 0,
      'user_id': 'alTlRb9qMBX11pARX05Big'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2009-12-10',
      'funny': 0,
      'review_id': '7M2GCeIba1uTJMJbVO7TKw',
      'stars': 1,
      'text': "I really don't understand how anyone can eat the food from here.  Granted, I am Chinese and enjoy authentic food, however I like good Chinese American food.  For someone who usually doesn't leave food on her plate unless it is really bad, I could not finish my meal because it was tasteless and below average.",
      'useful': 1,
      'user_id': 'e--whH51bx5mDaMo3aJ-hg'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2014-10-01',
      'funny': 0,
      'review_id': 'pxs5biP-IzmjXZpoh-iVHg',
      'stars': 2,
      'text': "Went in for lunch with a coworker today. The small portions are overpriced and really aren't small-half would've been more than enough. \nWe were served quickly and the staff was friendly! The restaurant was clean and tables were bussed quickly. \nI was disappointed with my Mongolian chicken. It tasted more like Teriyaki and was swimming in sauce. I had at least a 1/3 cup of sauce left on my plate after eating my food.\nAll in all, PeiWei's food will be why I don't return anytime soon even though their staff was excellent and so was the atmosphere.",
      'useful': 0,
      'user_id': 'Ns0hZ0xDOuVRMpHk0Q-5Yw'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2015-07-13',
      'funny': 0,
      'review_id': 'DW0B2tlav4Z7T1DTzAL7kQ',
      'stars': 5,
      'text': 'The food is great and customer service is the best! The Dan Dan noodles are dynamite but they come standard pretty spicy...  I placed a pick up order by phone and asked for them to be mild, but they ended up being  spicy!  When I got home and tried to eat them they were too spicy.  I called in and spoke to the manager and he took care of it completely and replaced the order for me at no charge!',
      'useful': 1,
      'user_id': 'SM20gx7YH0GtI5JOXhfXdg'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2016-02-14',
      'funny': 0,
      'review_id': 'nI2rhDM2CgNazMdhiODoRQ',
      'stars': 2,
      'text': "OH MY GOD, this place. I used to like Pei Wei until I moved to Surprise and this location ruined it for me. They consistently screw up on orders, have huge wait times, and run out of key ingredients on an alarmingly regular basis. We just spent an hour waiting for a take out order (kung pao veggies and vegetable fried rice), which was twice the time they quoted us, only to be told they were out of WHITE RICE (how does a chinese restaurant fail to foresee the possibility that they might serve a lot of white rice on a Saturday night?!). I may have chosen to stick around the extra 15 minutes if they hadn't already had to re-make half my order after making it incorrectly the first time. When I change up an order even slightly to accommodate for allergies/dietary needs, they have to re-make it at least 75% of the time.\nIf the food was consistently delicious, I'd rate it higher, but honestly, it's only okay.",
      'useful': 0,
      'user_id': 'R-kL1bocHgP4GW7Mgd-ZXA'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 6,
      'date': '2015-02-12',
      'funny': 5,
      'review_id': 'sb7iYaCc6ggpShtElkcwiw',
      'stars': 3,
      'text': "My wife wanted to try this place for awhile now.  One of her brothers friends used to be a manager of one in Colorado too.  Tonight we decided to try it out.\n\nI placed my order online.  Easy process.  My wife went with an order of Chicken Lo Mein and I went with Sesame Chicken/Steak with fried rice as well as an egg roll and 2 pot stickers.  Total for everything was $24.80 after tax.  I drove down and picked up my food.  The manager their was friendly, rang me up real quick and I was out the door in no time. This bag was heavy as hell too!   Here is how everything tasted.....................\n\nChicken Lo Mein-  Only had a little of this.  Good noodles, good white meat chicken.  My wife liked it.\n\nSesame Chicken/Steak-  I decided to try out 2 kinds of meat.  They let you mix and match which is cool...they charge you for the highest priced item.  I substituted fried rice for white rice (.99 cents extra).  This was touted as soy citrus glaze, red bell peppers, onions, scallions, ginger, garlic and chile paste and sesame seeds.  It was just alright.  The flavors didn't really do much for me.  It was a tad spicy from the chile paste but overall it was kind of bland.  I had some pieces of steak/chicken that were a bit funky too.  I would definitely try out a different dish next time here...if there is a next time ;-).  The fried rice was good!\n\nEgg Roll- Eh.  It was an egg roll.  Nothing fantastic about it but it tasted fine.  It came with a sweet mustard dipping sauce which was pretty tasty.  Usual fillings of pork, veggies etc.\n\nDeep Fried Pot Stickers-  I got 2 of these things.  They were good.  Small but good.  Straight pork filling in the middle of them.  \n\nOverall it was just alright.  I will say that they give you a GRIP of food.  The 2 entrees we got could have easily fed 1-2 more people.  For $24.80 it was a great value.  The food just wasn't the type of thing that  you would ever crave and keep you coming back though.  While I was eating my meal, all I could think of were better Chinese food places I have had before....wishing I was eating them instead.  It filled my belly, service was friendly and the value is definitely there.  I'll probably give them another try and order some other dishes to see if there is something I really like.  We shall see............",
      'useful': 7,
      'user_id': '6jz_Yr6_AP2WWLbj9gGDpA'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2009-01-24',
      'funny': 0,
      'review_id': 'j51qEbi4hMm6WHGlkt57xg',
      'stars': 2,
      'text': "Haven't eaten at this location in particular, but in general the prices are high for what you get.  It's nothing special and any other Chinese place in the city can probably beat it in the price/quality area.",
      'useful': 1,
      'user_id': 'UXZDRVdx8eJqdqb13Bcfcg'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2014-08-28',
      'funny': 0,
      'review_id': 'fKCizCcgm5lrdh0fxnOplg',
      'stars': 4,
      'text': 'Always on point. I get the Carmel chicken.. The crab wontons are a guilt must have. And I love there  tea too. Ate her for two years and always pleased. Thx!',
      'useful': 1,
      'user_id': 'EKTCccgwn9MAIDNSsDwuIg'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2015-11-01',
      'funny': 0,
      'review_id': 'TGG8n3Cb3klpPdamJ88owA',
      'stars': 5,
      'text': 'I love Pei Wei. The food is delicious. The atmosphere is chill. The restaurant was clean and staff friendly.',
      'useful': 0,
      'user_id': 'rHP3q9Ok1qu9_tJmIy4i9w'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2010-01-10',
      'funny': 0,
      'review_id': 'hQCBUqXmVB2QuQ2MVv3Apg',
      'stars': 5,
      'text': "Came here with my friend today cause we were starving and decide to go whichever next place we saw. And we were very surprised at how pretty the inside of the restaurant looked. The decor was amazing. I ordered the Japanese teryaki bowl vegetarian style and it was amazing. I loved the tofu and the brown rice and the veggies. My friend ordered the lo mein and she loved it. We didn't finish due to the fact that the portions are huge. (we didn't know it was a like share the bowl type of place, and when we figured it out we didn't do it because hers had meat). But overall this place was amazing. The employees are very very nice! And this place is vivid and just amazing. WAY better than panda express. I will definately come here again, and recomend it to everyone!!!",
      'useful': 0,
      'user_id': 'bwElUOvj3cIjMp2qp4vjeQ'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2010-07-04',
      'funny': 0,
      'review_id': 'P8h9lk3qwwzaYgM4ueCeqA',
      'stars': 5,
      'text': "Very consistent food. Definitely a second best to P.F. Chang's.",
      'useful': 1,
      'user_id': 'C38sl6tI_DVrpL5sg4Kmfw'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2015-10-29',
      'funny': 1,
      'review_id': 'LP1OdCHa7DMVpbLhB4dgMg',
      'stars': 3,
      'text': "Service is fast and on time. Love the online ordering-very convenient!  Food is always hot and delicious. However...NO FORKS FOR TAKE OUT?  That is poor management. Either you didn't order in time or there was a problem with the order...either way, hustle your booty to Walmart and buy a couple boxes: Unacceptable!",
      'useful': 2,
      'user_id': 'cs6biPoaG9SmtVB44hU9Sw'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2013-06-21',
      'funny': 0,
      'review_id': '_laP3dAOE--w6JMCG4HFjg',
      'stars': 4,
      'text': 'The food was fantastic! I highly recommend  the lettuce wraps.',
      'useful': 0,
      'user_id': '1fMDrDfB3IHPCka9SeSHsA'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 1,
      'date': '2010-07-02',
      'funny': 1,
      'review_id': 'RMuW5m75o_5RgL2mTMfg2g',
      'stars': 4,
      'text': "Then we went to dinner at Pei Wei. Not bad for an Asian fusion place. Filled with senior citizens, but that's not all that surprising. I had the Mango Thai Shrimp, apparently, which was good. Right amount of spice in the dish. Not too heavy, either. Steve and Chris had their own plates as well, both of which looked appetizing. Not crazy about the moderate wait nor the fact that strangers came up to the table with unnecessary parenting advice for Steve and Chris.",
      'useful': 2,
      'user_id': 'zLt0Qu_98ZGObYBTcG7jRQ'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2012-09-11',
      'funny': 0,
      'review_id': '9-mx4vAByTMxJDrLfNZZbA',
      'stars': 4,
      'text': 'We use them to cater family get-togethers. Love the vegetarian dishes, especially the baked tofu- yum!  The Mongolian beef is a favorite with family.  When we dine in we always get the lettuce wrap.',
      'useful': 0,
      'user_id': '33h6CGtB9MSYHi0-kbXMhw'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2011-02-27',
      'funny': 0,
      'review_id': 'FxAxYxj5SbiOtcK36bJxCA',
      'stars': 1,
      'text': "We went as a family of 6 and was treated very poorly by the manager at the  place. The food was greasy (I guess most Americans would love it since they don't see the difference) The manager had the audacity to compare this restaurant with PF Chang which is equally greasy and bad. I wouldn't recommend this place to any health conscious people. To top it all its expensive. Go to taco bell or China Panda and save your frustration and your appetite. Just being fancy is not the attraction motto for me. Maybe they will train their managers to change the attire of the restaurant. I wont waste my time here.",
      'useful': 0,
      'user_id': 'mCE5jUDzP-ihU_6cwQsiiw'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2016-05-31',
      'funny': 0,
      'review_id': '0sRld5Hk0O6JUGKpPlWgPQ',
      'stars': 5,
      'text': "I love their food. I have gone there multiple times and still think it's great. I especially love their soda machine.  The Mongolian Beef dish is my absolute favorite.",
      'useful': 0,
      'user_id': 'gRhtEXoxVpaccgvr3OkC0A'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 1,
      'date': '2015-01-21',
      'funny': 1,
      'review_id': 'g9IdtJr5oVv4Df0nY_IPng',
      'stars': 2,
      'text': "We are winter residents and always look forward to eating here when we return.  Sorry to say we were quite disappointed with our Pad Thai orders today, so much sauce that it resembled a thick soup.  The egg ingredient seemed to be overlooked and was in big slices rather than smaller curds,  when I asked a server about the sauce he replied that it wasn't really sauce but that the rice noodles were too wet.  Either way, it was not a pleasant dish today.",
      'useful': 2,
      'user_id': 'ljGTQD_-yC4iXievtgCsbA'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2009-10-06',
      'funny': 0,
      'review_id': 'i_g131Z2mXHBSSm5erJZbw',
      'stars': 3,
      'text': 'Very tasty Chinese food!  The prices are reasonable and serving sizes are generous.  I really enjoyed the Mongolian Beef!\n\nDont forget to grab a fortune cookie from the bowl on your way out!',
      'useful': 1,
      'user_id': 'faaOI6hU64h6SSaF0f11eg'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2010-10-04',
      'funny': 0,
      'review_id': 'pta_2pIbHJhx_XM1-rIHzw',
      'stars': 5,
      'text': 'All around great flavor and portions. Perfect for take-out always consistent the fam eats here about three times a month never had a bad meal.',
      'useful': 0,
      'user_id': 'd6om2E23bPoZFWzLss-rrA'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2008-12-02',
      'funny': 0,
      'review_id': 'Ty2JjQqEZrmbSO7eFwlyiA',
      'stars': 4,
      'text': 'The vegetarian options are pretty good, especially if you like tofu. The restaurant was out of napkins the last time I was there and the staff was friendly. I would recommend the take-out option versus dining in though.',
      'useful': 0,
      'user_id': 'O2U5kLpXDY-xjDppJ7d_Nw'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2016-02-13',
      'funny': 0,
      'review_id': 'hZsxP7BZi8seJQRKw6hxEQ',
      'stars': 4,
      'text': "Pretty good food a chain. You do get a lot of food for the price. However, every time I have ordered online there has been a mistake. Sometimes they tell me they never even received my order when I received the confirmation email. Also, if you in during a busy time and have a teenage girl as a cashier, you're chances of having a pleasant experience with her are slim to none. This has happened to me multiple times with multiple different cashiers. I have spoken to the manager and he is always very apologetic, but I guess you just can't get stressed teenage girls to be nice to customers. I will continue to go here as the food is good for the price and they are relatively quick with preparation. Nice options if you're veggies as well.",
      'useful': 0,
      'user_id': '5zkRD7dv8GqysVCvwPr-Ew'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 1,
      'date': '2012-07-08',
      'funny': 1,
      'review_id': '1GVnTlLTCeR6Ew8z2D1EFw',
      'stars': 5,
      'text': "What can I say, I'm a sucker for Pei Wei.  When we lived in California, I heard the place was shut down.  Glad to move back and see it open.  \n\nTake out rocks, call ahead.  They remember your phone number and name, so when you call they already know you.  They also save your last order to ask if you want the same, or to add anything to it... Nice touch.  I think I eat there too much! \n\nThe only thing I don't like, and I think I'm the only one who doesn't like it, is the soda machine.  Only because I always get a soda after someone has pumped out some nasty combination of orange soda and root beer, then I can taste it in my coke.  Wish they had some filter system.  Otherwise, I could eat here daily.  Find me here at least once a week! :)",
      'useful': 2,
      'user_id': 'Ov-0pRQxz5fTTElFE-WtTQ'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2013-05-23',
      'funny': 0,
      'review_id': 'lHZziOQQYBGuyx11nCGSnA',
      'stars': 2,
      'text': 'Took the family there last week.  Ordered fried wontons, sweet and sour beef, Mongolian beef, honey crusted chicken and pad Thai noodles.  \n\nThe Mongolian beef was delicious, however the white rice was undercooked, some of the rice was hard still and crunchy.\n\nHoney crusted chicken also quite good, same story with the rice.  Same thing with the sweet and sour beef.\n\nOrdered the Pad Thai without peanuts... Son is allergic to peanuts so we made it very clear we didn\'t want anything with nuts.  They deliver the pad Thai with peanuts all over the top.  Sent it back and they fixed it.\n\nThey brought us spring rolls which we did not order.  We said they were not ours we ordered wontons.  They bring us the wontons a few minutes later.  The wontons are less expensive than the spring rolls, we point that our to the guy at the register and with a bit of smart-alecky-ness says that he "let us keep the spring rolls" and "didn\'t charge is for all of our drinks" as if he did us a favor.\n\nOverall the food was decent, the rice was terrible and the attitude of the cashier was not what I expected with a $65 food order!!\n\nI wouldn\'t go back for that price and food quality but perhaps we caught them on a bad day..',
      'useful': 0,
      'user_id': 'QZmG0pJlPQUq0PudwDKI5g'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2017-04-07',
      'funny': 0,
      'review_id': 'hUnVNj83RehIogPzxKkR9w',
      'stars': 5,
      'text': 'I love the food and the containers it comes in. We eat there sometimes and also carry out. My favorite is ginger beef broccoli! Thank you for the emails especially. I really appreciate it',
      'useful': 0,
      'user_id': 'iSWxqsVgkzn_I3t5pAw66w'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2016-03-17',
      'funny': 0,
      'review_id': 'wmoivnocytzFnsJaB-5RIA',
      'stars': 5,
      'text': 'Pei Wei has become a favorite "take out" spot for both myself and my girl friend.  We drive a little further than the local Taco Bell - KFC combination fast food stop, but to eat healthier it is certainly worth our time. \n\nYesterday, after our Silver Crafting Class, we drove the 2.4 miles and not only enjoyed a relaxing healthy lunch, but then purchased "take home" orders so we would have our evening meal in the refrigerator waiting.  (I brought my husband home an order of his favorite Kung Po Chicken with brown rice.)  \n\nIf you haven\'t give Pei Wei a try, stop in some time. I believe you\'ll enjoy it -- that is, IF, you enjoy Asian dishes.',
      'useful': 0,
      'user_id': 'DAIpUGIsY71noX0wNuc27w'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2015-08-08',
      'funny': 0,
      'review_id': 'ZUKo5tY7ZXqzF8-Uu0R8Rw',
      'stars': 4,
      'text': "It's just above average.  Portions are decent and cost was ok.  Got in and out fast by phoning our order in.  Good was ok, it didn't floor me but it didn't make me not want to there again.",
      'useful': 0,
      'user_id': 't9lwePmlZ7Sl_wiw9SFSrQ'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 6,
      'date': '2008-11-27',
      'funny': 4,
      'review_id': 'l8cGGc6AaHCj8XomkYRzAw',
      'stars': 2,
      'text': 'In Surprise this is probably the best Chinese offering which isnt saying a whole lot. I really wish there was an awesome offering out there in terms of Chinese food.. Mexican food, oh yeah, some very good places mostly but Asian, youre better off staying at home and cooking. \n\nWe ordered the Asian Chicken Salad, chicken was old and had that old gross fridge taste.. Dan Dan Noodle, the noodles were over cooked and stuck together as one large soggy noodle.. Curry with Tofu, the curry had a weird perfume taste.. the honey chicken was pretty good as well as the hot and sour soup though neither made for good leftovers. \n\nWith Surprise having so many chains it wasnt surprising to see one of these finally opening up.. a far better choice than Panda Express.. kind of sad, huh?',
      'useful': 8,
      'user_id': 'p_azadim_uWFOXAhhKB3ag'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 3,
      'date': '2017-06-10',
      'funny': 1,
      'review_id': 'kAA5n5ie00OFi0w0K53WSA',
      'stars': 3,
      'text': 'Pei Wei is the fast food version of PF Changs. There\'s a long and interesting story surrounding Pei Wei and that of PF Changs. Let me just say that, a man named Paul Fleming (PF) and (Flemings) is responsible for many of the restaurants that you and I frequent. His ex wife and him went through a bitter divorce her nickname is (Flow) and she operates "Flows," Chinese restaurants. Flow is Chinese and married another Caucasian man 20-25 years younger than her, who speaks fluent Chinese. Anyways, Peiwei\'s in my opinion is overpriced for the small amount of you receive for the consideration you give. In other words, there\'s no value for your consideration. Yes, I do enjoy the PeiWei Spicy chicken, extra spicy with water chestnuts, but at $12.53 for the water chestnuts and extra chicken, it\'s just not worth it. The rice is almost always dry and is cooked incorrectly in massive amounts, as well as being a long grain, not short grain rice. NOT WORTH THE MONEY!!!',
      'useful': 5,
      'user_id': 'GG9tAoEC9sMa_DokSkkImA'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2013-04-26',
      'funny': 0,
      'review_id': 'k53Lj6HucbPbqZWYNFoBsg',
      'stars': 4,
      'text': 'I love the lettuce wraps here. They are simply the best.  I order carry out often from here. I occasionally dine in.  My problem with the dinning in is the seating. It is always difficult to find a seat.  The place really needs to expand.  If you have a party of more than 4 most days you can hang it up.  Other than the seating I like it. Large portions of rice and noodles.  A little stingy on the meat.',
      'useful': 0,
      'user_id': 'cPfnF9PAvVjamEdZM3wTzA'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2014-12-27',
      'funny': 0,
      'review_id': 'ULJhNKA4iikk9fVyvfPHqA',
      'stars': 2,
      'text': 'Just dirty ... Sometimes I wonder why we come back here. At the register the girl handed me a cup. There was particles of food on the inside. I handed it back and said "It\'s dirty, I\'ll take a different one." She put the dirty cup back with the other \'clean\' cups. I stopped and restated "No. It\'s DIRTY, you probably shouldn\'t give it to someone else." (Shaking my head)',
      'useful': 2,
      'user_id': 'MFKjHdwguK0kDzNteF07_A'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2017-04-23',
      'funny': 0,
      'review_id': 'En6K7NF2Sqxjc3NKLjx0cw',
      'stars': 5,
      'text': 'Hot,vegan,carbo load \nYou name it and they have it\nPricing is reasonable  and quality excellent \nNo disappointments here\nFive stars for the food \nWe try to stop by monthly',
      'useful': 0,
      'user_id': 'e3S63222bmnlIYLi2DCeQg'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2016-03-02',
      'funny': 0,
      'review_id': 'uiSS0Hd9L4krCvwX0jMugg',
      'stars': 5,
      'text': "I eat regularly (3-5 times a month) at Pei Wei, usually in Scottsdale or North Phoenix. My wife's doctor is across the street from this Pei Wei, so we have been stopping in when we go to her baby check ups. It is, without a doubt, the best Pei Wei I have eaten at. It is consistent, they don't get frustrated if you customize your order, and the cooks on always on point. \n\nWish this location was closer!",
      'useful': 0,
      'user_id': 'GjxM-LmQcta13_DiUniVgg'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2017-06-12',
      'funny': 0,
      'review_id': 'z30Tv_MrQzEow6WjLFmtXQ',
      'stars': 5,
      'text': 'This Pei Wei location has such kind, welcoming staff. They are so accommodating and hospitable. 5 stars, hands down.',
      'useful': 0,
      'user_id': 'C_DEhSnhqFbtJcPb39FALA'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2009-01-19',
      'funny': 0,
      'review_id': 'EvlwP4hCwnFdQyUvC6RFVQ',
      'stars': 4,
      'text': "Pei Wei is a fast, easy, cheaper PF Chang's - and that's what makes it great! If you like the tofu dishes at PF Chang's, you'll love Pei Wei. The food here is great. The service is exceptional. And the portions are huge! I usually end up taking about half the food home with me. Also, compared to other lunch spots near by (like Paradise Bakery), it's really not all that busy. Of course, it's not authentic Asian food, but you shouldn't be expecting that when you come here anyway.",
      'useful': 0,
      'user_id': 'wsDiT-IGGM8tvlaMwf1zwQ'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2010-12-29',
      'funny': 0,
      'review_id': 'mgNBEZoEznL5VYGAUmyAWQ',
      'stars': 1,
      'text': "The problem may be that I lived in the Bay Area for 30 years. There, bad restaurants expire in less that a year.  Pei Wei wouldn't last very long.  We had the Mongolian Beef and Kung Pao chicken.  The beef was in a salty, sticky sweet brown sauce that rendered it almost inedible.  The Chicken had little spice or sauce, although the pea pods were fresh.  Think Panda Express. Food is not serve family style. Food is ordered at the register and brought (eventually) to your table.",
      'useful': 2,
      'user_id': 'LSNwKpMPDsTjaAZEq5ud2Q'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2016-08-28',
      'funny': 0,
      'review_id': 'UQ68U4Z2JAhuBMOrAv1AGA',
      'stars': 1,
      'text': 'First time ever giving a review but felt it was necessary. I\'m just gonna go all in and say it... Pei Wei sucks ass. It seems to be subjective, I know. But in comparison to legitimate Asian food, wheather that\'s credible Thai, Chinese, Vietnamese, etc., I\'m not sure how anyone can make a compelling argument that Pei Wei is "good" food.. The crab wontons are usually a sure bet at any Asian restaurant but for some reason it just always sucks ass here. Fishy weird taste that\'s contrary to what they should taste like and the sauce is garbage. Teriyaki Chicken was bland and shitty. Pei Wei Spicy dish was average at best only cuz of the sweet sugary sauce that masks the overall flavor. Now my girlfriend and I just feel like shit.. Why? Cuz it\'s shitty food. Also saw on the local news that the location on Glendale and 7th st. had 4 health code violations. I\'m officially done with this place. Spend a couple extra bucks and go eat some authentic Asian cuisine at a family owned restaurant that actually knows how to cook real food.. Not some corporate food chain bullshit.. Thank you for reading and bless your heart.',
      'useful': 0,
      'user_id': 'd5xWk1GosWpEP7pGfLzh9g'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2015-11-07',
      'funny': 0,
      'review_id': 'wFP3lpVdl03_NxZoXL4Pzg',
      'stars': 2,
      'text': "The wait time for the food was way too long- 30 minutes for a bowl of spicy noodles with some veggies. I'm a guest from out of state but this is nothing better than my local Panda joint. They should probably let the customers cook the food cause we could do it faster. \nThe food was okay but the portions were a bit big. While I was waiting for the food (+30 min) I noticed a lot of people just ate a few bites and left the food on the table. So much food wasted- there must be a reason for that right?? \nThe bathrooms weren't clean, barely had soap no towels and toilet didn't even flush!! \nI hope next time I'm down here that it isn't this bad.",
      'useful': 0,
      'user_id': 'F783ardridpj3owUMHg-Fg'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2010-11-19',
      'funny': 0,
      'review_id': '0udTI0_Iy32PVH5ASUjfpA',
      'stars': 4,
      'text': 'I ordered the teriyaki bowl and it was great food for the price.  Quite impressed by the variety of options on the menu. Not sure if this location was the one I tried.',
      'useful': 1,
      'user_id': 'hxhKA0q9jLxTfUdg1LM9Bg'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2015-03-15',
      'funny': 0,
      'review_id': 'ylXGqKJWJkwCwO-mhKh12A',
      'stars': 5,
      'text': 'Absolutely wonderful. This place is clean, the staff is wonderful and go out of their way to help you. They stay a step ahead of you and get what what you need. \n\nThe food is perfect! I have eaten here many many times and it has always been fresh, hot, and delicious! I highly recommend this place, 100%!',
      'useful': 0,
      'user_id': 'IFBx9nPxyomWzj_2prR6bg'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2011-09-15',
      'funny': 0,
      'review_id': 'mUgMtcicQR39CJX9FCchTg',
      'stars': 4,
      'text': "Service is a 4  . Very sloppy also the hot and sour soup aint that  bad if u add enough chile paste and salt and soy sauce the gentleman named pop he is the manager great service from him and this other lady did not get her name unfortunetly  I've changed my rating 2 times that's how impressed I am",
      'useful': 0,
      'user_id': 'RhDwmkLWmC1SGF0VPV7xYQ'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2014-10-27',
      'funny': 0,
      'review_id': 'nsUHkKb3vIdwd0x7egzBUQ',
      'stars': 1,
      'text': "Love the food but it's great at all locations. The customer service is horrible at this location. Only two tables were occupied at the time we walked in for a carry out. We walked in and they looked at us like.....really? Go away!! Someone walked up to the register and he just looked at me. I asked him if he was ready to take my order and he just kept looking at me like I was wasting his time.  No thank you, nothing. Just a hand to take my money and then a bag in my face so I would leave and get out of their faces. Really???",
      'useful': 0,
      'user_id': 'CnSzFGho_Ofsda3DOg0cIA'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2017-03-06',
      'funny': 0,
      'review_id': 'M6qZU16SjXoyoLrmNENhQg',
      'stars': 5,
      'text': 'We are from the Midwest and are visiting Phoenix on a vacation. We stopped in this place for some quick Asian food. What a great surprise! I had the Mongolian chicken and it was great. Very similar to PF Chang. You order at the counter and they bring the food. I wish we had one of these where I live.',
      'useful': 0,
      'user_id': 'kq0emTsKE2DcDOWbe1NYRg'},
     {'business_id': 'jQsNFOzDpxPmOurSWCg1vQ',
      'cool': 0,
      'date': '2017-01-24',
      'funny': 0,
      'review_id': 'Wjcm9kzfQp3d2R6uqblzzg',
      'stars': 1,
      'text': "I usually go to pei wei regularly and am usually over all satisfied.  However today did not go well . First I was unable to order online,   when I  went to pick up the food I brought  a gift card with a remaining credit on it and the remaining total written on top with a permanent marker which would have covered the entire cost of my meal. When I handed it over to the employee  she swiped the card and simply stated there is a zero balance on this card. To which I replied that can't be correct because the last time I used it the employee wrote the available balance on the top of the card for me. The cashier didn't seem to take me serious and simply asked for another method of payment. I was in a rush it was my lunch hour so  I paid out of pocket. \nThe sauce on my food taste watered down over all a poor experience. To add insult to injury I later called the number on my gift card and they assured me that I did have a credit balance which leave's me wondering what are gift card's good for? \nBUYER BEWARE...",
      'useful': 1,
      'user_id': 'mIGfb8yh-zSOoh0U2Y8mSQ'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2011-08-21',
      'funny': 0,
      'review_id': '4RF8dMNBW-p2eTluPME_4g',
      'stars': 4,
      'text': 'Enjoyed the bright fun Mexican decor!  The food was delicious and reasonably priced!  And the margaritas were delicious!',
      'useful': 0,
      'user_id': 'rv6_U_4AsOQ-L50aNRuNNg'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2013-06-03',
      'funny': 0,
      'review_id': 'ClgrKJ6dqiM7vSKJBJ2w6Q',
      'stars': 4,
      'text': "I've been here at least 5 times now and each time is better than the last. I Iove what they have done to improve the building and add outdoor dining. Service is always good here and on par with other top places in Mentor. I really suggest the chicken soup on a cold day it's very yummy! Everything I've had on the menu has been great.",
      'useful': 0,
      'user_id': 'T5MGS0NHBCWgofZ6Q6Btng'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2014-03-15',
      'funny': 0,
      'review_id': 'IBCTqmvwvd5ZqQhuvFDNXg',
      'stars': 5,
      'text': 'Terrific service. The place was packed, but we were seated right away. Chips and salsa on the table as we were seated. Great food, big portions, fast service.',
      'useful': 0,
      'user_id': 'NtkMuGqcis30GjAkq91etA'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2014-06-09',
      'funny': 0,
      'review_id': '69kni-xG6qtg9y3Hq_zw5g',
      'stars': 4,
      'text': 'Ate here for lunch on a Sunday.  Arrived around 12:30, and there were a decent number of customers seated.\n\nService was friendly and fast, food was excellent, and they have DRAFT BEER, including a couple of Mexican brews.\n\nThis location is the best Don Tequila I have been to.',
      'useful': 0,
      'user_id': 'unEY79t6hHECP9Yd58R1dg'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2012-02-25',
      'funny': 0,
      'review_id': 'rQOasxLFCDNWLNW27VHnyA',
      'stars': 5,
      'text': "Been dining here since it first opened. Wife is from tx so this is her taste of home. When tx folks come to visit they ask to go to Don's. \nIt's very good tex-mex.",
      'useful': 0,
      'user_id': 'UwfgmOOul1fc79IcI5h2MQ'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2014-03-15',
      'funny': 0,
      'review_id': 'eGtRHSII_uNGVELSxjYNSA',
      'stars': 5,
      'text': 'If you have a hankering for Mexican, head on over to don tequilas. Fast service and delicious food with generous portions. We ordered the Guadalajara special tamale and the pollo adobe and they were both delicious.',
      'useful': 0,
      'user_id': 'kqsBiDRm1u34Q0RqN62QIA'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2011-06-14',
      'funny': 0,
      'review_id': 'O5UfgCBj4osaEeolc_muUA',
      'stars': 3,
      'text': "Kind of disappointed with the previous reviews in the area of customer service because honestly there was none. We were the only patrons in the restaurant and the waiter seemed  inconvenienced by us being there. After he took our order he went and sat down in another booth and started playing on a computer. We did not get our food quickly like everyone else has said, as we were the only people here it shouldn't take as long as it takes for me to finish my beverage. Maybe it had something to do with the fact that the waiter was farting around instead of doing his job? They got three stars because despite the service the food was AMAZING and quite inexpensive. If you order a quesadilla you may want to order a la carte items as well because the quesadilla was the size of a large Taco which is odd.",
      'useful': 0,
      'user_id': 'BYgAy3hu2s5GeiJ5WMuD0w'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2013-04-24',
      'funny': 0,
      'review_id': 'Dq7iR4uVsVWPfqKqaLtmtw',
      'stars': 5,
      'text': 'I eat here all the time and most of the staff are from Guadalajara.... Food is definately "americanized" and by that i mean tamed down in heat and changed a little for what americans expect.... But when i order my racos de carne asada with lime and cilantro and onions.... All i get are big smiles and HOT sauce.... So if your in the mood for authentic hotmexican food.... Just let the waiter know.. Hell guide you to a great choice!',
      'useful': 0,
      'user_id': 'iIZhrDYOmcyGdiWSWAldmw'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2017-07-01',
      'funny': 0,
      'review_id': 'H4ecwTdoO_sjwELf62BjwQ',
      'stars': 3,
      'text': "Inconsistent service, average food.....I've had better at Chipotle.....\n\nWon't be back.",
      'useful': 0,
      'user_id': 'tV0N0henYG4krjjA7glZhA'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2012-12-10',
      'funny': 0,
      'review_id': '0IOu6FA2x-rcvtXb8Xvheg',
      'stars': 1,
      'text': 'Always the same old bad Mexican food. At least this time I did not get the nearly 2 foot piece of hair in my Fajita Burrito.\nWhy many people I know, including my wife like to go to this place is beyond me.',
      'useful': 0,
      'user_id': 'Rt1sD4KdPD6Uquf9BIsw2Q'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2015-09-07',
      'funny': 0,
      'review_id': 'ysv7Ynllze9TV3GkzF0D3g',
      'stars': 4,
      'text': 'Very good, reasonable priced . The Texas quesadilla with chicken, \nSteak, shrimp, onion, peppers and cheese was large and very good\nThey have Dos Equis on draft!',
      'useful': 0,
      'user_id': 'xU3FI_O1XrjGRzOiFWvjBw'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2013-11-23',
      'funny': 0,
      'review_id': 'CwiGJRzxJ0wOfgmA4yCdig',
      'stars': 5,
      'text': 'What a surprise!  The most amazing Mexican food I have ever tasted and I am a foodie. My husband and I ordered the table side guac.  Delicious!   Texas margaritas. Delicious!  Seafood burritos with fresh cilantro!  I will be salivating for them soon!  All of the food was so fresh!  You must try this place. We will try the Green Road location next as it is closer to our home.',
      'useful': 0,
      'user_id': 'b7zWPM3MDi074PbSw9GSIA'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2016-12-16',
      'funny': 0,
      'review_id': 'DmVaJHiNO21faYMc74YegQ',
      'stars': 2,
      'text': 'Amazingly good Mexican food but they rip you off EVERY time.  They scratch your order on a piece of paper with no descriptions and prices never add up. Its ALWAYS more. We have checked this several tines and wrote down everything we ordered and the prices on the menu and its been 20-30% higher consistently every time.  Its too bad they scam their customers like this because the food really is awesome.  Write down everything you order and be prepared to question your bill.',
      'useful': 0,
      'user_id': 'vUeijF36ja09bwzjUkoNpA'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2017-06-04',
      'funny': 0,
      'review_id': 'Oj67TGc1ZBWYzzZi-7dIkQ',
      'stars': 3,
      'text': 'The service was ok, the salsa flavorful, and the food pretty good. They only serve pre mixed margaritas,  which is no bueno. I ordered the burrito fajita and it was great, but the kids nachos were bland and rubbery.',
      'useful': 0,
      'user_id': 'nbeZzsqfROHS9Ol2V4nyUA'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2014-03-21',
      'funny': 0,
      'review_id': 'UR_LseKj5wZjLo-tCwwMIg',
      'stars': 5,
      'text': 'We came here for the first time to celebrate my birthday.  We ate a ton of fresh chips in advance of the meal, the salsa was pretty mild.\n\nI had the Chori pollo, which is a burrito with chorizo and chicken with rice.  Man, it was amazing. Hot and spicy with peppers, I took some of the huge portion home and ate it the next day.  \n\nThe service was seriously, amazingly fast for a group of 4, and they kept our drinks regularly filled.',
      'useful': 1,
      'user_id': 'ciFYbNSdhRpHW0LofmmeEQ'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2013-03-20',
      'funny': 0,
      'review_id': '5DOAfsOxp0zfhMjR2PdFgw',
      'stars': 5,
      'text': "Yum!!! I love this place! Come almost every Friday. Try the Chicken Chili Verde, my first favorite dish. My new favorite, Burrito Chipotle!!!! So delicious! Don't forget the margaritas!! :)",
      'useful': 0,
      'user_id': 'mPfLvSeRYqdQxJQk5ANVUg'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2017-03-13',
      'funny': 0,
      'review_id': 'SsVEwBJDgATGEtZSRhNCGw',
      'stars': 5,
      'text': "We've eaten here 2-3 times a month for the last few years. It's my absolute favorite Mexican restaurant in the area. I've never had a meal I didn't like! The menu is so huge and full of options, even your pickiest eaters should be able to find something they can enjoy.\n\nI love how quickly the meals come out, which is especially nice when you're on your lunch hour. The portions are always great, and the staff treats us like family each time we stop in!",
      'useful': 0,
      'user_id': '8e2Khf95bZgBlkeMqh59fA'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2015-09-13',
      'funny': 0,
      'review_id': 'DTTZFF-izUvZbrmP_4b2Jg',
      'stars': 3,
      'text': 'We have just moved from Austin,TX to NE Ohio and are searching for our new favorite Mexican restaurant.  We were greeted promptly and brought to our table.  The chips were fresh and warm.  The salsa was a little blander than we care for.  Service seemed slow, but in all fairness it was Saturday evening.  I ordered the Lobster Enchiladas which had pieces of lobster sautéed with onion and tomato wrapped in a fresh corn tortilla.  It was average.  My husband had the Burrito Especial with a chicken burrito and a beef tip burrito.  Hubby said it was above average.  The rice had a nice flavor with an aftertaste of tomato and cilantro.  We were disappointed with the lack of complex flavor profiles that we have become accustomed to while living in the Southwest.  Overall, we rated it as adequate to good.',
      'useful': 0,
      'user_id': 'xevJes9a3TJOsuasqVpidA'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2015-01-14',
      'funny': 0,
      'review_id': 'loBnTqzypW71GSEsE8-R_g',
      'stars': 3,
      'text': "Maybe let's work on the chili rolenos. Do nice large poblemo peppers like el rancho and you got me as a frequent customer. A+ on the guacamole! Large portion. Nice!",
      'useful': 0,
      'user_id': 'fGgE1ASXCWfuw5edGCfCFA'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2014-11-21',
      'funny': 0,
      'review_id': 'iDRXhARsx77_IpWhey58Gg',
      'stars': 1,
      'text': "The food isn't bad, but every time I come here with friends they overcharge us. We end up getting charged at least a few dollars more than our menu items should be when we check out, and it's happened on several different occasions. \n\nThough the food may have been good and reasonably priced (on the menu, not on the bill), we will unfortunately never be going back. Very disappointing.",
      'useful': 1,
      'user_id': '9Y_HfxjyZU7ltQ3VY3Wj-A'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2010-03-29',
      'funny': 0,
      'review_id': '6sX7YcBhK0ASD8kLdVdCnw',
      'stars': 3,
      'text': "First time I eat here I came with friends who told me the food was the most authentic Mexican food they ever had, well it was not for me since it's not really Mexican but Tex Mex really. It wasn't bad but nothing to write home about either, I ordered a piña colada and the waiter spill it over me without replacing it not good at all, the place tries so hard to be unique and as Mexican as can be but it is just a caricature and a stereotype of anything Mexican honestly I was waiting for the three amigos to come out of the bathroom or something I despise decoration like this  and it affected the overall rating. Second time there food was not as good I ordered a dish that included carnitas in it but they were missing I told my waiter he apologized brought them in told me he was sorry but he charged me for them anyways even though they were supposed to be included. Not as bad as others but not great by any means.",
      'useful': 1,
      'user_id': '9CiRp71NmS_AP74rW-1gdg'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2016-07-14',
      'funny': 0,
      'review_id': 'oDOyMnSs70q3jSeCKkETwQ',
      'stars': 5,
      'text': 'Amazing Mexican place!! Very tasty and reasonably priced. We ordered Molacajete, seafood chimi and carne asada burrito. The best!! Molacajete is a must try! This is something new I have never tried before. Service is quick! Cute authentic decor.',
      'useful': 0,
      'user_id': 'yycHN--LCuCjYt6G-SBu8A'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2012-07-26',
      'funny': 0,
      'review_id': 'SgUSVEETtE_HcBmeKAsN_w',
      'stars': 4,
      'text': "Service is great. Food is texmex and very good. I wish they offered black beans, I'm from California where black beans are popular. Margaritas are great, I got a buzz from two. The place was busy. Mexican families were here  and that's a plus.",
      'useful': 1,
      'user_id': 'I7nW8TNt7h4Q6NnsJg3roA'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2010-05-18',
      'funny': 0,
      'review_id': 'LWsWwahRBEPbmiSlrwA31g',
      'stars': 3,
      'text': "The first thing I noticed here was the decor, which I have to say I liked.  Of course, yes, I enjoy lime green walls and bright pink color washes, but that's just me.  \n\nWe had three little ones with us, so you can imagine that we were a little less than organized, but our server took it all in stride, which we appreciated.  He was also super fast; I think all of us were surprised how quickly our meals came out. \n\nAs for the food, I feel like it's a little difficult to review.  I had never been here before and my Mexican palate is less-than-sophisticated, so I randomly ordered a plain enchilada and quesadilla.  The latter was pretty good, though I found the enchilada a little perplexing.  It was hiding under the sauce and I couldn't really tell what was in it.    \n\nThe chips and salsa were great, though.  And hot.  The one person at our table who can apparently drink tabasco sauce straight even admitted his eyes were watering a little.  Everyone else seemed to really enjoy their meals.  \n\nI'm not in Ohio much, but I'd happily go back and try something else.  Oh, and um, I wouldn't exactly object to painting my living room lime green, after seeing how cool it looked at Don's.",
      'useful': 1,
      'user_id': 'vikyvfRREkyvIFg1wtxk1g'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 2,
      'date': '2010-03-20',
      'funny': 0,
      'review_id': 'mvThe_xt5r-nXwM4btoxeA',
      'stars': 4,
      'text': "With everywhere else packed on a Friday night, we wondered in, not expecting much quite honestly. \n\nWe both ordered off the specials menu, and my husband barely managed to squeeze in his standard 'where's my burrito' before two large plates, overflowing with food, arrived in front of us.\n\nThe food itself was good. Mexican food can too easily be greasy, but this was not. The menu is nothing startling, but the classic mexican dishes were all on the mark. And there was enough left over for a second meal tomorrow.\n\nPrices were very reasonable - our total bill, including tips, was less than $25.  Definitely planning another visit.",
      'useful': 3,
      'user_id': 'xlp2qzyxgBg31wFHSDhKww'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2013-09-11',
      'funny': 0,
      'review_id': 'SzI_v454Dd6HmxSi-wSZPw',
      'stars': 2,
      'text': 'i don\'t understand all of the high marks for the this place. the chips were not good and the salsa wasn\'t good either.. the chips were bland, old and not seasoned at all, the salsa represented canned tomato taste, yuk, the chix enchilada was flavorless, while using white meat chicken it was not seasoned and very chewy, it had no flavor whatsoever and was definitely something "not" to order. the  most off putting thing about the place is the filthiness of all the furniture and the fact that we had fruit flies buzzing our table thru out our meal. while el rodeo and el patron are nearby, and have far better food,i cant understand what the attraction is to this place, will not be back',
      'useful': 0,
      'user_id': 'vkOWrnWgPhgpqev9ZanAnw'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2010-08-22',
      'funny': 0,
      'review_id': 'rwzyKNL1bA54asCoTGiQUA',
      'stars': 4,
      'text': 'On a recent trip to Kirtland, OH.\n\nDecided to try even though the outside appearance was somewhat questionable.\n\nBut nice simple decor inside.  Friendly waitress who gave good tips from the menu.\n\nWe both had the Tres Amigos.  Nice chicken, steak, shrimp combo.',
      'useful': 0,
      'user_id': 'eyYzBULJ8uIi41drYCctuw'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 2,
      'date': '2011-07-26',
      'funny': 0,
      'review_id': '1JgdQwRj4IrGaNfQIoxF1Q',
      'stars': 5,
      'text': "If you blink. You will miss this little eatery, squished into the corner of a small strip mall in Mentor, Ohio.\n\nI went with a friend of mine from the area, who was treated like a rock star when he arrived (well, he IS a rock star), and that's indicative of the casual and familial vibe at this restaurant.  I didn't take notes - too busy enjoying the company - but I'm pretty sure I got a burrito and it was delicious.  Margaritas were excellent (this much I do remember) and service was quick and pleasant.  Now that I know that this place exists, it gives me even more reason to travel to the northern tip of Ohio for good company and eats!",
      'useful': 0,
      'user_id': 'KQ-gI-xsZzq5xeHYev6eYg'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2014-08-14',
      'funny': 0,
      'review_id': '8sK2ci8cUOzRxKP1hpkPNQ',
      'stars': 5,
      'text': 'Good food, great prices, large portions, fast, friendly service. What more can you ask for. We highly recommend you try it!',
      'useful': 0,
      'user_id': 'is6jBVBI3fbjFOCD9cJwvA'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2015-06-05',
      'funny': 2,
      'review_id': 'z6wG7RrcONWJlxE8AJypeg',
      'stars': 1,
      'text': "This will probably be my last review of this location. I was at don tequila yesterday & the cashier was talking  to me but I couldn't understand so I laughed it off. \n    \nHe told me I looked sexy today and said something else. I said thanks and laughed so that he wouldn't be offended. He came around the counter and walked up on me then hugged me so I assume he was asking for a hug. \n    \nHe then whispered in my ear that he just wanted to feel my breast on his chest. Seriously....I don't feel like I need to wear jeans and sweaters in the summer because he is a pervert. \n\nIt was very disrespectful and unprofessional and I feel uncomfortable eating there anymore.",
      'useful': 1,
      'user_id': 'YBbNemmB8wWbKGr6V3bCAA'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2014-04-04',
      'funny': 0,
      'review_id': 'NSBC7JSjUn3ru4vf3f21Dg',
      'stars': 3,
      'text': "Sometimes you need a little spice in your life. Don Tequilla delivers. The neon sign beckons you inside and once there, sit back and enjoy.\n\nThe roomy booths offer space for cozy dinner or enough room for your whole brood or gang of friends. Nibble some fresh chips and salsa while you ponder your meal options. The salsa is fresh with tomatoes, cilantro and spice. In case that isn't enough heat for you ask for the special sauce from the kitchen for extra kick and spice.\n\nI ordered the chili rellenos hoping not to be disappointed - this gal spent some time in Arizona and loves her Mexican food. These didn't disappoint. Yum. Served with rice and beans with a drizzle of red sauce.\n\nGood enough to want to return and sample through the menu -- several other items caught my eye - fish tacos, nightly specials, burritos with tomatillo sauce!!! \n\nThe staff is friendly and patiently gives you time to check-out the menu and answer any questions. nom.nom.",
      'useful': 0,
      'user_id': 'CLHRQRNRrpj7Ht-oQSFf7g'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 1,
      'date': '2012-11-26',
      'funny': 0,
      'review_id': 'wPGXLAQX0z7FVh1PhInyLg',
      'stars': 4,
      'text': "i love this place!\n\nFirst impressions this place is a fun atmosphere filled with mexican clay rooftops and colorful furniture.  You get your chips and salsa as you sit down as usual while they take orders for drinks.  They got horchata here now, which is a + in my book, if you haven't tried it yet, i totally recommend to.  \n\nThe food came out rather fast and speedy on extremely hot plates.  The salsa and hot sauces there are bombbbbb. Ask them for homemade style hot sauces they make in the kitchen, it's different everytime you go in.  Some will leave you sweating and some will leave a sweet or an extra tomato flavor in your mouth.\n\nThe food i'd definitely rate 7.5/10. as i'm a hardcore mexican food lover.  It's nothing you find in texas or california, but quite damn close.  I always find myself coming back to this fun filled, warm, family owned place and always leave feeling satisfied.  for taqueria style tacos, try the tacos el pastor!",
      'useful': 0,
      'user_id': 'MGBv4e-Z6lfyUcSVwPXQPA'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2016-01-01',
      'funny': 0,
      'review_id': '5oLdxF5fJfwrCZ20XDFDXg',
      'stars': 5,
      'text': 'Authentic. Delicious chips and salsa follow you to the table on arrival - food arrives on plates too hot to touch. We lived in the South West for 9 years and ate at many Mexican places ONCE, before calling DonnyTs our favorite. We have been enjoying perfect food here for 6 plus years. Very little turn over in wait or kitchen staff... consistent quality food and service.',
      'useful': 0,
      'user_id': '4pVwuNpFwn8nWrSkT0urvA'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 1,
      'date': '2010-08-18',
      'funny': 0,
      'review_id': 'BaqYuJH1oIMNayOBW_AqFw',
      'stars': 4,
      'text': "Why hello service. Don't let the lime green walls scare you, this place is all business. Seated immediately, the waiter came to the table immediately, and the food came almost immediately after we'd ordered. Boyfriend got a combo platter with an enchilada and taquito and devoured the whole thing. I got a carnitas dinner platter and I could only finish half since I filled up on chips and salsa and guacamole. It was so good! \n\nThe only disappointing part was the dessert. We ordered flan, and when it arrived it was covered in whipped cream and strawberry sauce which I am not a fan of. The consistency was nubbly. Should've went with the churros.",
      'useful': 1,
      'user_id': '5z587IBRnjCbo51IaHNPzQ'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2011-10-28',
      'funny': 0,
      'review_id': '0yqWd0USYSpaPI7zUW4JAw',
      'stars': 1,
      'text': "Passed this place a few times and finally decided to try it. I got the #17 on the lunch menu. It was a burrito  with eggs, chorizo and cheese.  It was probably one of the most tasteless burritos I've ever had. It was covered in what appeared to be some sort of franks red hot sauce, and no im not mistaking it for enchilada sauce. The side of my plate was a pile of tomatoes shredded cheese and a piece of lettuce, a whole piece not shredded. Not happy with my lunch at all. The salsa and cheese dip were average, nothing to go crazy over. Service was quick and friendly. The food came out about 5 minutes after we ordered which kinda scared me. I like quick service, but also don't mind waiting a few minutes for some really good food. Overall the place was cute, people were friendly but food was less than OK. I wouldn't go back.",
      'useful': 0,
      'user_id': '-S1dz92Q3RPfHomiqEeP8Q'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2016-04-02',
      'funny': 0,
      'review_id': 'pVpmE6BWfK6W-1Zvat94NQ',
      'stars': 5,
      'text': 'This is our go to Mexican restaurant. The staff is kind and courteous and Javier is an amazing gentleman. Always prompt , fresh and delicious. Their mango margaritas and dos Equis pair perfectly with our meals. Very family friendly and great atmosphere. Really a great place',
      'useful': 0,
      'user_id': 'ludX46lEfeQ8TuHsK0D9zw'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 1,
      'date': '2012-09-12',
      'funny': 0,
      'review_id': 'd1y0PAqpOQ_Ml-xil4Pz6A',
      'stars': 4,
      'text': "Food was AWESOME. The normal tacos are ... well ... normal and typical. My wife ordered the chicken fajitas and they were very good fajitas. I ordered the burrito durango which was a 12 inch shell STUFFED with chicken, steak, and huge shrimp! I asked for it spicy... and it actually came with pretty good heat. The burrito was outstanding, which is why I gave it 4 stars instead of 3. Overall, we'll probably be back.",
      'useful': 0,
      'user_id': 'rK2aGPWEU7UVwz_fqKE8mA'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2017-06-02',
      'funny': 0,
      'review_id': 'yED_RnJWbt03uA2tW56zFA',
      'stars': 4,
      'text': "Food is always great especially Tableside Gauc but today service was really slow because of kitchen. Waiter couldn't have been nicer",
      'useful': 0,
      'user_id': 'XeOCjJwfLlKpkSd_oYxyGQ'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 1,
      'date': '2016-09-17',
      'funny': 1,
      'review_id': 'M8g0a_W1AcHxrOcyXG0paw',
      'stars': 5,
      'text': 'We had a really good meal here, service was quick and accurate.  Food was definitely better than most Mexican places around.  Margaritas were very good.  Most of all, the place was very clean!  Would definitely go again!',
      'useful': 1,
      'user_id': '3o2vT1qybhjpm_yChoLxlA'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2014-06-25',
      'funny': 0,
      'review_id': '3v3S0lYInTbqeN-lBh9VNA',
      'stars': 4,
      'text': 'Went on a saturday night on the first night of summer, and had a party of 12.   We were sat almost immediately.  Went there because they had catered a party of a friend of mine.\n\nOur server was professional, and bilingual.  But you could totally understand him and had a cool accent.  Americanized i suppose.  They do not use a POS system but use an old paper check system...so separate checks becomes an issue.\n\nThey put tons of chips and salsa in front of you...the chips are super crunch but almost no salt, and the salsa was bland for my taste.\n\nI had the el paso speciai; beef tip burrito and chicken burrito w/ rice and beans.  All came out hot but a little watery.  Dunno where that came from.  But I did ask for lettuce/tomato set as it did not come with any in or on it.\n\nKids food came out fast, they did have the world cup and that seemed to distract the waiters...but as far as Mexican food goes it was pretty good for Mentor.',
      'useful': 0,
      'user_id': '09jpZrKD_j0SG9WPBDSJQg'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2016-09-09',
      'funny': 0,
      'review_id': 'bBkEPhdZhni_vzvTKFv-oA',
      'stars': 3,
      'text': 'Your basic traditional Mexican style restaurant, never at in only ordered out.\nMade me wait for a hand full of my orders when I came to pick it up..but super tasty. With a lot of hit and miss items in the menu.',
      'useful': 0,
      'user_id': 'NvI6nkdKut_x5GommcngDw'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2017-01-14',
      'funny': 0,
      'review_id': 'Z8P9EK2sOgYnyG9jqH25Tw',
      'stars': 1,
      'text': 'Very bland. Salsa is like tomato sauce. Enchilada sauce taste like canned marinara.. just awful.',
      'useful': 1,
      'user_id': 'gf_VD1MJRnoqLDljDwiDnw'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2015-01-28',
      'funny': 0,
      'review_id': 'MjkZOjE7Wi5V7SjAhgcj-Q',
      'stars': 5,
      'text': "Been here a couple times and so far every time we've gone the foods been great, the service has always been friendly and the portions are amazing!",
      'useful': 0,
      'user_id': '2rIHr_3qznLHn73YfOwkWA'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2016-06-04',
      'funny': 0,
      'review_id': '2pfPmMCXr3KEdHlO7k_N-g',
      'stars': 5,
      'text': 'Tacos al pastor--delish! Salsa and guacamole was very fresh! Refried beans were super good. Service was friendly and fast. Beer was cold and the bathroom was clean. What more can you ask for?',
      'useful': 0,
      'user_id': 'o82jtAsa6rSgkl2jQIKfPA'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2017-07-04',
      'funny': 0,
      'review_id': 'V-YiU3DC3prLkOh1ByOUAw',
      'stars': 3,
      'text': "So, I thought the food was average. But what makes this place is the cool interior and super attentive and friendly bartenders and servers.  We felt like we were part of la familia. I haven't felt that way in a Mexican restaurant, probably ever.  Fajitas did not arrive sizzling. Bummer.  My bean burrito and cheese enchilada were standard fare. Nothing to shout about. The beer was extra cold so bueno for that.  Very clean restaurant.",
      'useful': 1,
      'user_id': 'Xxvz5g67eaCr3emnkY5M6w'},
     {'business_id': 'dwQEZBFen2GdihLLfWeexA',
      'cool': 0,
      'date': '2012-12-02',
      'funny': 0,
      'review_id': 'nLiRNtABmKq5-Vu2jsJQKg',
      'stars': 5,
      'text': "We've been going here several times a month ever since they opened. Some of the BEST Mexican food I've ever had at absolutely the LOWEST prices I have ever found for food this good. Waiters friendly and attentive. Nice decor. Nice new Bar.",
      'useful': 0,
      'user_id': 'TQTMic2HzXjXmeMls5l0-g'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2009-07-21',
      'funny': 0,
      'review_id': 'Z5l99h18E3_g1GLcDSsWqA',
      'stars': 3,
      'text': "I left Table 17 feeling very ambivalent. Meh as others would put it. Some things good somethings not so great but on the whole in between.\n\nThe room is simple, farmhouse chic: long harvest table, ornamental roosters and wooden chairs with cushions for seating. Sundays and Mondays they have a decent prix fix menu on offer or you can order a la carte from their shared plates menu. I went with the steak frites and salad off the table d'hote. Spinach salad came nicely dressed though the dressing was plain could have used more tartness. I had a few slight issues with the steak frites, the steak came closer to medium when medium rare was requested and was under seasoned. My accompaniment of frites (which were wonderful btw) on the other hand was too liberally salted with kosher salt so I guess I got the best of both worlds! The Bearnaise sauce sauce on the side was so-so and didn't do too much to perk up the steak. Dessert was chilled rhubarb with creme anglaise really lovely but too much sugar in the sauce and the sprig of thyme thrown in kinda messed with my tastebuds, I really should have plucked it out!\n\nMy server was a lovely affable gentlemen who was very friendly, warm and quick on his toes. I did like the service. But the food was so-so and I find that there are better bistro options around town.",
      'useful': 3,
      'user_id': 'djpMXOA1ic5wv3FPtubHNw'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2008-12-13',
      'funny': 0,
      'review_id': 'Z3Fw292i0Eg8liW0DT1jsw',
      'stars': 3,
      'text': 'for the time being, for all its worth, i am going to stick to the magical number three.  okay. lets break it down.\n\nthe service.  when we walked in the waitress seemed to be adept at being the problem solver, however, slowly this role for her, became more difficult and intrusive.  we were (ironically enough) a party of three at around one pm on a friday for lunch and there was not a three person table available.  there seemed to be only one three person table available for seating in general (lets give them the benefit and say there was 2 three person tables) and so we were relegated to a two person at the front of the restaurant decorated with a third chair.  not the best, but so be it; its one pm at a relatively new restaurant and i was prepared to deal.   i will say though, with no regret, that it is worth it for them to reconsider how many two person tables they have versus three.  if they are popular or getting to be, this is something they might need to reexamine.  the waitress seemed to be the only one serving (i would guess) the 17 tables that were almost filled at that time, and so, i can appreciate it was a hard go, but at the same time it doesnt excuse for me her terseness and impatience.  every person at a job has stress to deal with and if yours deals directly with customers, you should learn to keep the dealing with that pressure to yourself.  its not the problem of the customer.  its her job to figure out how to make that work for herself.\n\nfor lunch table 17 serves sandwiches, salads and a soup.  we each had a sandwich, a salad, a coffee or tea and the bill came out to be about 60 bucks with tax and tip.  not exorbitant and not cheap.  though im sure the quality of the ingredients used was high, the sandwiches felt like they could be made at home.  in conjunction with that sentiment i am eager to admit that it takes skill to know when to stop cooking something, to pair spices up with core ingredients to arrive at something special, and not any joe will have the wherewithal to claim he can do the same whether he thinks so or not.  but if you are going to charge me somewhere around 9 bucks for a sandwich (with no other sides except for a couple pieces of cauliflower and a sliced carrot - which were good, doused in some kind of vinegar or pickled sauce) then perhaps you should use dijon instead of yellow mustard, or applewood smoked cheddar instead of havarti.  its fair to assume for the general population that 9 bucks is a hefty price for a sandwich and with that in mind one might put their ingredients where there outstretched palm is.\n\nthat being said the sandwiches tasted great.  (one thing has nothing to do with the other should you be saying to yourself something along the lines of "well then what the hell is he complaining about").\n\nthe vegetables that came on separate plates alongside the sandwiches were  4 dollars each.  we had three kinds.  one was sweet potatoes dressed with chilies, toasted coriander, mint and olive oil.  another was called farro, an ancient tuscan grain dressed with extra virgin olive oil, cucumber, carrot, and fine herbs. and the last one was chickpea salad with celery, pickled red onions, rosemary, parsley and lemon.\n\ni enjoyed all of them, but they were honestly not representative of their ingredients.  i dont need to have my dishes overpowered by their listed ingredients, but if something says its going to be flavoured by mint, fine herbs and lemon dont you think you should be able to experience that as you taste it?  i think so.  again, to reiterate, i didnt mind the natural taste of the vegetable offered, but if thats the way it will end up tasting, perhaps you should say so.\n\nall my peeves aside, i am still interested in trying this place out for dinner; the feel of the place is great, cool and relaxing and as i looked at their dinner menu it looked quite tasty.  so perhaps once they get their groove on itll be more enjoyable.',
      'useful': 1,
      'user_id': '-pXs08gJq9ExIk275YLvPg'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2013-10-12',
      'funny': 0,
      'review_id': 'hsKINx1dIKeFTDe-ZlCvgA',
      'stars': 5,
      'text': "Love this place. I went there with me boyfriend to celebrate our anniversary and I am glad my colleagues suggested me this place. The food is really really good. They have a great cocktail list. \n\nWe shared the hot balls (delicious) and the rillettes (I loved it). Then, I had the beef tartare with the side of potatoes cooked in duck fat (awesome). To finish, I had the lemon pannacotta...this is by far one of the best dessert I've tried. I am not a dessert person and not a big fan of lemon but that dessert was just perfect!! \n\nThe service was slow so don't go there if you are in a rush! Our waiter was really nice  and even tried to speak in French as he noticed that we were speaking French. \n\nI will definitely go back there...I am sad this place is kind of far from were I live!",
      'useful': 1,
      'user_id': 'PTj29rhujYETuFlAZaDi3w'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2013-02-17',
      'funny': 0,
      'review_id': 'oviMS8F4ACflGysxsXKmew',
      'stars': 5,
      'text': 'Had a lovely evening last night at Table 17.  Great selection of wine and the owner made an excellent suggestion of something we had never heard of before - bearing in mind both price point, meal choices and preferences (though overall it was a bit of a pricey list, only one bottle south of $50, just).  \n\nLovely, vibrant atmosphere, the polenta was delicious as a starter (the chef tops it with a different finish every day - last night it was meatballs) - though the other ones we saw at other tables also looked sumptuous - and our meals/sides were excellent.  Also, the food prices\n\nAll in all a great evening  - could have happily ordered anything from the menu, and what we did order was delish.   \n\nDefinitely will go back to work our way through the rest of the menu.',
      'useful': 0,
      'user_id': '3hLMY2dBEP1kYbd_ywTsCQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2011-08-14',
      'funny': 0,
      'review_id': 'SmH05C7YViVmVjaL4FpHPQ',
      'stars': 3,
      'text': "Table 17 is a classic bistro with a very good menu, wine list and cocktail list.\n\nI've been many times for dinner as well as brunch.\n\nFood is always consistent, have never not liked anything I have ordered. \n\nDinner: have been numerous times have tried their salumi & cheese board ($15) with was good, not Black Hoof excellent, but still pretty good. Their steak frites ($21) consisting of an 8 oz top cut sirloin was cooked perfecting medium rare and served with a side of beranaise sauce. decadent and good. Have also had lamb here but it must have been a special because it does not appear on the menu.\n\nBrunch: You need to order a side of their thick cut double smoked bacon. Its to die for. They also typically have cinnamon buns at brunch that are baked in house. My favourite brunch item on the menu has to be Farmer's breakfast with consists of a scotch egg (soft boiled, battered then deep fried egg), a terrine with toast and a small side salad. Surprisingly it doesn't look like its on the menu anymore :( Have also had the stewed eggs which was also good.\n\nNote: If you are a member of Foursquare, check out their specials for check-ins, last time i was there for brunch you got a mimosa or caesar on the house with you check-ins.",
      'useful': 1,
      'user_id': 'zZy5Jljx7rEvISiJ2isJpA'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2014-11-12',
      'funny': 0,
      'review_id': 'wNe5wh4wXjdRfkb6agPyAg',
      'stars': 4,
      'text': "I've been here for a few dinners and brunches, and it has been a pleasure every time. \n\nThe service has always been very attentive and helpful. For one of the dinners, our group had trouble deciding on dishes so our server suggested that we pick a price point and let the chef put together a menu. It was a really fantastic experience!",
      'useful': 0,
      'user_id': 'B7IvZ26ZUdL2jGbYsFVGxQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2015-03-09',
      'funny': 0,
      'review_id': 'H_rqSpDAvTIWRG2-I6nOzw',
      'stars': 5,
      'text': 'Ok, full disclosure:  went here as a default because Ascari Enoteca was closed down due to a busted water pipe or something.  \nFood was AMAZING!  We ordered a bunch of appetizers and 3 mains, as well as dessert and cocktails......thought for sure we would get something that we did not like, but alas everything was delicious.  To quote an obnoxiously catchy song from The Lego Movie, "Everything is awesome...."',
      'useful': 0,
      'user_id': 'XCDxUzBSY8-JOvcp4jqpvg'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2014-10-29',
      'funny': 0,
      'review_id': '-8AcVBM6o3v0dbXhCSBOBA',
      'stars': 4,
      'text': 'We had an exceptional experience at Table 17. Chef John prepared an incredible fois gras app, the squid ceviche did not disappoint... the Rainbow trout was melt in your mouth delicious and we ended our foodgasm with the dark chocolate tort.',
      'useful': 0,
      'user_id': 'bzKH1kadsxohV4UgNc1wTw'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2013-06-08',
      'funny': 0,
      'review_id': 'c62RkSBhA3kmqcyyFn6dIg',
      'stars': 4,
      'text': "Found this place by chance and walked in for brunch on a Sunday. There was one table available luckily. There wasn't a lot on the menu, but all the items we ordered wew so fabulous!! And very out of the ordinary. Service was good too.\n\nFrench toast with creme anglais, pastry shell with onions, pears etc... all fabulous. It's a delight for the senses, this place.",
      'useful': 0,
      'user_id': 'Kf10dTiGlcnyhj8DBoZn1A'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2015-03-29',
      'funny': 0,
      'review_id': 'DMksEQKXo4s8Df59X6EvWw',
      'stars': 5,
      'text': 'By far my favourite restaurant in the east end. Always great food and service - and a cool and inspired cocktail list. Great for any occasion!',
      'useful': 0,
      'user_id': 'uenzd7mVPIhCdphDQNRKCg'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2014-07-29',
      'funny': 0,
      'review_id': 'zxVqUWJbDRhjwLKTsCt5eg',
      'stars': 5,
      'text': "I booked the private room for a small wedding reception recently. We had 14 guests, the room was perfect. Booking was incredibly easy and they took care of everything professionally and quickly. We were even able to come in the night before and decorate.\n\n\nThe dinner was amazing. We had the family style menu, which included: kale citrus salad, buffalo mozzarella, hunters mash, green beans, polenta with lamb bolognese, braised lamb shank, short ribs... There was more, but you get it. Lots of food for every one, lots to choose from, and so amazingly delicious. Everyone stopped talking at one point, they we're too busy eating.\n\nWe had a cake of our own brought in, so we got to take the dessert from the menu home. It was amazing, so decadent.\n\nIt was a fantastic dinner at the end of a fantastic day. I don't think I could have picked a better place to host my wedding meal.\n\nMy favourite was the polenta bolognese. I could eat that for days.",
      'useful': 0,
      'user_id': 'MFNZ5_mQIPOsGO-PBmf26Q'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2010-06-07',
      'funny': 1,
      'review_id': 'UbL9dgEYA7PeL-47W6m3KA',
      'stars': 3,
      'text': 'I wasn\'t super impressed but I also wasn\'t really turned off, I guess my experience was mediocre with a little disappointment. \n\nThe restaurant in itself is very down to earth with only a slight hint of class. The host and the servers are not snobby and seem fairly friendly in this Leslieville establishment. The patrons are mixed with both young and hip city dwellers along with older more experienced foodies dining out with other couples of the same.\n\nI found myself here ordering the much touted "Hot Balls" but found it just okay. The only hot ball that really stood out was the Goat Cheese with a honey like dipping sauce. It was actually a pretty amazing piece of creation. The other appetizer we have were the "Diver Scallops" which were cooked to perfection and came with a side of really thick and salty bacon. Our mains were the "Leg of Lamb" and "Wild Stripped Bass" both were great entrees that did not disappoint but at the same time didn\'t have a great WOW factor.  Desert was a bit of a disappointment, not much of a variety, you had a choice between chocolate, chocolate, or chocolate in some form or another. We had the homemade chocolate cheesecake, which really didn\'t amount to much for my taste buds. \n\nThe Wine list is very French and if you\'re going by the bottle they don\'t really have an inexpensive alternative. Every bottle is $50+ which can make for a very expensive meal. \n\nOverall the food experience was just okay with some ups and downs. But the real down of the night was the service, after sitting down to our table , it took literally 10 min for someone to show up to greet us and take our drink orders. The other downside of the service, they don\'t place the wine bottles at your table but at a communal table in the centre of the room, they\'re suppose to pour for you but I found that our server kept forgetting even when our wine glasses were empty. I had to reach over to this communal table to grab our own bottle of wine and pour every time which really kills the experience.\n\nI hope this was just an isolated experience for us as I think this restaurant is a real gem in the Leslieville area. If you do decide to go, make sure you have reservations as the restaurant seems to get filled up in all hours of the night.',
      'useful': 3,
      'user_id': 'Jyi0WJt0UfGdEg0grR38ZA'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2013-07-05',
      'funny': 1,
      'review_id': 'OrRzQ2F9vhKFB0idvvB_SQ',
      'stars': 4,
      'text': "I had a nice time at Table 17.\n\nIt was busy when we got there. We had reservations but had to wait at the bar (no big deal) and the bartender (he's in his early 100s) was a riot.\n\nI had the polenta to start. It was creamy and buttery, and the braised pork shoulder in white wine on top was oh-so-good.\n\nI had the short ribs as my main. The meat was tender and moist and the root veggies were perfectly cooked.\n\nWe shared the hunters mash as a side - I was expecting more flavour.\n\nTable 17 had a great vibe to it.\nThe food was good. The servers were attentive. I'd definitely go back.",
      'useful': 1,
      'user_id': '0HxInQ94hVHVlO1FGPWctA'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2015-06-14',
      'funny': 0,
      'review_id': 'CLOqBbXZuKdcFG7meiGYMw',
      'stars': 4,
      'text': 'We had the scotch egg and the tomato based eggs. Everything was delish!\nWhile I prefer to have my bread toasted I understand it\'s not the french way (and they would have accommodated if I asked!)\n\nThe potato rosti was totally worth breaking the diet for, and comes in 4 pieces and is easily shared!\n\nThe americano was outstanding they use amazing beans, and I wanted more. I had on of their $5 mimosas which is something that would keep me coming back weekly if I could !\n\nThe staff are all super lovely, and the ingredients were all do fresh (you can tell these things)\nA perfect brunch spot, no need to wait in line at the other two "brunch hot spots" in the area. \n\nHighly recommend this spot!',
      'useful': 0,
      'user_id': 'hnMURVuScWLJPovCAC5dng'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2012-11-07',
      'funny': 0,
      'review_id': 'AT6hOO-wA9TPfvWXcEJyaQ',
      'stars': 4,
      'text': 'my first time here. Very impressed. Food was very tasty. Our waiter recommended me the Trout and it was amazing. So fresh. It came with a side of vegetables and i got some steak fries with it as well.  The service was pretty spot on and we were always asked if  everything was o.k...which it was.',
      'useful': 0,
      'user_id': '6V2QIqiVY692Ncwev5kokw'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2009-03-01',
      'funny': 0,
      'review_id': 'vYjLW5T9-uqpUv7K_jXtEQ',
      'stars': 4,
      'text': 'Walked in on a cold saturday afternoon round 1330!\nwaitress Helen received us very warmly and sat us at the best table for two by the window in the sun with view onto queen st east.\nWe started with mimosas while studying the brunch menu.\nHelen was helpful, knowledgeable  and prompt.\nI ordered the seasonal scramble, which today was chorizo artisanal, and tarragon scrambled fluffyly together with country butter toast. Loved it.\nMy girlfriend had the french toast, cinnamon creme anglaise and berries which she thought was pretty good.\nWe then asked for a cheese plate which was only available for dinner.\nHelen accommodated us with it anyway:\n*Paillot de chevre\n*Mapledale 5 year cheddar\n*Ciel de charlevoix\n*Monforte toscano pecorino\n*Clandestin \nGreat little figs stuffed with walnuts dry crannies and almonds\nA good bottle of sancerre and "Voila"\n$118 for the both of us\nwill come back, loved it!',
      'useful': 0,
      'user_id': 'EiSkWQLm3zPy4BNqkDwHXw'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2013-09-26',
      'funny': 0,
      'review_id': 'OBIcZAxeCihJZeJpwcUCYQ',
      'stars': 2,
      'text': 'I was very excited to try Table 17, and unfortunately it was a huge let down. I was made to feel very uncomfortable from the moment that I walked in. It was terrible service, and I was very surprised to see this. The waitress asked us to order our appetizer and entree at the same time, causing us to feel very rushed. We waited about an hour for our dinner and were quite disinterested when the food even came.  The beef tartar was by far my best course. The scallops were fine, as to be expected. The broccoli Was good once you were able to get past that it was cold. Wine list by the glass was limited, and we never heard of any specials for the evening. We did receive bread with oil prior to the meal, but no extra plate to put the oil on. We had to get up ourselves and go to the bread table for a plate, balsamic vinegar, salt and pepper. An all around let down for a place that I had heard was good.',
      'useful': 0,
      'user_id': 'Ggl13LZFNdNO3z1fXj_1RQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2011-03-04',
      'funny': 0,
      'review_id': 'JQAM0HY2FM6XxL9yYCW8KA',
      'stars': 3,
      'text': 'Had an enjoyable meal at this Leslieville spot. Menu is light on  vegetarian options but is otherwise diverse. The tagliatelle was al dente and did not disappoint. My pal ate the steak and frites and licked the plate clean. My pomegranate cocktail was so sweet and satisfying\n that I had two.',
      'useful': 1,
      'user_id': 'm6GlY6JALeb0nGnE29FJ8Q'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2014-12-31',
      'funny': 0,
      'review_id': 'lJOWB2JeEsb6-z6-0e70Cg',
      'stars': 4,
      'text': "I've had the pleasure of dining at Table 17 on two occasions. One time we were a group of four and another time it was just me and my husband. The place is simple but beautiful - gorgeous reclaimed wood, candlelit tables and lush fabrics. It just feels good when you walk in. The staff on both visits were very friendly and attentive, and both times the meals were excellent. Their prix fixe meal is tremendous value for the quality of the food you get although just a heads up that some of the dishes are very rich, so go easy on the apps! (re: get yourself a nice salad if you order the short ribs or risotto). If you're feeling like a cocktail to start the meal (and who wouldn't?) I'd suggest the Bourbon sour. Yummy.",
      'useful': 1,
      'user_id': '1cb0oJc2pR2efodBQbXdvQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2012-06-05',
      'funny': 1,
      'review_id': 'pasBQx813wC1fBRw8fsvew',
      'stars': 4,
      'text': "I'm going to ignore the fact that the service was off and just chock that up to the fact that it was the poor girl's first week, so the cocktail spilled, the flatware dropped and the general feeling that the girl was nervous as all get out didn't really taint my evening because the food was excellent. \n\nOur group started off with the polenta special, a strong recommendation from the friends that suggested Table 17. As I understand it, the polenta to start rotates nightly... maybe weekly? This particular night it was served with some seasoned pork and Parmesan cheese. Thumbs way up for the polenta, the pork on the other hand... eh. Didn't have a tremendous amount of flavor. But no worries. Our entrees were bomb. Fish, filet, sides I don't think you can go wrong, and I highly suggest grabbing things everyone will like because you'll want to start forking at your neighbor's! I was impressed this place had such a strong cocktail program, figured wine was where it's at, at a place like this.",
      'useful': 1,
      'user_id': 'JUT0U3HTSB3kz9Wh7N0GqA'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 3,
      'date': '2010-12-14',
      'funny': 1,
      'review_id': 'WcgoxemFDJbv33uc4FLxew',
      'stars': 4,
      'text': 'I love this place.  It\'s definitely a little pricey, so it\'s more of a special occasion place.  Love the décor.\n\nThey have a gigantic table in the center of the restaurant where they slice your bread.  Kinda cool, but it takes up a lot of useful real estate.  Another 2 or 3 tables could potentially be in it\'s place.\n\nThe "hot balls" were highly recommended by a friend so we just had to order them.  I can\'t say I\'ve ever had anything like them before.  There are 3 different types:  Arancino - mushroom & Fontina stuffed risotto, Goat cheese - wildflower honey, and Arancino - spiced lamb stuffed risotto.  A little expensive for the size of these things, but delicious.\n\nI went with the fish (trout) for my entrée.  It was very good.  Nothing special or unique to report.\n\nI was very impressed with the service here.  I\'ve recommended this restaurant to others, and they\'ve loved it too.',
      'useful': 4,
      'user_id': 'AhoxHm569hH_PRkoegDwcA'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2013-10-07',
      'funny': 0,
      'review_id': 'k_swPjZERWnqRbP0Fuu5uw',
      'stars': 2,
      'text': "Considering the amount of rave reviews on here, I suggested our group of five celebrate a birthday at Table 17. Maybe it was an off night but I walked away with a bad first impression. So did my guests.\n\nThe dinner reservation was at 5:45 on a Saturday, so it wasn't like it was packed. Still took close to 40 mins until our cocktails arrived. \n\nOur server was very pleasant but the service steadily declined over the course of our 3 hour (!!!) meal. Scatterbrained service, slow bartender and even slower kitchen. It was clear the kitchen and bar were completely backed up. \n\nOur table was finished our bottle of wine before our mains arrived so I ordered another glass of wine so I could enjoy some with dinner. That glass arrived 20 mins after I was finished my main. Lots of apologies but little else.\n\nThat being said, the cocktails, hot balls and panna cotta were outstanding. And the space is gorgeous. However, our mains were overpriced and overcooked. We all cleaned our plates because we were starving (almost 2 hours to receive mains), but they just weren't great. The Halibut was the best of the bunch.\n\nConsidering the amount of solid restaurants in this city, I won't visit again.",
      'useful': 1,
      'user_id': '9FhtC5gKQvUCQMMGlcXW3w'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2012-01-09',
      'funny': 0,
      'review_id': '-4bMAu3ygzhG2VSn6ygyow',
      'stars': 4,
      'text': "Booked through OpenTable a few days before, for 8pm. Zipped right in to our table.\nHad the marinated olives and Diver Scallops to start...wow! The scallops were fantastic, some of the best I have had here in Toronto. I am usually not a big béarnaise fan, but found myself scraping the plate. The guanciale was a neat change too (different style of bacon).\n\nHad the pork tenderloin wrapped in pancetta for dinner, also fantastic. Well cooked and presented. Love the square plates.\nMy friend had the braised short ribs, but was mildly disappointed. Found them to be very chewy. She made up for it with chamomile grappa for dessert.\n\nThe cheese plate for dessert, was a LOT of cheese! Good size hunks of Ontario Swiss, Ontario 10 year cheddar, a blue (im not a blue fan, so didnt pay attention), a something else that was kinda gouda-esque, and wonderful creamy goat cheese. Plus bread, crackers, olives, honey and apricot jam. Massive. Took a piece of cheese with me to the car!\nMy friend had the chocolate trio. She didnt like the darjeeling sauce on the chocolate, so that kinda disappointed dessert.\n\nAll in all, i thought it was quite good, but rare occurrence in Toronto where i had a fantastic meal, but my dinner date's was ho-hum. We agreed though that we'd give it another chance based on the scallops alone.\n\nThe service we had was fine, gentleman with a bit of a british accent was quite good and knowledgeable. Too bad others had issues, thought our guy was fine.\n\nReservations are definitely recommended, was quite busy.",
      'useful': 1,
      'user_id': 'gXQfe8T1UvMpmR5rcWpGWw'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2011-10-16',
      'funny': 0,
      'review_id': 'gWxIIqLHfD0Pomiu3LilpA',
      'stars': 4,
      'text': 'I came here for weekend brunch. We arrived around 11:30 and there was no wait, lots of free tables. It was packed by 12 or 12:30 though. \n\nThe place is cozy and inviting. It\'s nice enough for a romantic dinner (NB: Sunday and Monday nights they have a $32 three-course table d\'hote, and bring your own wine) but relaxed enough for families with young kids. The chicken theme (in the logo, decor) reminded me of the old Bistro Bakery Thuet. \n\nThe brunch menu changes seasonally, so the current offerings are different from the menu on the website. I had the scrambled eggs -- smooth and fluffy with pecorino cheese and mixed peppers. It came with a green salad and warm baguette. It was filling without being too heavy or greasy. My DC had the Farmer\'s Breakfast, with a hard boiled egg in this deep fried batter, with an assortment of Ontario and Quebec cheeses (including a yummy Brie), pate, and tiny pickles. We shared an order of rosti, which was just right -- crisp and hearty. It\'s chunkier than the rosti at Le Petit Dejeuner, but not as thick as the rosti at Richtree/Movenpick. For drinks, I just had a glass of the freshly squeezed OJ. They do have $5 mimosas and caesars. DC had coffee, which was refilled diligently. \n\nThe whole meal for 2 people came to $40 including tax and tip. Pretty reasonable for a nice place and great food. Service was friendly and attentive. \n\nStreet parking on weekends is usually easy to find and $1.50 per hour. \n\nI checked-in and unlocked and offer for $36 for a "14 oz. (!!) Heritage Beef Ribeye with hen of the woods mushrooms and mint & orange gremolata" for my next visit.',
      'useful': 0,
      'user_id': 'alUuOskFSl1bODjnce2PpQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2014-09-28',
      'funny': 0,
      'review_id': 'WAzEW9fg-zy_eHE3_kRKeA',
      'stars': 5,
      'text': "I'll be frank: the only way we could afford this place was due to a gift certificate from a generous wedding guest. However, the food was right on the money. Good flavor; prompt service. If you have the money, go!",
      'useful': 0,
      'user_id': 'A-zo2jSQDnTgIoJg0sFevg'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2012-08-21',
      'funny': 0,
      'review_id': 'VAUqyws2WuYDvV3Uahx5Ag',
      'stars': 4,
      'text': "We were here for brunch a few weeks back - delish !  Service was great, food was fantastic, and as a nice surprise we didn't have to wait all that long for a table.  And a good brunch can't just rely on its food for good reviews, so the answer is yes, the Ceasars were awesome.  We don't really live in the area but our visit here was really good so we'll be back.",
      'useful': 0,
      'user_id': 'nQuC_UE-TYI5Epb5KR4AWQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2011-02-27',
      'funny': 0,
      'review_id': 'rUnkJt8Mm4OpvXzJac9P-w',
      'stars': 3,
      'text': 'I came here with my sisters for brunch on a Sunday so there was free street parking. (Yay!). \n\nThe decor is nice and the tables are well spaced out. We were seated by the bar area close to the entrance. They have huge windows which are nice but get in your eyes in the day so we asked the servers to bring down the shades. \n\nService was decent but not the best since the restaurant was pretty full and they only had two servers. \n\nThe brunch menu is pretty diverse and goes beyond the typical french toast, pancakes, scrambled eggs  with sausage/bacon/ham & hash browns deal. \n\nI had The Sloppy Guisseppe, which is a plate of a couple fried eggs over bolognese sauce and fried polenta and fresh greens with a vinaigrette dressing.  The bolognese sauce was really meaty and thick - delicious and paired with fried eggs, you just can\'t go wrong. The fried polenta didn\'t seem "fried" as it wasn\'t crispy; not even around the edges which made the dish mushy...but then again, that\'s probably why it\'s called the "Sloppy" Guisseppe. I would have preferred if I had at least toasted bread on the side for a crunch or simply just a different texture (the salad wasn\'t enough). \nMy sisters had the Pain Dore, which is french toast with cinnamon creme anglaise & berries. They liked it and thought the creme anglaise was a nice alternative to maple syrup.',
      'useful': 0,
      'user_id': 'qt1b6zXExL-uoJGRRouQYw'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2013-09-30',
      'funny': 0,
      'review_id': 'eAZDknO3E_za9AIGZjIDAA',
      'stars': 5,
      'text': "Had a very lovely meal here on Saturday. Came at 6:30 and it wasn't busy but filled up by the time we left. The crowd was definitely older, I don't think I saw anyone under 30 in there. Decor was very woodsy and rustic. But at the end of the day, it's the food that counts!\nWe had...\nHot balls: lamb risotto, mushroom and fontina risotto, and a goat cheese and honey balls. You get three little timbit sized balls for 7 bucks which I thought was very reasonable. They were so delicious, I wished we had gotten the double portion after we were finished.\nPolenta w/ Sausage Sugo: My affection for polenta knows no bounds so I loved every little spoonful of this. It came out on a slice of a tree trunk which allowed the polenta to slowly spread out perfectly over the dish. Polenta was creamy and had a nice umami flavour. Good liquid-to-polenta ratio made this comforting and incredibly addictive. Chef came out and laid the hot sausage meat sauce right on top of it. We loved it so much, we made it at home the next day...seriously.\nSteak tartare w/ frites and salad: Delish and flavourful tartare. The frites were incredible!! Get them!\n\nWill I be back? Yes! None of the dishes even remotely disappointed me and the service was solid.",
      'useful': 0,
      'user_id': 'FYAkhIj29gyWD_Lte1v7SQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2012-10-16',
      'funny': 0,
      'review_id': '4BEtZTKBXzska1oaB-mhAQ',
      'stars': 4,
      'text': 'One of my favourite restaurants in the neighbourhood. Food is consistent and the service impeccable. \n\nThe steak frites are the best in the city. Scallops, duck confit and short ribs are also regular favourites.',
      'useful': 0,
      'user_id': 'wnRv7PRPZ0jU_flnGJnYAg'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2011-06-15',
      'funny': 0,
      'review_id': 'ny9M2uEvdrxPpIxZ0B5euw',
      'stars': 5,
      'text': "Literally my FAVORITE new restaurant in the northern hemisphere. I've all but given up on Toronto until finding this gem through a friend in New York who raved about it. \nTook my vegetarian children for dinner, and we had an absolutely outstanding experience. \nEverything from the friendly and funny staff, to the custom vegetarian dinner that the teens devoured, was spectacular. \nThe highlights: Hot balls, Brussel Sprouts, Diver scallops, Tartare, and the pasta was so light and tender with incredible delicate flavors\nKudos.\nWe'll be back soon!",
      'useful': 0,
      'user_id': 'f-dzSIu2QURZ8Y_fxo7J-Q'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2013-11-15',
      'funny': 1,
      'review_id': 'elb_JjGey_SGKUyo91c8hA',
      'stars': 4,
      'text': "The first time I came to Table 17 was a little over a year ago. I took my grandmother and my aunt out for dinner. The moment I came to this place I was very impressed. \n\nFirstly, it was very conveniently located to us. Secondly, it has this very home-y feel and decor. Nothing felt overwhelming. We came on a Monday night. (To my knowledge Sunday and Monday nights Table17 offers prix fixe menu.) We all ordered different staters, steak and different desserts! However, what made me fall in love with this place is thee amazing home made olive oil. Their olive oil tasted amazingly fresh and you can't help but eat all of their fresh bread with it. Unfortunately, in multiple occasions I've asked if I could purchase a bottle and to my disappointment they do not sell their olive oil. :( \n\nThe only thing I found about this place that did not satisfy my expectations were tht the servers are very slow and not as hospitable as other servers from my other experiences. \n\nOverall, I definitely recommend this place. Especially their duck confit which is amazing! \n\n-CL \n\nPS they take reservations and it gets pretty busy so be sure to make them. :)",
      'useful': 1,
      'user_id': 'DuoBTfKNM67965BcrlYBYg'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2009-08-10',
      'funny': 0,
      'review_id': 'RNiNQyQ78zvKunbP09ChJw',
      'stars': 3,
      'text': 'On a stormy, humid night in Leslieville, my partner and I stumbled upon this welcoming looking restaurant with the rooster out front, and it turned out quite well.\n\nThough the category on Yelp is listed as French, I would say the food is much closer to Italian if you had to pick a cuisine.\n\nRight off the hop the service was questionable. Benefit of the doubt moment: it was after 8 pm on a Sunday -- which in Toronto means you\'re lucky if anything half-decent is open at all -- and the place seemed  understaffed. \n\nStill, having to stand awkwardly for 10-15 min in the non-foyer, while  waiters whisked by us was not too pleasant.\n\nAnd once we had a table, service continued to be spotty. Our waiter was friendly and self-effacing, calling us his "little neglected table." This was not inaccurate as even getting our initially requested waters (it was sweaty out!) took nearly 10 minutes, and later he left us with dessert menus for 20 minutes before coming back to see what we wanted, our appetites for anything sweet having seriously waned.\n\nFood-wise though, things were great. We weren\'t terribly hungry, and so avoided the table d\'hote fixed price menu going with the "shared food" section on the left column of the menu. It\'s a limited selection here, and we ordered about half of the available items, and were not disappointed. \n\nRich white polenta, brought out on a wooden cutting board was accompanied by an aproned chef who ladled a "sugo" of veal, zucchini and fresh basil on top. It was heart-warming, subtle and delicious, as well as a generous portion for two to share.\n\nAlongside the polenta we had two smallish ramekins. One contained "sauce & balls" a cheekily named but very tasty take on classic meatballs in tomato sauce. The other had what was possibly my favourite dish of the night, a simple roasted beet preparation with blue cheese. The multicoloured heirloom beets were "al dente" but tender and went amazingly well with the semi-melted cheese.\n\nWines were great as well. My glass of Rollegrosse red ($11) was smooth beyond even the menu description, and my companion\'s "Anna 6" white was very dry and very refreshing. \n\nIn addition to the service problems, my one other issue was the post-prandial whiskey, a single malt ($13) that came in a far-too large glass (we were informed the kitchen was "out" of snifters...) and which I found harsh and not to my liking. But that could be a matter of taste.\n\nTotal cost of meal for two: $77 + tip.\nThis would increase to over $100 easily if you go with the $30 table d\'hote option + drinks.\n\nI would happily give this place another shot, hopefully on day where service was at full steam. Until then I can\'t give it more than three stars out of five.',
      'useful': 3,
      'user_id': 'bCPO7i-x5eZ41lLVOwuQwQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 2,
      'date': '2015-04-15',
      'funny': 0,
      'review_id': 'v79TP4gkdC3ud0y9YjoFSA',
      'stars': 5,
      'text': "Food: 9.5/10\nAtmosphere: 8/10 \nService: 9.5/10\nOverall experience: 9/10\n\nA lovely experience overall. Went and ordered:\n- foie gras appetizer\n- polenta (main #1)\n- pasta special\n\nAll the food was great. You could tell it had been prepared with great care and came out piping hot.\n\nThe service was also fantastic. The server was extremely helpful in selecting a wine, and providing recommendations on the menu. We didn't feel pressured at all during the meal to hurry up or order more AKA increase the bill amount.\n\nTHE BOTTOM LINE: wonderful food, and equally as wonderful service.",
      'useful': 1,
      'user_id': 'wu-ijx0ZiBThoc9tRHf_Hw'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2015-01-27',
      'funny': 0,
      'review_id': 'y-ARDKgXj26RjFm7AhoCdg',
      'stars': 5,
      'text': "We went on a Travelzoo coupon and we arrived early for our 7:30 reservation. No problem, here's your table. Our server was fantastic! I can't remember his name, unfortunately. He made excellent suggestions, was very helpful, knew all the dishes, and the ingredients.\nWe had the Beef Carpaccio and Squid Ceviche in Gozpacho to start and were blown away by both!\n My wife was really looking forward to the lamb shank, which was not on that days' menu, so our server suggested the Venison loin. My wife had never had venison, so it took a little convincing, but she loved it.  It was probably the best venison I've ever tasted. I had the polenta with the wild boar sugo. very good. We also ordered the Roasted Jerusalem Artichokes, which were fantastic. \nThe wine was very good, and dessert was excellent, Maple Panna Cotta and Sticky Toffee Pudding.\n All in all an excellent dining experience.",
      'useful': 0,
      'user_id': 'B0izwVcG6y-NbWvfNBVSAg'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2015-06-17',
      'funny': 0,
      'review_id': 'ndqc1YQPz3kTYbJIFORrdw',
      'stars': 5,
      'text': "This hidden gem is wonderfully delicious. Ordered from appetizers to entré. Everything is great. Great service, great food, great environment. Can't complain about a thing here. Even their bathrooms are sparkly clean. Heard They're moving to another location soon or renovating. Better call ahead and ask. Duck confit and their pasta in beurre sauce is mind blowing. Worth a try If you're in the area. Something different for date night.",
      'useful': 0,
      'user_id': 'a39kizwaoThU6uLIv_Ke2g'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2012-07-08',
      'funny': 0,
      'review_id': 'K2BYOl6veq4_aykdrDqSYw',
      'stars': 4,
      'text': "We were back here this evening.  Delish!\n\nI ordered the beef tartare - it was pretty tasty, large enough to share as beef tartare is both an acquired taste and fairly rich on it's own.  My main was the leg of lamb, it was a little well done for my liking, but the flavour was quite nice - the morel mushrooms were loaded with a burst of buttery flavour.  \n\nMy bf started with scallops, they were quite large for scallops, and quite nice, and had a white fish, not trout, maybe halibut, that was also quite delish.  I love the decor, and upon reading other reviews am super sad that I missed out on a group on!  It was wonderful anyway, and I think a great place to dine.  \n\nWe went for an early seating (a bit before 6), and I'm happy to report, that next time, I'll either go for another early seating or something later (after 9 in the evening), as it can get quite loud, and that makes it hard to hear your dining companion!",
      'useful': 1,
      'user_id': 'J3ucveGKKJDvtuCNnb_x0g'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2014-05-21',
      'funny': 0,
      'review_id': 'mP2j4QGj72q5VM-teKKcgg',
      'stars': 5,
      'text': "Love this restaurant!!! Awesome brunch, $10 for anything on the menu. My hubby doesn't get super full off of the stuff, but I'm good with the amount. Awesome decor too!!",
      'useful': 0,
      'user_id': 'dvr-iz4B9eNzDZ-uvjvHPQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2014-06-01',
      'funny': 0,
      'review_id': 'WaS0TdcqyAN-T4C_N0X9RQ',
      'stars': 4,
      'text': "We went here for Mother's Day brunch.  I had a very good egg omelette with guacamole and my wife had the French toast. Very nice intimate setting and the menu has very nice choices.  We got our son a fresh berry yogurt which I ended up eating most of it because it was so good. My son was still quite happy regardless. We will definitely be back to claim our check in deal but that is not what would bring us back. The exciting dinner menu is why we will be back soon.",
      'useful': 0,
      'user_id': 'L3X_u_PWEglem1ETrPMjpg'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2014-01-15',
      'funny': 0,
      'review_id': 'vv4S6OxIfBGbLTu_qxjMVA',
      'stars': 5,
      'text': 'We had a great time. Dinner for 4 - with a healthy amount of food (or unhealthy - depends on how you look at it).\n\nThe polenta had their "reindeer sauce" which consisted of venison, spices, and berries. Delicious. \n\nOysters were tasty, and the braised pork was unbelievably good. No idea what they used for aromatics - but there was a hint of fennel that was incredible. \n\n$400 for 4 people and drinks. Not a value dinner, but enjoyable, and guarantees my return. \n\nOh. Peanut Butter Panna Cotta. Order it and thank me later. Servers were hilarious, attentive, and accommodating. I just can\'t see anything wrong with this place.',
      'useful': 0,
      'user_id': 'ncULBoGp-kRF3zExhbdDfw'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2014-12-28',
      'funny': 0,
      'review_id': '0IeuJHDMwJxHPMMs-Bt0BA',
      'stars': 5,
      'text': "I left Table 17 as pleased as could be last weekend- indulging in the very reasonably priced Sunday night table d'hôte we enjoyed the scallops, salted cod, polenta, boar sausage, panna cotta and gingerbread cookies with eggnog (!) We were particularly wowed by the boar but the entire meal was fabulous.\n\nAs a big bonus the place is BYOB with no corkage fees for one or two bottles.\n\nA couple other reviewers noted a lack of attentiveness on the part of the servers- I would contend that our experience was the complete opposite of this. The server was charming, present and exceedingly helpful. Maybe this can be racked up to visiting on an early December Sunday evening? In any case, it was exactly what I was hoping for and more.",
      'useful': 0,
      'user_id': 'LRKmxCcf6ZkxyujPdmp5zQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2011-02-04',
      'funny': 3,
      'review_id': '8pDkQMl1UYO7goTuKHDaNw',
      'stars': 2,
      'text': 'I wish I could share Avitania B.\'s Groupon enthusiasm because I found this place a big let down. I too had a $50 Groupon here and was really excited to give it a try. \n\nAt first I was concerned that this place may be too fancy and romantic for a casual dinner with a friend, but the other patrons made that problem go away fast! Picture It\'s 9pm and you are trying to enjoy your meal next to four tables of three super loud middle-aged women who are all gossipping and complaining about their "first-world problems": She only got a 1.5 karat ring... I hate it when my maid doesn\'t iron the cuff on my pants... blah blah blah. I honestly wanted to tell them to all shut up. This is by no means the restaurant\'s fault but killed the ambiance for sure.\n\nThe food itself was good but the portions are quite small. I also found the menu selection to be minimal with only 6 entrees to choose from including a super tasty but super small bowl of pasta with wild mushrooms for $18.\n\nI found the service friendly but very slow and inattentive. At times I felt as though I might need a bell to get the waiter\'s attention.',
      'useful': 3,
      'user_id': '2VKVhy1SwaixHCeiWglLUQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2012-08-03',
      'funny': 0,
      'review_id': 'kJsYjLDPMO4b65NX3-ocYQ',
      'stars': 4,
      'text': "This is one of my favorite restaurants in the city. I love that they aren't over-priced (not cheap either, mind you) and that they can usually manage to squeeze you in at the end of the service without a reservation (I'm a late eater.) The appetizers are where it is at here!",
      'useful': 0,
      'user_id': 'M5Ci03Ce9gab2jahundlRQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2013-08-26',
      'funny': 0,
      'review_id': 'VyDNI9gAjAcTnSgW_Y4OBQ',
      'stars': 4,
      'text': "Had dinner here once and it was ok, but the brunch was great. I had the Sloppy Giuseppe - Two sunny side eggs top a crispy, fried polenta cake, surrounded by a moat of Bolognese sauce. Awesome. I'll definitely be back for that alone.",
      'useful': 0,
      'user_id': 'vMrnL8HA0OvnME8PC0Dg1A'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2014-06-01',
      'funny': 2,
      'review_id': '1goCEpy95A_Vxyo-IT1Yag',
      'stars': 4,
      'text': 'Is the food at Table 17 excellent? Yes.\n\nIs that enough for me to give it five stars? No.\n\nI\'ve been twice, once for dinner and once for brunch. On both occasions my meals were delicious, fresh and expertly prepared and presented. Wine and coffee top notch also! Yet, I still left hungry, and somewhat frustrated.\n\nWhy? The portion sizes at Table 17 are just too small, even for the average appetite. Add in frankly exorbitant costs, and it becomes clear that this place just ... well, just could be so much better. My French toast for brunch was served with about five berries -not even a salad - and must have amounted to about three slices of bread in total. $10 before tax.\n\nThe dinner menu was infuriating in its unnecessary pretentiousness. It took me about five readings of the menu to actually understand what was on offer. Less of the fancy culinary jargon and arbitrary French/Italian synonyms for basic food items, please. My chicken mushroom tagliatelle was flawless as it was - was there really any need to describe it as "Hen of the woods (is that the new word for chicken?!) hedgehog mushrooms, with tarragon beurre fondue"?! \n\nAgain I must reiterate the pricing. Two mains, one shared side and a glass of wine came to $70. \n\nThankfully, the friendly service and general lively ambiance is spared the unnecessary pretensions of the menu. And if the food continues to be mouth-wateringly delicious (and trust me, it is), then I can\'t see why I wouldn\'t return to Table 17, for a special occasion. \n\nThis restaurant could be a true five-star establishment, if only it had bigger portions and a more down-to-earth, straightforward menu, that would allow the world-class food to speak for itself.',
      'useful': 1,
      'user_id': 'Oxe8DXjBOi6C-cR2w64zOg'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2013-04-10',
      'funny': 0,
      'review_id': 'EGHciRl1EuEL16fXNvrFfg',
      'stars': 4,
      'text': 'Brunchy time and we have a winner.  To start, we got a seat right away on Sunday for brunch.  I highly suggest going around 11:30 as you just miss the 12pm rush.  It is a stylish restaurant and the service was top notch.\n\nFor my meal, I had the french toast.  It was good - light and perfectly cooked.  The cinnamon anglaise tasted like melted ice cream.  There was an assortment of berries as well.  Their orange juice was freshly squeezed and we were in and out in under 45 minutes. \n\nDefinitely will be back for another brunch extrodinaire!',
      'useful': 0,
      'user_id': 'Yl7OYdHuYmr7K-IW9_ayng'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2009-06-06',
      'funny': 2,
      'review_id': 'enYFb4pnwyarggyWSbCAVA',
      'stars': 5,
      'text': 'What an experience: from the shabby-chic decor, to the wonderful service, Table 17 was an all-around treat. \n\nFirst off the "hot balls" ( I know, I know), were phenomenal: arancini stuffed with risotto, another with goat cheese, and finally the crab. Each came with their own dipping sauce: amaaazing.\n\nFor dinner, the lamb tenderloin with a parsnip puree was cooked to perfection, a friend ordered the steak frites (delish), the beef tartare was tender and spicy and you can\'t really go wrong with an arugula salad, which was fresh and tasty. The presentation of the halibut wrapped in ramps (mild leek), was a show-stopper (as was the pomegranate mojito). \n\nGreat place for a date, for a nice celebration, or just a nice culinary experience.',
      'useful': 1,
      'user_id': 'asPq38KJqzxeLotQCdxQgg'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2012-06-07',
      'funny': 1,
      'review_id': 'dNrRyGjK9ur9UBZGMeBWyA',
      'stars': 4,
      'text': "Was here for my first time just last week. Loved it. It was highlighted by the Soft Shell Crab and the Hunter's Mash. Also tried my first pint of Chambly on a recommendation from the bartender. I will be hunting this beer down in other restaurants I visit. \nAlso, they are very quick to respond to any questions you have via Twitter.",
      'useful': 1,
      'user_id': 'XXxrsJ3KpTFybcBFpZLYiQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2012-01-16',
      'funny': 0,
      'review_id': 'pTq1ez9s0SNxaSlSxLBGGg',
      'stars': 4,
      'text': "Bottom line: great food, cute place, fabulous dinner. Would definitely go again. On the pricier side for the neighbourhood, but nothing ridiculous. Remember to make a reservation.\n\nWent to Table 17 for dinner last night. My dinner date and I stupidly didn't think to make a reservation at 7:15 on a Saturday night (dumb, I know). So I definitely recommend a reservation. He had been to Table 17 before, I had not. Of course there were no tables when we arrived, but the staff were more than happy to seat us at the bar, which was thankfully empty when we arrived. Usually not a fan of having a nice dinner at the bar, but the barman managed to do a good job of staying out of our face, but being attentive at the same time. Of course the seat wasn't the comfiest, but that was our fault.\n\nFood:\nThe server/ bartender made sure we knew all the daily specials and even recommended a few and took our drink orders (just beer). He also brought us a basket of bread, though it almost wasn't necessary because our appetizer came quickly.\n\nTo start we shared a Buffalo Mozzarella salad (arugula) with spiced pumpkin puree and honeycomb - yum! My date wasn't too sure about the honey at first, but actually really liked it, was a nice flavour with the mozzarella. The warm pumpkin  with the cheese was my fave part - so delish. \n\nFor my main I chose the pasta. It was capellini (I think, can't quite recall)...with gorgonzola cheese, roasted hazelnuts and I'm sure I'm forgetting a few ingredients. It was awesome - great mix of flavours and the pasta was cooked perfectly and served hot. I was skeptical about the hazelnuts at first but they were a great compliment to the gorgonzola. The portion was pretty big for this type of place, so my date had a bit of my pasta too and he also thought it was yum.\n\nMy date had the beef short ribs and said they were excellent and that the meat was falling off the bone. (As someone who does not eat beef or pork I was assured this was a good thing ha). It came with root vegetables and parsnip puree. Just as we were digging in our server brought a little bowl with extra root vegetables because the chef had suddenly realized my date didn't get enough. Once he pointed it out we realized there were only like 3 mini veggie cubes in his dish so it was nice that the staff noticed before we even did. \n\nWe contemplated getting dessert for a minute, but were pretty stuffed. Maybe next time I'd skip the appetizer and try a dessert.\n\nOnly main downfall was sitting at the bar, it's not the comfiest to sit on a stool and it felt a bit crowded when other patrons were seated at the bar. It was also a little chilly/breezy as it's near the door and it was cold out. But that was our own fault.\n\nNot sure what the total bill was in the end for the 2 drinks, appetizer and two mains, but our mains were around $20-25, some on the menu were a little pricier.\n\nOverall great food, good service, cute place, fairly priced (though pricier than some of the other places in the neighbourhood). Definitely recommend.",
      'useful': 1,
      'user_id': 'g5wYoa19wv1X0UOYVX1IlA'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2012-02-03',
      'funny': 0,
      'review_id': 'SQrS_X4z4_HolRUZ-9rHfQ',
      'stars': 4,
      'text': "If you want to have a more quiet place to meet up with your spouses parents for the first time - that's the place to go on a Saturday morning. Make sure to come early as they don't accept reservations for less than groups of 6!!",
      'useful': 0,
      'user_id': 'iF3V9eqIVAjX8sj96hJ_sw'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2013-02-23',
      'funny': 0,
      'review_id': 'CWOGe1a0q-FR_ofzEt-MEQ',
      'stars': 5,
      'text': 'We really enjoyed this place.  Food and service was excellent.  Table 17 has different and unique things to try! I highly recommend the goat cheese balls, they were fantastic!!',
      'useful': 0,
      'user_id': 'RYP7YVMcYPfO78H1OTzt7A'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2008-09-29',
      'funny': 0,
      'review_id': 'F7SZgPC5chCn4DItevIJGg',
      'stars': 4,
      'text': "I went here for the brunch the other day in this little bistro-style restaurant in Leslieville and I have to say I was pleasantly surprised. \n\nThe restaurant itself has a really cool decor - very minimal but rustic decor, brown paper tablecloths, quaint, romantic tables and nice, high ceilings. I couldn't get over how much personality the restaurant has, especially considering how new it is. \n\nAnd obviously its age has no affected its popularity. The place was packed at about 1 pm on a saturday. But considering how tasty the food was, I was surprised how busy it was. I ordered a mimosa and the toritilla - a sort of spanish-style dish that takes eggs, potatoes, cheese and bakes it into some quiche-type thing. It was served with a fresh tomato dip and some greens and it was absolutely delicious. I also sampled my boyfriends cheese plater, toasts with a variety of cheese and fruit, and it was also very tasty. \n\nMaybe it's because it's new, but the service was the one thing that didn't make this a 5 star occasion. Our waitress was slow, took forever to take our order and then forgot what we wanted. But the food was so good that it didn't ruin the experience.",
      'useful': 1,
      'user_id': 'y3MNMa0SG_cHC1cwmmntpQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2009-07-07',
      'funny': 0,
      'review_id': 'bqJP8gO-f2OadqqdXFdhPw',
      'stars': 5,
      'text': "Great decor, great food, great drinks.\nFood was really good, not a huge selection on the menu, but it was also a set menu for the night, so can't comment on the normal menu. Food was very tasty, and there was a very different selection of draft beer, all ones I hadn't heard of, but the one I tried was very good. Will definitly be going back!",
      'useful': 1,
      'user_id': 'nZiF42dKhSImm-1oOXxLVA'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2014-09-10',
      'funny': 1,
      'review_id': '650QSmmJ_MaF4WN1V_gFZA',
      'stars': 3,
      'text': '"Table 17 is the rare restaurant that\'s both a neighbourhood hangout and a culinary destination," is the opening line as you land on their homepage, a quote from Toronto Life from 2013.  That\'s a tall order to fill and sets the standards high.\n\nI wasn\'t a fan of Table 17 back then and I\'m not surprised that my most recent dining experience was no different.  It satisfies the former part of the claim (being a neighbourhood hangout) but is far from being a culinary destination.\n\nThe kitchen seems to work along at its own pace which errs on the side of slow while servers try their hardest to service the two-roomed restaurant.  While I wouldn\'t say they\'re under-staffed I often found myself turning my head in all directions to flag our server down.\n\nA meal at Table 17 errs on the side of pricey and while I have had many expensive meals in my lifetime as a food blogger; this one definitely errs on the side of overrated.\n\nI\'ve always lived by the mantra that if a restaurant\'s bread was not up to snuff that it would be an omen of the meal to follow.  Such was the case with the baguette sloppily appearing at our table; one has to ask for the communal balsamic vinegar and olive oil bottles because the servers can\'t be bothered to offer it.\n\n\nOysters come with a trio of condiments;  two mignonettes and the standard horseradish.\n\nWe opt for the halibut ceviche which, with its pretty plating is a disappointment as it turns out to be bland and flavorless.  The saving grace are the pea shoots, flavored with a peanut, chili and lime vinaigrette that is stellar and full-flavored.  I would happily eat a plate full of that as an appetizer.\n\nFor our mains we settle on an Ontario leg of lamb; which came deboned and served atop a bed of fava beans, roasted spring onion and butter poached black trumpet mushrooms.   Our table of 3 simultaneously shared a single thought - where\'s the rest of it?  To add insult to injury, the lamb was lukewarm (and the reason was in no way attributed to my occupational hazard of photographing the dish).\n\nThe Lake Erie pickerel is equally disappointing as we begin to pick at our food by this point.  The radish fennel & grapefruit "salad" which to me resembles more of a slaw is bland and lacking oomph.\n\nOur dinner suddenly finds salvation in the Spaghetti Al Anatra; its texture loosely reminding me of ramen cooked to a perfect al dente as the duck confit does its magic at flavoring the dish.  The crispy duck skin is a little scant but we happily devour the noodles.\n\nSide dishes are available à la carte and we opt for the sautéed kale.  It\'s nothing to write home about but gives us our daily dose of vegetables.\n\nAs with most things, what goes up must come down and in aviation-speak our dinner had a hard landing.  A dark chocolate torte that was reminiscent of a two-bite brownie drizzled in cheap store-bought chocolate sauce with some berries thrown on top.  The Panna-Colada was equally pedestrian as it loosely resembled those cheap complimentary desserts you\'d get at the end of a Chinese dinner - all gelatin and not much else.\n\nThe message at the end of the day is - don\'t let the appearances at Table 17 fool you.  While everything looks great from the outside, the experience lacks substance.\n\nOur meal with 2 glasses of wine and 3 coffees came just shy of $300.  Suffice it to say, this will be the last time I dine at Table 17.',
      'useful': 1,
      'user_id': 'HFItzRohDHZvcKDrM6ABZg'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2015-06-29',
      'funny': 0,
      'review_id': 'cYuLXekMwyxb7bCNes-JdQ',
      'stars': 4,
      'text': 'Stopped in right after lunch shift for a couple drinks. Bar tender was really helpful, friendly, and made for a great weekend afternoon drink. That said, they were definitely on the pricey end.',
      'useful': 0,
      'user_id': 'SRSmzTPA2dX2Zpfa2uE4sA'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2009-08-09',
      'funny': 0,
      'review_id': 'Bxxq3FoDgeyCN94P_dje_w',
      'stars': 3,
      'text': "Great service, the food and drinks were a miss.\n\nLimited menu so we went with the waiter's recommendations. The hot balls we were excited to try. 1/3 made the grade. Each hot ball is something different, fried risotto, goat cheese or crab croquette with its unique dipping sauce. Loved the goat cheese and honey, the other two were okay.\n\nPOLENTA - served hot on a board with a duck confit topping, pepper and parmesan cheese. The far and away highlight of the meal. Came served on a board with a few sprigs of fresh basil and the chef served the topping at the table. Could have eaten this for dinner alone and left very impressed.\n\nMains were lamb striploin with a ratatouille and a steak with the bone left on. Both were disappointing. The steak was very tough and overdone. The lamb was tender but lacked flavor and the sides seemed to be leftover add-ons. \n\nThe recommended Malbec was spot on. The decor was interesting, low lighting and some strangely heavy music detracted from the atmosphere.\n\nOverall, I'd have to be convinced to go back. Maybe if someone promised me I could get two polenta and some wine.",
      'useful': 1,
      'user_id': 'TZelLnaGbHgaG7LBI6rfgg'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2014-07-18',
      'funny': 0,
      'review_id': 'fdsak3Yi-XGcD4GCeMvpOw',
      'stars': 4,
      'text': "I went here with a group of 4 late on a friday night at around 9:30 pm. We had a bottle of wine, foie gras, risotto, the ribeye, and a few other sides. Everything was perfect. Service was great. Prices were not cheap, but worth every bit. This is a place is good for 2 to 4 people who don't mind sharing some delicious plates and paying for quality.",
      'useful': 0,
      'user_id': 'TYiDRwfIUBEos45ERdzeAw'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2013-02-28',
      'funny': 0,
      'review_id': 'rlEsmhITN6zfzHWepYuT0g',
      'stars': 4,
      'text': 'PEI oysters with the lime-tequila sauce were great.  The menu on the website was out of date, so the bearnaise I was expecting to have with my steak frites was not available. However, the chimichurri sauce that replaced it was very good.',
      'useful': 0,
      'user_id': 'He-fxlDE9JB3hOlmQ0gFIQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2014-06-14',
      'funny': 0,
      'review_id': 'Xn3eG_RukBlHubUivYT85g',
      'stars': 3,
      'text': 'Had a travelzoo voucher for this place. Went on a weeknight with my bf. waiter asked to take our coats, I paused to think if I wanted it taken and decided yes. Waiter remarks "might as well get your money\'s worth, right?". I was a little offended at the comment, I\'m not going to lie to you. I am 28 year old professional but am constantly mistaken for a teenager- I think he judged us based on my "age" and our voucher. \n\nAfter that initial snafu his service was fairly good while he was around. He seemed genuinely interested in our opinions on the menu and food. Had the beef tar tar, which could have been better. The taste of Dijon mustard was very strong and the meat was cut up almost to mush (rather than small chunks). Waiter changed shifts while we were eating but we were never told and were made to wait a long time. \n\nThe restaurant crowd wasn\'t my style either. Seemed like accomplished mid life adults who care a lot about $ and image. I prefer a more down to earth crowd (despite $ or age) and atmosphere- this could have just been the night we went (Tuesday). Food was good  but seemed slightly off mark. I expected better based on the reputation and price. \n\nI honestly felt like we were given the "pretty woman" young+poor treatment. Guess you can\'t tell by looking at us that we have good dual incomes and no children to eat up our $. \n\nBc we had the voucher I didn\'t mind as much that we weren\'t taken seriously and that the food was lacking- we were underpaying for it anyway!\n\nAll in all the food is a lot better than most restaurants. I think my dish was possibly good but not put together with has much care on a that particular night. My mashed potatoes were absolutely fantastic. Deserts were good too, nice and light. \n\nI wouldn\'t chance going back and getting similar service while paying full price. It wouldn\'t be worth it.',
      'useful': 1,
      'user_id': 'uyEIiUd0qGkZySz4kbA0uQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2013-05-06',
      'funny': 0,
      'review_id': 'kLMNy32I-Y0B8f6NbaO5pQ',
      'stars': 4,
      'text': "Rarely do I ever venture east of Yonge. It terrifies me. But when my BFF requested we go to Table 17 for her birthday, I put on my big girl pants and made my way to the East. Walking along queen, east of Broadview, I was amazed at how much has changed (I lived on Broadview and Dundas years ago and this was NOT the place to be). \n\nOnce we arrived at Table 17 I was really impressed with the overall vibe. Warm, inviting, with minimal segregation from the kitchen to the dining area (I love seeing who's making my meal!). \n\nI had oysters to start and the beef tartare for my main. It was chalked full of flavour. Everyone else at the table really enjoyed their dishes as well (Some fish, some pasta... no weak meal was served). \n\nThe server was wonderful. He had a great balance of being helpful and present at times and then absent when he wasn't needed. \n\nGreat venue... worth making the trek across the yonge street tracks.",
      'useful': 0,
      'user_id': '6X9PioodbNo5nnkZTFWhWQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2012-02-04',
      'funny': 0,
      'review_id': 'SbP4WoInuFF3KxY33mocOQ',
      'stars': 4,
      'text': "Had brunch again at Table 17 for a third time and each time it's been better and better.  Last time I had the daily quiche and it was good.  Today I have the Alsatian tart with salad and I enjoyed every bite of it.  \n\nI'm really shocked to read all the reviews about bad service.  Having been to this restaurant six or so times now, I have never once encountered any rudeness.  Maybe I've just lucked out.",
      'useful': 0,
      'user_id': 'VVm-TFCpi9M1-k8ED0l1eA'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2014-08-24',
      'funny': 0,
      'review_id': '3ljuhghMnJDB_ZuGiVtRvQ',
      'stars': 5,
      'text': "I am reluctant to write a review for Table 17 only because I want it to be Leslieville's best kept secret! I went for brunch with my husband and a friend. For Saturday brunch at prime brunching hour, we were quickly seated. The place wasn't full like all the surrounding places so not only were we served quickly, but we were also not rushed out the door. It made a great easy Saturday morning. \n\nServing staff were friendly and helpful. We loved that their bread comes from Bonjour Brioche, the best bakery east of the Don! My tart was so flaky and warm. Great sweet and savoury flavours in an appropriate portion for brunch. Everyone's plates were well-presented. \n\nThe decor had rustic charm and we didn't feel crammed in like many small brunch spots. \n\nOur coffees and water were constantly refilled and I spent an easy afternoon sitting and catching up with friends. What more can you ask for to make an enjoyable and easy-going Saturday brunch? \n\nI would definitely return to Table 17.",
      'useful': 0,
      'user_id': '3MbAc0Q7WEIGBIihksNTJA'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 2,
      'date': '2011-05-25',
      'funny': 0,
      'review_id': '_lRk2d0vsLvJNrpzn_e85w',
      'stars': 5,
      'text': "The only downside about this place is that somehow I've only just discovered it!\n\nMy friend and I walked in around 11:00am on a Saturday after a workout. The restaurant was reasonably busy but there was no line-up - a welcome change for weekend brunch in Leslieville/Riverdale! The interior is dark wood/brick and quite cozy. Realistically, we were a little under-dressed for the occasion but they welcomed us in any event.\n\nMy friend had a dish (whose name escapes me) but consisted of eggs served on bolognese sauce on top of polenta. I had the quiche with a tomato salad. Both were really fantastic. Brunch entrees were reasonably priced at about $10.\n\nDuring our meal we saw a server walk by wih an order of frites and quickly decided we had to order them. No regrets here - crispy, salty, and delicious! At $4, totally worth it.\n\nI would absolutely return and recommend.",
      'useful': 1,
      'user_id': 'tYk7mMGGFl3gLfmhST5L-A'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 2,
      'date': '2012-03-19',
      'funny': 1,
      'review_id': 'XLcIW09xJTIgUzjwUkT-5A',
      'stars': 4,
      'text': "Dropped in on a Sunday for Table 17's BYOW Table D'Hote.  There was a '96 Bourdeaux that just had to be drunk and so my friend and I spent a lovely Sunday evening dinner obliging.  \nTable D'hote gives you a  selection of pork belly, steak frites, trout and a vegetarian option with a selection of first courses and desserts to go with it.  \nNeither of our dishes disappointed.  \nWhat we got was high quality, well prepared bistro fare at a very reasonable price.  Service was efficient if not overly friendly. \n\nI remain a fan of this cosy Queen East bistro and wouldn't think twice about many repeat visits.",
      'useful': 2,
      'user_id': 'gMfkWTxRVZtJAMm_adQKsQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2013-02-12',
      'funny': 0,
      'review_id': 'zQiUh7rGAGFoVE6n2mpvfw',
      'stars': 5,
      'text': 'The fish croquette, pork belly and collard greens price fixe for $35 were to die for. Uncorking two bottles of Chateauneuf du Pape for free was simply priceless.  I am just not used to actually getting such value for money at restaurants in Toronto.',
      'useful': 0,
      'user_id': 'hrA2TKOiPrJ0Va3ceAgq9g'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2010-02-12',
      'funny': 0,
      'review_id': 'Ing_TePSktj9qCvI-z07Hg',
      'stars': 4,
      'text': 'Nice atmosphere with good food and attentive service.\n\nThe much-touted hot balls appetizers are a little over-rated: they are basically breaded, deep-fried balls of something with dipping sauce and were edible but nothing to write home about.\n\nFor entrees we had the delicious beef short ribs and black cod.  Both were cooked to perfection and in manageable portions (I often feel that mediocre fine-dining restaurants throw large portions at you to compensate for weak flavour).\n\n$250 for 4 people with some booze.',
      'useful': 2,
      'user_id': '3Le78qZoyKE0Od0sGU8EBA'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2014-03-28',
      'funny': 0,
      'review_id': 'pPPk-A3Hx6YApitBccDWsA',
      'stars': 3,
      'text': "Lovely ambience, we had a nice table for 2 by the window. I'm sure the table beside us could hear our conversation as we could hear theirs when there was a pause in ours. \n\nThe service started off being attentive and slowly started to slip (e.g. My date had to go find a waiter to get me a side plate for me to put shells from my seafood linguine, despite me having asked for it 5 min earlier). The fresh oysters were amazingly delicious and fresh! \n\nI enjoy dipping my bread in olive oil and Table 17's olive oil was no better or worse then any other nice restaurant's. My seafood linguine was average. I believe my date had lamb and he said it was fine. We also ordered a bottle of wine and felt their wine selection was appropriate. Overall, we weren't blow out of the water and have made no plans to go back as we were expecting more. Make sure you make a reservation!",
      'useful': 0,
      'user_id': 'zbSBmoujGgIPuNWQcny14g'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2010-01-05',
      'funny': 2,
      'review_id': 'J5x0w4x7D1mkUAtUbK9dMw',
      'stars': 3,
      'text': "The summary of 3 stars for Table 17 is that it's too expensive for it's location and quality.  It's still pretty good, I just don't like seeing a cheque for $60 for brunch.  \n\nThe good:\n- The food is good and the ingredients are top quality\n- Sundays and Mondays it's bring your own wine with no corkage fees\n- Brunch has a 14oz Blanche de Chamblis with a splash of OJ for $4 which is the best value on this menu\n- They do potato rosti right\n- Sarah T orders TWO sides of bangers because she loves them so much... I really do like their bacon\n- Good service both times I've been\n- Proximity to my house\n- they use opentable\n\nThe bad:\n- It is too expensive.  The breakfasts don't come with sides, and sides each cost $4. So scrambled eggs with toast is $10, plus bacon is $14 plus hashbrowns is $18. I could maybe stomach $14.95 for a complete brunch, but I hate paying 10$ for eggs then having to add on the things I feel should come with it.  Maybe I'm not classy enough to eat bacon here.\n- It's not as good as Le Select brunch, which I don't mind splurging $60 on once in awhile. I won't spend my $60 here instead of there. (both times I've been here Sarah T picked.)\n\nNo funny stories.\n\nWould I go here again?  Maybe.  I might try it out for dinner one one of the BYOB nights.  Man, this place is just too expensive.  I know that's all I keep saying about it, but they charge too much for the food here.",
      'useful': 8,
      'user_id': 'AlXx0P-OhUylep0jNi773g'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2010-07-16',
      'funny': 0,
      'review_id': '1K1YYeyrQfpaYc021nWKYg',
      'stars': 4,
      'text': "This was a nice little surprise.  My Friend and I were planning on going to The Comrade which is 4 units West of Table 17 but their food selection was limited.  The Comrade bartender recommend this place so we thought we'd give it a whirl.  \n\nThere was a good feel to this two roomed restaurant, classy but not fancy.  There was a long wooden table in the middle of the 2nd room where the wait staff cut bread and got their serving supplies from which I thought was a nice touch.  The waitress we had was extremely helpful in her recommendations, both in food and drink and cute as could be.\n\nI will return.",
      'useful': 0,
      'user_id': 'Ce9f0fGVphaywABOvwGOCw'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 2,
      'date': '2009-08-07',
      'funny': 2,
      'review_id': 'DexBF1k8zV3lW9sNzpB0jg',
      'stars': 3,
      'text': "I can't seem to make the half star symbol work because I'd give Table 17 three and a half stars. I found myself there late last night where I met up with an old friend for dinner. I have been wanting to go there for a while so when I found myself on the eastern side of the city dealing with a late day work issue I seized the opportunity. \n\nI can't say all that much about the decor as we sat on the side western side of the restaurant right by the door (in all fairness, we just walked in and had no res so we knew it would be hit or miss). The other side of the restaurant, from what I could see, looked lovely and it appears that they have one of those large chef's tables as well. \n\nSpecials of the day included an unusual app of deep fried Zucchini flowers stuffed with ricotta (3 on a plate). The waiter explained that these flowers were only available for about two weeks of the year and it was a very special menu addition. The main menu is not that large with approx. half the mains being some sort of fish (arctic chair, salmon, halibut, scallops). \n\nWe started with the charcuterie board because I wanted to compare it with the charcuterie board that my fellow yelper, Deanna and I shared at Swirl the day before. The bread they served was very unusual, some sort of blueberry bread with walnuts in it. Very fresh but very heavy. The board consisted of four different types of meats, a small terrine of chicken pate, a couple of dried figs with walnuts shoved in them, no cheese of any sort, nor any savory (I have this thing for olives!).  Pate was pretty good but not anywhere near as good as the pate we chowed down on at Swirl. Would have loved the figs if maybe they had squeezed some mascarpone into them and then shoved the nuts in. I'm such a foodie critic. To drink we ordered the McManis Cabernet Sauvignon ($55) and it is a very reliable bottle of red. \n\nI opted for (let's be predictable Christine) the steak frites and my friend had the poached halibut. The steak was delicious and cooked to order (medium rare). The frites are so crispy, crunchy and tangled, just the way I love them but what the hell is up with all the salt? I mentioned this to the waiter when he asked me how my meal was and then to his credit he brought out an little bowl of frites sans salt. He insisted. He also apologized for the kitchen as he said sometimes they don't realize how much kosher salt they are tossing into the bowl when they shake up the frites. The halibut was meaty but bland as hell. Then again, it is poached. Great if you are on a diet. Personally I felt it could have been poached in perhaps a court bouillon with lots of fresh herbs. My friend began picking at my plate which tells me that she wasn't sated by her dinner because she dived right into the bowl of extra frites. \n\nWe passed on dessert. After all that meat there was no way I could eat dessert (urp!). All in all, I liked Table 17. The service was great, our waiter as friendly and stayed on top of making sure our glasses were looked after. Still, something just felt slightly off for me and I can't quite put my finger on it. Perhaps it may have just been our location inside of the restaurant, I never really felt like I was in it.  I would be willing though, in all fairness, to give Table 17 another go.   The lamb sounded wonderful as did a couple of the specials of the day.  I so gotta learn to step outside my steak obsession!",
      'useful': 2,
      'user_id': 'EhO5C7t3yfGFLvsTyK5PPw'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2013-07-15',
      'funny': 0,
      'review_id': '74P8qKUUGy9yQ5j_smzkrQ',
      'stars': 4,
      'text': "Table 17 is a truly awesome restaurant and worth a visit. My girlfriend and I stumbled across it for brunch one weekend and were impressed. We then returned for Sunday dinner and realized they have no corkage fees that evening, what a nice surprise.\n\nThe people running this restaurant get it, down to every detail. The servers are very friendly and helpful. The decor is nice, nothing overdone, just comfortable and 'rustic chic' which fits the type of place they are and food they serve. \n\nSpeaking of food, it's the best part. I don't have a high end palate, but I can say that I enjoyed every dish and felt the pricing was fair. Servings were generous so you won't come away hungry. \n\nBravo guys. Bra-vo.",
      'useful': 0,
      'user_id': 'gW4UsZSf81Xqc29eE2yqCg'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2012-05-26',
      'funny': 0,
      'review_id': '4rnjBGZ8nJQlYgyBQxLIuA',
      'stars': 5,
      'text': "We're regulars here. Delicious brunchs! Delicious dinners! Service is quick and helpful. Is a dressed-up/relaxed place. Bonjour Brioche have a line, come here!!!",
      'useful': 0,
      'user_id': 'fCy0RLrVg6xyhaNRpZtBIA'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 3,
      'date': '2010-09-08',
      'funny': 0,
      'review_id': 'eOa6zF8E1CC3ZW5poylHgA',
      'stars': 3,
      'text': "I had very high expectations of Table 17 as it was recommended to us by a bartender at Origin (where I had a phenomenal experience - see my review) and a good family friend (who has exquisite taste). What do they say about having high expectations? You will usually be disappointed. Sigh. \n\nAnd so it goes...\n\n I loved the space at Table 17. It felt classy rustic, if that makes sense. The intimate bistro (composed of two small rooms) had wood floors and walls, antique lights and a subdued color palette. I felt relaxed. \n\nThe service was superb. Our waitress was more than a joy and spouted all the day's specials by memory and was well-versed in the few wines we were deciding between. She was attentive, humorous and gave great recommendations. \n\nThe food? It was good, but forgetable (don't hate me, Table 17!). I wanted them to elevate the flavors. Punch me in the face and make me giddy. Make me moan with pleasure (food here people, get your mind out of the gutter!). Make me want to lick the plate clean. \n\nLike the last green leaf on a tree whose leaves are changing with the Autumn air, the dolce forte wild boar polenta was the sparkling highlight in the crowd. They change the polenta frequently, but it's a must order. Even if you don't like polenta, order it. The presentation itself is worth it. You are brought a thick slice of wood where the polenta is perched. Two table spoons of fresh herbs and red pepper flakes accompany it. Chef John Sinopoli comes out himself and scoops the pipping hot wild boar out of the pot and onto the polenta as he explains the ingredients and care that went into it. It was delicious. Not too gamy, not too heavy, just right. I would have liked it a bit saltier, but maybe that's just me. I have an addiction to salt (blush). \n\nWe also ordered the chilled white gazpacho (definitely recommend), charcuterie board (typical), warm roasted beets side dish (yummy but I wanted more goat cheese!), summer arugula salad with blue cheese and peaches (the peaches were lovely) and the steak tartare. I am a sucker for steak tartare and always order it when it's on the menu. Table 17 serves it differently, they mix beets into the chopped beef. While some may like this, I did not. It tasted like beets, not meat. I missed the salt (shocked?), the capers, the onions, the quail egg on top, the savory flavors of a traditional steak tartare. \n\nI know every restaurant can't be phenomenal, but at those prices, I am a bit tougher on the food and my expectations are a bit higher. I liked Table 17. I would give it another shot.",
      'useful': 3,
      'user_id': 'Je9hP1439B3jp4y-nBgiMw'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2012-02-06',
      'funny': 0,
      'review_id': 'IJu2nUgiiEyvKjd4mA59gw',
      'stars': 4,
      'text': "Appetizer:\nHot Balls: Arancino - Mushroom & Fontina stuffed risotto, Goat cheese - with wildflower honey, Arancino - Spiced lamb stuffed risotto with mint\nMy friend asked me which one is which as she could not decipher. The flavours of each is not very strong....like it didn't punch you in the face. The mushroom flavour was not very full. The lamb flavour was very slight. The one with the most flavour is the goat cheese ball. It really had to be elevated by the sauce at the side. I actually had no issue with that because I often find the gamey taste of lamb and goat distasteful. She on the other hand was slightly disappointed about the legendary hot balls from Table 17. \n\nDiver Scallops: Diver Scallops with Roasted with fresh bay leaves, served with sauce béarnaise & guanciale\nI liked this dish. It was again on the heavier and saltier side. Everything plated works together harmoniously. The scallop balanced the béarnaise and the guanciale perfectly. The texture of the scallop is escalated by the crunchiness of the guanciale. \n\nEntrée:\n\nHaddock: Haddock with chanterelle mushroom, sprouts, truffled potato, radish\nPerhaps our expectations for fresh seafood is a little high. We both felt the haddock is lacking in freshness. Fresh fish tends to hold together better. The sprouts were supposed to reduce the heaviness of the dish but it just wasn't enough. \n\nDessert:\n\nLemon Panna Cotta with Blueberry compote. This dessert hit all the right notes! The panna cotta was silky and smooth (like my legs...just kidding!) The delicious blueberry compote adds fruitiness (DUH!) to the plate! The lemon balm gives it a bite of citrus taste. It was awesome! The combination just sings in your mouth! I can eat that all day! Nom nom nom! \n\nIn the end, I wish the chef would balance the dishes with more acids, spices and fresh herbs instead of sodium and grease. We are definitely anomalies here. Table 17 had garnered a lot of loyal patrons over the years. I really wanted to fall in love with this cosy little restaurant!",
      'useful': 0,
      'user_id': '7_RaCe5zzPBYWm9znlffUA'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2012-03-23',
      'funny': 0,
      'review_id': 'fbUfvWQw9ScdGinxahPutg',
      'stars': 4,
      'text': "I am a fairly regular dinner guest at Table 17, and thought it was time to leave a review.\n\nAmong the cosy bistros popping up in Leslieville, Table 17 was among the first, and is among the best for its consistency in quality of food and service. I also really appreciate the Sunday & Monday BYOB nights, and wish other restaurants would take Table 17's lead on this generous offering.\n\nI am always happy to see my favourite mains such as the braised beef short ribs and steak frites on the menu (and I happily order them over and over each visit - the fries appear to be crisply double fried in duck fat in traditional Belgian style. Yumm!) \n\nThat said, the menu does not seem to have changed very much over the years, and I almost wish they would take that risk and do a bit of a menu refresh. I still give it 4 stars because I go there when I crave my favourites. When I'm feeling adventurous, I just go a little further west on Queen to Ruby Watchco.",
      'useful': 0,
      'user_id': '3TdL8QTy4XOWo6BVQJIqQg'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2013-06-24',
      'funny': 0,
      'review_id': 'PFqnL8q2ipKj-eZWAZy2Zw',
      'stars': 3,
      'text': 'Taste good but bad value. Portions are way too small. We both had a app main and desert and still very hungry.',
      'useful': 0,
      'user_id': 'eV5usRjY2cDqNKVv8wXroA'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2014-07-07',
      'funny': 0,
      'review_id': 'FhpVX2jrqiF1oDghyyA-5w',
      'stars': 4,
      'text': 'Very good food and great atmosphere, friendly service. Bottom line: impossible to go wrong with a visit.',
      'useful': 0,
      'user_id': 'MC8wS2NGRMSfO5fFWU7b4Q'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2016-07-22',
      'funny': 0,
      'review_id': 'Jl_JjxrOReAXpWppP9m9Kw',
      'stars': 5,
      'text': "Went to a Brunch party and it was so cool and awesome, the food and service where amazing and the stuff where so funny!! \n\nKnow I here is shut down and closed that's sad :(",
      'useful': 1,
      'user_id': 'f3HeJmAQYWbbuyVVzxstOQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2015-01-22',
      'funny': 0,
      'review_id': 'kgVI4dW1kWlBJqFI6187YA',
      'stars': 4,
      'text': 'What a relaxed place to enjoy a nice dinner. Table 17 is intimate, split into two rooms.  The service is very personal but not invasive.  Tonight we got the right amount of attention without feeling rushed. \n\nThe menu is well edited with interesting selections. We choose the artichokes and  bocconcini as the starters followed by venison and steak tartare, with sides of amazing Brussels sprouts and frites. All excellent, but the venison was truly a standout. \n\nI have been here a couple of times, but tonight was the best. Certainly will return.',
      'useful': 0,
      'user_id': '8xArUHrVys0dH17X36KyJQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 2,
      'date': '2010-10-25',
      'funny': 0,
      'review_id': 'wcgtJ2FMQjcOJsa_glqP2Q',
      'stars': 4,
      'text': "I think the main problem with this place is that it is too small to accommodate people who may want to hang out after dinner and have a few drinks. There is a bar, but it's small. \n\nAs for the food: quality ingredients, food cooked simply, and the right way. The trout was prepared perfectly. Overdone fish can be a problem, but not here. \n\nThe food was also not over-seasoned, which is one of my peeves with restaurant food in general. The flavors of the foods themselves are predominant here. Not that I don't enjoy spice, but there is a balance.\n\nThe duck rillette was really delightful, and I am generally not a huge duck fan. I'm not sure if that was orange zest or I imagined it in there, that's how subtle the flavors can be, but  I loved it. Especially the piquance of the pickled carrots with the toast and meat. Would have bought a quart of the pickles if we had room in our luggage. \n\nThe duck-fatted mash was bit too heavy for me that night, but my fiance had no trouble finishing it off. They give you a good heaping pile.  The greens were perfect to me--again, very simply prepared. \n\nThe arugala salad had the perfect amount of dressing on it. A little too much goat cheese on it for me though, could have been balanced more in that way for me. \n\nWe might even go back before we leave Toronto  to try the Polenta, which unfortuntely they were out of.",
      'useful': 1,
      'user_id': 'NL8ULDSd476vP3PtU4pvfw'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2014-02-16',
      'funny': 1,
      'review_id': '15jv3wYvtnS3bv6dDAKyIw',
      'stars': 2,
      'text': "Disappointing experience.  The restaurant was packed and my friend and I were crammed into a small table where we were basically touching elbows with the tables beside us.  The food wasn't anything special which surprised me - I had been intrigued to try this place for a while but I won't be going back.",
      'useful': 1,
      'user_id': '3gw82ABfZVOD9Yg51wTmiA'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2012-08-05',
      'funny': 0,
      'review_id': 'Jd-Twvmb3acE3fT8UH7JFA',
      'stars': 3,
      'text': "Went for Sunday brunch. Nacoise salad for me, scramble of the day for my friend.  Waiter had an attitude. Food was good / acceptable. Decor was lovely (especially the bathroom!), however  I wouldn't go back again.",
      'useful': 0,
      'user_id': 'NFH6lgwwub14W-sR7m40hA'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2011-06-23',
      'funny': 0,
      'review_id': 'p9cWIp_b_73CtKam8u3EIQ',
      'stars': 4,
      'text': 'Table 17 is a quaint find in Leslieville village perfect for a good catch up with your girlfriends or a dinner date.  The restaurant is divided into two rooms - the first room you walk into features a long bar and some seating and the room towards the side which we get seated in, is simple, farmhouse chic.  The menu has amazing sharing plates and we do choose a couple to start with. \nWe were very impressed with the presentation of the polenta du jour dish.  This is a must at Table 17!  The fluffy, white polenta, dished out on a wooden cutting board was accompanied by the chef who ladled a "ragu" of steaming lamb on top of it. There was chili and dried basil on little spoons for more flavour.  It was heart-warming, a generous portion - ideal for our group of 3.  The lamb ragu was flavourful and the polenta with some grated Parmesan cheese and cracked pepper was still light as air.\nAs a note, Sundays and Mondays it\'s bring your own wine with no corkage fees with decent prix fix menu on offer',
      'useful': 0,
      'user_id': 'OHfTeAhGpJtUMEOe7QLwgw'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2013-10-05',
      'funny': 0,
      'review_id': 'Actc5kcYxlR8uYCYrh68yQ',
      'stars': 2,
      'text': "Meh. Service was bad and food wasn't much better. The restaurant was clearly understaffed and our server rushed us all night.\n\nTo be fair, the hot balls and kale salad were very good appetizers. The main course was a huge disappointment. My fiancé ordered the trout on a recommendation from our server. It was bland and undercooked. The avocado mousse tasted processed and fake. I ordered the Simcoe County Pork Chop after our server highly recommended it over the steak frites. Super bland, no spices. \n\nMaybe it was an off night as there seem to be mostly positive reviews, but we left disappointed.",
      'useful': 0,
      'user_id': 'b6QQV4qVoyQ4rCQpj1yj_g'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2013-03-23',
      'funny': 0,
      'review_id': 'ZFj_gIgu0V2qIez1aET5Tw',
      'stars': 5,
      'text': "I had a groupon certificate for Table 17 and I'm not usually one to purchase groupons unless it's a place I really want to try or already like, We walked in without reservations, which I am also not prone to doing, but they had a small table available that was out of the way and quiet.\n\nI've heard others complain about the service here. I found it to be fabulous, and I'm in the industry myself, so I know what the job entails. Our server explained the specials and gave us some great recommendations for drinks and menu items.\n\nThe hot balls (arroncini) were fabulous and small enough to add on to any meal without spoiling your appetite. Lamb leg and merguez sausage wad fabulous and surprisingly lean.\n\nDesserts looked fab, although we were too full, but I loved that the cheese platter was priced with a glass of tawny port. Delicious.\n\nOverall, a great dining experience and will be back for sure!!!\n\nP.S. when you use a groupon, please please please remember to tip on the amount before discount. Your server worked just as hard as he or she would for any other patron and it's in very bad taste to not acknowledge this.",
      'useful': 0,
      'user_id': '_t3BJzyGaqr9mcDazYiYAQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2014-01-04',
      'funny': 0,
      'review_id': 'wq8HbFMyDGAMMdSChyS7tA',
      'stars': 5,
      'text': "Had dinner with some colleagues on Monday Dec 30, 13.  The food was ah-ma-zing!!  In addition it was BYOW night and no corkage fee - just some extra special icing on an already delicious cake!  You can't go wrong at table17.",
      'useful': 0,
      'user_id': '2Ez3VPd47JGv_hv975J-Uw'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2014-08-23',
      'funny': 0,
      'review_id': 'EKIrMW8Do8rR3OtdQY6f1Q',
      'stars': 3,
      'text': "I'm confused as to how to rate Table 17. When I lived in Toronto I ate here quite a bit for brunch and so when visiting the area again I was excited to go back. \n\nThe food and service were good, but my confusion lies in the cost of the Caesar at brunch time. It used to be $4.50 or $5. I saw they have a super special one on menu for $12 but I cannot justify paying that much for a Caesar. So I ordered a regular one, followed by another regular one. I was shocked when the bill came and they were $9/each. Ouch!\n\nMy bad for not reading the menu fully and noticing they are not part of the discounted brunch drinks anymore.",
      'useful': 0,
      'user_id': 'imPfhpAk61DqSW6K47D4QQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2014-03-22',
      'funny': 0,
      'review_id': 'hypJVgWGTNxWCeNBjFwfUg',
      'stars': 5,
      'text': "Came for brunch with my wife and two young boys on a Saturday. We had the scotch eggs and Giuseppe eggs. They have $4 kids meals too. We got the French toast which was two slices with fruits. We also got a side of rosti. Everything was really delicious and unique. I've been searching for a great French toast and this one was right up at the top. Full of flavour but not too heavy. Prices were all pretty reasonable around $10-$12. We'll be coming back for sure!",
      'useful': 0,
      'user_id': 'u04pbgaUuqqy0UnMI05p8A'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2012-04-10',
      'funny': 0,
      'review_id': '3FGjx0dgPKY1tCBYLxaoEA',
      'stars': 4,
      'text': 'What a great little restaurant in Leslieville.  I was pleasantly surprised at the quality of the food and the service.  Not too expensive and the the variety on the menu was a nice change from many other restaurants in the city. \n\nCall me a fan. I will definitely go back!',
      'useful': 2,
      'user_id': 'ZUsS5CjWh4YrPVi4Hi3Y6Q'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 2,
      'date': '2010-10-25',
      'funny': 0,
      'review_id': '2v685kARRKjNxjxAaHYbfA',
      'stars': 3,
      'text': "For a small place, Table 17 is very accommodating.  I was a part of a very large party (10+ people) who could not make up their mind.  Various members called the restaurant multiple times to change our reservation.  I was horrified that when we showed up, they were going to turn us away (I would!).  But, they did not.  Everyone was very friendly and welcoming.  Our table was waiting for us when we arrived, and we were all served very quickly.  The waiter made excellent suggestions for drinks, apps, and entrees.  Everyone was happy with their meal.  The ambience is nice, quiet yet trendy.  The bathrooms were immaculate and while we were in the front room near the bar, the back room had a center table as a servers station for bread and wine bottles.  The only downside is that the entrees were very small - fortunately they weren't horribly expensive.   The drinks on the other hand.... Let's just say I'm very happy I wasn't paying ($15 for a basic martini).",
      'useful': 2,
      'user_id': 'p8rCTA139YIM6DQNq5s5XA'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2014-07-27',
      'funny': 0,
      'review_id': 'vcaoApX4YBRLYuKV28Eezw',
      'stars': 4,
      'text': "I have only been for brunch but was impressed.  Very friendly staff.  Somewhat traditional French brunch menu, and well executed items.  They have some fun cocktails, and the ambiance is great.  The nature of the menu (not traditional brunch obviously) precipitates only infrequent visits (I'm and eggs benny guy on Sunday mornings) but when we go we love it.",
      'useful': 0,
      'user_id': 'pQbLLV8uzry5S7EX114gUg'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 2,
      'date': '2010-11-24',
      'funny': 1,
      'review_id': 'k5R2b5ZdTsxH5dFl5JdUpA',
      'stars': 4,
      'text': "All hail the mighty Groupon, without which I never would have experienced this place. After a really long, rainy, and cold school week during which my meals mainly consisted of stuff I grabbed from convenience stores on campus, I really needed a good dinner. Table 17 hit just the right spot. We started with a charcuterie plate, which was all kinds of good and porky. The standout starter for me, though, was the Ribollita soup -- a white bean soup with pancetta and kale. I could live off bowls of that stuff. For my main, I chose the braised short ribs. They were fork tender, but slightly dry and tough on top -- maybe I got the top of the pot or something -- but still delicious and just the perfect comfort food that I needed.\n\nService was great for the most part, but they were experiencing problems with their credit card machine that was a little frustrating -- they didn't seem to have a separate phone line for their credit card machine, so every time someone would call the restaurant, it would kick the credit card machine offline. Consequently, it took FOREVER to get our card charged for dinner. The food was excellent and I'm definitely coming back soon (dying to try their BYOW Table D'hote on Sundays and Mondays), but next time I visit, I'm totally bringing cash.",
      'useful': 1,
      'user_id': 'Uwu72w77MPox942_GnCS7g'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2015-06-14',
      'funny': 0,
      'review_id': 'FHWSAPYcX0qY9DAKcdcw3g',
      'stars': 4,
      'text': "We've been here for brunch a couple of times. Had the rosti both times, as well as the neapolitan eggs and the weekly scramble. Both times everything has been excellent and the service is great and friendly. Also entirely reasonably priced. Recommended.",
      'useful': 0,
      'user_id': 'dRlA0iymGFcNzsfWA7P-Jg'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2013-04-28',
      'funny': 0,
      'review_id': 'L3DqaARjr6I7RziyxMFERA',
      'stars': 4,
      'text': "I've been to Table 17 twice, and since the first time I came here, I have been anticipating the next time I would reunite with the sticky toffee bread pudding!\nWith a groupon in my hand and the final days ticking before expiration, I knew I had to get in here for what I hoped to be a fantastic dinner.\n\nThe tables were a little bit close, but not intruding and the waiter was patient and informative with the questions I had (I always have questions!)\n\nFor our shared appie, we went with the Burrata because I have been looking for a good burrata dish since I moved back to TO and had not found it yet.- I tried one at Reds last year and thought that it was not creamy enough and unfortunately, this one missed the mark as well. I thought that there was not mixture between the mozza and cream, so the soft texture of the burrata was lacking. The flavours and other components of the dish paired very well.. just wish there was more creamy goodness.\n\nIt happened to be Good Friday and I really did not plan to eat meat that night. I was tempted to go with the shortribs because I knew I would not be disappointed from my first experience. I was debating between the pasta and one of the fish dishes but there was one component of each dish that did not speak to me.\n\nOn a whim, I went with the scallops at the very last second. It came in a creamy broth (was not a fan at first read) but it had been quite some time since I had some good scallops and I definitely had set some high expectations. On first bite..... I was grinning ear to ear... the scallops were perfectly cooked, loads of flavour in the broth, and I could not get enough. I'm just not a super big fan of clams, so I cannot really comment on them. None the less, I tried to scoop up ever last ounce of the broth on the plate! The short ribs were tender (husband's review), and very tasty.\n\nFor dessert .... I did not have to look at the menu - sticky toffee pudding it is.... it came out hot in a ramekin, loaded with caramel sauce and apple butter on the side. Rich, decadent, tasty, mouth watering..... it was just like I remembered and more! S had the banana cheesecake, which I was hesitant about because I'm not sure how that could go well together... but it was actually surprisingly subtle but evident in banana flavour. With a touch of peanut butter to finish it off, the dessert was a great choice.\n\nI'm so glad that my second experience was just as good as the first. As someone who doesn't repeat restaurants often, I can vouch that this one will not disappoint.",
      'useful': 0,
      'user_id': 'uHZr3XbjKvRTwdXLXrFuvg'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 3,
      'date': '2011-08-04',
      'funny': 2,
      'review_id': 'VjDmZq8wyyAjBarXceWstQ',
      'stars': 4,
      'text': 'I can never resist hot ballZ (sexy menu item here at Table 17).\n\nConveniently located near Jilly\'s (hey!!!), Table 17 is possibly a little too beautiful for the neighbourhood, but I suppose that makes it a true diamond in rough because it\'s such a nice place.\n\nI tried the Ontario lamb with merguez sausage (lamb x 2!) for my entree. Lamb was cooked perfectly (medium rare) and the merguez was delicious. A perfect plate.\n\nThe hot balls come 3 ways and each is served with it\'s own dipping sauce. I think these are a "must have" if you come here.\n\nThe service was excellent and our water & wine was always kept topped off .\n\nIt\'s a small place with only a few tables, dark hardwood, large mirrors, chandeliers and a nice long wooden table in the middle with food artfully displayed. Definitely a great date spot.\n\nThe wine list is extensive but the price point was a bit over the top - no bottles under $60 and the 1/2 liters started at $45. Apparently on Sunday/Monday you can bring-your-own-wine, but not sure what the corkage fee was.',
      'useful': 2,
      'user_id': 'TbhyP24zYZqZ2VJZgu1wrg'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2012-10-16',
      'funny': 0,
      'review_id': 'GAcivlxYCCapdmMzBgjFIQ',
      'stars': 4,
      'text': 'I have been hear a couple of times and overall have enjoyed the food.  I had the trout as my main and it was very fresh and perfectly cooked.  It came with some pretty boring side vegetables but overall the meal was satisfying.  Its a little on the pricey side but our server was great and I enjoy the vibe.',
      'useful': 0,
      'user_id': 'hWJdT09U7oHcSoc_pYr0Tw'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2011-11-14',
      'funny': 0,
      'review_id': 'VKcXJGdqyB0EO5LLfGeHfw',
      'stars': 3,
      'text': "Table 17 is okay. The hot balls are pretty standout although for $12 for 6 ($2 a ball) I'm not sure I would order them again (thank you miracle of groupon-type discount!). The goat cheese balls were excellent, reminding me I never get enough goat cheese in my diet or to my liking. I would not recommend dipping them in the accompanying honey. The sweet just masks the wild goat cheese taste. Actually, it would be nice if you could order 6 goat cheese hot balls straight up.\n\nMy pickerel was a big bowl of meh. It tasted like something you might get during a summer/winterlicious. Being from the Lake Erie region, I like a tenderly cooked pickerel that allows you to enjoy the delicate texture and nuanced flavor of this white fish. But Table 17 prepares it extra crispy, burning the joy out of this fish. \n\nSlinky woman ordered the arctic char and seemed to have no complaints.\n\nPortions are a touch small given the prices. You get a couple slices of decent bread before your meal.\n\nService is excellent.\n\nTable 17 seems a nice addition to the aspirational bars that have cropped up in Leslieville along Queen street. The only thing I hate about Leslieville is it seems so dark. The strip needs some BIA work.",
      'useful': 1,
      'user_id': 'mb_8jXannipO5T5V5kGXiQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2010-01-17',
      'funny': 0,
      'review_id': 'IN36oY4LbL3Ts_yXXBRTmw',
      'stars': 4,
      'text': "Quite a great meal. Came early and even though we booked using OpenTable and it was busy, we were allowed to sit. \n\nSpecial was Foie gras and said yes before the words were out of the servers mouth. Shared tartare and it was glorious with a very large hint of moutarde, but U dug it. Shoulder of Ontario lamb was great with a hint of mint in the salsa verde. I'm not a dessert guy but the Gateau Elvis with chocolate, caramelized bananas and peanut butter was awshum. \n\nNice wine list. Will return soon. \n\nOnly thing hindering a 5-star is the fact that we had to top up our own wine a couple times.",
      'useful': 3,
      'user_id': 'FTNaQZ3t0dsVWw1WZUQGFg'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2011-10-24',
      'funny': 0,
      'review_id': 'SMO2bJwqn3ZVfkAeRADmqg',
      'stars': 3,
      'text': 'The things that I liked:\n\nInterior was homely, lots of wood used on the walls & furniture, there was a chandelier & the bar had accommodation to dine at if you chose to.\n\nServer was a very lively & attentive British man.\n\nSome qualms I had with the place was that my drink was a bit warm & my food not warm enough.\n\nThe steak frites was excellent & cooked medium-rare. Steak tartare could have been a bit colder as the dish is meant to be served cold.  Shared PEI Oysters were very fresh & succulent.\n\n For the prices paid, I expected a little more out of Table 17. There are better options out there for the price range.',
      'useful': 0,
      'user_id': 'sfKwAw0FD8KuWp4Hjv0cxQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2014-08-23',
      'funny': 0,
      'review_id': 'Ek69QrhQtlFH92BlbIzBhQ',
      'stars': 4,
      'text': 'My experience with Table 17 were excellent.\n\nI do like the French Countryside deco giving the interior a homey and farmhouse vibe.\n\nWe were greeted warmly at the door and the service was really attentive for the whole evening.\n\nWe shared:\n\nLamb Polenta ($14)\nDuck Confit ($26)\nVanilla poached rhubarb ($8)\n\nEverything was well prepared with great care. I even enjoyed my Lamb Polenta and I am not a lamb or polenta lover!\n\nThe restaurant did get busy even on a weekday evening. I guess a lot of people were taking advantage of their prix menu $35 during Sunday and Monday with BYOW.\n\nWill definitely return when I am in the neighbourhood.\n\nPS: Thanks Amy L for the invitation',
      'useful': 0,
      'user_id': 'CxDOIDnH8gp9KXzpBHJYXw'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2011-05-07',
      'funny': 0,
      'review_id': 'yJoJhj-UyS2JnvjBJg_Qmg',
      'stars': 5,
      'text': "wow this place is amazing for brunch, french style cuisine, free parking on side streets even on weekends! bonus! i didn't know there were coupons and deals at this place as I was reading the reviews but I would still pay full price to come here, gotta try dinner someday. good vibe here, good neighbourhood, good food! no line-ups for brunch either",
      'useful': 0,
      'user_id': 'NNOLEPL8DidbqMCH3Qkecw'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2013-10-06',
      'funny': 0,
      'review_id': '5gx2BOdilpycJSbcWZEKqA',
      'stars': 2,
      'text': "I wanted to go to Table 17 since I moved down to Leslieville and was excited to finally find an opportunity to go. My party of 2 got there around 9 pm and waited about 10 minutes to be seated. Unfortunately that's how the night continued. The decor is nice but very loud, with a large table in the middle of the main dining room for the servers to cut bread which we didn't receive until about 20 minutes after we were seated. We ordered drinks and decided none of the entrees looked appealing enough to pay about $28 bucks plus for, so we shared appetizers. We waited about 15 minutes for our drink. When we finally received our food at about 9:45 it was nothing special. To top it all off even i got the bill and they charged me for an entree I didn't even order. There are too many options in the area to go back for a mediore experince.",
      'useful': 0,
      'user_id': 'JyLTAZZZB9L40Q3S9b1jlw'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2013-11-14',
      'funny': 2,
      'review_id': 'NyYZ78XcczwgmXPqUc8cwg',
      'stars': 4,
      'text': 'My girlfriend took me here for my birthday what a surprise! I had their fresh oysters which were incredible tasted like they were fresh out of the ocean. What I loved the most here was their own homemade olive oil, simply the best olive oil I have tried I can just sit there the whole day having their freshly baked baguettes dipping in their basalmic vinegar and olive oil. \n\nGlad my girlfriend showed me this little place book a reservation today you wont regret it!',
      'useful': 2,
      'user_id': 'U_6JkfbEYRG-lb4pbbtlLA'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 1,
      'date': '2011-05-04',
      'funny': 0,
      'review_id': 'bLHMKw5LEXUCodI5-F8yww',
      'stars': 5,
      'text': "I had the most wonderful experience at Table 17 recently. I took advantage of the Telus Taste of Tuesday promotion in which you get a free appetizer and dessert with the purchase of an entrée. To top it off, parking was free at this part of town after 6 pm on weekdays. \n\nMy boyfriend had the potato and bacon soup. I had two delicious spoonfuls of the soup. I enjoyed chewing on the bits of bacon, which added a delectable smokiness to the creamy soup. \n\nI ordered the appetizer special of the day - chicken liver parfait. I absolutely loved this! The liver spread was delightfully creamy and I loved that intense flavour of the liver, complemented by a dab of tangy Dijon mustard.\n\nSteak Frites - 8oz thick cut top sirloin, served with lemon mayo - was executed to perfection. The steak was juicy, flavourful and amazingly tender. I really enjoyed the meat, but what stood out the most for me was the lemon mayo, which was, in one word, addictive. I wanted to eat it all right off a spoon. \n\nI had the striped bass, served over sweet onion purée and tartar garnish, with beer battered sweet dumpling squash and hen of the woods mushroom. I enjoyed all the components on the plate, and everything worked well together. The fish was perfectly cooked and the skin was nice and crispy. I wiped my whole plate clean and wanted more. \n\nFor dessert, our first choice was the maple panna cotta, topped with candied bacon strips. The bacon's saltiness really brought out the sweetness of the panna cotta and also provided a great crunch. The panna cotta, which came out resembling a block of tofu, was creamy and satisfying.\n\nOur other dessert choice was the honey and pistachio tart with crème anglaise. I loved that there was a generous amount of nuts in every bite. I wished the pistachio could be a little more toasted and crunchier, but the tart was, nevertheless, delicious. \n\nIt was a delicious meal and a great experience altogether. I can't wait to go back.",
      'useful': 1,
      'user_id': 'Jt1zgNmwz_jheOSmEsljPQ'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 2,
      'date': '2011-06-09',
      'funny': 0,
      'review_id': 'X1Vimh5En427lvZQ4v2vKw',
      'stars': 5,
      'text': 'After trying to find a table at various downtown restaurants downtown and frowned upon for expecting to "walk-in" on a Saturday night we happened to find parking (another surprise for a Saturday night) right in front of Table 17 as if it was meant to be. We walked in expecting to be turned away, especially since it was quarter past 10pm,  and were graciously seated at a pretty table by the window. The atmosphere was perfect - mellow but happy crowd lingering over dessert, attentive hostesses dressed casual and candlelight in a rustic contemporary decor.\n\nThe menu isn\'t extensive but offers a good variety to please any dietary preferences. We ordered the soft shell crab to start and it was set on top of a green chutney which elevated the taste of the crab and was just delicious. I had the lemon caper pasta which was really good and my DH ordered the lamb which was falling-off-the-bone goodness on a plate. We finished with the toffee cake dessert and it is high on my list of favorite desserts. It was decadent, luscious and big enough to share (but who wants to!).\n\nA great evening and I am glad we didn\'t get a table anywhere else - Table 17 rocks!',
      'useful': 2,
      'user_id': 'OFawbcy1lqVOeRtDTOnSpg'},
     {'business_id': 'mr4FiPaXTWlJ3qGzp4-7Yg',
      'cool': 0,
      'date': '2013-01-08',
      'funny': 0,
      'review_id': 'HAz9jl1PJnnnN_n_Y2XQeQ',
      'stars': 5,
      'text': "My partner and I went to Table 17 for dinner while on a visit to Toronto, we both felt that this was a very good restaurant based on both food and service.\n\nOur hotel was in the financial center, so we wanted to explore a different part of Toronto, and the server at another restaurant said good things about Table 17. The atmosphere there is rather romantic, with candlelight and a somewhat dark setting. It's not a very large restaurant, but the small space doesn't make it too loud.\n\nThe menu we had was prix fixe with an appetizer, entrée, and dessert. Since there were two of us, we were able to try a good portion of the menu, and we were rather impressed with what we got. My pork belly entree was very tasty and light.  My partner had the steak frites for entrée, and I while normally tend to stay away from steaks at a non-steakhouse (because you never know what you're going to get), this steak was tender, charred, and very well-executed.\n\nThe service was definitely a bright spot here. Our waitresses very friendly and indulged all are crazy questions and requests.\n\nOverall, I would definitely recommend Table 17 for either a group dinner or romantic dinner. There's one table right by the window that was open when we arrived, and we asked to sit there but the hostess said that people reserve some tables by number because they're good. If you care, that one is ...Table 23. ;)",
      'useful': 0,
      'user_id': 's19c8t_yAmFrby90LjDk8g'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 1,
      'date': '2017-02-25',
      'funny': 0,
      'review_id': 'kMrg_M_HOuTTH52AE6v_lg',
      'stars': 5,
      'text': "Smashed burgers done properly in the heart of Weston village. Very happy that we have a spot like this now in the area. All the right elements that make a great diner style smashed burger ( freshly ground steak on site, no fillers, execellent layer of crust on the patty, can't forgot to mention house cut fries ) . I opted for the signature z- burger ( double cheese burger with lettuce, tomato, pickle, onion strings,z-sauce)with an order of Greek fries( Caramelized feta,evoo, and what I think was tzatiki sauce, awesome) , my partner went for the sorry eh burger( bacon cheese burger, fried egg, tomato, onion string and z-sauce) with a classic poutine done right.  Definitely worth trying this place .",
      'useful': 1,
      'user_id': 'SJu5NWsA_DrPXu9ibN9y5A'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 0,
      'date': '2017-06-04',
      'funny': 0,
      'review_id': 'U4dQK77vqV7eWC-o9B9T_A',
      'stars': 5,
      'text': "The burgers here are the best burgers I have ever had, to top it off the service is even better. I was in the car waiting and the cook brought us the food when he didn't have to. Overall I will be coming here again!",
      'useful': 0,
      'user_id': '5IDStuxulcMAid2BLU4L_g'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 1,
      'date': '2017-04-08',
      'funny': 0,
      'review_id': 'tegon_HXy6YneQJ4-Y5h4Q',
      'stars': 4,
      'text': "Great fresh burgers seared on a flat-top, and excellent fresh-cut fries and poutines, served up by an owner who cares.\n\nFreshly-ground daily, a 4oz. scoop of AAA beef is dropped on the flat top then smashed flat and seared until it's nicely crusted, then served on a sesame seed bun. \n\nThe beef is juicy and flavourful, but not dripping with grease, like some places. Mark the owner tells me that he purposely set out to make decadent angus burgers, trying to keep it healthy, without sacrificing flavour.\n\nTheir poutine is huge, featuring thinner fresh cut potatoes, double-fried and topped with legit fresh white cheese curds and delicious beefy gravy.  \n\n*Don't forget to check-in to take advantage of the 15% discount special.\n\nIn a city where gourmet burger places are popping up every day, this is one of the good ones. Definitely check it out and help keep this place alive, we need to keep joints like this around and successful.",
      'useful': 1,
      'user_id': 'uf95iifC_TQ-DtDs0rtiXg'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 0,
      'date': '2017-06-02',
      'funny': 0,
      'review_id': 'HhmVl2tYCzBZGEjzUGk24Q',
      'stars': 5,
      'text': "Finally checked out this place yesterday after hearing great reviews about it... and it did not disappoint! My husband had the Z burger, I had the Heat burger and we shared the chef's poutine. Everything was amazing!!! Patty was very fresh and cooked to perfection and the toppings just topped it all off! The owner and his father were both so fantastic and went out of their way to ensure we were enjoying everything! The place is very clean and lots of room for seating. The mama in me also noticed a high chair which is very convenient for families with babies/toddlers :) Definitely check this place out! Nice to see some good spots being put in Weston... will definitely be back soon!!!",
      'useful': 0,
      'user_id': 'y2NiFFhNzQVSuW4MoEG0PQ'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 1,
      'date': '2017-01-30',
      'funny': 0,
      'review_id': 'Y3J0KfuOWk9iycOKUjwXRQ',
      'stars': 5,
      'text': 'Excellent and friendly service and even better burgers! Tried the Z-burger and it well surpassed my expectations. Highly recommend Zeal Burger.',
      'useful': 1,
      'user_id': 'n2p6VyO3dW46U2iCjiDYDg'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 1,
      'date': '2016-11-26',
      'funny': 0,
      'review_id': 'xDE4sPobqhn2mEjfnurHeA',
      'stars': 5,
      'text': 'Really good burger and fries. Always fresh, you can tsste the quality of the ingredients. Reasonably priced for what you get. Will be a regular customer.',
      'useful': 3,
      'user_id': '0ckMhO2TSrxLn5dxj0fbdA'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 1,
      'date': '2016-12-27',
      'funny': 0,
      'review_id': 'CY8Lahy5xwz63lR8OoTm1A',
      'stars': 5,
      'text': "Love this place! Got their Signature burger, The Z Burger but the mini one! Their Z sauce is like a Big Mac sauce but way better!! I wish I lived right next to this place. Can't wait to try the Dunamis burger next!",
      'useful': 2,
      'user_id': 'VKMVDCf9gZwZcjZFxGv2NA'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 1,
      'date': '2017-04-27',
      'funny': 0,
      'review_id': 'veHCfFqF6mjoC-LkGpCqrA',
      'stars': 5,
      'text': "Had a Z burger with a fried egg and guac, one of the best burgers I've ever had the pleasure of devouring. The smell of those burgers on the flat top would make a vegetarian reconsider their dietary choices. I'm ecstatic that they opened a legit burger joint in the Weston and Lawrence area. I love the area, grew up there, so any new business, especially one of Zeal Burger's quality, is a welcome addition to the 'hood. Thumbs way up guys keep it up! You deserve every star. Believe the hype.",
      'useful': 1,
      'user_id': 'Uge5lXCFpnWB7btHDwn0rA'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 1,
      'date': '2017-06-08',
      'funny': 0,
      'review_id': 'ZOAnTmATcq9OG0KymswfBA',
      'stars': 5,
      'text': 'Just had the zburger and poutine, if you want want a real filling homemade touch than this is the place. I definitely have a new hamburger stop, maybe I can can get a job there just to eat the food!!? \nCheers!',
      'useful': 1,
      'user_id': 'qWnEowA-nKE6eugdeKOmiw'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 1,
      'date': '2017-05-23',
      'funny': 0,
      'review_id': 'p5WYETQospqA4ZCLfnXItg',
      'stars': 5,
      'text': "This was an amazing experience. The server/cook is such a friendly and caring host. He consistently makes sure that his customers are enjoying the food. He also makes sure the food is nice a presentable and treats every order and burger with perfection. I've tried The Heat and it's a pretty solid burger. It's a nice mix of ingredients. The fries were also pretty good. The area is spacious, clean and well maintained. \n\nI've never met someone who is so dedicated to perfection when it comes to serving food. I will most definitely be returning.",
      'useful': 1,
      'user_id': 'JULZNfDh0PJCpw0wyIULrQ'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 1,
      'date': '2017-06-25',
      'funny': 0,
      'review_id': '7iL5lmSEjKhxgmCuUrVpHw',
      'stars': 5,
      'text': "The dude here is the nicest guy. Knew everything about the food and made us feel welcome. Burgers were amazing and worth the extra money. Skip Harvey's for this.",
      'useful': 1,
      'user_id': 'YnYcJQgeoLt2AW5Zi8quig'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 1,
      'date': '2017-02-11',
      'funny': 0,
      'review_id': 'Uno26gf78bXx0oY4H3qALA',
      'stars': 5,
      'text': "Tastiest burger I've ever had! I highly recommend this place, you won't be disappointed:)",
      'useful': 1,
      'user_id': 'rtRAalLgvvqgQkCijhLkOw'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 1,
      'date': '2017-04-08',
      'funny': 0,
      'review_id': 'svPvZfknc8mtDB4icvwBdA',
      'stars': 5,
      'text': "Honestly the best burger joint I've ever been too, hands down! I came here a couple months ago for the first time. I ordered the Sorry Eh burger and told them I prefer no egg and they were more then glad to leave it out. The feta poutine was amazing as was the Nutella milkshake! Very friendly owner and the other guy was as well! Keep up the good work and wish you all the best in the future!",
      'useful': 1,
      'user_id': 'D98mJS9A8r-coH37lZjnjw'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 1,
      'date': '2017-04-04',
      'funny': 0,
      'review_id': 'B8bSQAwvodDFEEs2kVRrCw',
      'stars': 5,
      'text': 'This is by far one of the best burgers in the city.  I live in the area and been meaning to try these guys for a while now.  Finally went into grab their Zeal burger with bacon and cheese.  OMG!!!! This burger was juicy with tons of flavour and cooked to perfection.  That was yesterday and I went back in for another one today -had to it was too good.  Might even go back tomorrow for another!!!\nVery clean place and great service!! Highly recommend to any burger lover.  \nPS\nAte it too fast to take any pics to post.',
      'useful': 3,
      'user_id': 'ptrfQk5cVmfOgs6pHY85Nw'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 1,
      'date': '2017-07-06',
      'funny': 0,
      'review_id': '3X_vXAwhiROkspZFBZhR_g',
      'stars': 5,
      'text': "I read all the reviews and decided to give this place a try. Absolutely the best hamburger I've had in a long long time. Love the fries and the feta, the beef is grilled with a nice crust. Very friendly guy behind the counter, he walked us through the menu and made some recommendations. I am so happy that this place is so close to home.",
      'useful': 1,
      'user_id': '1LcId4o9Ge0F0KXrUXhuyw'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 1,
      'date': '2016-12-27',
      'funny': 0,
      'review_id': 'zIitRcnEgDzMq7L6Kw6m9A',
      'stars': 5,
      'text': 'Just had my first ever Zeal burger - I never did a review before but I felt like I had too here!  From the second I walked in with my guests, we were greeted with a great friendly smile!!!  The burgers were cooked in front of us and looked soooooo good!!!!  The fries are cooked to perfection and the burger is definitely memorable!!!!  100% recommendation!!!!!',
      'useful': 3,
      'user_id': 'ZQ6cfhFhjqql_wTsBxYVYQ'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 0,
      'date': '2017-03-05',
      'funny': 0,
      'review_id': 'dE4kukIrmS1E8vlURN1l5A',
      'stars': 5,
      'text': "This had to be one of the best burgers I've eaten in a long time! Everything was fresh and very tasty. I can't wait to go back and try the rest of the menu. I would definitely recommend this place to people.",
      'useful': 0,
      'user_id': 'mG8XVKC6ze4ohTbBKvcWHg'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 1,
      'date': '2016-11-16',
      'funny': 0,
      'review_id': 'S29p8jT6JMtvMTPbnjN-Bw',
      'stars': 4,
      'text': 'First time here since the Sang\'s Chinese restaurant at this location closed. Nice 50\'s decor in Coca-Cola red, white and black. Black marble table tops and black metal chairs. The burgers are a 10 for taste, freshness, all fresh toppings and juiciness say both my daughter and I. My daughter loves the soft fresh sesame buns and wants to know where the owner buys them. The fries are the old fashioned hand cut fresh kind and yes, they do have malt vinegar as well as white vinegar, which "chips" IMO should have in a good burger place (daughter of an Englishwoman here). Quite reasonable prices. Single cheeseburger combo with a can of Coke or other soft drink $8.85. Add $1.00 for bottled pop. You order your burger & fries at the back counter and take your can/bottle of pop from the fridge opposite the counter. The owner says he\'s trying to cook things the old fashioned way and welcomes customer feedback. I wanted to try the dessert of the day but unfortunately all they had was Nutella milkshake, and I don\'t like Nutella. Apparently they had an Oreo cheesecake cupcake yesterday but were sold out of it. I gotta try that!',
      'useful': 2,
      'user_id': 'W6CrBiuPNADGDIH_m39_aw'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 1,
      'date': '2017-02-24',
      'funny': 0,
      'review_id': 'SdjEuABgdjl6jxpiKkReAQ',
      'stars': 5,
      'text': "Burger patty is ground from AAA meat and made daily you can tell from the taste that it is fresh. Fries are not greasy. Burger was done right not overly seasoned or drowning in toppings. I can tell the cook/owner loves what he's doing as you can see and taste it in the food. Customer service was top notch, place is very clean. I live in Markham and will drive to this place again probably next weekend!!",
      'useful': 2,
      'user_id': 'Hta27zzUlf4M-5mH2yYzBw'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 0,
      'date': '2017-06-17',
      'funny': 0,
      'review_id': 'QedukfbFo4SMOahmWYpFxQ',
      'stars': 5,
      'text': 'Wonderful service with a smile. The owner does everything himself (two thumbs up). One of the best best burgers in the city for sure. I wish they were open later though, just a thought.',
      'useful': 0,
      'user_id': 'yKv3Nt93xU61hYn6DObZAw'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 1,
      'date': '2016-11-14',
      'funny': 0,
      'review_id': 'hyDZcbKsYK4C_6N5h3qm1Q',
      'stars': 5,
      'text': "Excellent - so happy I discovered Zeal Burgers.  Much better than 5 guys or other places (and much more inexpensive - everything is included in combo!).  Meat is superb, bun is the freshest and fries are homemade, non-greasy.   The eating area is also extremely clean and cheerful. I don't live in the area, happened to discover it when I was visiting a friend but will certainly be travelling here all the time - certainly worth it.",
      'useful': 4,
      'user_id': 'stsJOpjpZI_jnm1V6XB-ww'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 1,
      'date': '2016-10-29',
      'funny': 0,
      'review_id': 'x5sf5UtVAmz-j2fKXMbO5g',
      'stars': 5,
      'text': "Fantastic burger! Definitely on par with five guys and I would say it's even better. The z burger had delicious onion straws and all the ingredients were fresh. The texture of the patties were unlike any place I've ever been to, a must try!\n\nAlso worth a mention, the fries are very tasty (not greasy like most places). I didn't get a chance to try the Z poutine but it sounds like perfection.",
      'useful': 3,
      'user_id': 'CGjRewYS0p_WEm6ocQJFrw'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 2,
      'date': '2016-12-23',
      'funny': 1,
      'review_id': 'sbQLXZiV_z44LSk0qEIy0Q',
      'stars': 4,
      'text': 'So I was randomly in the area, I had some time to kill waiting for someone so Zeal  Burgers was in area,  claiming they were great gourment burger and after having the Z Burger I cant say I disagree\n\nThe Z burger was a double cheese burger, two fresh grilled 4 oz patties, topped with their onion "stringer" Which is their take on mix between fried onion and onion rings, so it still is deep fried but not as large as onion rings, which is tasty add with their z sauce it was an amazing burger',
      'useful': 1,
      'user_id': 'bJTKxBIvb_AR8d4SuLPV8g'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 1,
      'date': '2017-05-26',
      'funny': 0,
      'review_id': 'PDK0KXnGm7GVUpige9nAQA',
      'stars': 5,
      'text': 'Ordered the "Sorry, Eh" burger with fries and it was a really solid burger. It was cooked perfectly with a nice fried eggs and their signature Z sauce is a much improved version of the Big Mac sauce. Their fries were a little bit bland, but still acceptable. Although a bit pricy ($10-15 usd) I would highly recommend this place.',
      'useful': 1,
      'user_id': 'YG9wYveha8KgQY9OnsX43Q'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 1,
      'date': '2016-12-23',
      'funny': 0,
      'review_id': '8x174dIsUo1-VdYqHUKSjg',
      'stars': 5,
      'text': 'Loved these burgers!  Fresh ingredients, flavours mesh perfectly for a mouthwatering experience.  My kids loved the fries and their cheeseburger. My husband devoured the spicy burger even though he "wasn\'t hungry" and I loved my feta.  We will go back often.  The owner is a great guy, made us feel really welcome as we had just moved to the city.',
      'useful': 3,
      'user_id': '-7B__mas1m1a6rLPL4Puqw'},
     {'business_id': 'JB8-8TtNYX-vLqN7cz-zHA',
      'cool': 1,
      'date': '2017-04-15',
      'funny': 0,
      'review_id': 'bays9EorJBAnRl0W3w12DQ',
      'stars': 5,
      'text': "Good food, good people, good prices. About 14$ for double patty and a couple extras and fries \n\nThe parking is a bit tricky if you don't wanna pay since the location is located on a main street",
      'useful': 2,
      'user_id': 'N9PQbJqnmVAK-wjgiaaGnw'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2011-12-20',
      'funny': 0,
      'review_id': 'ABeny93AMSYm7Hn8UXPrRQ',
      'stars': 5,
      'text': "My go to place to study!  They play great music at a volume where you can still hear yourself think.  It's almost never crowded, and the people that come there generally are reading or studying.  They have an amazing tea collection, with equally good food and even grilled cheese sandwiches (with the option to add tomato :) )  One of the walls has some cute benches that are padded, making it a perfect place to wrap your hands around your warm cup of tea and read a book.  I agree with Rachel C, I wish they had high tea, that would make it perfect!  The only drawback is that it's small, and there aren't a lot of places to sit.  If you are the mood to try a variety of tea, or just need a place to study/relax this is it!",
      'useful': 1,
      'user_id': '3KkT6SmPFLGvBS1pnDBr8g'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 1,
      'date': '2012-11-01',
      'funny': 1,
      'review_id': 'KiGRD0ZhuDF1K1xLc0jjGA',
      'stars': 4,
      'text': "This place is ridiculously cute, colorful, and cozy, with a huge selection of loose leaf teas.  They offer teas by the pot only, except for iced teas (during warm months) which come by the cup.  They range from $3 to $4 per pot depending on how fancy the tea is.\n\nThe staff here has been friendly every time I've been.  They are always keen on offering suggestions if you ask!  Last time, I was struggling between going for a fruity tea vs. a chocolatey tea, and the girl working suggested this chocolate peppermint tea blend that was not on the menu but that she had made for herself the other day.  I tried it, and it was AMAZING!\n\nMy experiences with the food here (from at least a year ago) have been great as well.  They have their own pastries as well as a selection from a nearby bakery.  Savory offerings include grilled cheeses, soups, and sometimes specials.\n\nOne small gripe I have is that their teapots tend to, for whatever reason or another, leak some tea leaves and residue when you pour into your cup.  I suspect it's because the filter is not super fine and so small bits from the tea mixture can fall through.  It's never been a major problem, though.\n\nYou can always find fun, seasonal specialty drinks here too (e.g. hot apple cider in the fall), in addition to coffee, if tea isn't your thing.  Newspapers and reading materials are usually available to pick up, and there are board games in the back.  This is a great place for a rainy (or cold) day!",
      'useful': 2,
      'user_id': 'TsBUWbhRhuiEMXb56kL0Cg'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2008-01-17',
      'funny': 0,
      'review_id': 'nGV40rXMRL_IN6lWpyrsmA',
      'stars': 4,
      'text': "You can purchase from a huge variety of loose leaf teas. It's one of the few places I know that really pays attention to brew times and water temperature. The smoothies are fresh and taste very natural. If you order in, they'll give you your drink in a cup that looks smaller than if you order to-go. They have nice metal counters with lots of plug in places for your laptop. The benches aren't terribly comfortable, but the couches are. Plus, the people behind the counter are great!",
      'useful': 0,
      'user_id': 'Um2iec4NKMXVpJEME3PfKg'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 10,
      'date': '2015-02-10',
      'funny': 8,
      'review_id': '8FvVPvTaMJAM5drD18JxAA',
      'stars': 4,
      'text': "I dropped by Te Cafe a week ago with CM to check out their offerings. As a tea lover, having passed it many times on Murray, I'd long been curious about it. My friends who live nearby never wanted to go, however, since they prefer Dobra Tea across the street. Last week, CM wanted to get a hot beverage in the area, and he was game for checking it out, so I was able to do so.\n\nWhen we went to order, the menu had loose leaf teas listed by type (columns) and by grade (rows), which determined their price.  While you can buy them by the cup, if you're with other people or you're planning on hanging out for awhile, it usually makes more sense to get a pot, since each pot provides 2-3 infusions, and hot water is free.  We decided on a pot of the milk (?) oolong ($4.50), recommended by the barista, which was on the lighter side and was quite pleasant.  Once we'd had a pot and refilled it with hot water for a second infusion, we also got a pesto grilled cheese sandwich with tomato as a snack ($4.75), which was crisp and super-flavorful (thanks for the recommendation, fellow Yelpers!), and complemented the tea well.\n\nMy favorite part of Te Cafe, however, was the board games that can be found on the back shelf and front table.  Unfortunately, not all of them are complete, as CM and I found out when we picked up the Ultimate Scenario Game (?), but those that are make for a fun time!  We ended up playing dominoes--which I'd never played with as a game before (just setting them up in rows and knocking them down ;-) ). Since the first two games went quickly, with each of us winning one, we played best out of three. The last game went on to the point I thought we were going to run out of dominoes, but I eventually won.  Yay!\n\nWhile I understand my friends' preference for Dobra, given that they have a more extensive menu of tea and snacks and a more unique Asian tea house-like atmosphere, I enjoyed Te Cafe for a change of pace. I'd definitely return for another visit in the future to sample more interesting teas--and to play another round of dominoes!",
      'useful': 14,
      'user_id': '135DbbQnr3BEkQbBzZ9T1A'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2017-05-01',
      'funny': 0,
      'review_id': 'CBamcMDNj6fCU5JbkRFgfw',
      'stars': 4,
      'text': "Came on a weekday afternoon to study and it was a good environment to do so. I got salmon sushi for a great price, and the pots of tea with free refills are a great deal too. The employees are friendly and attentive. This place seemed like a nice casual place to spend a few hours studying, and I'd definitely recommend it to others.",
      'useful': 0,
      'user_id': '3ew6BEeK14K6x6Omt5gbig'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2017-06-22',
      'funny': 0,
      'review_id': '9bdoZr1hs6ATyKNQPrRUYg',
      'stars': 5,
      'text': 'Amazing people \nLoveeee the owners. Best sushi in town best service and nothing beats their prices \nGood for bubble tea lovers',
      'useful': 0,
      'user_id': 'scnvdFMXTQ4o2DBJpHLbaw'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 2,
      'date': '2011-07-11',
      'funny': 1,
      'review_id': 'FuESGmmv_tsmKERvHQXXjg',
      'stars': 5,
      'text': "This place hasn't changed a bit. Still awesome tea, great service and friendly faces. Although, the owners have changed, Te Cafe remains the same. \n\nSo good to be back and sipping on my favorite iced lavender lemonade.",
      'useful': 3,
      'user_id': 'g0EQGDEVFl4DMN6jfarJFg'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 2,
      'date': '2015-10-15',
      'funny': 1,
      'review_id': 'uKl2HGMPzg2CazrvXYC6YQ',
      'stars': 4,
      'text': 'Mulled Apple Cider Tea. Deliciousness in a hot steaming cup.\n\nLots of space to lounge in. Friendly service.',
      'useful': 1,
      'user_id': 'hcZqq-a16ZTjaM2p2MljTg'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 1,
      'date': '2015-10-10',
      'funny': 0,
      'review_id': '-p0bL6keK6bJ963Z6MycJg',
      'stars': 5,
      'text': "First visit here the other evening and it turned out to be the perfect place to sit and relax after a long busy day at work.  I didn't even look over  the tea selections because this café had me the minute I saw the featured drink listed on their outdoor sidewalk sign.  I thoroughly enjoyed sipping  my  hot cider tea drink on  the cozy corner seat near the window while reading and watching the world pass by.  Truly a good place to unwind and be refreshed.",
      'useful': 2,
      'user_id': 'ZLS7cwa1UplSB8nRrwrHIQ'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 3,
      'date': '2016-11-12',
      'funny': 3,
      'review_id': 'ziGTjPAhjPkmb3KUvntfZg',
      'stars': 5,
      'text': 'LOVED THIS PLACE.\n\nIt\'s like a coffee shop (/it is a coffee shop ha), but also has sushi. Good vibe and fairly quiet, which I always appreciate. There were about 5 other people working in here when my friend and I stopped in for lunch - and it\'s sad that this is rare - but everyone respected the fact that everyone else was working and stayed reasonably quiet.\n\nThe sushi rolls were good and priced well for sushi. They were a good size too, which meant I ate just the right amount of food for lunch by ordering one roll with a seaweed salad.\n\nThe best part for me was by far the service. Very friendly throughout my whole experience. As my friend and I were leaving, the people behind the counter even asked us if needed anything else. We said we were all good and they sent us off with a "have a good day" and a smile.',
      'useful': 5,
      'user_id': 'f44In4p5PicSF4E4GaeTrw'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2017-03-25',
      'funny': 0,
      'review_id': 'Zu4Xl5Ab4h6XhpbkGfiEGQ',
      'stars': 5,
      'text': "This place doesn't even need anymore 5 star reviews. But it's just so damn good that I had to.\n\nThe employees and owner(s?) are soo nice and recommended some pretty good teas! They also have a convenient water boiler where you can refill your own pot. \n\nTheir sushi is spot on. Honestly, better than some sushi places around the area.  The ambience is really nice, although wish their tables were a tad bit bigger. Still, a great place to study!",
      'useful': 1,
      'user_id': '9tIjFbmboV3ioyng_G2Pgg'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2013-02-03',
      'funny': 0,
      'review_id': 'ImmGjseTAFTfhgcLIKKspg',
      'stars': 4,
      'text': "Te Cafe has an extensive selection of teas including dozens of green and black teas as well as a few other variations; you can also order coffee, hot chocolate, and pastries. In terms of ambiance, it is one of the coziest establishments that I have been in making it an ideal place to read, study, or complete some work. It's also a nice venue to meet up with a friend for a pot of tea and some quiet conversation. There are also some board games (including Chess, Backgammon, and Dominoes among several others)",
      'useful': 0,
      'user_id': 'SqVUNVeYJblyoUz4e-Fiqw'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2007-07-18',
      'funny': 0,
      'review_id': 'E0mubEQsoe-Ag84dF4RWPw',
      'stars': 5,
      'text': "I came here with a friend and her young child for my first visit.  What a nice and friendly place!  The staff was super sweet and helpful and suggested some different teas for my friend.  I ended up getting a white tea which was nice.  \n\nThe cafe has both tables and comfortable side seating (long comfy bench).  I believe they also have a couple of comfy chairs.  In addition to the internet, they have made this a cozy place with games that customers can play with - both for adults and children!  My friend's child ahd the time of his life.  \n\nI'll definitely be going back!",
      'useful': 0,
      'user_id': '7sAkIM47_LE99Hyg5lwdrg'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2013-09-01',
      'funny': 0,
      'review_id': 'f_rTGnBD1UvYKUp77xipIw',
      'stars': 5,
      'text': "Went here with my girlfriend and really enjoyed this place. Great atmosphere. I'm not that into tea, but I had a couple cups from the pot my gf ordered and enjoyed it. We ordered 2 biscotti, a medium coffee and a pot of tea (which comes with a free refill) and it cost just over $7. The Barista was really nice and helpful.",
      'useful': 1,
      'user_id': 'tBwWTZHMSR2m0XkmuXrWpA'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2015-11-08',
      'funny': 0,
      'review_id': '3zrel1UtG993XTc9BcVq5A',
      'stars': 4,
      'text': "This is a cute little shop that you might miss if you're not paying attention. The person behind the counter was very helpful and the tea I ended up getting was very delicius. The atmosphere is nice and relaxing, it's worth stopping in.",
      'useful': 0,
      'user_id': 'HoM645ouLZA0grMBtj4w2w'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2007-06-28',
      'funny': 0,
      'review_id': '46TNgyRfXo8NP5hydQFD_Q',
      'stars': 5,
      'text': 'Good tea, comfy chairs.  I hear the food is also good, but have not tried it.  Nice quiet place to relax.',
      'useful': 0,
      'user_id': 'YCZA7dlc3Rdj80Egh3__JA'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 1,
      'date': '2006-07-16',
      'funny': 1,
      'review_id': '5Lz5wCZiKd28fXYDV1p36A',
      'stars': 5,
      'text': 'With wireless internet and a relaxing ambience, this was a place I was looking for during my two years in grad school.  Coffeehouses in Pittsburgh often were too loud and rickety uncomfortable chairs.\n\nThe owners are often attentive and bring a certain unique warmth to the place.  Sure, a large part of the cafe is devoted to them.\n\nI loved coming here for an after-dinner conversation with a friend or for studying.  Normally I would order a pot of darjeeling or a green tea.  Great selection of teas.  Very attentive service.  They give you a timer to help you remember how long to steep the tea.  The cups are all selected for the particular type of tea being served.\n\nWhen I was in the area, the cafe closed on Saturdays (being in Squirrel Hill), but I hear the hours may have changed.  Also, the owners seemed to have hired some of the younger crowd in the last few months which changed the pace somewhat.\n\nStill, when I visit Pittsburgh, I will return to this place.',
      'useful': 3,
      'user_id': 'BoUGdFkiU6EVsHGSvt-Uwg'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2008-09-10',
      'funny': 0,
      'review_id': 'A2ONjv4-0-RbzUScRHTEJQ',
      'stars': 4,
      'text': 'I\'m writing from Te Cafe right now.  I love this place and always made an effort to come here when I was visiting Pittsburgh from other locales.  It\'s sad that I don\'t come out here more often, now that I live here again...\n\nAnyway, they have a fantastic beverage selection (not just tea), high-quality loose-leaf teas you can buy by the ounce, comfortable enough seating, free wifi, and friendly staff.  The music\'s a toss-up; right now, it\'s some pretty good oldish rock, but apparently there is "wispy Lillith fair crap," sometimes?  On the bright side, if you show up on a bad day, it is definitely quiet enough to allow you to just listen to your own on headphones.  And the clientele is pretty varied, which I also rather like.',
      'useful': 0,
      'user_id': 'lPeHcLi-6y4nIEdPmDedtQ'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2009-10-24',
      'funny': 0,
      'review_id': 's8USKjytCBnirArAjujQsA',
      'stars': 5,
      'text': "Stopped by Te when I was in Pittsburgh and loved it.  Quaint little place in Squirrel Hill with a fantastic selection of loose teas.  The staff knows their teas and is friendly/helpful.  And the atmosphere is comfortable and neighborly.\n\nIf you're new to tea, this is a great place to get started and learn more.  And if you're a tea veteran, you'll be happy with the variety.  Cream Earl Grey is delicious, as well as peppermint rose.",
      'useful': 0,
      'user_id': 'uNWIwGh50410cZLgFo6g_Q'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 6,
      'date': '2014-10-16',
      'funny': 4,
      'review_id': 'YtMoh__UWg9x-frr43bqTg',
      'stars': 5,
      'text': "Te Cafe is going to be one of my favorite places in Squirrel Hill.  I feel it.\n\nSo here is the quick summary:\n-  Loose leaf teas served by the cup or pot at reasonable prices!  Cups of tea were around $2, and I got a pot of peaches and cream tea for $4.01 with tax.  The Cream Earl Grey is amazing as well!  There is also free hot water so you can keep going!\n\n- If you are not a hot tea drinker, there is also iced tea, lemonade, and hot chocolate!\n\n-They sell grilled cheese sandwiches with pesto for $4.50 ish.  I don't remember the exact price, but I remember that the pesto grilled cheese was amazing.\n\n-This place gets packed with wifi hogs.  You've been warned.\n\nI really like the vibe of this place.  It's a good place to catch up with a friend or study, and I like that it's on the quiet side of things.  They usually are out of most of the teas on the menu, but there are still at least 50 types.  I will definitely be returning here soon!",
      'useful': 5,
      'user_id': 'Ic6-gs1_FjrWGx6JIr95Mw'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2017-04-02',
      'funny': 0,
      'review_id': 'tzUnUNWTxqTS_NgpAGePBA',
      'stars': 4,
      'text': "Not as fancy as Dobra Tea across the street but it's a comfortable place to meet people work, study, read comma or relax. They have a great selection of Teas and coffee the staff is personable and if asked will give her a recommendation without a lecture about tea.",
      'useful': 0,
      'user_id': 'A4SLfGedOniAWjjIwC5iOw'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2016-08-06',
      'funny': 0,
      'review_id': 'xawyoCQGgQkUcLQm3hW6MQ',
      'stars': 5,
      'text': "Love how they serve their tea if you get a pot of tea. It's also a great study spot, especially since you can get sushi for lunch here. I usually get the spicy tuna roll, and it's a pretty large portion size. There are plenty of plugs and the wifi is usually pretty good too.",
      'useful': 1,
      'user_id': 'ySq-gVgm0gxmllf0ZFz-mA'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2008-09-11',
      'funny': 0,
      'review_id': 'aJXNCEdwEfSJk8E9o6BEhw',
      'stars': 5,
      'text': "Huge fan of Te!  Always a great experience every time I come.  The atmosphere is cozy, nice friendly staff who all know what they are talking about.  Not sure what to order? they will for sure help you out!  I'm a fan of Dragon Pearl Jasmine...yum!  Great place to read a book, do some work, or just hang out with friends.  \n\nThey have awesome fruit smoothies, made with REAL fruit!  I've gotten them for breakfast a few times, also NYC famous H&H bagels, you cant go wrong grabbing something on the go here.\n\nI get some tea to take home from time to time, there are tons to pick from its hard to choose!\n\nTheir daily teas are always changing and are a great way to try something you may not think you would have chosen yourself.\n\nAll in all I think Te is great, I don't how they do it but they do it well.",
      'useful': 1,
      'user_id': 'EsJmtRBbnUEzLx2cGhIBdA'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2012-12-18',
      'funny': 0,
      'review_id': 'rtlv8tD24Gila1o6_aHOPQ',
      'stars': 5,
      'text': 'Certainly up there as one of my favorite spots in town to study.  The overall ambiance is very conducive to being productive: comfortable chairs, good lighting, a great selection of music at the appropriate volume, and of course the warm tasty beverages that are an absolute steal.  This place has the most robust hot chocolate lineup I have ever seen, with the Mexican chocolate being my favorite.  Free wi-fi, with no necessary password, makes things all the more convenient.  I also love how this place caters to those who want to study/read.  Sure light conversation does occur, but I hope it stays that way such not to detract from the great study vibe this place engenders.',
      'useful': 1,
      'user_id': '32M_2x0IzeIxcJucQp6BOw'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2013-05-13',
      'funny': 0,
      'review_id': 'R8SVECqA0rWwa0FCd-ViNg',
      'stars': 1,
      'text': "I came in for the first time, sat down to setup my computer, and within 2 minutes had a server telling me that if I do not order anything I will have to leave. I'm not sure if I was being judged by the way I looked (wearing sweatpants and ratty t-shirt), or if this is standard practice. I was planning on staying and purchasing a drink, but found this treatment extremely rude and uncomfortable. Do not plan on returning.",
      'useful': 1,
      'user_id': '3ACrcBGuulDP4q_IicHJ3w'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2011-09-21',
      'funny': 0,
      'review_id': 'gTYNjQmNvUC6cIKaYaRo2w',
      'stars': 5,
      'text': "Amazing place to read or just relax, lots of loose leaf teas to choose from. Some of their lower priced teas aren't thrilling (especially the greens) but you get what you pay for. \n\nThe only problem I see arising is, if you buy a pot which is the best deal, you get a neat tray that comes with a timer, a cup for drinking, and a cup to hold the used leaves. If you're drinking on your own or with a single friend, that's fine and dandy- but it's near impossible to have a proper meeting with three or more people without forcing everyone to use inefficient mugs instead. I might be mistaken but they really should have party tray type tea sets. \n\nStill, 5 stars for sure. Excellent service, eclectic music (classical one day, indie the next), good ambiance, good price, decent hours (I wish they would open a bit earlier, but they close so late that it makes up for it), and some really amazing teas mixed in. Free wifi makes it even better.",
      'useful': 0,
      'user_id': 'BKMP5NtMa-VqP_lnWjbOYw'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2016-06-19',
      'funny': 0,
      'review_id': '4TLx9EUZOPuej3FhT1p4eQ',
      'stars': 2,
      'text': "Great selection of teas. sushi are fresh, but I don't like the sushi rice they used, much sticky than I thought for sushi rice",
      'useful': 0,
      'user_id': 'sV-rtJQGY48VO1fqWefUaw'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2008-02-05',
      'funny': 0,
      'review_id': 'Zfb77iqBJKxZutGh0nud9w',
      'stars': 5,
      'text': "I'm a graduate student here in Pittsburgh, and this is one of two main places I go when I want to get work done. The tea selection is excellent, and the staff was excellent - everyone knows the teas very well.",
      'useful': 1,
      'user_id': 'Z3ACEiYK1hY-YXpPQ3s10A'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2015-12-05',
      'funny': 0,
      'review_id': '3HDCrtkYkoBK93AQWd4hVQ',
      'stars': 4,
      'text': 'I only wish they offered sugar. However, good atmosphere. If you wanted to have a decent conversation with a friend of interest, this place is currently ideal. The Chai is amazing.',
      'useful': 0,
      'user_id': 'kHIKU_7cvSQVYTI1Iiw9EA'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2016-11-01',
      'funny': 0,
      'review_id': 'oVfKbZcICdgZatDWd_MtSg',
      'stars': 5,
      'text': "A neighborhood gem.  Don't come here looking for lattes or other fancy coffee drinks, though - it's a TEA shop.  The sushi is pretty good by Pittsburgh standards (no offense, Pittsburgh, but sushi ain't your strong suit).  It's bright and sunny, which is a huge plus, but that means it can get a little warm; layer appropriately.",
      'useful': 1,
      'user_id': '4YwesZEX8VPPJOKLfPD7lg'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2008-09-26',
      'funny': 0,
      'review_id': '6XwvR5K_VNsK3bJia4BPQw',
      'stars': 5,
      'text': "Oh what wonderful, delicious, unique drinks they have! I'm a tea person and I was extremely impressed by their selections (with prices better than Teavana, mind you). It's small but beautiful (think Ikea beautiful). And damn good people work there. :)",
      'useful': 1,
      'user_id': '6CoMpwA8D2s00RfzpKDuOg'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2012-06-11',
      'funny': 0,
      'review_id': 'pItymWcrt6jB6h_vZCtusQ',
      'stars': 5,
      'text': 'I love Te Cafe. I buy loose leaf tea here and also like to come in and occasionally drink a pot with a friend or my fiance. The atmosphere is great. I love it here.',
      'useful': 0,
      'user_id': 'GvYzuy5nwG9QWC7Mp11dHg'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2011-02-24',
      'funny': 0,
      'review_id': '26jzh7o4_D5ifUIPwqn_-g',
      'stars': 5,
      'text': 'I have trouble ordering tea when I\'m out because it\'s such a simple thing to make. I know there are a lot of nuances to the preparation, like everything else, that can give you a plethora of subtle variation that you wouldn\'t get if it was made incorrectly. However, I still can\'t stand paying for it.\n\nHowever, Te Cafe has a great selection of gourmet teas that make the stop worth it. It\'s the kind of stuff that, unless you were a tea connoisseur, you would never be tasting. Their menu is full of teas all with names that vary from mundane, to Japanese, to just plain odd. I\'ve been to the Te Cafe a few times but last night I ordered the "Kokeicha". It\'s a kind of smokey green tea that you steep for 3 minutes. I tried to order a tea by the name of "gunpowder" but they were sadly out of it. The man behind the counter informed me that the "Kokeicha" was very similar.\n\nThe tea was great and, honestly, even when something\'s not your cup of tea you have a fun experience. It\'s something you usually don\'t get to try on your own. Who would honestly buy 23 dollars worth of loose leaf tea that they\'ve never tried? I\'ll leave it to the Te Cafe to let me sample them a pot at a time.\n\nEvery time I enter the cafe the staff are friendly, accommodating, and most of all informative. They know their way around tea so someone who is completely ignorant, as I am, can get a little guidance. It\'s a fun place to relax and enjoy a beverage you usually wouldn\'t; how often do you drink gunpowder at home?',
      'useful': 2,
      'user_id': 'knDrH7HggUkpMSO92bzJLQ'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2016-05-29',
      'funny': 0,
      'review_id': 'zFYHaBth2JyaK2VaTyzF1w',
      'stars': 5,
      'text': 'Great selection of fairly-priced teas. Good food, too. Could be bigger, but never had a problem finding a comfortable seat. Not much to say, check it out!',
      'useful': 0,
      'user_id': 'A8_ovU0s8KGipI8DPPzXzA'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2009-08-11',
      'funny': 0,
      'review_id': 'DnOWvfCUtGZlfiAJm13b0g',
      'stars': 5,
      'text': 'This is my favorite cafe in Pittsburgh.  The ambiance is nice, somewhat modern and chill, but still comfortable.  There are board games available for playing, but most people seem to either be there with one or two other people to chat, or on their laptops making use of the wi-fi. The staff is very knowledgeable about different kinds of tea, and if you need help picking something, will ask you questions to figure something out - green or oolong?  floral or fruity? - and then bring out several teas for you to smell before choosing one.  My favorite so far is the silk oolong, which is a smooth and slightly creamy oolong tea.  I also really like the tea cups they use for Chinese teas.',
      'useful': 0,
      'user_id': 'Q1HKO4Q4S_r0JCwdugs3Og'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 1,
      'date': '2017-01-18',
      'funny': 0,
      'review_id': 'g9vdZh-qikRXXqlTZq_l_g',
      'stars': 5,
      'text': "Te Cafe is one of my favorite spots in Sq. Hill! In the winter, the windows are usually nice and foggy, providing the coziest feel when you're settling in with a hot cup of tea. They also have a nice selection of pastries and sushi, so you can snack or have a tasty meal while you're there to compliment your drink. Their chai tea latte is deeeeelicious as well, highly recommended! Great prices, great service, great food & drink :)",
      'useful': 1,
      'user_id': 'MoFYvOoufOMyVTmUFAhRxQ'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2016-03-11',
      'funny': 0,
      'review_id': '2PKj9KBB4qSNW4hFl9TNKw',
      'stars': 5,
      'text': "Really delicious tea and a good atmosphere. Everyone who works here are nice and helpful. I never feel rushed when I work on my computer. The owners are also great, and bring a nice sense of community to Squirrel Hill when you walk in. \n\nI tried the Sushi here for the first time, and was surprised that they made it on site. You get a lot for $8-$9, and it tastes yummy. Would recommend. \n\nThe mulled tea is my favorite though. I'll miss it in the summer.",
      'useful': 1,
      'user_id': 'UcwzDDDK4dElIwBLTH4UwQ'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2010-08-08',
      'funny': 0,
      'review_id': 'D7N8Fw8WxsZT86S__k00tg',
      'stars': 4,
      'text': "Good selection of tea, plentiful seating, nice atmosphere. I had an iced hibiscus tea that was pretty great. I will definitely be returning.\n\nAlso, they have outdoor seating (though it's on the Murray Ave sidewalk and a little noisy) and a clean bathroom.",
      'useful': 0,
      'user_id': '9iNyv-Kf4B748GsL0d4QxQ'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2017-05-22',
      'funny': 0,
      'review_id': 'uV0ElhSj-gNRWqlHppN8fQ',
      'stars': 5,
      'text': 'The best hidden gem sushi in pittsburgh! Great tea selection also.. highly recommend for delicious inexpensive delivery sushi',
      'useful': 0,
      'user_id': 'YTb6dkfoYlYAzXI8veXgZA'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2012-12-09',
      'funny': 0,
      'review_id': '9mvAc-aujfSYPP-2K6-3Lg',
      'stars': 5,
      'text': 'Perhaps one of the best tea/cafe experiences in Pittsburgh. Hands down, this place is off the charts. The ambiance, employees, and tea are the best in Pittsburgh. Their selection of tea is impressively large. The cafe is also a great place to study.',
      'useful': 0,
      'user_id': 'nLXCAodRN6Z6O4-CjshgPQ'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 1,
      'date': '2006-09-24',
      'funny': 0,
      'review_id': 'nXUw865N8wCXCoS0sWzQEQ',
      'stars': 5,
      'text': "I can't think of a better to way to spend a lazy sunday afternoon than a pot of oolong, a copy of the sunday times, and just sitting back and relaxing at the Te Cafe.",
      'useful': 1,
      'user_id': 'k6Y8yT_EJtL5hhwnYPDUfg'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 2,
      'date': '2017-06-24',
      'funny': 0,
      'review_id': 'c_IJF6qqXY8f2wjPGt5EKA',
      'stars': 4,
      'text': "I've come here twice and have to say their tea selection is fantastic. There are so many different kinds of teas and I appreciate the variety they offer in terms of food + ambiance + teas. \n\nI would give this place 5 stars but I am not a fan of their bubble tea. They microwaved the boba (they pre-make it and then put it in the fridge) and the flavor didn't even taste like tea. It was $4 and I don't feel like it was worth it. I was hoping it would be better since they do specialize in tea but I do realize that doesn't necessarily mean their bubble tea would be good since in my opinion that is a whole different category.\n\nI would come back strictly for their hot teas but definitely not the bubble tea.",
      'useful': 0,
      'user_id': '3EN4rQgpR8cR1NxuJNzi9Q'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 1,
      'date': '2008-09-10',
      'funny': 1,
      'review_id': 'JB8Nwp6J-jX-jiQbWPfn3g',
      'stars': 4,
      'text': "Every tea and tea-related concoction I have had here has been great. The atmosphere is bright and warm. They must have taken the other poster's music complaint into account, because I have heard a wide variety of great stuff every time I have been in there.\n\nThe staff is always super-friendly (without being annoying) and the clientele is usually a good mix of studying students/laptop users and people actually having conversations.\n\nIn short, good tea, good tea-related creations, good service, good music, good place to study or have a chat.",
      'useful': 1,
      'user_id': 'cQpTkatAqnOBCarVNwixPQ'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2008-03-15',
      'funny': 0,
      'review_id': 'AyHH7u7bFe6EIGa5309bCg',
      'stars': 3,
      'text': 'Te Cafe does tea well.They have an outstanding selection of teas, and they are very careful with brewing and all of that.\n\nThey also have a nice selection of board games and big tables to play on, which is a nice addition.\n\nOne thing kills it for me: the music is ABYSMAL. New age, strummy folky wispy Lilith fair crap every time. I still keep coming back from time to time when I have a tea craving but WOW.',
      'useful': 0,
      'user_id': 'yKFeQD-gcsPKj1BIs813oQ'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2009-11-07',
      'funny': 1,
      'review_id': 'RBEoSzBeoqAaEXfxYlrzRw',
      'stars': 4,
      'text': "Even if you're not in the mood for something from Te Cafe's diverse tea selection, there are plenty of other drinks that satisfy: tea lattes, lemonades (both plain and with unusual flavors like ginger, lemonade and hibiscus), hot chocolates and more.  The pleasant staff is knowledgeable about its teas and can offer great recommendations.  The loose leaf tea is also sold, and some of the more unusual tea blends can make neat gifts for people.\n\nThe atmosphere is calm enough that I can concentrate if I bring work to do.  It's also really helpful that in terms of food, Te Cafe offers more than sweets--while I loved baked goods, sometimes I need something more than sugar to fuel my noggin.  Grilled cheese sandwiches and vegetable soup are recent additions to the cafe's menu.  While the soup is clearly from the can, it and the grilled cheese are nourishing.  I definitely appreciate not having to pack up all my study materials and leave in search of a decent meal.  \n\nI really like the feeling of community that Te Cafe fosters with an area for posters and fliers for upcoming local events and work by local artists on the walls.  The cafe also hosts occasional poetry readings, but unfortunately I've never been able to attend one.",
      'useful': 2,
      'user_id': 'zh8-XfjqlwAxf--RS4azzg'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2012-11-30',
      'funny': 0,
      'review_id': 'ODzDGcRzGUw2J-Yvxfjz9Q',
      'stars': 4,
      'text': "This is my favorite place to go in Squirrel Hill. There's so many different drink and tea options, I have to keep going back to try everything! The atmosphere is really nice and chill too- perfect for reading, sketching, studying, etc. I only wish there were more pastry/dessert options, other than just tarts and biscotti. It may actually be a good thing (for my wallet) though... haha.",
      'useful': 0,
      'user_id': 'hHhUIFHQucto-L3GDY8XxA'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2010-12-29',
      'funny': 0,
      'review_id': 'lvMMuZZqR6uQ79O-XcfjXw',
      'stars': 5,
      'text': "Started going to the Te' when they first opened. The two ladies that run the place are very nice and good to talk to.\n\nThe teas are excellent! I have yet to find a better place to get good, loose leaf tea. The presentation is top notch and the variety is excellent.\n\nThe atmosphere of this little tea shop cannot be described. I would stop by towards the end of the day to finish up some work. The relaxed setting and fairly quiet place made it easy to focus and get things done.\n\nI don't work in the area anymore, but I'll use any excuse to get back there.\n\nSeriously, it is worth fighting to find a parking spot in Squirrel Hill for this place.",
      'useful': 1,
      'user_id': 'mit8WLpdSlOrRZD71SOgjw'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 1,
      'date': '2015-11-14',
      'funny': 0,
      'review_id': '6kpcse53pqmuKjwuCrKy5Q',
      'stars': 5,
      'text': 'They have outlets almost everywhere and the seats are extremely comfortable. I had a Kyoto Cherry Rose tea that was absolutely incredible. The staff really knows their tea, the presentation is wonderful, and the atmosphere is beyond pleasant. They also have a few board and card games. Go for work or a game night with friends!',
      'useful': 0,
      'user_id': 't3FKu7xMv6UkyGegAewWCw'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2017-07-12',
      'funny': 0,
      'review_id': '0Hv-VzdIdn-oraQfIvBKCA',
      'stars': 4,
      'text': "This place is super cute! I went here with my bestie (visiting from CBus) and she said it reminded her of Zen Cha, which I can see. You can order pots of tea, sushi, other drinks and real meals! We got a pot of their Apricot black tea to share. It had enough for about 2 refills and the set it came with was adorable! Taste was rich and fruity and made me want to go back for another flavor!\n\nMy friend got their sashimi bowl and it looked awesome! It made a nice, light lunch for her and the price wasn't outrageous. I see that they have crepes too so will definitely have to drag the hubs here to order all new stuff! \nThey also have some games in the back but I was happy just chatting with my friend in the laid back atmosphere. \n\nSome other reviewers mentioned this but the smell of this place really wasn't appetizing--must be due to the combo of fish + tea? The owners were nice enough but they don't start selling their savory dishes until after a certain time which was not listed on the menus. So don't come here too early on the weekends!",
      'useful': 0,
      'user_id': '8DclOINSPldMTgiEbuCdoA'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2012-04-02',
      'funny': 0,
      'review_id': '3p37CToxlfadSTt5-sSBqA',
      'stars': 4,
      'text': 'Delicious green tea latte and tea mulled apple cider. Cozy atmosphere, generally quiet, would recommend for reading or studying and enjoying good tea.',
      'useful': 0,
      'user_id': '6lu-nRaojUcVdP4nhi3Asg'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 1,
      'date': '2015-01-25',
      'funny': 0,
      'review_id': 'BMhy7jzIDZThq8ft4R_K1A',
      'stars': 4,
      'text': "Nice basement/attic feeling where it feels like it's probably what your house feels like",
      'useful': 0,
      'user_id': 'OVyBgRQORTrSNodNYPxJlQ'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 2,
      'date': '2012-06-23',
      'funny': 3,
      'review_id': 'xUq3iv0vyx4_sAhPX4GQ4Q',
      'stars': 5,
      'text': "I can't believe all this time I did not write a review for Té Cafe. Since I came to Pittsburgh 4 years ago, this place has been my favorite location for some serious work. Hmmm, maybe I didn't write the review because subconsciously I wanted to keep this little gem all to myself. \n\nThey have an extended selection of teas though I usually just go for the easy option: the tea of the day. I've gotten their yerba mate every once in a while and I like that they tell you to properly time it at 5 minutes. Nowadays they make smoothies too though I personally haven't tried them (it's nice and quiet without the constant blender buzz). For snacks you can go with the grilled cheese sandwiches or some of the regular coffe shop sweets. Sometimes I complain about the lack of options in the snack department but I have to admit it seems wise to keep the food choices simple, after all you have more than 50 teas to chose from.\n\nIf you're looking for a mellow place to work (with comfy chairs) and few people mostly working at their computers or reading their ipads this is the place for you. \n\nSide warning:  I love the comfy chair and I'll give you mean stares if you happen to occupy it :)",
      'useful': 5,
      'user_id': 'VwAhjZPN7zJuyG03P8mp_w'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2017-03-01',
      'funny': 0,
      'review_id': 'hHbG4oTPwLcrl2MZ20Af0Q',
      'stars': 5,
      'text': "I love this place, it's become my go-to take out sushi place. The owners are very friendly people, the atmosphere is charming and quiet, the tea is very delicious and varied, and the sushi is excellent at a cheap price. I come here at least once a week now.",
      'useful': 0,
      'user_id': '0zwj4cI7rhhQM9Ck8ZnIGg'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 1,
      'date': '2014-09-29',
      'funny': 0,
      'review_id': '1Y61Yy0hZtv4jzTqlUh4-w',
      'stars': 5,
      'text': "I love buying a pot of tea and studying here. It's always quiet, there's a ton of outlets, and you can always get free refills of hot water. It has a great, relaxing environment. Their lavender lemonade is great as is their plum berry white tea. I wish they had more room. The one thing they can definitely improve upon is their bathroom. The toilet is almost always broken and it never seems too clean.",
      'useful': 1,
      'user_id': 'CMjjwZ43THQpVS9VwbpWOw'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 1,
      'date': '2010-09-02',
      'funny': 0,
      'review_id': '0ZZXh9XpkNnuUyHEP1jmHA',
      'stars': 5,
      'text': "Not a tea drinker myself, but I stumble in here about three or four times a year to buy tea for my mom for various gifts. Every time I've been in, the staff's  been great with suggestions, samples, patience, etc. Went in again yesterday, and again...totally great. For being a clueless meathead, I was treated with no condescension, just the barista's enthusiasm for tea.  \n\nOne of the good businesses in Squirrel Hill I'm happy to support.",
      'useful': 0,
      'user_id': 'cKTuL7ZUKZv027HVxQWkNA'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 1,
      'date': '2014-07-07',
      'funny': 0,
      'review_id': 'rfqHyUT7f0mJIvB7qFQc3w',
      'stars': 5,
      'text': "I recently  moved away from Pittsburgh, but this is one of the spots I miss the most. I spent many hours here working on my thesis, but also many hours here just hanging out with friends or reading a book. The people who work there are very friendly and very patient. When I first started stopping in, I knew very little about tea, and they always took the time to explain different properties of each type of tea and were able to make thoughtful suggestions for things they thought I should try based on my current mood and goals for the day.\n\nOne of the best parts of this place is the insane selection they have. I felt like there was always something new to try. I loved the hot cider in the winter.\n\nThe space was almost always neat and clean, including the bathroom. I enjoyed the little flowers they kept on the tables, and the music selection was what you would hope for in a cozy little tea shop. The atmosphere is inviting, and only rarely would it be too crowded to find a place to sit. The wi-fi was much more reliable than other places in the area.\n\nI love this place! If I ever make it back to Pittsburgh, I'm going to make it a point to stop in. Keep up the great work!",
      'useful': 1,
      'user_id': 'OKUs_ZvoDYkgg-bFRfmHCg'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2011-11-19',
      'funny': 0,
      'review_id': 'mp70-GC6K-GUWhr0zhMVzg',
      'stars': 5,
      'text': "This is my favorite cafe in Pittsburgh. I'm not much of a coffee drinker, I prefer drinking tea or smoothies if I'm sitting at a cafe and Te Cafe is perfect for that. It's a very chilled out place, got big windows and all kinds of seating. It's not the fanciest cafe around, which is actually nice, because it has a very comfortable vibe to it. They have free and unlimited wifi. The ambience is great for doing your own thing, but I've also played board games there.\n\nThey have all kinds of teas there. They have specials every day in each variety, and they give you a pot of it, which is around 2-3 cups. You can also order a pot of any loose leaf tea on the menu. Or, you can buy loose leaf tea as well. My all-time favorite is the Panetonne. It's a black tea with vanilla and orange, and just the right amount of it too. I also really like their Green Tea smoothie, they make it with Maccha powder. They make grilled cheese sandwiches and have some biscottis and muffins.\n\nGreat place, check it out!",
      'useful': 0,
      'user_id': 'uFinfsLRjbiEdiW1qgpRHQ'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2009-08-09',
      'funny': 0,
      'review_id': 'B-f8PvfKHiMus0vkeB--6A',
      'stars': 3,
      'text': 'Really interesting tea flavors. Stop by if your in the neighborhood.',
      'useful': 0,
      'user_id': 'JVJOfFqPmZEvkRznzF8w-w'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2015-08-28',
      'funny': 0,
      'review_id': 'GINnZIuD2_Y2gbnVvPn2AA',
      'stars': 5,
      'text': "That's my second time here, and I'm so happy that i've discovered that place! Very comfortable and relaxing atmoshere, delicious teas and huge variet varieties them! Friendly, always kind and heping stuff! That's one if my favorite places now!!!!",
      'useful': 0,
      'user_id': 'QNps8E2YEMweAnWdqPZxhQ'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2013-06-05',
      'funny': 0,
      'review_id': 'kMfF-7-zB65p7fTf79CWbA',
      'stars': 4,
      'text': 'Aw, I love it. It looks like a really great study area. I would come here to study. Also they give you a pot of tea with free hot water refills. That would be perfect for studying. Also, the guy working there was really helpful with suggestions. And the place inside is just super chill.',
      'useful': 0,
      'user_id': 'qpnrU31DaVDjxO8gxkrQGg'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2016-08-14',
      'funny': 0,
      'review_id': 'kPF3A3_90Q76Z3-zjWkn9A',
      'stars': 4,
      'text': "Can't vouch for the sushi as I only stopped by for a lunch time snack.  Tried the sandwich of the day which was a grilled cheese pannini and I have the iced chai latte.  Both were tasty and reasonably priced.  Staff was friendly and place was clean and comfortable.  If I was living in this area I could see myself coming in here on a pretty regular basis.",
      'useful': 0,
      'user_id': 'Ed_8rtw2VLvS-la9nWjWBw'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 1,
      'date': '2016-04-18',
      'funny': 0,
      'review_id': 'DfHmJy7B_IiOZtF4fOBizg',
      'stars': 5,
      'text': "With a calm, peaceful atmosphere, a very friendly and knowledgeable staff, and a wide selection of great quality products, I would definitely recommend this place.  Both the iced chai tea latte and the sparkling tea were excellent, and I'm sure that their other stuff is great too.\nI'm definitely going to have to make a point of coming back.",
      'useful': 0,
      'user_id': 'sV3xGQ-mFG48hzJcArlLZw'},
     {'business_id': 'MKd_EUD0PG__cOfIbeqYAg',
      'cool': 0,
      'date': '2017-06-20',
      'funny': 0,
      'review_id': '4MwWqMejF_5Lm3HJlPvjFA',
      'stars': 2,
      'text': "I wouldn't have guessed that a tea cafe would sell sushi but my friend recommended to try this place. Ordered the Twin City Roll which had tuna, salmon, avocado, and cucumbers. There was more sushi than rice so the ratio was good. The sushi looked good and I thought the price was reasonable. But when I was eating it, the sushi didn't seem fresh. There was a strong fish smell and the aftertaste left a slimy feeling on my tongue. The interior of the cafe also had a weird smell... I don't know how to describe it exactly. I don't think sushi and tea smells go well together? I don't know if I would recommend coming here for sushi. Maybe their tea is better as it is a tea cafe.",
      'useful': 1,
      'user_id': 'bryAPtwnnMYjbzoOWHzbWg'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 0,
      'date': '2015-07-29',
      'funny': 1,
      'review_id': 't9Ay7vjt_FkLRyfDVrbfXQ',
      'stars': 4,
      'text': 'This is my 9 year olds review.     Hamburgers are great and the cheese curds are awesome.   There is a Great Lake View',
      'useful': 0,
      'user_id': '0J4xtoM6MeqFr_mizTNvhw'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 0,
      'date': '2016-07-08',
      'funny': 0,
      'review_id': 'DoOr0Vsn9cxwzU9M-HPQFQ',
      'stars': 2,
      'text': "So I know Christy's is a Madison tradition with those lovely views of Lake Waubesa. But this place is a one-track pony... location, location, location. This is a family-owned bar/restaurant and the owners know they have a captive audience. The beer is cold and the food is average, but the service is atrocious. On a recent beautiful Saturday afternoon, there were only two servers for the outside seating area.  There's no host, so we (4 adults and 1 kid) sat at a picnic table and proceeded to wait at least 10 minutes to no avail. We eventually went inside to order drinks and lunch. At no time during our visit did a server ever approach our table. I'd certainly be inclined to write off this awful experience to a bad day or poor staffing, but unfortunately this is more the rule rather than exception. One really weird thing is the Friday Fish Fry that features a special menu available ONLY inside, this isn't communicated very well;  so you'll grab a table outside, wait to order only to learn you can only order from the pizza/sandwich menu.",
      'useful': 1,
      'user_id': '5JOP01e-jC-N3OrnevM7bA'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 0,
      'date': '2015-06-07',
      'funny': 1,
      'review_id': 'C7d9s1Oo2wtPkMh6JvQDoA',
      'stars': 5,
      'text': 'Great scenery. Solid food. Great energy! A must in the summer!!',
      'useful': 0,
      'user_id': 'Y9UTvawzqt1gObIsPxPQaA'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 0,
      'date': '2016-08-14',
      'funny': 0,
      'review_id': 'KhuDGyZaiRQ7ogBWGHqYgg',
      'stars': 2,
      'text': "This place has such potential but the service SUCKS! And I wrote that in capital letters on purpose. You literally have to stare a waitress down and even if you make eye contact you're lucky if they will stop at your table. Come on! We boat on the lake and wish there were better options than this. I guess it's Waypoint every time! As a side note: I'm writing this on a Sunday afternoon as I've been sitting at an outside table for over 15 minutes without one of the three servers currently working even acknowledging us! I really wanted the Shrimp Christy's. (Because it's fantastic!) Oh well. There probably won't be a next time either!",
      'useful': 1,
      'user_id': '4RptW32AaMcCzEArZjjitw'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 0,
      'date': '2016-09-05',
      'funny': 0,
      'review_id': 'qqiIYIsE_6eKjeLRn004VQ',
      'stars': 1,
      'text': "Unless you're a local yokel McFarland douchebag, you won't get any measure of service here.  After waiting more than 15 minutes for a server outside, I went inside to inquire of outdoor food service.  Of course, since I am not one of the regular inebriated bar flies, getting the attention of a bartender is nearly impossible.  I guess I'm just not loud enough, drunk enough, or both.  And once I did get his ear, the next local loudmouth douche bag stumbles up to the bar and interupts me to order his next round of Captain and Cokes (classy).  So we left and will never be back...unless of course we lose all sense of decency and purpose and relocate to McFarland.",
      'useful': 2,
      'user_id': 'tlQ2MDwfE1TBCNmsofeGfg'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 0,
      'date': '2016-05-05',
      'funny': 1,
      'review_id': 'c3Qu6sAMJTOgwyjnWbFhMQ',
      'stars': 4,
      'text': 'Great volleyball courts! The sand is soft but at night, some of the lights will shine right in your face and obstruct your view while you\'re playing.\n\nThere is a hose and area to rinse off afterwards, by the dock facing the beautiful, glistening Lake Waubesa.\n\nA cherry bomb will run you $7 and a Vodka 7 will run you about $5 or so.\n\nBe careful when you order the pizza, you start with either a 12" ($8.99) or 16" ($10.99) cheese and each topping is $1.50! My friends and I weren\'t aware of that and got a little carried away when we got our tab for all the drinks, pizza and wings was just under $100!\n\nOn Monday nights they have a AYCE taco bar with ground beed and chicken, as well as assorted toppings. The pico de gallo was very fresh, the shredded romaine was crisp, and the salsa had a nice bite to it. Every time you were ready for more, just ask the nice bartender to give you more shells (corn hard shell or soft flour tortillas).',
      'useful': 5,
      'user_id': 'd5WLqmTMvmL7-RmUDVKqqQ'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 0,
      'date': '2015-03-06',
      'funny': 1,
      'review_id': 'UtsYhJchMT1Y_JsB9SU_5g',
      'stars': 5,
      'text': "Great food, great environment, low key, full bar plus a tiki bar in the summer! Best nachos I've ever had in my life. Seriously. Overall just a chill, consistently great place that in the summer is the place to be if you like the sun lake and boating. Been looking for a place like this",
      'useful': 0,
      'user_id': 'uVsoP6rc9fHhjtVienl68Q'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 0,
      'date': '2013-08-29',
      'funny': 1,
      'review_id': 'RUdGyQeJuddLPu6FqE12lw',
      'stars': 3,
      'text': 'Menu is short. Excellent black bean veggie burger and seasoned waffle fries.  Old fashioneds are made from scratch/muddled.  I had excellent service while there--nice, competent bartender came outside to refill drinks several times.  It was a comfortable experience, relaxing outside watching the waves.  Veggie people beware taxidermied animals inside.',
      'useful': 0,
      'user_id': '6uLqCFKpK7nav7Zp_3TfRA'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 0,
      'date': '2014-01-18',
      'funny': 1,
      'review_id': '4ed5B3N9Y8Cuo2bnm43Nig',
      'stars': 5,
      'text': 'Good food, drinks, and views.  Great place to come by boat in the summer!',
      'useful': 0,
      'user_id': 'hT_2tGVf4b6ahqFj6GpApw'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 0,
      'date': '2015-02-03',
      'funny': 0,
      'review_id': '2KoVbCwHp49aUnvflElEbw',
      'stars': 2,
      'text': "Enjoyed a beautiful day on the patio last Sunday listening to the Blue Olives (they're great!) and sipping beer. Ordered the nacho's and was fully disappointed-a somewhat small pile of flavorless corn tortillas and minimal cheese, chicken, peppers, and a clump of black olives. It appeared to have been placed briefly under a hot broiler as the top was charred and any buried shredded cheese was unmelted. The result was a very dry dish that required multiple cups of salsa to provide flavor and moisture.\n\nNote: This review was written during the summer of 2014 but I just found it in my Drafts-- sorry for the delay!",
      'useful': 1,
      'user_id': 'VvwrCpL2_uMrRumFWo7nLQ'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 0,
      'date': '2012-02-27',
      'funny': 0,
      'review_id': 'iurpwMMbZ3CeNDd9GX1JIg',
      'stars': 2,
      'text': "Love going here during the summer.  Nice bar outside and great people and boat watching.  Went here for dinner last week and it was just Meh.  Food was bland an uninspired.  What the waitress offered as the special was not what was delivered.  After we asked, she still didn't know what was included with the dinner.  Walleye sandwich was just OK.  Service was on the slow side too; one server for the entire 2 floors.  The lady tried hard, so she earned a good tip.  The kitchen and management could do a lot better.  Probably will not go here again for dinner.  Too bad because we do like the place overall.",
      'useful': 1,
      'user_id': 'BcLEkxjjYX-L3jKdUKtWnw'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 0,
      'date': '2011-07-16',
      'funny': 0,
      'review_id': 'eYxRUyzyPkQJ63ypRjVHqQ',
      'stars': 1,
      'text': "I have been to Christy's quite a few times over the years and have never had a very good meal; however, last night's meal was a disaster! It started with the bartender getting our drink order wrong, One member of our party received a different potato than ordered, 1 had a different salad dressing than ordered and I had the worst fish fry EVER. It could have been good if the breading wasn't soggy on one piece and excessively crispy with holes in it (looked exactly like a funnel cake!). The baked potato tasted dry and leftover. The one high point was a waitress (first night on the job) that was very friendly and actually cared about our poor experience. Christy's is back on my list of places to avoid.",
      'useful': 2,
      'user_id': 'uzojZqd-h289ibuB6wPZ9g'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 2,
      'date': '2013-08-09',
      'funny': 1,
      'review_id': 'E1pjnmEX71QOZvDAGx3gJA',
      'stars': 4,
      'text': "I'm very surprised to read all the negative reviews. We were made to feel very welcome as soon as we sat at the bar by Tim the bartender. He was unfailingly polite and friendly, as were the people who sat near us. It made me wish we had more time to hang out and soak up the lovely waterfront location. \n\nStarving, we ordered a pizza and both enjoyed enjoyed the hell out of it. The toppings tasted very fresh and the crust had a slight flakiness to it. The only drawbacks I can think of are the plastic cups the beer is served in and that I'd love it if another taphandle were devoted to something a bit more local to Madison, like Ale Asylum, Karben4, or even Tyranena (from Lake Mills).\n\nAs we left Tim thanked us for stopping in and asked us to come back the next time we were nearby. I 100% plan to do so.",
      'useful': 1,
      'user_id': '1_9EFD8ZdRVVI1ylu2ks_A'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 0,
      'date': '2008-04-13',
      'funny': 0,
      'review_id': 'Fzg9uXBGOZgrEZO53W5diQ',
      'stars': 3,
      'text': 'Decent but not great.  OK as far as bar food goes.  Very smokey.',
      'useful': 1,
      'user_id': 'reN-Y7lIjv3VW21I4Z8WHQ'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 0,
      'date': '2013-06-20',
      'funny': 0,
      'review_id': 'ZDZ7Nu88u4jwIG1Yh_5KrA',
      'stars': 1,
      'text': 'Worst service ever. we arrived at 7 pm on a Friday night. Since it\'s on the lake, were not surprised there was a wait, but didn\'t really seem to be enoughpeople waiting for 1-1/2 hour wait.  We sat on the deck with a couple of drinks, and enjoyed that. At the 1-1/2 hour mark, went in to see how much longer and were shocked to see the mostly empty dining area in the bar. Maybe the upstairs dining was busy? Nope- one table of 4 waiting to be served, and 2 large tables of people who had finished eating and were doing their party speeches. We think they just forgot us.\nBrief wait for the waitress.10 minutes later got our salads for 2 of us, soup for 2  (husband wanted cole slaw, but they were out). Soup was good. Salad was very small, with a huge carrot \'coin\' and cucumber slice. italian dressing was blah, bleu cheese dressing had no bleu cheese chunks.\n30 minutes later, caught the attention of our oblivious waitress to check on the status of our order, and her comment was "didn\'t they tell you it would be an hour wait?" Wouldn\'t that be the waitress\' job? After another 20minutes, and the 1 hour mark, i went to find a manager. Bartender blew me off, so I went to find the host,who apologized, said it was coming out now, and said they "were just catching up". From what? The bar area dining was almost empty, and we were still one of two 4 person tables up in the big dining area.\n5 minutes later, with no waitress, no food, we left. We will not be back. Maybe the food is good, can\'t comment on that, but the aggravation is not worth it.',
      'useful': 1,
      'user_id': '_-Zgo91EBqEz5LiQtmZVDg'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 0,
      'date': '2017-06-11',
      'funny': 0,
      'review_id': 'NyWxv3NWZGkRym94DN8jOA',
      'stars': 2,
      'text': "I've been here a few times. The cheese curds and fries are nice. Onion rings ok.  Wings my kids love but I think they are too smothered in sauce. Each time I've been here It was by boat so we've sat out side. I would have rated it a 3 or 4 except each and every table has an ash tray which is gross and they allow smoking in the entire area. My kids 6 and 11  were with us and one has asthma so we had to move to get out of the smoke. Very inconsiderate allowing smoking, only 3 out of maybe 40 people were smoking so maybe inconvenience 3 people and not everyone else. Going forward Ill probably just get takeout.",
      'useful': 0,
      'user_id': 'QDzMM2YcB0LDLJVHDqrz3g'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 0,
      'date': '2015-08-01',
      'funny': 0,
      'review_id': 'reRtpwsYPjGLyeCVAaOpLg',
      'stars': 1,
      'text': "We tried the fish fry a few weeks ago after hearing great reviews from friends. We weren't disappointed. Everything was wonderful. Tonight we went back and our experience was the opposite. The lobster bisque was cold ( Apparently, they forgot to turn the burner on and we were the first to mention it. We were there at 8:00pm.). The fish was okay and the waffle fries were cold. My husband asked why we didn't get any bread. He was told that they ran out. We found that interesting since people who were seated after we were had bread. Not sure we need to go back.",
      'useful': 1,
      'user_id': '3z6t58i1h7tkDSaaGu-ukw'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 0,
      'date': '2014-07-18',
      'funny': 0,
      'review_id': 'Eh9x3hf6bA6p1KNkNlMkPw',
      'stars': 1,
      'text': 'Just got back from this restaurant for the third time. It is close to where we live and like the convenience of going there. This was the last time though. When we arrived we where told we needed to wait 5 to 10 mins for a seat. After 15 mins we where seated in a room with 10 open tables (not sure why we needed to wait).  About another 15 mins went by and 2 other tables where seated and served drinks and still no one waited on us.  No one ever did so we left. \n\nThe other times that we have been the food is only mediocre for bar food at best.',
      'useful': 4,
      'user_id': 'aJBDMS1-fLSxpEdj12I6dw'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 0,
      'date': '2012-08-08',
      'funny': 0,
      'review_id': 'FJ8RHbfzzbVg3hIB7QDdOg',
      'stars': 1,
      'text': 'This place is good only for the outdoor atmosphere. Their online menu does not correspond to their actual menu, buy they don\'t seem to see this as a problem. I requested my favorite beer: "Oh, we\'re out of that." Then, my second favorite: "We\'re out of that too." As a vegetarian, I had to order a black bean burger, which was burned, as was the bun. I asked for a dark or spicy mustard. No, they just have the mainstream basics. Combine this with extremely mediocre service. I will never eat there again.',
      'useful': 3,
      'user_id': 'O7Wru5nlVMwXzuI1-CNG4Q'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 0,
      'date': '2017-04-09',
      'funny': 0,
      'review_id': 'Y_DH_03zyYBmsLVkVaA0cA',
      'stars': 3,
      'text': "Unfortunately we had really high hopes for this place mostly because we noticed All You Can Eat Taco Bar on the website but apparently the websitr was not correct/up to date so we were left with other options. The waffle fries were delicious, wraps were average and a little pricey, and service was fair (even in the off season). Can definitely see the appeal during the summer but it won't be a place I seek for good eats and drink.",
      'useful': 2,
      'user_id': 'sxVgVtnnxyJlOB550cy5sA'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 0,
      'date': '2010-08-08',
      'funny': 0,
      'review_id': '_slgwd3lV-935lVEYZHeYA',
      'stars': 3,
      'text': "I was actually quite surprised at how good the fish fry was.  I had some great hashbrowns with it and my wife really enjoyed the salad that accompanied the meal.  If you're a craft beer lover like myself, you'll be disappointed in their beer selection, but Christy's is worth a visit solely for their fish fry.",
      'useful': 1,
      'user_id': 'YJInht2-Z0Pj9CH3ML1R3g'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 0,
      'date': '2015-07-06',
      'funny': 0,
      'review_id': 'oHm8T7HOpB8CAy6aHpOZhQ',
      'stars': 3,
      'text': 'Drove out on the bikes on a gorgeous Sunday.   It wasn\'t overly crowded.  Overlooks the lake and many people dock their boats and enjoy the patio and food.   Lots of people taking food back out onto their boats.  \nUpon arriving, there is no sign telling you to sit down or see the hostess for a seat.  We walked around and into the bar and then we finally were told by the bartender that we can sit where we want.   \nThe menus at the table were all different.   Three different menus, one omitted the burger and the other one didn\'t have the "baskets" on it.  Weird!!  The waitress knew that something was wrong with the soda guns and brought us 2 pitchers of essentially colored water.  When we finally mentioned it, she acknowledged that they were having problems with it.  After a while she brought us another one and said she hoped it was better.  It finally was.   The orange and lemon peels thrown on the grass, could be cleaned up as well.  \nOverall, the food was good.   Burger was juicy and flavorful and he bacon wrapped shrimp was really yummy.  The pizza at the table next to us looked delish and may be ordered if we return. \nThe place has some potential but a little makeover would be great!!!',
      'useful': 2,
      'user_id': 'fzWuWObdqMv-81BUk6Z8hA'},
     {'business_id': 'xOx-N9G7AteZTnZfVFfNHA',
      'cool': 0,
      'date': '2014-12-16',
      'funny': 1,
      'review_id': '8ia9RSWTgh-ZFqhftS8ylQ',
      'stars': 5,
      'text': 'A great neighborhood hangout with good food and great beer.  This is THE place to be on a summer night, and usually something fun happening on the ice in the winter.  The place is loaded with character(s) and is a true Wisconsin tavern on the lake.',
      'useful': 0,
      'user_id': 'u9aPEXqLiJKT84t9bzm-kA'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2017-05-15',
      'funny': 0,
      'review_id': 'onfGsGa0z59efaD6URgAEA',
      'stars': 4,
      'text': "A really good sandwich place. It's location is a little unusual but very convenient to the surrounding airport hotels. The turkey, avocado bot was great.",
      'useful': 0,
      'user_id': '2ik193bx9ElgztaB0Gf9qg'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2016-02-29',
      'funny': 0,
      'review_id': 'CI2hQUaR8KR8SGXvM1t0GQ',
      'stars': 5,
      'text': 'The food was definitely solid but what stood out the most for me was the friendly staff and the iced coffee they whipped up. Wish I had got their largest size. Highly recommended!',
      'useful': 0,
      'user_id': '9HGer_DiSPF3xJVXIcQM4Q'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2017-03-31',
      'funny': 0,
      'review_id': 'CsmCJ-2V3MyTYETirfBo0Q',
      'stars': 4,
      'text': "Great place to grab lunch. The location is a little hidden. It's in an office building. There is shaded outside seating. The food was fresh and my sandwich tasted great. My only complaint was that the potato salad had too much dill in it.",
      'useful': 0,
      'user_id': 'pwpQncGpmxgJ01r2KI67qw'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2016-09-14',
      'funny': 0,
      'review_id': 'hgovU_roMRHwRsYRdy0yYQ',
      'stars': 5,
      'text': 'What a find, right behind the Radisson, fresh cheap and delicious.  Friendly staff. Give it a try.',
      'useful': 0,
      'user_id': 'SuPR-q-vaOkdY1K3IbMYYw'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2013-01-28',
      'funny': 0,
      'review_id': '40nKNzdl3ulfPRKVGDgz7g',
      'stars': 5,
      'text': "I work in a office building right around the corner from them.  The place is small, but once you find it, you'll keep coming back.  Honest to God, the food is addictive.  My faves are the Italian Hoagie (reminds me of home back East) and the Chipotle Chicken sandwich.  And I also enjoy their flavored iced tea to go along with it.\n\nStaff is friendly too which makes the experience even better!\n\nI'm a fan!",
      'useful': 1,
      'user_id': 'h7J6pJrLCzUu-LLpuQpxGg'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2016-03-17',
      'funny': 0,
      'review_id': 'rOC3Hs18POWCqbituKs0lg',
      'stars': 3,
      'text': "I can't figure out why their food isn't that great. They seem to use good quality products. Been there for breakfast and lunch and its meh.",
      'useful': 0,
      'user_id': 'GKcIOfDnPSrveGRi4BLMmA'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2016-03-21',
      'funny': 0,
      'review_id': 'F4E_0pzw0uZLhuHLfLVudg',
      'stars': 4,
      'text': 'Only open for lunch spot, a nice little sandwich shop hidden behind the Radisson. I had the Grilled Fiesta Chicken Wrap and really enjoyed it. Pretty large sandwiches from what I can tell and looking forward to trying other options, as well. Being from out of town, you can tell it has its regulars and people really like this lunch place.',
      'useful': 0,
      'user_id': '-AUqoiMsKwilppN1671nHQ'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2017-01-03',
      'funny': 0,
      'review_id': '97f1lNOtVnT25y_Qs1_PGA',
      'stars': 5,
      'text': "After arriving at PHX airport we searched for a nearby lunch option and this was well reviewed.  It was a little difficult to find.  It's hidden in a larger building with very little signage.  Also the parking lot has signs about hotel guests only etc.. so that was a little unnerving.  They were closing soon when we came in, but we ordered a BLT and a southwest chicken panini.  Both were ready quickly and were very good.",
      'useful': 0,
      'user_id': 'ni9ClI2mhAeAJUSaSY58jQ'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2012-02-03',
      'funny': 0,
      'review_id': 'Y3y1EdnVyhtTdn0asawa9g',
      'stars': 3,
      'text': 'I have visited The Lunch Lounge 3 times since one of their workers stopped by my office and dropped off a menu and some yummy cookies. The first time I wasn\'t sure what to expect so I got a Ceasar Salad hold the Ceasar please. It was okay, the chicken was good and none of the leafy greens were in bad condition. I guess I was expecting a wow factor when it came to the $6 salad I had received. they do brew Starbucks so I had decided to get one of my favorites. Hazelnut latte... only they were out of the Hazelnut Syrup. I instead got a caramel latte. Not my favorite, but it would do.\n\nMy second visit I was very excited to try one of their specialty sandwiches the Deluxe Roast Beef Press (only with 9 grain instead of sourdough) ! However when I arrived they had run of or roast beef, but I was told they were expecting their shipment to arrive tomorrow. Still hungry and not wanting to wait till tomorrow to eat I decided to try the Vegetarian Delight. $8.00 later my meal was... okay. Good sized, but in the end not what I was really hoping for when I had first came.\n\nToday I went to give it one more try. I got the Sandwich "your way" which I ordered Roast Beef, imported Swiss, on 9 grain with Lettuce and Spicy brown mustard. I also got a huge chocolate chip cookie and a Caramel Macchiato to wash it down. A total that came to $13.38. I arrived at my office and to my surprise my 9 grain bread turned into Sourdough - My least favorite bread... \n\nTo say the least I really wanted to like this place, but I am not blow away by The Lunch Lounge. I have given them a 3 for effort. - Also to add I found out this last visit that they only have soup on Tuesday\'s',
      'useful': 0,
      'user_id': '6ai3vG2vlcQXTINdbO5H3w'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2015-06-19',
      'funny': 0,
      'review_id': 'R3gJ_j4bWlkO8QD32Di06A',
      'stars': 5,
      'text': 'Happy to have found this place! Great food, decent prices, friendly staff. This makes me a more-than-once-a-week patron! And they will even accomodate an off-menu request as long as they have the ingredients. Thanks, Fred!',
      'useful': 0,
      'user_id': 'Zh_ochroi2vle8ZZPZ8dUw'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2016-12-30',
      'funny': 0,
      'review_id': 'uhxRYN8DsM2K1YWiuaTroQ',
      'stars': 5,
      'text': 'I absolutely love this place! In fact, probably my favorite sandwich spot. A hidden gem. I eat here atlas once a week. I have yet to have one bad thing.  My favorite sandwiches are the Queen of Clubs ( like the King of clubs, but with only a double decker not a triple decker) and the tuna melt. The cobb and garden of eden salad are my favorites. Their sandwiches and salads are pretty big, so go with an appetite. Its a mom and pop sandwich shop, which I love. Try it out! You will not be disappointed!',
      'useful': 0,
      'user_id': 'P7Bwn2qljq8qqKoZEXbsGQ'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2016-01-25',
      'funny': 0,
      'review_id': 'b7K6FGNibHQ6TH4j7feW6w',
      'stars': 4,
      'text': "Tucked in behind the Radisson at the corner of an office park, this is a great alternative to the food in the local hotels.\n\nI had the Cobb salad. Despite the generic iceberg lettuce it was enhanced with fresh spinach, avocado and egg and everything else was fresh. Good dressing and generous portions.\n\nAs others have said come before noon or after 1 o'clock if you don't want to wait in line. Three nice tables outside with a little lawn view.",
      'useful': 0,
      'user_id': 'IoLMXv6wzfKaisN60u5iyg'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2013-11-27',
      'funny': 0,
      'review_id': 'vsiKrkAlOXCEXiuN6SeF2w',
      'stars': 5,
      'text': "an excellent place for lunch, the food is great the service too! always friendly, never pushy, as one guy said, i like the 'tude of the lounge lady...i call it the the lunch lizard and its off the beaten path but you wont go wrong with thier specials or anything on the menu for that matter. the outside seating is nice, i try to eat here regularly...beats any chain sammich shop!",
      'useful': 0,
      'user_id': 'Rs0jSJBwEuv5SJ16m1UWhw'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2015-07-27',
      'funny': 0,
      'review_id': 'bVF8fvEiLfs4pnVwvPJy5w',
      'stars': 5,
      'text': 'i am not a fan of turkey so when i asked the sandwich guy for his recommendation, and he suggested the albuquerque turkey, i was unsure, nevetheless i went on to try it and boy! Was i glad? The sandwich comes with   fresh cole slawish salad, bacon cooked just right crisp and retaining its flavor, generous slice of turkey and to top it off chile mayo which is really what separates  this sandwich from the other diners... .all in all a great sandwich...made to perfection. I want to come here and try other items on the menu... They all look good....hopefully on my next trip to phoenix.',
      'useful': 0,
      'user_id': 'mtg2AObS7ZxtjJ47WSsfig'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 2,
      'date': '2017-02-17',
      'funny': 1,
      'review_id': 'iCtyeCryz3FzGARhTjbEvA',
      'stars': 5,
      'text': "Great service in a quick-bite-to-eat, make-your-own-sandwich spot. Most of the business probably wanders from those office buildings. People looked lost and in their own heads as they would methodically make their way. It is a sunny day at 74° but it's clear that's it's Friday at the office.\nAnyway back to the food-\nChoices for breads cheese meats veggies\nChips and soda \nPretty standard with more selections available too\n\nSandwich was well prepared and tasted as expected.\n\nNote: Almost gave them four stars because the cashier is a Colts fan but we all make choices and their good service and food shouldn't suffer from his choice to root for the Colts. I joke, I joke. Also, Go Patriots. :)",
      'useful': 1,
      'user_id': '6IrzPt91rwVgH52o8_ytNg'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2016-11-05',
      'funny': 0,
      'review_id': 'Hk33_V_Ltk708njGR-kT9Q',
      'stars': 4,
      'text': "Good food at a small lunch restaurant,  it is a good place to pick up lunch if in the area but it is not a destination lunch spot. More of a business man's quick stop.",
      'useful': 0,
      'user_id': 'B0ASQntm8Sy4aV1CXJi_rQ'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2014-07-17',
      'funny': 0,
      'review_id': 'nWk7otoD_ZL42bbbK4-jVg',
      'stars': 5,
      'text': "I have been to the Lunch Lounge many times for lunch and love the chicken salad.  It is the best!  I also love the BLT's.  Today I stopped in for breakfast and had the breakfast panini sandwich and I am once again impressed.  Thank you Lunch Lounge!!!",
      'useful': 0,
      'user_id': 'Aw9YiasaOs9zikh1XsbI9g'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2017-04-13',
      'funny': 0,
      'review_id': 'kmZTkEDMApvsuRwBrbgWbA',
      'stars': 1,
      'text': "Ordered the Southwest Chipotle Chicken Panini and it was still frozen in the middle.  FROZEN! I couldn't believe it; it took them a long time to have my order ready but still manage to not cook it all the way.  I didn't realize it was frozen until I got back to work to eat it.  By then, I didn't have time to return to have them make me a new one.\n\nThis is an expensive place for just sandwiches only.  My Panini alone was over $8 after tax.  Not a good value because it doesn't even come with a drink or side.  I appreciate and support local businesses whenever possible but after this mishap, I won't be back.",
      'useful': 0,
      'user_id': 'SR9KgvoaxcBsSd4Gds8LnQ'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2014-04-30',
      'funny': 0,
      'review_id': 'n3oI45ffNqHRriyyS8dY_Q',
      'stars': 4,
      'text': "I eat here at least once a week all of their sandwiches and salads are always fresh and delicious and every 9 salads or sandwiches you buy you get the 10th on FREE. Goodstuff! Glad it's near my work.",
      'useful': 0,
      'user_id': 'FL9L2jIG9QT4ODYE_VdEgA'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2014-05-20',
      'funny': 0,
      'review_id': '0W5SEy35PV93rjcAj3X1bQ',
      'stars': 5,
      'text': "Love this place.  Weekly specials are amazing.  Great, fast, and friendly service.  I'd highly recommend to anyone.",
      'useful': 0,
      'user_id': 'Lqp0uX1ShEpxgq1Rou7iHg'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2017-02-03',
      'funny': 0,
      'review_id': 'vShpdW9kU92417H9HvigDA',
      'stars': 5,
      'text': "Today I had the hot pastrami with on the side. The sandwich and the slaw: EXCELLENT!  The staff is great. I'm flying today so I also purchased a pb&j on grain bread for later.   I'm sure it will be awesome too.",
      'useful': 0,
      'user_id': '7i6GtPosZxmiAFtTZcuggg'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2015-08-26',
      'funny': 0,
      'review_id': 'GbzOy5nb6g9apvc5sb8lhQ',
      'stars': 5,
      'text': 'Excellent little hidden spot. Great menu with fair prices. Had the Breakfast burrito with a cup of coffee. Outstanding burrito! Loads of cheese & stuffings. The menu on yelp is a little outdated but the prices are more than reasonable. Definetly give this place a try.',
      'useful': 0,
      'user_id': 'Gq8OwMLyRRdL71NWp6VM9A'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2017-04-03',
      'funny': 0,
      'review_id': 'RN6EcYhy7JzlTgiu1ku_4g',
      'stars': 5,
      'text': "Great sandwich shop near several airport hotels. They also serve salads at lunchtime, and breakfast starting at 7:00 AM. Alas, wasn't in town long enough to sample the breakfast, but lunch was fantastic. Small but very popular at lunch when I was there",
      'useful': 0,
      'user_id': '4tGyRRF7ygEnwIs4ZqLBZg'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 1,
      'date': '2015-04-16',
      'funny': 0,
      'review_id': '6WWIKC_GYHFZKKZ8-mCrSg',
      'stars': 5,
      'text': "Had lunch here. Very close to work but only found out about it through yelp. My sandwich was delicious and the staff was super friendly.  I'll definitely be back.",
      'useful': 2,
      'user_id': 'JM18LKd3By6gL3WOFXhaMA'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2015-12-08',
      'funny': 0,
      'review_id': 'nzIHJdGBbkLFeUCBflmzOQ',
      'stars': 5,
      'text': "For a little sandwich shop in an office park, this place is great. \n\nDelicious, big breakfast burrito.  No problem to substitute avocado for the meat. Hot tasty coffee. friendly staff. Outdoor seating option.\n\nAll for under $6.\n\nSo glad I didn't have the more expensive breakfast in my corporate airport hotel.",
      'useful': 0,
      'user_id': 'v7omYtBpRag3wsStEwXYpg'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 1,
      'date': '2014-11-22',
      'funny': 0,
      'review_id': 'Xd7fDGNp6wc_8mjPs0Unvw',
      'stars': 4,
      'text': "I stumbled upon the Lunch Lounge while staying at the Aloft last week, and was thrilled that I did. It's kind of hidden from the street behind the Radisson on 44th St, but that didn't stop me from taking the short walk over from the hotel. Cute little place with great service and good eats! They have all kinds of sandwiches and salads, and couldn't decide what I wanted since the menu looked so good. They serve Boar's Head meats, which makes for an awesome sandwich. My friends who were staying with me raved about the food and said they wanted to come here for lunch again tomorrow. Only not so great thing was that they were out of the soup of the day, which we wanted to try, but since we did come about a half hour before they closed I could live with that. Wish we had more deli-type places like this in Milwaukee! Yummmmm!",
      'useful': 0,
      'user_id': '7a2OVx6IJTMpUtYDLPZhwA'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2013-04-05',
      'funny': 1,
      'review_id': '18RML0hSOiL4oWSpHefNfA',
      'stars': 5,
      'text': "What a find!  While staying at the Aloft in Phoenix for business we found this on a morning walk.   \n\nWe had the breakfast burrito and a breakfast panini.  Both were excellent!  The burrito was so large we agreed that it could have fed us both.\n\nThe lunch menu looked great and I would like to try some of the sandwiches.\n\nWe are here for a few more days and will eat here on Monday.\n\nIf you are staying at a hotel in this area be sure to give it a try for a good quick breakfast or lunch.  You won't be disappointed.",
      'useful': 1,
      'user_id': 'bCj_W-14a7SmGKyYufPzZg'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2016-02-19',
      'funny': 0,
      'review_id': 'x3Yy3gLYyTkd98_pH_ux8w',
      'stars': 4,
      'text': "Excellent sandwich luncheon spot!  The Albuquerque Turkey Panini is fantastic!  Green chili mayo?  Who knew?  I had to buy some to go!  Fresh vegetables and Boar's head meat. Warning if you order a cold sub sandwich the bread is a bit too much.  It's loaded with goodies but it's too much bread.",
      'useful': 0,
      'user_id': '02Z0sSGvQRm2rgQwUmNAJQ'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2016-07-07',
      'funny': 0,
      'review_id': 'gKMHY8pY2yLELjYqqa57ng',
      'stars': 4,
      'text': 'Great breakfast panini for $5. More affordable than a breakfast or brunch chain for sure. Nothing for breakfast is over $5 which makes this a spot I will recommend and more than likely come back to next trip. Iced coffee is only $1 till 11 and breakfast is served til 11.',
      'useful': 1,
      'user_id': 'k-p5RfLnAPEherauN4CxUQ'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2013-09-25',
      'funny': 1,
      'review_id': 'F5Hj9ozgOJ6VILZGmNeY8g',
      'stars': 5,
      'text': "Amazing lunch.  Fresh Boars Head meats. Fresh bread (marble rye is awesome) and tr staff is great.  One of the best sandwiches in the Phoenix area.  Don't miss this place. You won't be disappointed!",
      'useful': 0,
      'user_id': 'cwAxa2L4T0t5Hu1zgb5YVQ'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2015-12-20',
      'funny': 0,
      'review_id': 'w4xgWD2tc9llQwqHl9C9kA',
      'stars': 5,
      'text': "Worth finding - a little awkward the first time, but I don't regret having to drive the block an extra time! The Italian sub was loaded with meat - just like back home (Boston) - salad was fresh, bread was fresh, meat was just YUMMY!! Nice surprise - we will all be back. BTW - lunch for 4 + drinks was $30.00 ... Great, clean, fresh and friendly!!",
      'useful': 0,
      'user_id': '24Azqoft4fexUdM9yQbqrg'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 1,
      'date': '2014-03-01',
      'funny': 1,
      'review_id': 'GXtY_8FkuWCVfWjjBuHw9g',
      'stars': 1,
      'text': "Food is good and location is convenient. Service is terrible - even the owner - the older woman acts like its an inconvenience to help people. We have had the opportunity to give her more business but decided against it because of the service. Don't waste your time.",
      'useful': 1,
      'user_id': 'Ru2NSw1CzoGAsXYvDGf0Uw'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2016-08-04',
      'funny': 0,
      'review_id': 'RH2Hzv7_erHGpDlXMvYWdw',
      'stars': 4,
      'text': "Restaurant/Ambiance: Small cafe tucked away in a corporate office park. Has a Modern but relaxed cafe vibe inside. Has about 12 or so tables inside with 3 tables outside. Typical clients are business execs and tradespeople on lunch break. No issue getting a table or parking space. Little difficult to find though.\n\nFood: Great selection of sandwiches. Quality meats and fresh veggies. I like the southwest chicken chipotle sandwich. Also excellent salad menu. They also have a great breakfast menu but I've never been there for breakfast.\n\nService: Fast service, able to move the lunch rush pretty well.\n\nOverall: Great little lunch cafe!",
      'useful': 1,
      'user_id': 'CQ7tEwDL_5jpEJzDKKYv4Q'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2014-11-20',
      'funny': 0,
      'review_id': 'X3J4JtADK48m9DXjuWZmBA',
      'stars': 5,
      'text': 'SImple to find and great custom made food. Sandwiches, Chips and Drink. No frills but delicious.',
      'useful': 1,
      'user_id': 'LGeBGv_vXRSBvIq2ypABxg'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2016-10-27',
      'funny': 0,
      'review_id': 'N0cm6cj-c3EWx5Y2FdkZNw',
      'stars': 5,
      'text': "First time for breakfast. Waiting for someone at sky harbor. Breakfast burrito outstanding. A meal in a tortilla. Very tasty and really inexpensive. I'll be back!!",
      'useful': 0,
      'user_id': 'tkpIV6DjjNKsbmSB2OkP_Q'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2016-03-26',
      'funny': 0,
      'review_id': 'U6an2qDggnvuWWFMo6V2CA',
      'stars': 5,
      'text': "Great little local place with tasty food.  Be diligent in finding it as it isn't obvious from the main street.  Minimal seating for eat-in so plan ahead.  Friendly staff, good prices.",
      'useful': 0,
      'user_id': 'xodu7YZipff5EW5QbpT0aQ'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2016-03-10',
      'funny': 0,
      'review_id': 'fdGbPDqSpnhjj1UdVyYdhg',
      'stars': 5,
      'text': "If you work in the area and have not been here, GO IMMEDIATELY!  I have worked across the street for 15 years and I am so thrilled to finally have good food in the area.  The Chicken Fiesta Wrap is my favorite (I CRAVE it) and the California Club is really good too (crispy bacon....yummmm). If you can get there before they run out of Tortilla Soup it is outstanding. The food is always consistent and they don't seem to cut corners. Reasonably priced for what you get and what's in the area. Fast service, wifi on site, nice staff...I seriously cannot say anything negative.  Thank you for saving our lunch dilemma in the area!",
      'useful': 0,
      'user_id': '9EBTKHoCH1sl3X0MX8txjg'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2016-06-08',
      'funny': 0,
      'review_id': 'CEyzTVZ2MD7mwhwxY5B6JA',
      'stars': 4,
      'text': "I picked this spot out because of Yelp. I was looking to walk from my hotel (Aloft) and it's just down the street. Pay attentuon to the signs. It's in the back of a building that blends in with the rest.\nI had a Cobb salad and it was fresh and ready in about 3 minutes. Seemed like good friendly sevice. Portion was perfect and prices were reasonable.",
      'useful': 0,
      'user_id': 'QT76PggHNvPtzgOJ-zns6w'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2015-05-21',
      'funny': 0,
      'review_id': '2-g2Fhn2qAAYI0aaVPJKUg',
      'stars': 5,
      'text': 'Awesome find! Stopped in for breakfast on the run and will probably come back for lunch. The guys are super nice... Coffee is amazing!',
      'useful': 0,
      'user_id': 'QZ0vkltdo_JYmh65SoFqiA'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2015-07-31',
      'funny': 0,
      'review_id': 'A6vxSSNe2x2c_kdHtvF7hA',
      'stars': 3,
      'text': 'Great little sandwich shop tucked into the office plaza (Follow the sandwich board signs to find it). I had the lunch roast beef special and it was really good. The red potato salad seemed to be missing something for my palate, but over all it was really good! Meat was tender, taste was good and no overpowering sauces. \n\nIts a bit cramped in there so a togo order is a good idea, but its close to my other office so it makes for a perfect stop. Service was fast considering the crowd at lunch and the patrons and employees seemed to know one another.\n\nIts a nice place if you are in the area, stop by and have lunch!',
      'useful': 0,
      'user_id': 'sOCuf-jZo43DUzWTDKiR9w'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2017-05-14',
      'funny': 0,
      'review_id': 'N75A6yCqjZk0hrthXcSbNg',
      'stars': 4,
      'text': "It is hard to locate but just look for a small sign stating there are just hiding in the back! (: The set up is nice with an option to eat indoors/outdoors although I wouldn't recommend bringing large parties. Customer service was good and the food was delicious! I had to go with the BLT which, they did not shy away with the bacon. I wouldn't mind visiting back for lunch again.",
      'useful': 0,
      'user_id': 'aemN874BzsJo6NBEIbygfA'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2017-01-10',
      'funny': 0,
      'review_id': 'XcARF3jBueZcmXNyfr6gNQ',
      'stars': 2,
      'text': 'I feel like this place has gotten worse over time, and it feels as every time I try to go back, they fail to meet the mark somehow. I used to frequent it when my work first moved into this area, about 4 years ago, but I think today was the draw for me. They forgot to include dressing with my $9 salad. Also, without the dressing, the chicken has a strange taste, almost questionable, which makes me wonder how it has tasted in the past...at least I got extra spinach. The initial reason I started going here less often was a noticeable decrease in their salad sizes a few years back, hence I ask for extra spinach. One good thing they have going for themselves is that the veggies they use in their salads always taste fresh, for that reason I gave them two starts instead of one. However, I expect better for the price they ask for their food. Today was my last visit and I will make sure to spread the word at work.',
      'useful': 0,
      'user_id': 'ktdSILM4nVCPAzVVVQmB_w'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2015-11-12',
      'funny': 0,
      'review_id': 'j6vBl6VAbNWnOszna9C2bA',
      'stars': 5,
      'text': 'Great little lunch spot!  Fresh food.  Something for everybody.  Long line at lunch though.  We had our food within 15/20 minutes from the line at the door.',
      'useful': 0,
      'user_id': 'APnNt9CdoZ3n3EAIB2nMog'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2013-11-04',
      'funny': 0,
      'review_id': 'LIFGdvp1ftt7FSknVReeGg',
      'stars': 5,
      'text': "Was in Phoenix for business and discovered this friendly, wonderful  haunt  near the airport.  I had the best lunch ever -  a Southwestern chicken chipotle panini.  OMG!  The guy behind the counter turned out to be the owner, and himself grew up in a family owned East coast deli.  \n\nI had several expensive meals during my trip, but nothing was as tasty as my Lunch Lounge experience!  I made sure I had time for a breakfast stop on my way back to the airport., and was again wowed with a huge breakfast burrito, great latte and a smile - all for less than my daily NYC latte and muffin.  \n\nMy  client has never been to The Lunch Lounge, so next time I am in town, we have a lunch date at the best undiscovered charming cafe in Phoenix!  I can't wait to share the secret!",
      'useful': 0,
      'user_id': 'oBQL575GZLzElgQg7wrq9Q'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2015-07-18',
      'funny': 0,
      'review_id': 'n4z4QbIaMv9feBBja7X3Tg',
      'stars': 4,
      'text': 'Tasty and fast.  They primarily serve the corporate crowd so keep that in mind if you visit during prime lunch hours.',
      'useful': 0,
      'user_id': 'zTK1nPD2Hpa-ksSXsE-JzQ'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 1,
      'date': '2013-06-20',
      'funny': 0,
      'review_id': 'eizIAeDy6K9bVFqjQ_j9mg',
      'stars': 2,
      'text': 'I visited this restaurant because I got a groupon now deal. When I walked In and the older woman standing there took about 5 minutes to acknowledge that I was standing there at the counter waiting to be served..\n\nFinally she said "whenever your ready..". I told her hi how are you, giving her a nice reminder that that is how your supposed to greet new customers in your establishment. That on its own turned me off because I hate giving my money away tobusinesses that don\'t appreciate me.  I don\'t feel like I am asking for much in exchange for coming back and supporting local businesses. \n\nAs I was standing waiting for my food I overheard the employees (including my o so pleasnt serve lady) complaining about the groupons that kept coming in.. while I was standing there! Rude! \n\nThe older man behind the counter making sandwiches was very nice. He is the only reason I am giving two stars. \n\nThe food on the other hand was decent. Too bad I won\'t ever becoming back just to be treated like I am bothering them.. next time I will just eat mcdonalds breakfast like I wanted anyways.',
      'useful': 1,
      'user_id': 'e0TeYGOQRM2PrYCAdho7lg'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 2,
      'date': '2012-10-21',
      'funny': 1,
      'review_id': 'nN70gmUQEXvIJZedynlbwg',
      'stars': 3,
      'text': "Tucked away in a very boring looking business park, you can sneak in for a great sandwich. Go around the corner and look for the Boar's Head sign. I had the Thanksgiving lunch special - turkey, stuffing, cranberry jam, and I held off on the cheese. Nicely toasted bread - all in all, a decent lunch!",
      'useful': 1,
      'user_id': 'vUUWdvgtD0EPqDOzTv6iag'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2017-06-26',
      'funny': 0,
      'review_id': 'SpJQn8ijiLFPs9n31XSJTg',
      'stars': 3,
      'text': "I had the Egg-Cellent Burrito. Its name is not representative of the taste. It indeed did have eggs, but not the greatest tasting. They pulled the burrito out of the microwave so I assume that's how they cook it. I'd recommend checking something else out next time.",
      'useful': 0,
      'user_id': '69nezrTL38skxMaA9CO5VA'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2014-10-21',
      'funny': 0,
      'review_id': 'hpX2k1MvzRAF8naLfaUYWA',
      'stars': 5,
      'text': 'I stopped in last Friday morning and thoroughly enjoyed the breakfast burrito. The service was fast, and not only was the breakfast burrito reasonably priced, but it was also enormous.',
      'useful': 0,
      'user_id': '_k3o_AKj-S9JD0q6AhSXdw'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2017-03-14',
      'funny': 0,
      'review_id': 'sQYnC14kF_Hkubr6x-zPvQ',
      'stars': 5,
      'text': 'Yum! Good food and they deliver on Amazon Restaurants! I will certainly be ordering again. I was particularly happy with the carb free option for breakfast. We had a happy breakfast crowd.',
      'useful': 0,
      'user_id': 'YcmNpPM0ag94g4T0zAtdcg'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 1,
      'date': '2015-09-30',
      'funny': 0,
      'review_id': 'C4J79LVbr991w48PzKy9yw',
      'stars': 5,
      'text': 'Awesome meal.  Very willing to make adjustments and recommendations. Would come back here in a heartbeat.',
      'useful': 0,
      'user_id': '0rFoL4NHtwN4qVNK8HhXWA'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2015-01-28',
      'funny': 0,
      'review_id': 'LRwvRtyuITMr8juNqBGe9Q',
      'stars': 5,
      'text': "This place is ridiculous! Wasn't expecting much from a deli sandwiched between office buildings, but wow! This place is just across the street from where I work and everyone is blown away by the food. The staff is super friendly and the food is really fresh. The owner took my order a few weeks back and recommended the Stromboli since I always ordered the Roast Beef Press (which is fantastic!). O. M. G... the Stromboli is the best sandwich I've ever had. Inside out sub roll, pesto, ham, pepperoni, salami, cheese, and cherry peppers - toasted on a panini press. Seriously? On my day off from work, I drove 30 minutes back there to take family there for lunch. If they served dinner, I'd have to think about moving closer.",
      'useful': 0,
      'user_id': 'U_FfJTKoLsutJf-r98H9EA'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2014-08-15',
      'funny': 1,
      'review_id': 'ETamajeJMUVK0kjqTlV8Ug',
      'stars': 4,
      'text': "Just paid my first visit to this place and I will definitely be back.  I tried the Southwest Chipotle Chicken Panini and my co-worker had the Hot Roast Beef Press.  Both were good choices, but seeing how happy she was with her choice I have to come back to try that one next.  Plenty of off-street parking.  I only wish this place was open on weekends (although given their location I can understand why they are not).  Maybe open a second location near the business park at University and Elwood (hint, hint)?  I'm pretty sure it would be a big hit.",
      'useful': 0,
      'user_id': 'LZdMLoFIpW8X0QdIZOVGbQ'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2016-02-15',
      'funny': 0,
      'review_id': 'zFXJ1g4xkD4-WLANb_7tkQ',
      'stars': 5,
      'text': "The Lunch Lounge is absolutely delicious and fresh! Everyone is always there with a smile on their faces serving up yummy sandwiches, wraps, and salads. I haven't had one thing on the menu that I have not enjoyed. It is the perfect mom n pop shop of a deli!",
      'useful': 0,
      'user_id': 'OW0PZxoGFGSOthAmN_cSUg'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2016-10-19',
      'funny': 0,
      'review_id': 'jeONDYVsH6xxKd0K18yrlw',
      'stars': 3,
      'text': 'I got the southwest chipotle chicken panini and it was good but not great. It also cost $7.99+ tax which is too expensive for a sandwich with no sides or drinks. I would come here again if a friend wanted to eat here but i wont come back again on my own accord.',
      'useful': 1,
      'user_id': '7aHlhpJFiAdFHVhs6OHP6w'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2016-10-04',
      'funny': 0,
      'review_id': 'S22_jgLFNA8AQyGM5SusMg',
      'stars': 4,
      'text': 'Really good sandwich. I love having my lunch here. I usually buy Veggie Sandwich and I ask them to grill it. Tastes really good and affordable too.',
      'useful': 0,
      'user_id': 'XQfRpQv5q0uiP95QzIynug'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2014-01-31',
      'funny': 1,
      'review_id': '43ngKD54LnCu_BUrNWy92w',
      'stars': 5,
      'text': "Hidden cafe with outside seating.  \n\nDon't let the name fool you...they also serve breakfast. \n\nGreat prices including my 1$ coffee!",
      'useful': 1,
      'user_id': '_JpgkISb5Bd2MyiN614QJw'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2016-02-05',
      'funny': 0,
      'review_id': 'ow9-GQeZy9f8PnOa4KSnng',
      'stars': 5,
      'text': "Needed to find a quick easily accessible place for my kids and myself to eat and this place was in walking distance right behind Radisson hotel where we're staying at. Was able to find something for all of us. Pb&j's for the kids and a hot pastrami for me. Both were delicious and the staff was friendly. Quick, easy and much healthier than the fast food chains. Would of course recommended and return to the lunch lounge. Might actually get me some coffee there tomorrow morning.",
      'useful': 0,
      'user_id': 'bBAwihLfU7N2ombsDY5A0A'},
     {'business_id': '_mqUzNXs_sJ1EJYgYZYszg',
      'cool': 0,
      'date': '2014-03-25',
      'funny': 0,
      'review_id': 'LV_chTALhXSNHwkpgCybbw',
      'stars': 4,
      'text': "This place is located in a prime spot for travelers as well as office workers. \nThe menu is a diverse and creative mix on standard lunch box fare. \nWhether it is breakfast or lunch you can't go wrong with trying this place for a simple, but gourmet, breakfast sandwich or a dressed up salad for lunch. \nI only gave them four stars for this type of venue because I felt they were a bit understaffed when I arrived around 10am. Don't hold this against them. It was a bit of a wait for a sandwich but the quality is well worth the wait. The dill on the breakfast sandwich was a nice touch. \nThe outdoor seating was nice on a cool morning.",
      'useful': 2,
      'user_id': 'BS7etqD8Y2_zgZdFQWMliA'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2016-09-13',
      'funny': 0,
      'review_id': 'QqXa7pnQYcyyUHI2uo1J9Q',
      'stars': 4,
      'text': "Much better! I frequent the Charlotte area for work. I previously tried the Golden Crust and honestly I was not impressed. I just went  back this past Saturday not for any cooked food but for prepackaged goods. I shared my dissatisfaction with the manager and she shared things have changed for the better.  A new cook has been hired and she offered samples of oxtail which was tender, well-seasoned and delicious. She also shared a hot flaky fried dumpling, YUM! I'm pleased that the management and cook has changed they are now representing Caribbean food well.",
      'useful': 0,
      'user_id': '9dWPoN02z3JBbmqjtz2_tg'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2017-06-22',
      'funny': 0,
      'review_id': '77vyq7BxQ9QBRdys1W_aDg',
      'stars': 4,
      'text': 'What I had was good .. jerk chicken combo with rice and cabbage and gravy on top , and a patty .. very tasty, will stop back 3.6 to 4',
      'useful': 0,
      'user_id': 'Cqn_766h_JpQoV6fwizJhg'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2011-06-25',
      'funny': 0,
      'review_id': 'kPp5xyH5QMr8wxYz6ylNFw',
      'stars': 4,
      'text': 'I was SO excited to see Golden Krust outside NY so I had to stop in while in the shopping center.  My curry goat was SO delicious and had all the right flavor and spice.  The flavors reminded me of the great food I used to get when I lived near a Jamaican area in Brooklyn!',
      'useful': 2,
      'user_id': 'DDFAI_WROA_J8FF8dWe_6Q'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2017-03-24',
      'funny': 0,
      'review_id': '8eLLSpaJnzNF-wckRiQnQQ',
      'stars': 2,
      'text': "Prepare to be flexible because a lot of things run out.  The food is ok but I tried it because of the lunch specials. I wanted Mac and cheese they'd ran out. Ok, sucked that up because I prob didn't need it anyway. The vegetables and rice to me was very bland. The staff is not friendly and customer service is poor.  Don't come here with a lot of expectations.",
      'useful': 0,
      'user_id': 'zSblIQYZhN0gauAhIx46Tg'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2015-01-03',
      'funny': 0,
      'review_id': 'bQH4t-oqo-5JUlTLq_pn1w',
      'stars': 3,
      'text': "I really enjoy Jamaican food and was excited to see a Golden Krust in Charlotte.  The food was average and I didn't get to try the items I really wanted or any patties. I came in at 11:15 and they told me that most of the menu wouldn't be out til noon.  \n\nI may come back to try the patties.",
      'useful': 2,
      'user_id': 'QQ7sxRNs0A4uFK6pQE4AYw'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2012-07-08',
      'funny': 0,
      'review_id': 'abzG_1BqpTq0sR2Ol099AA',
      'stars': 4,
      'text': "I've been going to Golden Krust since the franchise came to Charlotte. Food is always great. The prices are reasonable. Service is just OK. On busy nights like Friday and Saturday they have run out of certain items.  I don't know how authentic the food is but it is on par with other Caribbean spots in town. Over all, I do recommend Gold Krust to get your Caribbean fix.",
      'useful': 1,
      'user_id': '9khBWjeKA9rHjT9Wo63WrA'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2011-03-15',
      'funny': 0,
      'review_id': '75wHJ12WyXoxmpQ6fFFlQQ',
      'stars': 4,
      'text': "Golden Krust is a bakery first and their specialty is the patty, and other baked goods, they also serve Jamaican staples such as oxtails, fish, jerk chicken, and curry chicken with all the fixins(rice and peas, plantains, and veggies), Ting, and other Jamaican drinks. With that said, if you're looking for patties look no further because they're simply the best ever! Buttery, flaky, fat-laden pastry, filled with chicken, beef or shrimp! Yes, it is a chain and the patties are shipped in but they're baked in house daily and trust me they're addictive! \nNow, the food overall has been fresh and pretty solid, but over the years it has been a tad inconsistent, but everyone has a bad day so I haven't held that against them. I had my very first oxtail here several years ago and I've been ordering them ever since, but I must say my favorite is still the curry chicken. The oxtails and curry chicken are both seasoned just right with plenty of gravy.........it's all about the gravy. I know most people think of jerk chicken when coming to Jamaican restaurants but since I'm not a jerk type of chic, I can offer no insight on that entree:-) Golden Krust has been serving up solid Jamaican food in Charlotte for years and though there are other decent Caribbean spots in the city, I continue to return to the Krust.......",
      'useful': 4,
      'user_id': '9IRuYmy5YmhtNQ6ei1p-uQ'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2015-11-10',
      'funny': 0,
      'review_id': 'LydIJlouXND8lXDTzSBUyA',
      'stars': 4,
      'text': "No frills, just good Caribbean curry to go. Goat curry lunch special is under $8.00 with tax. And they have other specials daily while they last. This is a franchise so like all franchises there are certain guarantees in terms of food style and quality and ingredients, but there's also a sense of family here which doesn't exist in all offbeat chain restaurants.",
      'useful': 0,
      'user_id': 'Oau7UrS5KRIB6iYSSsuWtw'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 1,
      'date': '2010-08-15',
      'funny': 2,
      'review_id': 'XEdHAeMYhH5iF0kTA5Npuw',
      'stars': 2,
      'text': 'Golden Krust is decent Caribbean for fairly cheap prices.  If you don\'t have high expectations for service or want to be "wowed" with food, you won\'t be too disappointed.\n\nOn both occasions, I\'ve had the dinner combo ($8-$10).  The food itself is pretty good.  I had the Jerk Chicken and the Curry Goat, both of which were not too spicy. The plates are served with generous helpings of rice and red beans and another side, usually steamed cabbage or collards.  The meat portions was kinda slim.  But the service?  A total take-a-number-and-wait-for-me-to-move attitude pervades this place.  The line was long and the staff were slow to move.  Geez!\n\nMy buddy and I sat in the back dining hall, which they closed off with a wet floor sign as they were closing in an hour.  RUDE!  I had to walk around it to get a drink refill.  If you want cheap Caribbean food for takeout, this might be a OK place.  I\'m sure that there is a Caribbean joint with more personable service and exceptional food.',
      'useful': 5,
      'user_id': '7IZQ06zkYSCLQM4JU9kFvg'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2017-01-21',
      'funny': 0,
      'review_id': 'JeQ7nexlSkcuq35CBIiG1w',
      'stars': 5,
      'text': "I am so glad that they got one of these stores here in Charlotte North Carolina I have been trying to find it I found you found Golden Krust Bakery you must eat their the best you're making pulled ever wish they had more big location that is just a little too far for me to get but I get there when I can try it you will love it Golden Krust the best ever",
      'useful': 0,
      'user_id': 'NJ8i8HeGJUDFlzKj6S7Jow'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2017-07-13',
      'funny': 0,
      'review_id': 'SAzBU_Sx7Cfq7CGYptTMNw',
      'stars': 5,
      'text': "If I could give this place 6 stars or more, I DEFINITELY WOULD! I always recommend this place to everyone. Outside of the place doesn't due justice to what deliciousness is served inside!",
      'useful': 0,
      'user_id': 'HS3y3GltOH_pJ37-EJwOng'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2015-12-07',
      'funny': 0,
      'review_id': '_yBv4ZtH7ifEeudNmbFyPQ',
      'stars': 1,
      'text': "Food is just okay.. Their portion sizes for the amount you are paying is terrible.. You get more Styrofoam than food and I don't like paying $10 FOR Styrofoam. They need to give you a better value. If you are going to cut portions on the meats at least put a good portion of rice so you feel like you got something..  The ones in NEWYork are much better. imho",
      'useful': 0,
      'user_id': 'w3ReHv4j5f-gnzHIz4Krbg'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2016-06-01',
      'funny': 0,
      'review_id': 'ZnIYRO5zXuW8eE4Qiqwovg',
      'stars': 4,
      'text': "Pretty good Caribbean Food, Nice location, and decent service. They use to be a lot better years ago. It's a little overpriced for the portions you get, but overall it's ok and I'll probably return",
      'useful': 0,
      'user_id': '17TiV5o3EsIIbyFgSpBmXg'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2015-09-18',
      'funny': 0,
      'review_id': 'ifGCxe9H9GmiA9_Yh8-wQg',
      'stars': 2,
      'text': "Bland food. I don't even know how you can make jerk chicken bland but they managed. The customer service is mediocre. I just recommend going elsewhere.",
      'useful': 0,
      'user_id': 'XjuZm_TSZwlPq28jF8w0Rg'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2014-09-11',
      'funny': 0,
      'review_id': 'Ub7ZiK9pfK4CVM3JFSAOHw',
      'stars': 2,
      'text': 'I have been to this restaurant quite a few times and always received pretty decent service. The food is always tasty and never really found that service was an issue until my last visit.\nI had a Jamaican visitor and she want hard dough bread to go with our dinner. Now I could have gone a couple of different places but I chose the Bakery. I love this bread but I typically only eat it when in NY, Jamaica or Florida because I need my hard dough to be as fresh as possible. \nWhile the seemingly nice woman was ringing me up I politely asked "how old is the bread?" Now I know this is called a bakery but everyone knows their baked goods are shipped in. She responds by saying "the bread is fresh." I say your idea of fresh and mine maybe 2 different things. She became a maniac but in a calm way. She asks: do you want the bread or not? At the time I really just needed to get the bread and leave, dinner was waiting. But not before I explained that as a consumer and patron of this store I have a right to know how old something is in a darn bakery that doesn\'t bake anything.  \nAfter the fact, i was very upset because I could have gone somewhere else, and for her think that it was OK to respond in that manner bothered me. Why was she so combative in the first place? In the end, she told me their shipments come every Tuesday. So this would be the best day to visit if you need baked goods. I myself, will never spend another nickel there.',
      'useful': 3,
      'user_id': 'E1szdQHhFDwO115pGDh_Xg'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2012-09-23',
      'funny': 0,
      'review_id': 'RJOCu7omsWzeLGaw191eOQ',
      'stars': 1,
      'text': "This place is NOT Caribbean food! It's terrible. You must ask for a drink separate from the combo even though you've ordered a combo. They're not going to offer it to you either. The food tastes horrible. I wish I could sit outside and turn people around to save their souls. But unfortunately this is all I can do. Take heed and go elsewhere!!!!!",
      'useful': 1,
      'user_id': '-8mGZ-pJi-NcjZckuz1M7A'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2017-05-20',
      'funny': 0,
      'review_id': 'IDAs8s_uBf2Ag7JfCzKdSA',
      'stars': 1,
      'text': "The jerk sauce tastes like bbq sauce from a bottle mixed with ketchup! I should've known to keep walking when I didn't see any customers. Never again.",
      'useful': 0,
      'user_id': 'wsCRhiX1UpakSeDvo9DHQg'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2017-04-19',
      'funny': 0,
      'review_id': '8xeVKHqYWyL7kc7GW6ot1A',
      'stars': 4,
      'text': "Golden Krust really does live up to the name. I ordered Jerk Chicken with a shrimp Jamaican patty. The Jamaican patty had such a crispy, well put together crust. It was probably the best Jamaican patty I've ever had. If you're in the Charlotte area & you like Caribbean food, it's definitely a spot to look into.",
      'useful': 0,
      'user_id': 'El0_mZrcFQmyW7LNz41R5Q'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2014-10-10',
      'funny': 0,
      'review_id': '8qWThGPh_hFoUBLzonAKPA',
      'stars': 3,
      'text': "Love the veggie patties, when they are available...haven't really eaten anything else there.",
      'useful': 1,
      'user_id': 'FSJDnMPHNOpZrqR1wHp53A'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2015-08-20',
      'funny': 0,
      'review_id': 'ZoliATYd76iNDQM-R6mS1g',
      'stars': 4,
      'text': 'The food here is really good, if you are in the mood for Jamaican food this restaurant is great. The sell cakes and patties which are very good. They get a four star because their portions could be better for the price.',
      'useful': 3,
      'user_id': 'bm2tUuhog7tMu7EqfN7BNw'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 1,
      'date': '2016-07-26',
      'funny': 2,
      'review_id': '7PUEJnGT_R9KjUC43uAckQ',
      'stars': 2,
      'text': "It has been awhile since I've been to Golden Krust and I remembered this place being pretty good but things have sadly changed. Walking into the restaurant it was extremely humid and hot so I assume the air condition was not working. My family and I proceeded to order since we were starving. We ordered the Curry chicken with rice and beans and cabbage, Oxtails with rice and beans and cabbage and Jerk Chicken with rice and beans and cabbage. We also added 2 spicy beef pattys, they did not have any more regular beef pattys, I was disappointed. With it being so hot in the restaurant I wanted a Large ice water BUT their fountain drink machine was not working and they did not have any bottle waters. The only drinks that were available were bottled juices that I was not interested in. So I walked over to Shane's to buy a large drink. After returning with my large drink in hand my family and I ate dinner. The curry chicken was my favorite out of all the entrees we ordered but I was disappointed in my plantains were very soggy and not fried long enough. The oxtails disappointed me the most. They were not seasoned and did not really have a taste. Overall my experience here was just ok, I hope Golden Krust improves the obvious issues and bring back the original Golden Krust.",
      'useful': 1,
      'user_id': 'uaeU6FBaY5by7V1Cj_2RpQ'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 2,
      'date': '2015-12-22',
      'funny': 3,
      'review_id': 'mLpQ8o2UVydo5SIYPicYkg',
      'stars': 3,
      'text': "The roti here is an improvement on the other Jamaican restaurants I've eaten at in Charlotte. I'm from Brooklyn, so I'm used to having maximum food for a great price. Not so in Charlotte, for the minimal amount of oxtail, the price could be a bit lower. There is no reason to pay a high price for lots of rice and cabbage but a scoop of oxtail. \n\nI'm definitely not expecting oxtails to dominate the plate, but the serving my daughter had was enough to convince me I won't be getting anything other than beef patties here. That is, when they actually have any available.",
      'useful': 6,
      'user_id': '7U3H1iJZ04FbadJtt2Npjg'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2015-07-22',
      'funny': 0,
      'review_id': 'qhN6T0i8Bcz77Gw0lZXrsQ',
      'stars': 3,
      'text': "Right off the bat, don't go to this place expecting warm and cheery service. Just focus on the food to keep your satisfaction levels in check.  I met some friends here after work based on their recommendation. When we got there, it was empty (this was around 5:30 on a Monday evening) but definitely picked up as time passed.\n\nThis was my first time eating Jamaican food (I know!) and I wasn't all that hungry so I hemmed and hawed over the menu for a little bit. I went with the spicy chicken Jamaican patty and a pineapple-flavored Jamaican soda. There's a case of bottled Jamaican sodas or cans of American sodas (coca-cola, pepsi, or mountain dew). FYI - they don't always have all of the patties listed on the menu available to order so just ask before you get too committed to a particular flavor. The woman went into the back and came right back out with my patty. Nice and warm, ready to eat.\n\nI went into the back seating area while my friends finished placing their orders. The patty was a fluorescent yellow color which threw me off a little bit but when I bit into it, my fears were long forgotten. The patty crust was flaky and had a little bit of a sweet flavor that balanced the mild heat from the meat inside. The Jamaican soda was very sweet, which should come as no surprise since its second ingredient is high fructose corn syrup. It's definitely not anything I would probably have again but it was good enough to wash down my patty. I got to taste a bit of my friends' meals and I really, really enjoyed the jerk chicken. The chicken was juicy and extremely flavorful. It's definitely some I would order on my next visit. Also, the coco bread was/is a winner.\n\nOverall, a good spot to get your introduction to Jamaican food and/or to indulge your already long-held affection.",
      'useful': 1,
      'user_id': 'SBi0TYUSIrct8WSTyK6DFg'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 1,
      'date': '2013-05-02',
      'funny': 0,
      'review_id': 'tfNlQpBAJgZepWO6IXhIxg',
      'stars': 3,
      'text': "This is a nice spot for some of your Caribbean needs.  You should definitely check it out at least once if you find yourself in the University Area.\n\nIt's small and tucked away in the shopping center, but you visibly see the sign for the place.  The menu isn't that big, so you for sure will not be overwhelmed with too many choices.  I have gone here twice now and definitely recommend the Goat Curry or the Oxtail meal.  I don't know about you, but when I ordered these, I also asked for extra gravy.  It was great to have the extra gravy with the rice and cabbage. Note that when you order the meal, it doesn't come with a drink though.  Kinda a bummer on that part. \n\nThe place is alright, pretty small.  They have tables near the register, and also near the back.  I haven't been towards the back to see how many tables there are as I have only gotten take out.  The prices are a little pricey to me, but the food was worth it.",
      'useful': 1,
      'user_id': 'hhQUVO2AqBdSeQcPo19Bsw'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2015-05-15',
      'funny': 0,
      'review_id': 'WUJLbMGtLXenOauafm7BOw',
      'stars': 2,
      'text': 'Was given three prices for The ox tail combination additionally they did not have any fountain drinks or ice had to pay an extra dollar for a canna Coke in addition to the meal that should have included a drink, according to the menu',
      'useful': 0,
      'user_id': 'yIYXU7Upih6ihm9usrBCJw'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2015-01-31',
      'funny': 1,
      'review_id': '1ltFxd_dxJ55ZiEGtdhimw',
      'stars': 3,
      'text': 'Food is good, price is good I can\'t take that away from them to be fair. However I ordered a combo that included a drink and Jamaican bread(forgive my ignorance). I paid with my debit card then when I got my food that\'s when the woman informed me their drink machine was down, and they were out of the Jamaican bread. Which she then immediately followed up with " no refunds".',
      'useful': 3,
      'user_id': 'TqoxE8B0r4CUWpGYAHXoOg'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2016-11-07',
      'funny': 0,
      'review_id': 'C0QZoRpYdaWsU4ne5TUihQ',
      'stars': 5,
      'text': "Delicious food!!! I have been coming to this Caribbean Restaurant since moving to Charlotte 6 years ago. I must admit there was a decline in the food quality about a year ago due to a change in chefs. I did stop coming during that time. I stopped in about a week ago to give it another try. I spoke to the owner and the pleasant staff and found out they have a new chef which they gave rave reviews. I was pleasantly surprised that the were indeed correct. I food was scrumptious  just as I remembered!! Keep up the good work!! I'll be back and I'm putting the word out.",
      'useful': 0,
      'user_id': '3oXOq3p_XzMvmL6G2ALYMw'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2013-05-23',
      'funny': 0,
      'review_id': 'f6snZFRJ2SadUP9Km7nnOw',
      'stars': 4,
      'text': 'FOOD IS GREAT.  The service is average.  The prices are cheap so you tend to get idiots in there who have no manners.  I recommend, taking the food to go or sitting in the back.',
      'useful': 0,
      'user_id': 'iN5kL0PrH0lb_69fZ6difQ'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2017-06-09',
      'funny': 0,
      'review_id': '-hAtmQFWrDePTOioOGAW3w',
      'stars': 2,
      'text': 'This food was not my favorite I spent about $12 and barely ate any of it. I went on my lunch and was driving and eating very disappointed. I may try this place one more time in hopes of better flavor.',
      'useful': 0,
      'user_id': 'sbtSsIQKYRPmS4KJ6jdYnw'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2008-12-31',
      'funny': 1,
      'review_id': 'TK-4DQpBNzvtfGSPskSQQA',
      'stars': 4,
      'text': 'Tis De Place Mon!  First Off me Being from one of the best cities in the world. NYC i know good west indian food and this hit the nail on the head! Great to the point service, with hardy plates not for the quickly filled stomachs here! Try the cuury goat! and the beef patties!! and a DG Soda!   best in carribean food in charlotte, nc! This is truly the mirror image of true carribeen food! anything else is just a watered down xerox copy.  then even have the spice bread and pastries!  Best for lunch $5-6, and dinner $8-10.',
      'useful': 3,
      'user_id': 'ifZh_oEWxd05Ci_RQVoL9Q'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2016-03-12',
      'funny': 0,
      'review_id': 'df6kl4vRmARX0ydiQt4SeQ',
      'stars': 2,
      'text': 'Ive been going here for yrs ...as a NYer I\'m familiar with the chain. The foods good but I\'m never going back. They\'ve lost me and numerous family members. I wonder if they changed management or something because their portion sizing are horrendous! Theyve scaled back so much and still charge the same thing! I\'m not paying for Styrofoam! Then if you ask for more rice you get told "no!"  What kind of service is that?! I got a whole lot of gravy though! Ugh theyre rude too! Now the food Id rate a 4...maybe 3.5. But the service and portions a 1. Shame I stopped getting plates and stuck to beef patties but then they stopped serving cheese beef and never bothered to have them again or remove it from the menu. Ill be going to the other Caribbean restaurant up the street from now on.',
      'useful': 0,
      'user_id': 'b6icLzaIuloPmFvWuuUIsw'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 6,
      'date': '2016-06-17',
      'funny': 3,
      'review_id': '3IZqYKXIG7qmPr1wgxeKKg',
      'stars': 4,
      'text': "Love this place for spicy patties! The patties are worthy of 6 stars, the only reason I gave the 4 stars is because every time I visit the music is blaring and the air conditioning seems to be turned off or not working. They have a dining area, but I can't imagine eating here. And be prepared for a wait, if you have been to the Islands then you know the slower pace you have to get used to. \n\nI have tried several of their curry dishes, the goat was my favorite. And if you're looking for a cold bottle of Ting or a piece of rock cake, this is your place. Did I mention the spicy patties? Wow, they are delicious. Growing up in Miami, we had a favorite place to go for patties. Here in Charlotte, this is our favorite place. I usually pick up 6, and I'm sure they would hold up fine for a couple of days in the fridge. They never last that long.",
      'useful': 6,
      'user_id': 'uT-yzDkY79szXYNWk7zDzQ'},
     {'business_id': 'wtazPNyIbsRMHmrpEYbqZA',
      'cool': 0,
      'date': '2017-03-18',
      'funny': 0,
      'review_id': 'U38h1YLDdZZCEr5MbKLujw',
      'stars': 5,
      'text': "If I could give this place 10 stars I would !!! I loved the food and the service was VERY FAST ! I loved everything and the small plate filled me up I couldn't even finish. The prices are GREAT and it's seasoned so amazing! I felt like I was on the islands. I would go back and take everyone in my family if I could I will definitely be returning my next trip to North Carolina. I loved it !",
      'useful': 0,
      'user_id': '30XooIBLkNu5kuLyplk99Q'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2017-04-01',
      'funny': 0,
      'review_id': 'BLIJFaJZ-_fOcBs16fL_6g',
      'stars': 5,
      'text': 'Loved, Loved, Loved. It is a simple place, but you dont come to this place for the decor. You come for the damn good food. \n\nPlease note* I am just a customer. There is no dicount for posting nor any compensation for this rating.',
      'useful': 0,
      'user_id': 'jgzD7eBwZrasqy6wUy122w'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2016-11-05',
      'funny': 0,
      'review_id': 'VuKbGklNbOESJSx76_EjyA',
      'stars': 5,
      'text': 'Is a small restaurant food is good! Also the owners are very friendly they make sure u are satisfied!',
      'useful': 0,
      'user_id': '2v_meK453YAWXz4NjJ9abA'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2017-06-01',
      'funny': 0,
      'review_id': 'HBaAmcS9zp5rY1qiMuWygA',
      'stars': 5,
      'text': 'Best Mexican restaurant in Vegas. Meat is super soft and very tasty. Fried fish is delicious. Salsa is so yummy! Very friendly service.',
      'useful': 1,
      'user_id': 'cdFWtOgA1PAkNYkiwzUJbQ'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2017-07-16',
      'funny': 0,
      'review_id': '41ORR2OPi2CP0FAoiWjwWA',
      'stars': 5,
      'text': "I  would've like to take a before picture when it looked all nice and pretty, but it was so delicious I just couldn't wait. I had the chicken chimichanga and my friend had of the chicken burrito supreme. I will never visit Robertos, Pepe's tacos, etc. again. #obsessed",
      'useful': 1,
      'user_id': '60ENRcuFKNf9oeA2zWROFQ'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 67,
      'date': '2017-06-20',
      'funny': 42,
      'review_id': 'Oxz26pqpIb7dDVeuUzNZlg',
      'stars': 4,
      'text': "We had originally planned on eating at the Sinaloa food truck across the street, but they had a really long line, so we decided to give Mi Pueblo Taco Shop a try. Walking inside I was greeted by a sweet young girl who also took my order and she was nice about going over the menu and what the specials included. Inside was clean and decorated nicer than I expected. There was also a good amount of customers eating inside which I took as a good sign. \n\nI noticed they had different daily specials on the windows out front and they are totally worth checking out. I was here on a Friday and the special that day is 2 enchiladas, beans, rice and a drink all for $5.95 before tax. Great deal! The rice was lacking a bit on flavor and while the beans could have used more cheese, they did taste good and I liked the few tortilla chips on top of them. The enchiladas had a decent amount of cheese inside and the sauce tasted good. I'd for sure come back for this special again.\n\nAl Pastor Taco ($1.75) - The tacos come meat, onion and cilantro. There's a salsa bar with veggies and containers. I grabbed some of the red and green sauce and favored the green. My taco was filled nicely and had great flavor. \n\nCarne Asada Burrito ($5.99) - Here's where things got odd for me. Hubby and I intended to share this and he always prefers just meat, cheese and sour cream in his burritos. The girl who took our order said he couldn't have it that way and that the burrito had to have beans and rice inside. If the burritos aren't pre-made this shouldn't be a hard request at all. I saw a different girl putting meat on the grill and then put the burrito together, so I can't figure out why the burrito couldn't be customized to have the rice and beans left off. The burrito was a nice size for the price and did taste good, but it would have been better enjoyable if it could have been made the way preferred. \n\nGreat weekly deals and decent Mexican food. I'd come back.",
      'useful': 66,
      'user_id': 'Fv0e9RIV9jw5TX3ctA1WbA'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 1,
      'date': '2016-06-12',
      'funny': 0,
      'review_id': 'JGIVj26nGtLgzsL4zeaP2Q',
      'stars': 5,
      'text': "One of the few times I have given a business five stars. This restaurant has very good food everything on the menu is fresh and very very tasty. I've been to this restaurant probably 10 to 15 times and have never had a bad meal. Is the owner and staff of this restaurant are very nice and courteous and provide a good service and excellent food. I highly recommend this place for anything Mexican on their menu you will not be disappointed.",
      'useful': 1,
      'user_id': 'UKFgwgNM1a7Gmr0UIGPzpw'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 2,
      'date': '2015-08-01',
      'funny': 1,
      'review_id': 'w3ezGmzYbucBCNB66ilE4Q',
      'stars': 4,
      'text': "Las Vegas does not have a lot of good Mexican food unfortunately. So, when we do find one, it's always pretty exciting. This place was good. Reminded me a little of Texas, in terms of quality. \nEnchilada was very tasty, great sauce. Rice and beans were both delicious. Chile Relleno needed a tiny bit more flavor, but had plenty of cheese. Great little salsa bar. Had a bite of gf's Huevos Rancheros and we agreed that they were pretty awesome. We'll be back soon.",
      'useful': 4,
      'user_id': 'LnGDsxHFPhoCKT8sFGaViA'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2017-03-11',
      'funny': 0,
      'review_id': 'wM9a-pB0MSnyIJ4eMmXBUw',
      'stars': 5,
      'text': "Trying to curb my taco cravings, checked yelp for a taco spot not too far from the house, when I came across the review for Mi Pueblo Taco Shop and they have a perfect 5star rating. I haven't got my food yet, but the smell coming out the kitchen is amazing. It's a Friday night in Vegas, at a hole in the wall taco shop. The single file line is out the door.....Now that's a recipe for GREAT Mexican food",
      'useful': 0,
      'user_id': 'E1b2_YD1uwufHmCsrKteCw'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2015-07-22',
      'funny': 0,
      'review_id': 'fEpol5iWEiqKdbrIuHOc3A',
      'stars': 5,
      'text': 'My daughter and I stopped in yesterday and truly enjoyed our meal. So much so we added an order to go. Just might have to give up my usual taco spot!!!! Food was great. My only critic no chips and salsa before the meal and only gives you a couple chips stuck in the great beans!!! Also might look into making up paper menus. Nothing to take home to read. The service impeccable. Will be back for sure.',
      'useful': 1,
      'user_id': 'JoATwObKNywEmAduyKdm6w'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 1,
      'date': '2017-05-24',
      'funny': 0,
      'review_id': 'rqCifVSFO9vNV9S97NoQng',
      'stars': 5,
      'text': "The carnitas sopés are the best! Carne Asada tacos are in second place... \nI eat there at least twice a week, and it's fantastic every time!",
      'useful': 1,
      'user_id': 'BYL1E668zAEJNimiFPSA-Q'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 1,
      'date': '2014-09-05',
      'funny': 0,
      'review_id': 'XgoMPW26QuJ6duS08QXc-g',
      'stars': 5,
      'text': "Talk about authentic Mexican food!! there i was in search of a new place to eat close to work. I was craving authentic tacos, as i reminisced on my favorite taco shop in san francisco.\n      First stop was Cardenas. It was closer to me but the lines were ridiculous and the environment chaotic. Needless to say, i left in search of greener pastures. Second stop, the las vegas chain Rigos tacos. The place gave me a creepy vibe plus their soda machine was not working. I'm all for hole in the wall places but this place was too much. \n    As I drove aimlessly i came across this place, Mi pueblo Taco shop. Boy was I pleasantly surprised!! my  long search of a mom & pop taco shop proved fruitful. The lady at the counter and the cook turned out to be husband and wife, and the owners. They were both extremely friendly  and the food was delicious!! they had weird pricing quirks but hopefully they figure it out. overall, 5 stars and will definitely come back again.",
      'useful': 2,
      'user_id': 'OVFCRSZ_yFEc1rAlye_Z5g'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2017-05-20',
      'funny': 0,
      'review_id': 'AL3myD9GD5Zj7YuVLq0N1w',
      'stars': 5,
      'text': 'I stop by here on my way to ed fountain park when my son has soccer games.  I usually get tacos and tamales for the team and they love them more than pepes.\nDefinitely a go to place for great Mexican food...cant wait to try other items.',
      'useful': 0,
      'user_id': 'Jl-0N4zQ6VCbvk7DFpNfJg'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2017-05-12',
      'funny': 0,
      'review_id': '6EYXqTZHxfoZgA4gIrDVrQ',
      'stars': 4,
      'text': 'Very clean and friendly service. I had the Carne asada torta and my friend had tacos overall fairly good.  For a quick bite not bad.',
      'useful': 0,
      'user_id': 'C6ezDVKNyQz-aKr4OOjC4w'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2016-12-11',
      'funny': 0,
      'review_id': '0dP8Woa5xr1eNJ4eJlojlw',
      'stars': 5,
      'text': 'I was surprised it was this good it was like a home cooked meal... I purchased one posole, 3 carne asada tacos, 5 fried fish tacos, a combination of carnitas, 1 chicken nugget plate, 2 orchatas and my total was $51 not bad at all.',
      'useful': 0,
      'user_id': '59-R2EWXP6EUelGzuM1UsQ'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2017-07-15',
      'funny': 1,
      'review_id': 'sG2U7rH5NSTYR3-DumVT7w',
      'stars': 2,
      'text': "TLDR: Lunch Specials; salsa bar available; excellent service; subpar food; convenience charge for card payment\n\nI was really craving excellent chips and salsa so I usually have a go-to spot for Mexican fare. Learning that they were closed, I found something in the area with excellent reviews and was happy to find this location with a Yelp check-in.  I skimmed the reviews and saw several 4/5 star reviews with a couple of 2 star reviews so I decided to give this place a shot. \n\nService was excellent and I was impressed with the politeness and grammar used in both English and Spanish upon walking in and ordering. It reflected that the business cares about the customer and how they are perceived when spoken to by the various clientele. \n\nI immediately placed an order for the chips and beans and was happy to see breakfast items on a menu. My foodie friend ordered a pastor burrito and I ordered chilaquiles with green sauce. Unfortunately, this is where I should've listened to the 2-star reviews. \n\nWhile I believe that the food is made fresh in-house, the quality lacks and pales in comparison to the service. The beans lacked flavor, excluding the cotija cheese and the chips were stale. This is probably a storage issue. The salsa were flavorful, but lacked spice. Interestingly enough, the green salsa was the spiciest. \n\nMy friends pastor burrito was just ok. Not fantastic, not terrible. Just ok. It probably is similar to Roberto's, my terrible baseline for Mexican food flavor and service. My chilaquiles were tragic. The rice was pretty bad and the green sauce was pretty boring. It's sad to say that Del Taco has better rice, beans, and green sauce. Needless to say, I didn't enjoy the dish and I won't return anytime soon.",
      'useful': 2,
      'user_id': 'cQzo2WX1TyPEmS5jPzOUoA'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2017-03-26',
      'funny': 0,
      'review_id': 'Eb3mICEEu0E8obitUjI8sA',
      'stars': 5,
      'text': 'Their food is delicious!!! I got Huevos Rancheros and my bf got menudo. He loved it! The lady at the register was friendly and super nice! The coffee is really good too!!',
      'useful': 0,
      'user_id': 'ozS37Fa6pFDEEMuqU3qmvg'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2017-06-12',
      'funny': 0,
      'review_id': 'jM1RJRrF4OUU0OyZSTgFeg',
      'stars': 5,
      'text': 'Great tacos de carne asada.  I really liked the salsa.  I added rice and bean and I am glad I did.  Delicious and very affordable.',
      'useful': 0,
      'user_id': 'hcK85MiLdfYYrIOuzKtJQQ'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2015-02-09',
      'funny': 0,
      'review_id': 'LbwHTp9GEKk5xoQzIjITeg',
      'stars': 5,
      'text': "Wonderful divey taco shop with awesome Mexican food. Highly recommend the tacos dorados, cheese enchiladas and chips with the salsa bar. The flavors are amazing! They must do most of their business as take out since it is so small, but the eating area is clean and nice. Don't be swayed by the strip mall. Definitely worth a stop.",
      'useful': 3,
      'user_id': 'Rh2YtAmvRxDonUqZlxJDIw'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 3,
      'date': '2016-11-03',
      'funny': 0,
      'review_id': 'PM5aFLiwbKWDm_AtC71-3Q',
      'stars': 5,
      'text': "I walked in my scrubs and the owner asked me how I found out about his place.  I told him Yelp and that I was in the area for a school assignment and lived across town.  He never heard of Yelp and didnt know that his place has a 5 star rating.  He was wearing an SD cap and I found out he lived in San Diego.  If you ever had authentic San Diego Mexican food you know this place is legit, because it reminded me of that.  I found this to be a unique experience with genuinely nice restaurant owners not catering to Yelp reviewers.  Hopefully that won't change after my random stop there.\n\nI had a carnitas burrito, cachete taco and carne asada taco, and a horchata (no not Smart Chata).  Overall the carne asada was the most flavorful with a nice charred taste.  The cachete was tasty.  My carnitas burrito was good too, similar to the SD burritos I miss.  Tortilla, meat, rice, beans, and since I'm originally from San Francisco I added some sour cream with guacamole.  The salsa was also fresh and tasty.",
      'useful': 3,
      'user_id': 'd9y5MjVOgxcP4Zc7qjowdA'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2015-02-10',
      'funny': 0,
      'review_id': 'SxPbDeghypuf3ESF3CzXQw',
      'stars': 5,
      'text': 'THE MOST AUTHENTIC MEXICAN FOOD IN TOWN. \nDo not sleep on this spot, if you are ever in Vegas definitely stop by... I cant think of my favorite dish because you cannot go wrong with anything on the menu; The menudo The cocteles, The enchiladas. Mm everything is bomb! and always great customer service and very clean nice little shop. Love it.',
      'useful': 0,
      'user_id': '5IgLVsxowG4gcIJK8ovSzQ'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2015-10-20',
      'funny': 0,
      'review_id': 'gonKSjSZrYkOzhdHBaCDeg',
      'stars': 5,
      'text': "Great place if you want some traditional fresh Mexican food ! The over ceviche which is one of my favorite dishes, to to be honest it's one of the best ones I had !",
      'useful': 0,
      'user_id': 'vb9UuBS_qwUR-EGsJKlSFw'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2017-06-24',
      'funny': 0,
      'review_id': '_dROxY7jxG8hH4eGOi6-nA',
      'stars': 5,
      'text': 'They have tamarindo! My girlfriend and I have been looking for weeks and finally found our new go to spot!  Thanks!',
      'useful': 1,
      'user_id': 'IeVYKQZObeFnKTa04oB5Gg'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 2,
      'date': '2016-07-05',
      'funny': 1,
      'review_id': '6SaVHkF1qwexY-nbGaxlPg',
      'stars': 5,
      'text': 'Craving for authentic Mexican Food and feeling adventurous and out of your comfort zone. Gotta try this humble and family operated place. Everything I had was simply delicious and would go out of my way to venture in this neighborhood for Tacos and Mexican food\n\nThe cashier/server was very friendly and asked if it was my first time here. She answered my questions and made me feel welcomed. I originally was going to order Tacos only but the enchilada combo grab my attention and got the combo with a single order of Carne Asada Taco on the side because I had to get a taco too. \n\nWhen the order finally arrived at my table. I got the three different salsas that was available at the salsa bar. From the first bite of my taco and enchiladas 1-beef & 2-chicken combination along with the fresh salsas. Every bite was amazingly flavorful and authentic. The whole experience was great and felt like I was taking a short vacation in Mexico!',
      'useful': 4,
      'user_id': '6pzA8EDhHgW3A5-rBObhBA'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2014-11-20',
      'funny': 0,
      'review_id': 'FiNxxZQA_b6HEuTumHaEag',
      'stars': 5,
      'text': "Man this place is bomb dot com - \nLet me tell you when it comes to Mexican good I'm picky  there's so many spots here in vegas so when something really catches your taste buds then you know lol",
      'useful': 2,
      'user_id': 'Nym1l6I0gMFrr9WCAcMO0w'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 1,
      'date': '2017-06-16',
      'funny': 0,
      'review_id': '7NzfZCe-pdsY0oMJr17MZw',
      'stars': 5,
      'text': "We were in Vegas for a few days and after eating at a number of the expensive resort offerings we found this little diamond in the rough. This place ROCKS! Totally authentic mexican food. The tacos and flautas are legit and hit the spot dead on. We are mexican and we know our cuisine and you can't go wrong eating here. Just stop reading this review and try it. Like now.  Go!",
      'useful': 1,
      'user_id': '2qLnlH24c6wdPn5XvG6DLg'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2017-05-05',
      'funny': 0,
      'review_id': 'D26ZjupzxrjQHEt-wtCN7w',
      'stars': 5,
      'text': "Yes its definitely a five star hidden gem. A great pit stop on the way to Moab. Everything was fresh right down to the rice. A nice older couple seem to run the restaurant. The prices are fair, it's probably not not in the best part of N. Vegas but worth trying out, we'll be back.",
      'useful': 0,
      'user_id': 'WRWvu_McOkXguooZftZJmw'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2017-05-05',
      'funny': 0,
      'review_id': 'Y5kb-t6_mRVHnr2Yv778ig',
      'stars': 2,
      'text': "I ordered carne asada, fish, & chicken tacos. Tacos are really small & over cooked. Was really disappointed, expected a lot based on the reviews I read. You're better off going to a taco truck. I really hate how over priced tacos are now. Wouldn't mind paying $2-3, but at least give me what I'm paying for.",
      'useful': 0,
      'user_id': 'CL2Zdl7Uynf4h7QwxMNkuQ'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2017-07-16',
      'funny': 0,
      'review_id': 'onvMwlNZZjZ4mIz_2z7kvA',
      'stars': 5,
      'text': "MI Pueblo is the BEST TACO SHOP IN VEGAS.\nThe owners are so sweet and kind. Their food is absolutely delicious. I have had every single meal available here, and I NEVER once had anything that was bland or stale. They Cook everything fresh, despite what others think. Even though I'm guessing those people have never seen the cooks in action. This place is 1 IN A MILLION! I've been going since they've opened in 2008. I have never regretted going there. I ABSOLUTELY LOVE THIS PLACE AND THE OWNERS! It's such a shame when people check in on here to get free food and THEN POST NEGATIVE REVIEWS ABOUT THE FOOD THAT THEY ACTUALLY HAVE TO PAY FOR LOL. \nYOU WONT FIND MANY CLOSE TO HOME, DOWN TO EARTH AND DELICIOUS PLACES LIKE THIS.",
      'useful': 1,
      'user_id': 'Wt5mt4Byw5Mn3U41jUnCAQ'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2017-01-03',
      'funny': 0,
      'review_id': 'EgQjBwQ1cA0ymPZ-uYgGHw',
      'stars': 5,
      'text': "I haven't eaten yet but if it tastes half as good as it smells I'll be happy. The staff was super friendly thumbs up!!",
      'useful': 0,
      'user_id': 'OMnGx8677VNH3-n3FGujYw'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2015-05-12',
      'funny': 0,
      'review_id': 's2GQlpCicvkBv0vzDj235Q',
      'stars': 5,
      'text': "By far the best burrito and salsa I have ever had. This place is way across town from where I live but i go there just to get my burritos, yes if I'm gonna go across town I better get two burritos to go.",
      'useful': 0,
      'user_id': 'wwucRFQsufzBZBxsNBDkwQ'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2016-07-22',
      'funny': 0,
      'review_id': 'eDQEgk6uoZEATcDzwAi10w',
      'stars': 5,
      'text': "Legit food!! I had the chilaquiles with red sauce comes with eggs,  beans and the best rice I've had in a long time.. Definitely recommend this spot. They know how it's done",
      'useful': 1,
      'user_id': '8ToVL-FXEu5kjh3LC7kBRQ'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 2,
      'date': '2016-12-01',
      'funny': 2,
      'review_id': 'EEAdWv92XwDgFnJOmacXHA',
      'stars': 4,
      'text': "Came here while visiting my friend in Vegas.  We wanted tacos and I'm picky about Mexican food.  This place totally hit the mark.  Not only was it delicious, the lady taking orders was super sweet and a really good salesperson.  I was debating on getting carne asada fries because i had already had fast food that day.  Here's how the convo went:\n\nHer: do you eat them [carne asada fries] every day?  \nMe: well, no...\nHer: then they're not so bad then\nMe: good point\n\nSalsa bar was on point and the spicy carrots and onions were crispy and super hot.  Yum.",
      'useful': 2,
      'user_id': 'u5o_QDwRPQW8gjgFudu9bg'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2016-10-14',
      'funny': 0,
      'review_id': 'Z5bwRMa0dR_e5Pm_cz5zog',
      'stars': 4,
      'text': 'Once again. The good was hot and fresh tasty, the customer  service  is friendly  and personal. This what to expect from a family owned restaurant.   The burritos  are very nice sized /over sized for your $$ the torta  is good also,  all the food is good sized portions',
      'useful': 1,
      'user_id': 'zc6cucEAy6Qxkxr9ZUkqbw'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2017-02-19',
      'funny': 0,
      'review_id': 'LVRBgIK3CP5Ap__LlczgRg',
      'stars': 5,
      'text': 'This place is special. All their food is made with the care you would expect at a family gathering. The hospitality is incredible. This is NOT your typical Mexican turn-and-burn kind of place. Every dish I\'ve ordered thus far has been absolutely fantastic...Well assembled and delicious. Personally, I love the "caldo de pollo" (chicken soup). I could eat it every day! This establishment is a definite winner in my book!!',
      'useful': 1,
      'user_id': '0Fhcuf1rKxSZOWQ4J56udw'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 1,
      'date': '2016-06-16',
      'funny': 0,
      'review_id': '4UPAFxc0r-KrJ20p4LOrRA',
      'stars': 5,
      'text': 'Best fish tacos! I also like the Carne Asada fries. The first time I had a chicken chile verde burrito.  But once I tried the fish tacos, I was hooked! The other burritos are good, too. My husband loves the pickled/peppered veggies. He gets a bagful and eats them while we wait. Family owned and operated. Friendly, cheerful staff.',
      'useful': 1,
      'user_id': 'TqoqUHUCiMVG4_PEZK4qkQ'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2016-05-19',
      'funny': 0,
      'review_id': 'kC5ecZO35TPFuSV_H9poew',
      'stars': 5,
      'text': 'The food is very tasty, generous portions.  Had the combo with cheese enchilada, Carne Asada taco and chili rellenol with rice and beans.  Hubs had the Carne Asada plate .',
      'useful': 0,
      'user_id': 'qDdc-EvTGorahsfXnBYAEw'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 1,
      'date': '2015-07-29',
      'funny': 0,
      'review_id': '_SfOk_nyyitCBgZSEi6G_A',
      'stars': 5,
      'text': 'I really enjoyed it here. The cheese enchiladas and carnitas taco were really good. The staff is very friendly, always nice to see positive happy people. I will be back for sure! give this place a try, you will enjoy it.',
      'useful': 2,
      'user_id': 'VDjzt6N--TMXH4E7Y6ialQ'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 2,
      'date': '2017-05-14',
      'funny': 1,
      'review_id': 'qbkhfoOAP43YREAH2t-PTQ',
      'stars': 4,
      'text': "I really liked the 'mom and pop' feel of this little Mexican restaurant, and the quality of the food was really good. Solid four stars. The complimentary chips and beans I got for my Yelp check-in were excellent. I upgraded to a larger size beans in order to share. You can tell these beans are the real thing.... slow cooked with love for the flavor and not smashed too much. Yum.\nMy husband has the carne asada burrito, it was huge, their version came with rice and beans inside the burrito, a little different than other taco shops I have been to.\nI had the combination plate, Chicken Enchilada, Chiles Relleno, and Chicken Taco. This plate came with rice and beans. Everything was excellent, but the star of the show was the taco!\nMy next visit I will definitely try some of the other tacos on the menu. The salsas in the salsa bar were amazing, I put it on everything!\n\nSo here is the bad news about this place........ the neighborhood! This shopping center and parking lot is scary! There are so many people loitering in front of the next door liquor store. I am glad I was with my husband, I really would not have felt safe coming here without him and coming here after dark.......",
      'useful': 3,
      'user_id': 'FEwPxhjdVKxSPVQQAHX2gg'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2017-02-19',
      'funny': 0,
      'review_id': 'bepPxZRTYhy4jWfWu_eCHg',
      'stars': 5,
      'text': 'This place is great! The food is amazing. Had a plate of chilaquiles and my wife had the machaca. The portions were generous and tasty. I highly recommend this place if you you in the area and want great Authentic Mexican Food!',
      'useful': 1,
      'user_id': '7JQsASQOgJAPuneUTxYZEg'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 1,
      'date': '2014-10-30',
      'funny': 0,
      'review_id': 'K9RjHW__5WqYMZDdkFsneA',
      'stars': 5,
      'text': "Outstanding. Visiting Las Vegas and happened upon this place. Having been raised in San Antonio, I feel like I know great Mexican food. And we found it here. The food is very authentic because they know how to build the meal with a great foundation. The chips and homemade tortillas are to die for. We all had various tacos, chicken and asada. Their burritos are huge!  The best part of the meal was the carnitas nachos. Don't miss out on them. When I'm back in vegas again, a meal here will be on my itinerary.",
      'useful': 2,
      'user_id': 'y74WFWeQHLCC9AbxuRWcSg'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2013-09-26',
      'funny': 0,
      'review_id': 'wL9hpAfkMx6FueuuoMOo4w',
      'stars': 5,
      'text': "Dont sleep on this spot! Consistently great, authentic mexican food. We've tried many other places but continue to return here!",
      'useful': 2,
      'user_id': 'o5hZLSgek20FFV0-aHY6TA'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2015-11-13',
      'funny': 0,
      'review_id': 'I_xF8TrcJkiamX3lDBlstg',
      'stars': 5,
      'text': "Seemed questionable from the outside especially only the outside thou when u walk in the atmosphere changes the cashier / sever name is Sarah she was really helpful I been here twice now I've tried the cabesa tacos, the lague  tacos, fish tacos, the shrimp cocktail  and the carnitas torta with fries all of it was outstanding I am surprised on how good it was they have a large variety of menu options which in sure there all good the cabesa was tender and juicy the torrilla they use was thin and delicious tasted home made fish taco was fresh it was fried still good thou love how fresh the tomatoes where , shrimp was perfect and Devined  correctly had a good amount of avacodo too which I loved , the carnitas were tasteful and not overcooked lettuce was fresh and crunchy over all I would say they put a lot of love in there food which makes it super good new spot for my Mexican food cravings I highly recommend this place",
      'useful': 0,
      'user_id': 'Ozg8dKXznd911fQx-6qy1g'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2016-06-28',
      'funny': 0,
      'review_id': '2mEJoiRnFmlyJKe10lpi_w',
      'stars': 5,
      'text': 'Amazing, fresh and inexpensive! First time here and I love it.',
      'useful': 0,
      'user_id': 'yvWZJA-h-UEC_Xf51eE0OQ'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2017-02-15',
      'funny': 0,
      'review_id': 'AQgNtmL_BZ1sNw0UsekBtw',
      'stars': 5,
      'text': 'Was in town on a short layover and was looking for some real deal Mexican! This place was so good! The sopes were amazing!!!!!  Will definitely be back when in the area!',
      'useful': 1,
      'user_id': '7vomvJjoNBthj7iVo1lFqg'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 1,
      'date': '2014-04-28',
      'funny': 1,
      'review_id': 'Pmbh65APyWtZ2A9UNkNZfQ',
      'stars': 5,
      'text': "Only 1 review?? C'mon! This place has awesome food!! It's rough around the edges but don't let it fool you. Tacos! Burritos! 3 different soups!",
      'useful': 3,
      'user_id': 'w7Q-WMxPuznP-P2Y79ICng'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 1,
      'date': '2015-11-03',
      'funny': 1,
      'review_id': 'Nyb_AnA1nP6u1uPQ2p_RNg',
      'stars': 5,
      'text': 'Mi Pueblo Taco Shop is a family owned treasure in Las Vegas , Nevada.  Owners Juanita and Manuel make homemade Mexican food in small batches and treat E everyone like family.\n\nIt is outstanding.\n\nPaulette Motzko',
      'useful': 1,
      'user_id': 'vlcsGHaSPAS_Bb59EW6E8g'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2017-07-21',
      'funny': 0,
      'review_id': 'iHJQRpIvKhStGT6ruIaqQA',
      'stars': 5,
      'text': "This place is great. I come here almost every day. Today I had a California burrito which was incredible. The fish tacos are always delicious. I really enjoy coming here, the staff is nice and it's always clean.",
      'useful': 4,
      'user_id': 'yianVcvmHzUE08AVOz1iKw'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2017-07-08',
      'funny': 0,
      'review_id': '2z563U4pdgYWkYKfpYhhBg',
      'stars': 5,
      'text': 'Finally found the BEST Mexican restaurant in Las Vegas!! The food is not only fresh but comes in large quantities! Will come back again and will recommend this place to my family and friends!!',
      'useful': 0,
      'user_id': 'nx7twsWNlAMrsr1CBIW6AA'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2016-07-29',
      'funny': 1,
      'review_id': 'ulzjicEUXoVff3d7VWzkeA',
      'stars': 5,
      'text': "Bomb! Fresh ingredients! \nYummy to my tummy! \nI would highly recommend this gem. This meal hit the spot, next time I'll try the enchiladas.....",
      'useful': 0,
      'user_id': 'nbVr7Q8j48hBODKrMOZdKQ'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2016-12-29',
      'funny': 0,
      'review_id': '8-sknGIGoDXqudrGyMNS_g',
      'stars': 5,
      'text': 'I will say our expectations were guarded when we pulled up, but step inside and receive a warm greeting and look at the menu, you know all will be good, and it was. The burrito was tasty and plentiful, "sandwich" with the fried pork, Pastor tortas, absolutely satisfying, simple and perfect. The hot salsas were fun and tasty. Enjoy',
      'useful': 0,
      'user_id': 'pmPsoV2jamfOgWOgiZxLmA'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 0,
      'date': '2017-03-26',
      'funny': 0,
      'review_id': 'aWeHWmKbYRWbNh5_rSHd3Q',
      'stars': 5,
      'text': 'Small mom and pop joint. Food is really good here and service is great, super friendly and welcoming.',
      'useful': 0,
      'user_id': 'ynaNvoqkUEm0Uoh06EltZQ'},
     {'business_id': 'OQcvO5P3gH0cuJ-bPXwfQQ',
      'cool': 1,
      'date': '2015-12-05',
      'funny': 0,
      'review_id': 'Sz7Q5Ylx6zeuxPqKx1Ohbw',
      'stars': 5,
      'text': 'Very good I recommend the fish tacos!!! And carne asada tacos! All with red salsa!! Bomb!!!',
      'useful': 1,
      'user_id': 'bmjZVlu-GdjK6f_c8FLNvw'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2015-11-17',
      'funny': 0,
      'review_id': 'VRx9QZhMV0JmUIwYdYWftQ',
      'stars': 5,
      'text': 'Love the place. Been going for years. The coffee is really good too. I always get the special meal and my wife the dosa. Both are great.',
      'useful': 0,
      'user_id': 'gTWPMVLcOrTb3OSy0XgTGw'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2016-11-05',
      'funny': 0,
      'review_id': '5gEdpTSiXkbKzxwAuTvcDQ',
      'stars': 3,
      'text': "Great place to enjoy South Indian food\nThali was good along with mini tiffin \nPretty busy on weekends \nDon't forget to try the coffee ( served in traditional stainless steel utensils)\nWashroom pretty clean",
      'useful': 0,
      'user_id': '8q711of5XKzLGJ400PI09Q'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 1,
      'date': '2016-10-09',
      'funny': 0,
      'review_id': 'hmHpfSZfXTuNi0vsReTq8w',
      'stars': 3,
      'text': 'We went during Thanksgiving weekend and realized they only had a buffet for $14-15/person. \n\nThe dosas were brought fresh to your table. Dosas were made from chickpea and rice flour (i.e. not authentic). But all restaurants here use this method rather than actually soaking "washed urad dhal" and rice and then grinding them and allowing it to ferment (have a bit of a sour flavour which I love). \n\nSambar was warm but the idlis and vardas and other dishes were cold. Sambar tasted okay. \n\nThree types of chutneys as well as lemon pickle. \nI liked the hoppers dish.',
      'useful': 0,
      'user_id': 'zbSBmoujGgIPuNWQcny14g'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2014-10-19',
      'funny': 0,
      'review_id': 'GOUoPL6Y_aicIvoP4B3aEQ',
      'stars': 3,
      'text': 'The food was quite good but the service could be better. We went there 1/2 hour before closing and were rushed to order and our food wasn\'t even hot. They wouldn\'t provide refills to the "sambhar & chutney" as the kitchen was closed!!',
      'useful': 0,
      'user_id': 'QEt3srla7bfPICwlXqRsyQ'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 1,
      'date': '2016-03-26',
      'funny': 1,
      'review_id': 'pshC96Vt8phDg335MR6uxQ',
      'stars': 1,
      'text': "I used to love their ala carte menu...now they have all day buffet very bad and stale food.....avoid at any cost....we can't even order ala carte. ....",
      'useful': 2,
      'user_id': 'P5f5Dx3_rL0znuE4wy77xQ'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2015-04-28',
      'funny': 0,
      'review_id': 'z1IPBvG4czEcwzVYYRXgGw',
      'stars': 4,
      'text': "I'm really not sure why this place gets such bad reviews - maybe the quality of the food has gone down over the years, or maybe people are comparing it to some of the more fancy/expensive dosa places in the area. \n\n I thought the Chana Bhatura & the Dosa that I ordered were delicious and I'm a complete sucker for Sambar. For the price and the quick service, I thought it was a great alternative to some of the other choices in the area.",
      'useful': 0,
      'user_id': 'IL1-dyCe_zSqhmt0ExL_Jg'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2017-02-21',
      'funny': 1,
      'review_id': 'Xn4pIF9paC7QjZrR3oVdCQ',
      'stars': 1,
      'text': "I have been to the Scarborough location and had a decent experience. I took my family here (12 people) after calling ahead and being ensured that there would be minimum wait time. When we reached here the line was out of the door but we were told it wouldn't take long. \nAfter 45 mins(!!!) of waiting we finally got a place to sit. We were happily looking forward to some tasty masala dosas. Well that's the first thing your order at a Dosa place right? Well guess what, they ran out of potatoes.... You can have a palak paneer dosa sir, our server said. Well I didn't come here and wait 45 mins to eat a palak paneer dosa. We walked and instead of having a nice time at dinner, we ate greasy Domino's pizza instead.\n Interestingly upon further investigation, I discovered some older reviews with the same experience. I wish I had read them before. At this point I don't want to set foot in at least this location of Saravana Bhavan. I might consider giving the Scarborough location another shot. With so many options around, please do yourself a favor and go to some other South Indian place in Mississauga where they don't run out of potatoes....",
      'useful': 3,
      'user_id': '1mmaT_rJbI0Y8NJDyJcWRw'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2014-06-07',
      'funny': 0,
      'review_id': 'O-DBxzWowHr1vi5LuB_x6A',
      'stars': 4,
      'text': 'I am a big Saravana Bhavan fan. Back home in India as well as here. new to Mississauga. food is good. enjoy the buffet specially. Thali with tiny cups and variety dishes is another favourite specially the vata kolambu. I love the cabbage curry too. \nsamabr vada is another favourite. \nI was wondering about the tip money I left for the server. I did not see the server come back for it. but supervisor/manager lady took the money and went inside. I am wondering how much does the server get? I went twice after that day, but did not find the server who had served me. other restaursnts have a tip collection box and the money goes there. \nI was charged extra for a coffee I did not drink I brought it to their notice and it was taken off without an apology. \nwell, I love the food, and I am willing to overlook these thing.\nD James, \nEtobicoke',
      'useful': 0,
      'user_id': 'kZpxNuWO25elzvBpAvAyRQ'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2014-07-15',
      'funny': 0,
      'review_id': 'cHA14ugNdhBISxWIKw_HSg',
      'stars': 3,
      'text': 'EDIT:\n\nI tried this place again almost after 2 years and it seems to have improved quite a bit. Possibly because of the new management. The food is much better now and we visit this place a couple of times a month and are pleasantly surprised with the changes. \n\n2014 Review:\n\nI\'m really disappointed with the degrading quality of food and service at this place. I\'m not a fan of this place, but decided to give it another chance as my friend wanted to eat here.\n\nThe place has been upgraded, which is about the only good thing with this restaurant. The new management is pathetic and the food equally worse. We went in on a Monday night for dinner and the waitress took our order and took close to half and hour to get our appetizer. The vada was spoilt and when we mentioned it to her, she just ran away and came back after a while to say that "This is how it is" !!! and she did not even apologize. How Rude!!!! She seemed to have a lengthy discussion in the Kitchen and another waitress came to our table and served our food almost 40 minutes later. She did however apologize for the delay. Much better than our previous waitress. We were appalled at the quality of service here and vowed never to come back here again and recommend people from going here ever.\n\nIf you\'d like to get good South Indian food in Mississauga, goto either Udupi Madras Cafe or Chutney Swamy or Guru Lakshmi.',
      'useful': 0,
      'user_id': 'mT_0RP1y9qYyfU4Kd2SRTg'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2012-10-02',
      'funny': 1,
      'review_id': 'ckevdpWvkhPiIlXN3609Ug',
      'stars': 4,
      'text': 'Been in to eat and also have picked up takeout. Great menu, fair prices. Food was solid. I will return.',
      'useful': 1,
      'user_id': 'CXwHjHxwGDzRTgvU_NdLIQ'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2014-03-22',
      'funny': 0,
      'review_id': 'th_T0lKPrYCt6lcIECM-vA',
      'stars': 1,
      'text': "Terrible service - they can't handle pressure\nFood served cold. Sambar that's key to a South Indian Dosa meal must always be served hot but was served cold. Their excuse for cold sambar was that they're just preparing it because of the rush. Shouldn't that mean the sambar should be hot?!?  And their servings for sambar are too small. They skimp on the chutneys. \n\nWe'd ordered a Kara dosa which they got wrong. Not authentic at all. \n\nHot coffee served cold. And when we asked for it to be brought back and served hot we got a lame excuse that the milk was just poured fresh, what fresh from the fridge?!?  When they did take the coffee back, it never was served again. And we still got billed for it. We asked for water several times.  Not served!!!  They finally brought out a jug of water as we were putting on our jackets to leave.  \n\nThis is the 3rd time in as many months that I've made the mistake of coming here. But no more.  \n\nThis place sadly does not have its act together and gets a big fat zero in both the food and service department. One of the staff even have the excuse that cold food is a 'kitchen problem'. Hello, is management listening.  \n\nWith this sort of customer service I seriously wonder why the place is busy or popular. If I could I would give them negative marks. One things for sure., they've lost an avid foodie customer.",
      'useful': 0,
      'user_id': 'h3q-Y8pf1-_0jKgdEj29Uw'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2014-04-12',
      'funny': 0,
      'review_id': 'COoskhNoQrbjpcxCh5BdYA',
      'stars': 1,
      'text': "This place has lost my vote completely .... Used to be an awesome South Indian buffet for breakfast with hot kaapi served with idli dosa and wada and a background music of suprabhatam to augment your weekend into a bliss... But off late the service is unbearable... I used to go as the food was atleast better than the others... Today the food also was a huge let down! Sambhar with no sambhar aroma just daal with tempering of curry leaves and mustard seeds and idli, dosa, wada all undercooked and stone like texture and not to mention cold ..... And a background music of titanic? Really? This has to be a management issue.. Even the rava upma was disastrous ... The table was not even served water! \nI would just like to add is this restaurant was special to me and I used to come here very often... This is a yelp alert to other people to avoid this place and not to get their heart broken like me! \n\nIf you are still thinking to try this place let me tell you they don't have gulab jun the only star entry of this place and have some cheap kesari... \nThank you for not going and trying out Cora's adjacent to this shit",
      'useful': 0,
      'user_id': '348o9qgR_hiNe25Rs-CDsw'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2014-09-01',
      'funny': 0,
      'review_id': '1eZJbt6Y7b54CGli3YeW7A',
      'stars': 5,
      'text': 'Absolutely love the food! Brought many people here for lunch and dinner and they also love the food. I highly recommend masala dosa and gun powder idlys.',
      'useful': 0,
      'user_id': 'r0_9MfX9tzTKvuIQFwKuQg'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2012-02-09',
      'funny': 2,
      'review_id': 'OTgGe54jpKgIJ0e0wzUGHA',
      'stars': 1,
      'text': "Had heard so much about Saravanna Bhavan- we just moved here from California and was excited to give this place a try. The service was non-existant...it was as if they were doing us a favour by serving us lunch. The supposedly spicy 'kara' dosa was tasteless and the potato stuffing inside was a violent red colour- using red food colour does not make anything spicy people!\nMy friend ordered another dosa which was so greasy that it had no taste but stale oil.\nThe thali looked OK but tasted zilch. The little bowls that the sambar came in...that was a mystery...some of them were stuck inside the other- probably the dishwasher could not get them separated- so basically that meant that the lower one was dirty. When we complained the server mumbled something and then never showed his face. \nNever going back here again",
      'useful': 3,
      'user_id': 'cW1v5mgoC8KAZsHanDLAUw'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2017-05-05',
      'funny': 0,
      'review_id': 'zB-uYinco4c9u14GtYBn3Q',
      'stars': 1,
      'text': 'How many of you are aware that you get charged 10% gratuity on your bill plus 13% HST on the tip amount?  I am not aware that this automatic gratuity mentioned in the menu card. Can someone advise  if this is legal?  Anyway I think it is a dishonest practice. I am really disappointed.',
      'useful': 0,
      'user_id': 'Ur0gkDyk310NsZu4V-krqg'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2017-05-15',
      'funny': 1,
      'review_id': 'Lc7GzIzD7n06dfyqvihKRg',
      'stars': 2,
      'text': "So I'll start by saying I had a terrible experience at the flagship location in Chennai, India in 2005. Since then, I swore off coming to this place ever again (no matter what corner of the world). \nSo after successfully boycotting this restaurant for 12 years, I was forced into going on a recent family dinner. \n\nNow I'll tell you why I will not go back again:\n\n1. They sat us directly next to the cash and a waiter just hovered outside of the register directly behind my seat. Table was not clean! \n\n2. We were a party of 6 and received 3 menus. When I asked for an additional 3 menus, I was handed one without a word and the waiter walked away. Another one walked by and I asked for an additional two menus and was asked why I couldn't share and rolled her eyes?! (Ummm ok)\n\n3. Six people, 5 spoons. It was a mission to get another spoon. \n\n4. One of our appetizers came after our mains because we asked if it was coming (I'm guessing they forgot to put in the order)? \n\n5. Not enough napkins. We asked for more and got 2. (Maybe they're saving money by reducing napkin waste?!) \n\n6. Had to ask four times for extra sambhar before getting any. \n\nOverall, the food was not anything special. There are many other dosa places that serve far better quality food and actually care about their customers. The service is horrendous. We went on a Monday evening around 6pm. They were not busy and many servers were just hanging around chatting with each other. When interacting with customers they behave in a manner that leads me to believe that the customer is a burden and they are not interested in ensuring a positive experience.\n\nThank you for reminding me why I had not eaten here since 2005 and why I should never eat here again.",
      'useful': 0,
      'user_id': 'QgiR84sGGz1jpGLMs-ZGSA'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2016-01-26',
      'funny': 0,
      'review_id': 'cW9XQINtwasnXh3FoZk-mg',
      'stars': 3,
      'text': '#SOUTHINDIANFOOD #SARAVANAABHAVAN #YELPGTA\n\nI\'ve been to Saravanaa Bhavan multiple times, a couple of years ago. It never was my "go to" place for dosa, but after an extremely unhealthy-greasy feeling at Guru Lukshmi one time, Saravanaa became a recent first-pick. \n\nMy dad and I decided to try the weekend vegetarian buffet. The restaurant gets super busy and for the South Indian food lover, this could be paradise. However, for a person like me who goes for selective items only (dosa, idli, chutney), this didn\'t exactly strike much of a chord in my heart. The dosa was subpar and I found the supply limited and fleeting. Every 20-25 minutes, the staff put out dosas that were gone in 5-10 minutes. Same story with idli. \n\nIt was fantastic having all-you-can-eat coconut chutney though. A lot of places skimp out on that one in particular. \n\nOverall, an okay experience. I probably wouldn\'t return for a buffet. I\'d like to select my kind of crisp dosa next time.',
      'useful': 4,
      'user_id': 'mOVyk3O18VY5nrUMTZgM1w'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2014-01-09',
      'funny': 0,
      'review_id': 'iI4MCm3h1CBH4mIm9mDRnA',
      'stars': 1,
      'text': 'Dosa is sometimes seen as a delicacy. Most do not make this at home. So when looking for a dosa fix, and a proper one, I have tried to come here.\n\nI came here about 3 times this year.\n\nThe service is minimal. You get your food, which comes in small portions, and you get out. You pay at the counter at the front. So that kind of tells you what type of set up this is.\n\nZero ambiance. hard cold seats, storefront setting, in a plaza. Walls peeling. \n\nIt is all veg food - no meat. \n\nThe dosa was barely filled. I mean, barely. I would say two tablespoons of filling in the large paper dosa.\n\nI would rather go to Annjapar a bit further south.',
      'useful': 0,
      'user_id': 'eOTzkS-by4pVwwCVrifmqg'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 1,
      'date': '2014-04-21',
      'funny': 1,
      'review_id': '_AQBiBk4NGuFS5MuIoyGfg',
      'stars': 3,
      'text': 'This Saravanaa is not as well run as the one in Scarborough. The food is not that good. The dosa was cold & okay. The sambar tasted like spicy north Indian dal! I am surprised since I was under the impression that this Saravanaa in Mississauga will be better than the one in Scarborough.',
      'useful': 2,
      'user_id': '0CSlu7ZhK0SfEmo_1WkcnA'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2016-10-15',
      'funny': 0,
      'review_id': '62rQaTXb0XMo3D8FfVbJBw',
      'stars': 3,
      'text': 'Three stars for the absolutely amazing delicious food. They lose two stars from me because their service is HORRIBLE. Not only are they EXTREMELY slow, they don\'t actually care about it. They forgot half of our meal about 20 minutes after the first part was brought to the table they "remembered" one dish was completely forgotten about, had to ask (the girl who was sitting on her phone in the corner) to please it to us. We couldn\'t even get them to bring the bill after asking twice. \n\nIf you don\'t mind waiting or being ignored. At least the food is very good.',
      'useful': 0,
      'user_id': 'K-q7oE90OWNZ8IOXQqXsAQ'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2015-09-21',
      'funny': 0,
      'review_id': 'ULFWaUwYi7PftxKIKIGu8g',
      'stars': 5,
      'text': "Ultimate south indian breakfast buffet! Totally satisfied with this place. The staff is friendly,the food is fresh, Dosas are crisp, chutneys are delicious and sambar is to die for! If you're planning to catch up the weekend buffet and don't want to wait then be there by 10.30 a.m.",
      'useful': 0,
      'user_id': 'qB6k6ubk1B6SkJYOWbdhSw'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2012-03-19',
      'funny': 0,
      'review_id': 'vP8swFbBl9WEPUXgWAFM_g',
      'stars': 2,
      'text': 'Good value for food however I have always had an issue with their cleanliness. Overall not super clean but ok. Food quality is ok. Overall value 5/10',
      'useful': 1,
      'user_id': 'w2C0_M6WCfxYuY7ZvoSLvA'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2016-05-14',
      'funny': 0,
      'review_id': 'TkaQjEzPUfCNUqGDKxeIBQ',
      'stars': 5,
      'text': "again - as usual - it's amazing!!",
      'useful': 0,
      'user_id': 'PbuXIB-7MRXcf1Ueg_g2lA'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2014-10-21',
      'funny': 0,
      'review_id': 'xp63FhRxJYXL20WRvt3DJQ',
      'stars': 3,
      'text': 'It seems that a lot of reviewers have already shared their strong feelings on the flavor (or the lack thereof) of Saravanaa Bhavan. To be honest, it\'s not terrible like some of the reviews would have you believe. But it\'s not particularly amazing either. Maybe if we had ordered from the menu, it would have been better, but we opted for the buffet option, and it was all... one-note... Not bad. Just not mind-blowing. It was food stuffs that helped fill our bellies before we left town. It\'s closer to a [*][*] experience in that "I\'ve experienced better" but a lot of the [*] reviews are unnecessarily harsh. \n\nI didn\'t touch the sambar. Sambar is one of those things that varies quite a bit even between two people from the same part of India. Personal experience has taught me that ingesting sambar (especially sambar that you\'ve never had before) before a long road trip (or in my case, a long road trip and then a long plane trip) is asking for trouble and shaky leg syndrome.',
      'useful': 0,
      'user_id': 'itI6CqeR92ikei1n86N5sg'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2014-02-01',
      'funny': 0,
      'review_id': 'iktzgZrYTmQceUqHLUOZNQ',
      'stars': 3,
      'text': '5 stars if you eat at a saravana bhavan in Chennai,India, but not here. They are not able to create the same magic here.',
      'useful': 0,
      'user_id': 'HPaMg9PTxf-7uQaatFtpgg'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2017-06-15',
      'funny': 0,
      'review_id': 'frDZjONa7ubQhGmP2YzSTw',
      'stars': 4,
      'text': 'Excellent for vegetarians and healthy food. Variety of dishes. If you want taste something different this is the place to go.',
      'useful': 0,
      'user_id': 'XcO7fb0q-LukfudaBMNPIA'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2016-06-25',
      'funny': 0,
      'review_id': '0ra9e8IxU5kJlwxih-soqA',
      'stars': 2,
      'text': "On the name of Chennai's Saravana Bhawan we visited here three to four times but not good as other branches are. Last time visited Delhi's Saravana Bhawan and it was really good. We can't even compare this Saravana to Delhi's Saravana. I didn't liked there sambar and dosa was not crispy at all even filter coffee was like Tim Hortons! Also the thali is limited... Can you imagine?!? For the first time I experienced this kind of thing... We just needed 1 extra poori and for that only we had to pay. We visited Bahrain's Saravana also... Even that branch is also far better than this one!",
      'useful': 0,
      'user_id': 'BCjsgRFDCxGQHr0hVlwqvg'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2015-08-13',
      'funny': 0,
      'review_id': 'OVZ-TCbfh7fSu0vKVLZRrw',
      'stars': 3,
      'text': "I would like to update my review on this place based on my experiences since.\nI decided to try something different other than the usual thali. So I tried their Kara dosa, which came out to be a thick, soggy dosa with a sweet tomatoey sauce. Then when I visited again I decided to give their dosa's another shot and opted for their Mysore masala dosa, which I have loved in other restaurants. They warned me that this dosa is supposed to be spicy but I can handle heat so I was fine with that. But again, I ended up getting a overly thick dosa that was too soggy and tasteless. My brother got the chole bhatura, which he enjoyed but complained that the bhatura's were a bit too oily. \nSo I feel that ordering something other than the dosa would be your best bet because their dosa's are just not worth it and you can get much better ones at Guru Lukshmi or Udupi.",
      'useful': 0,
      'user_id': '39VYc2nWSdvdTCldsDJ7qw'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2013-04-23',
      'funny': 0,
      'review_id': 'l7WiMOZPfP6UZhn_obnLCQ',
      'stars': 2,
      'text': 'Been here a couple of times and it was ok for most part. Not spectacular. The food has been ok, but we\'ve always had good service. Returned last week with baby in tow, only to find no changing table in bathroom of this "family" friendly restaurant. Not returning.',
      'useful': 0,
      'user_id': 'QsIDgqf4n71PjI1l2J9O9g'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 2,
      'date': '2009-09-16',
      'funny': 1,
      'review_id': 'X-caW4AMdMOBpDvqwGgiSg',
      'stars': 5,
      'text': 'For my money, Saravana Bhavan is just perfect. food for a great price.  Crispy, smooth dosas and bubbly, chewy uttapam.  The Sambar is just like Dosa in SF, packed full of leaves, seeds, and spices that hit your taste buds in all the right places! \nThe staff are super nice and kept making sure we were satisfied. "More sambar, More Chutney?" \nYes, please! \nAfter getting way more than 2 people could eat (come in groups) we packed it up and of course left with more sambar and more chutney.  \nSat and Sun, the buffet offers all the south indian food you could eat, (AYCE) and desserts. Try just a bite of these super sugary, fried desserts and think about jogging back to downtown TO.',
      'useful': 2,
      'user_id': 'gPK2AKr-UA_7hZus80SSng'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2012-12-17',
      'funny': 0,
      'review_id': 'SmFAqExw0wY1itq8hqIwyQ',
      'stars': 4,
      'text': "Been to this location thrice at different times. The service has been pretty consistent - quick and efficient. I will not expect more from an Indian restaurant.\n\nThe food has been really good for the most part. Here are the things we've tried - \nThe vada sambar is top-class; fresh, hot, soft and super tasty.\nThe bisibele bath is good with plenty of good flavour. The onion rava masala dosa is crisp and filling - the best I've had. One star off for the rava idli that lacked life... Otherwise you cannot go wrong with anything here!",
      'useful': 1,
      'user_id': 'Rk_O89vyLCcEa3Xq6RYftQ'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 1,
      'date': '2009-08-12',
      'funny': 0,
      'review_id': 'A3ndd6Za9IwGl-e4WpqE9A',
      'stars': 4,
      'text': "This is going to be a biased review because this chain is from my home city in South India. \n\nNow that's out of the way, this is the best place to get authentic south indian food. If you are a newbie to the cuisine go on Sunday for the buffet when you can try a bit of everything \n\nIf you familiar with South Indian food, the mini Idlis, paper masala and the adai aviyal are something that you can always bet will be great. \n\nIt's a completely vegetarian joint, so meat eaters might be tad disappointed - the flavour will make up for it, trust me ;) \n\nEnjoy.",
      'useful': 3,
      'user_id': 'VH1tcxstMMtiLQ9ArdzmOg'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2017-06-04',
      'funny': 0,
      'review_id': 'jBpGidedpQRcQsbE-aHfZg',
      'stars': 4,
      'text': 'The idlis are so good! The service is wonderful. Rava kesari is the best. Cheese Kara Masala Dosa is also a must try. The quality of food and the speed at which its served is amazing.',
      'useful': 0,
      'user_id': 'ff_rH18q-V_VhrKSzZZ16Q'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2017-02-22',
      'funny': 0,
      'review_id': 'XKmM3gy7DTUkaKsN6tIs2Q',
      'stars': 3,
      'text': 'This place is always busy. We went for family day lunch. We waited for 10 mins after which we were shown our way to a table. After siting for 15mins (and waiting for the menu)  we were "informed" that they only have all day buffet that day. \n\nThe dosas were served fresh and piping hot and aloo masala was a favourite at our table. The rest of the food was average; uttapa, Idli and medu vadas were all cold. For dessert they had gulab jamun (avoidable) and rava kesari (yum). \n\nIn all a good place for South Indian food. The dosa and the rava kesari are a must have.',
      'useful': 0,
      'user_id': 'ieLHSYDzCnLbX1IjZBUdRA'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2016-02-09',
      'funny': 0,
      'review_id': 'bmQ2jcgCKNryYtcczKe_1A',
      'stars': 5,
      'text': "I really love this place. The paper dosa is so crispy! The potato masala is very mild and the chutneys have varying degrees of spicyness. We also love their Rava Kichadi, Parotta's with potatoes and a delicious cauliflower dish. The Mini Tiffin is a great place to start and try a bit of everything. And don't forget to have some Rasmalai for dessert!  Enchanting!  The Rava Kesari is also delicious.",
      'useful': 1,
      'user_id': 'wkqy2o_qF0S_Z4vyG6pdIQ'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2013-10-18',
      'funny': 0,
      'review_id': 'Qoc_JCZgk9Sv94fpwq_bdA',
      'stars': 4,
      'text': "I love the manager here, he knows us for a very long time. I like their food. The servers change so often that we don't like it but still they are much better than a lot of other restaurants.  \n\nFood - 4/5\nService - 4.5/5\nManagement - 5/5",
      'useful': 0,
      'user_id': '3bCDpygbXAkRDWxQ6_Nf6w'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2014-05-26',
      'funny': 0,
      'review_id': 'OLSuyCoeOLaOgSfHeXsBcQ',
      'stars': 1,
      'text': 'I am a huge fan of Guru Lukshmi and they set a really high bar for dosa in GTA. Unfortunately, Saravanaa Bhavan was a big letdown. We went in pretty late on a sunday afternoon. The decor has been recently upgraded which gives the place a clean modern look although it is still based in a strip mall. \n\nMost tables around us were having the thali which should have given us some strong signals. \n\nWe ordered the vada sambar but the quality of the sambar was not good. The vada was not crispy as well which let the whole dish down. \n\nThe onion and chili utappam was the biggest disappointment. It was so thick and dry inside that after a few bites we just did not have the appetite to finish it. The chillies were huge and it was quite a task to remove them from the almost thick pizza crust like utappam. \n\nThe saving grace was the onion rava dosa but they had stuffed pepper in it which was annoying since it was quite strong. The accompanying chutney portion is too small and they were not flavorful at all. \n\nI am never going back again! Go spend your money at Guru Lukshmi. I am surprised they are a global chain with such poor quality of south indian food.',
      'useful': 0,
      'user_id': 'B4Rwp0SUYwO8lgSHVT9w5g'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2017-05-23',
      'funny': 0,
      'review_id': 'cdpFy5i5V1cPWe4jFLuoWw',
      'stars': 1,
      'text': "Disappointed ! They don't seem to function to the level they built up their restaurants in Chennai.\nThey don't have a clue about the importance of ambience in a restaurant .\nSacha",
      'useful': 0,
      'user_id': 'XV52fNAaWKDinoKy4bo39w'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2016-01-08',
      'funny': 1,
      'review_id': 'FovEkq4pj5U1uewA75Clvg',
      'stars': 1,
      'text': "This is the worst place to have South Indian food whatsoever. I am a South Indian myself and wanted to take one of my colleagues to try south indian food.\n\nInitially I thought of Guru Lakshmi but then I decided this place as Guru Lakshmi tends to get way busy. I have tried Saravanaa Bhavan in Dubai and this place comes nowhere close to it.\n\nI ordered dosa and madras coffee. The coffee came before the dosa. I requested the server to give the coffee a little later as it would get cold before the dosa would even arrive. But guess what, the server plainly refused. I didn't want to create a scene in front of my colleague. Finally the dosa arrived and guess what again, it had hair on it. Yes the dosa that I wanted my colleague who never tried south indian food in his life, was trying it for the first time and the dosa had HAIR on it !!!\n\nAt this point I was going to almost lose it. But I kept my calm and requested it to be changed and hoped that there would be no cockroach.\n\nTotally ruined my experience and I'm sure my colleagues as well. \n\nPlease folks go anywhere to try south indian food but not here. For the same price you will get better south indian cuisine with NO HAIR in it !",
      'useful': 1,
      'user_id': '6Yn7CvXitWjWo5OYDA9jSQ'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 1,
      'date': '2014-10-16',
      'funny': 1,
      'review_id': 'oaCwkb91TzpaKIZEMv1o3Q',
      'stars': 4,
      'text': 'I tired their Special Thali today ( only thali on the menu ) and it was fantastic, it was a gloomy day and it was just pefect, 14 items all veggie, it took about 10 minutes to get my order ready, that is kind of odd I would say as they were not busy all all when I entered, that being said it was worth it..the food you could just tell was fresh and clean.\n\nThey also renovated, the place looks way better than before, they added some booths as well . so if you are in the mood for some spicy south indian cusine, give this place a try.\n\n\nPlease note: If you are trying south indian food for the first time, it is pretty spicy :)\n\nthanks for reading \n\nPS. the service is a bit  odd but not rude. so dont be surpised if you find your server to be a little odd :)',
      'useful': 1,
      'user_id': '_cj4j6-FUBda7AvWrclHTw'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2012-06-26',
      'funny': 0,
      'review_id': 'ASe8Hk6voOIytmrxDlkJbw',
      'stars': 1,
      'text': "went there for dinner last night. Service was average. Food was horrible. Sambar and Chutney tasted bland. Dosa was tasteless.\n\nWon't be visiting them soon.",
      'useful': 0,
      'user_id': '2w8ltW7s3iuMCu-ibASohA'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2014-05-18',
      'funny': 0,
      'review_id': '5qIsq76JTyO3fmED2ujdew',
      'stars': 2,
      'text': 'Saravanaa has gone on a steep downhill trajectory over the past few years in food quality and service. The only upgrade as of now is the restaurant itself. A few months ago it was very unattractive, almost cafeteria style, with hard seats like you would find in a food court. Kudos to them for getting a massive facelift. I wish I took pictures, but it looks lovely inside. They even have an electric fireplace (for some reason) in the tiny entrance. \n\nBut ambiance aside, the food itself did not get a similar makeover. I will never again order a masala dosa from this place. There was a mere SMEAR of potato bhaji, a few tablespoons at most, in my dosa. The chutneys were pretty bland and also in small amounts. The best thing was the sambar, spicy and warming. The dosa itself was very oily.\n\nMy dad had the Saravanaa Special Meal, which consisted of ten TINY dishes (curries, chutneys, curd and a gulab jamun), rice, two papad and two pooris. As is the apparent custom here, the poories were miniscule. They were baby poories. I mean, how much are they saving really by cutting the quantity of their menu items? \n\nAnd it\'s not like this place isn\'t hopping. We got there at 7pm for dinner and already a queue had formed. This was the trend for the rest of the night, and people even crowded outside and left when they were told it would be a half an hour wait.\n\nThe waitstaff were pleasant enough, although our server tried to take away my plate before I had finished. They also left the front door open for a while, telling us "the air conditioning was not working". It is necessary to note that it was 10 degrees tonight, not a balmy summer at all. \n\nThis restaurant is NOT good for groups. There are tables of 4, and they can put a few together if needed. But do not come in with a group of 15, like people did tonight, and assume you will get a table quickly. Check ahead of time if you can make reservations.\n\nMy mom had the best meal tonight, the Mini Tiffin. It came with uppama, mini idlis in sambar, a mini dosa (with probably the same amount of bhaji as my dosa!) and a sweet. The sweet sheera was delicious, with cashews and raisins. Sad I didn\'t go for that.\n\nPROS:\n- nice ambiance, comfortable booths\n- delicious, spicy sambar\n\nCONS: \n- oily dosas with virtually no filling\n- crowded early, stays that way through dinner\n- NOT good for groups (call ahead)\n- hasty servers who do not consider the weather\n- miniature food',
      'useful': 0,
      'user_id': '1mie5crWiX4eMNEhmd5LDA'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2015-07-13',
      'funny': 0,
      'review_id': 'i447aU58uZJs-Df94sKeGQ',
      'stars': 4,
      'text': 'Great Masala Dosa - the mustard seed flavour of the Masala is perfect. The Parotta was good but not excellent: a bit small and not quite the right flavour compared to other Parotta I have had. The Rasmalai dessert is fantastic - although the presentation is nothing like the menu picture (no saffron strands) and suffers as a result.  I will definitely come again if I am in the area. For those new to South Indian food, it would be a great introduction, but those who are picky may not like it as much.',
      'useful': 0,
      'user_id': 't21OyjIVoBMRmCIaElwuEQ'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2016-11-22',
      'funny': 0,
      'review_id': 'tLQ-BpRYz1HvoHPWhSVlTQ',
      'stars': 2,
      'text': "We usually go to Anjappar Chettinad one weekends when I have a strong craving for a South Indian breakfast. My review is going to be a mild comparison of the two. \nWe decided to try Saravan Bhavan a few days ago as I like trying different options. \nThe breakfast buffet on Sunday is from 10-12 and we had arrived at 11:10 to quite a line up. We were seated at 11:30. \nThe first thing I noticed was that the dosas, idlis and vadas are much better than anjappar and are really well made. The sambhar and tomato chutney is delicious as well. \nOne of the reasons I crave this breakfast is because I love Rava Kesari and Payasam. The rava kesari was quite tasteless and the payasam seemed to have been thickened with starch rather than reduced the way it should be. \nWe asked for two coffees as we noticed that there were no beverages in the buffet area (In comparison, anjapar has coffee that you help yourself to) and they brought us two very small coffees that were quite good. We didn't know until it was time to pay the bill that each of the 150 ml coffees were $2.25 each adding $4.50 to the total. \nFor two people with coffee the bill came to $29.90 plus tip. \n\nI have rated it two stars because of the overall experience - food, service and pricing. \n\nFinal verdict: Liked- vadas, dosas, idly, sambhar and taste of the coffee\nDisliked: Service, wait time, pricing, poor tasting deserts, overall experience. \n\nWon't return.",
      'useful': 1,
      'user_id': '-9RU4LuI_TfYgv9rBijJoQ'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2016-11-12',
      'funny': 0,
      'review_id': 'SKTpq87BB53yLy4o7xQr1A',
      'stars': 5,
      'text': 'Love the dosas here. The size of a dosa is smaller here but it is also cheaper. A great bang for your buck. Service is prompt and staff is friendly. The restaurant is usually packed and is also quite spacious',
      'useful': 0,
      'user_id': 'hFefNv6jWn3_cSjN_UwXNg'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2016-09-17',
      'funny': 0,
      'review_id': 'JBEv2_R-4aATxy9mmiKrNQ',
      'stars': 4,
      'text': "Somehow after we moved to Mississauga this Sarvana Bhawan was just a block from our new place so obviously we had to check it out.  Come from Madras (the birth place of this chain) I would say it's the same flavour and taste as back home.  \n\nIt doesn't mean that's it's a great place to eat out south Indian food around here.  There are better options for sure. But if you are looking for good old Sarvana Bhawan taste this one pretty much nails it bang on.",
      'useful': 0,
      'user_id': 'z-9-XZdrmRrOFpMKAQ2rvQ'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2014-06-18',
      'funny': 0,
      'review_id': 'QbJiK8Zj9XLssurUafOWfQ',
      'stars': 5,
      'text': 'Best south indian food in Toronto. Everything freshly prepared and I am surprised at all the negative reviews posted by some members. The service was slow but maybe due to the large number of people.',
      'useful': 0,
      'user_id': 'DvLjx73mypBzwke8DEy9vg'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2016-12-13',
      'funny': 0,
      'review_id': 'VyOKSOtxe39K2CrwPRXHXg',
      'stars': 5,
      'text': 'Awesome authentic food and well priced\nCan be quite busy over the weekends\nService can be slow, but the food more than makes up for it',
      'useful': 0,
      'user_id': 'L6HGM8pmm2xCC8RTFqv0og'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2014-04-12',
      'funny': 0,
      'review_id': 'pxK_sHN16OwRbe2LBPLfNw',
      'stars': 1,
      'text': 'New Management is crapy. Quality of food very low. Used to a big fan of this restaurant but will not recommend it to anybody. Please dont eat there.',
      'useful': 1,
      'user_id': 'ipbFsb710sHyYQcH3xbVkQ'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2013-07-28',
      'funny': 0,
      'review_id': 'K3HZb8rF07FpWDJFXGmu4g',
      'stars': 4,
      'text': "This place serves the pure vegetarian south Indian cuisine .. even though I am not a pure vegetarian, I do eat out at this restaurant sometimes when I am in a mood for a light meal .. the thing with this kinda food is that no matter how much you eat, it doesn't makes your stomach go rumbling or feel heavy .. so I guess I can go here once in a while ..\nThe ambience isn't that great .. some of the tables are lined up in the middle of the path and I hate sitting in this area. The service is alright, not the fastest but reasonable . Everytime I've been here, I've noticed that the place is kinda crowded .. so there is a lot of noise but then if I choose to eat out then I should not complain about the noise I guess.\nFood is good if you are not looking for the typical spicy taste that Indian food is known for. Their spice level is pretty low.",
      'useful': 0,
      'user_id': 'm0uVfBkFiTbzg1GedPmBug'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2015-08-02',
      'funny': 0,
      'review_id': 'gPHffHb7Pg_Erl_1dGkeQw',
      'stars': 4,
      'text': "The food here makes me forget that I'm not eating meat. I'm a carnavore and make few eceptions about not having some meat on my plate, but I have to say I don't even miss it when I eat here. My wife and I have only ever gone for the buffet. I eat my vada with sambar,  my dosa with rice and tomato chutney, and my puri with potato or rice. It tastes amazing and I've never left without being so full that I'm uncomfortable, which is a good thing. The only reason I don't give them five stars is because they need to expand. It's always so busy that the service is close to chaos. But I'd much rather have great food and slow service than great service and terrible food. I recommend this place to any meat lover like myself to prove that the occasional meat free meal can be very enjoyable.",
      'useful': 0,
      'user_id': 'dcdVcbfwiS5iz7ys1QSWWA'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2010-07-13',
      'funny': 6,
      'review_id': 'e3XMzPuzQa-okjIR4nGo7w',
      'stars': 1,
      'text': "This was THE SHITTIEST first-dosa-tasting of my life. I know it's impossible to rate my first-tasting of anything, but I'm pretty sure when you get a sour stomach and horrible shits after being at a restaurant - it was a horrible first experience. \n\nI really hope dosas don't actually taste like what they did here. The food item itself can't be that horrible or else no one would eat it, right? But you see, I'm traumatized to a point where I might not even consider re-evaluating what they taste like and just leave it at that. \n\nThe service was shoddy. The drink prices (non-alcoholic) are shitty for the amount you get, which is close to nothing. The venue looks like a McDonald's in an airport food court. If this is what they think south India should be, then I recommend that Britain reoccupy. \n\nThe only thing I'd recommend is the Madras coffee, to-go. That's the one thing they didn't manage to screw up royally.\n\nAnd paying $30 for all this, that was the rotting cherry on top.",
      'useful': 4,
      'user_id': 'Hg1EF9PjGfcKBSNMjvWBeQ'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2017-05-09',
      'funny': 0,
      'review_id': 'Icjrm_OvZnSJaGKujCEAtg',
      'stars': 4,
      'text': 'The place for Dosa, in my books!\nThis place has always provided the crowds with some great vegetarian food as well as my favourite, dosas! I am not much of a vegetarian, but i would go here for Dosas as well as other tasty dishes. \nThe service is good, not the best, but still good.\nThe place is clean, and has the sink to wash your hands, without going to the washrooms...a must for your seasoned foodie.',
      'useful': 0,
      'user_id': 'p-6RZe5JAtxB17AS4wOaXg'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2014-04-21',
      'funny': 0,
      'review_id': '1PutBL8y8MTFFpyNEQzO2A',
      'stars': 2,
      'text': 'This is a poor version of this esteemed franchise. Sambhaar lacked any tamarind and looked and tasted like daal. Chutneys were also subpar. The dosa was strictly ok. The Scarborough branch is much better. Very disappointed.',
      'useful': 0,
      'user_id': 'Aqtnyl-rhUZF_zxDXlf0sw'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2012-09-10',
      'funny': 0,
      'review_id': 'dsb7RjU_QuzAvKkemfMaVA',
      'stars': 2,
      'text': "I love a good fixin' of dosa every now and then. When I do have it, I always go in with expectations (which I'm working on) of the dosa rice crepe itself having to be crispy, the potato filling has to be spicy and spread evenly inside the dosa crepe. \n\nCraving dosa, I went to Udupai first (one of my favorite dosa restaurants) but they were closed on a Tuesday for lunch! Hence, I ended up coming to Saravanaa Bhavan due to proximity & time crunch. I guess it was a good excuse to try it out. \n\nHere's what I found at Saravanna Bhavan: \n\n-Mom and I ordered 2 Spicy Kara Dosas. We found them to be rather soggy and oily \n-The potato filling was blood red in color, making me think they had added in a ton of food coloring \n-The dosa crepe was overly thick\n-Chutney's were okay; however they really skimped on it. I also found that the chutneys were rather inconveniently served to the left into three portions of the thali (steel square plate)\n-The sambar was okay; nothing special. For some odd reason unbeknownst to me, they had used 2 katoris (small steel bowls) which were stuck together to serve the sambar in \n-As for service, there was only one person serving when mom and I went here for lunch making service slow. \n\nSure it was decently priced; but the quality of food and service was worth WAY LESS than the $8 we cashed out for each dosa. Unfortunately, food and price do not go hand-in-hand here.",
      'useful': 1,
      'user_id': 'kD9Gr2IxZQv0CNp5D-Bsjw'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2014-06-24',
      'funny': 0,
      'review_id': 'Y2G4L0PGa-Pjtgq4p7rj7g',
      'stars': 3,
      'text': 'I have been to Sarvana Bhavan a zillion times, okay probably not zillion but you get what I mean :) if you are a true south indian food loyalist you will like the food. It offers good south indian food without the bells and whistles like Chinese hakka Masala dosa or the likes!! The food is very true to its roots and the chutneys and curries are cooked in chennai - tamilian styles.  \n\nThe place recently was revamped and is more organized and looks much neater. The servers are polite and the food is delivered without a very long wait time. \n\nI ordered the thali and it was very very delicious !',
      'useful': 0,
      'user_id': 'RgJ_nkjVYUKzCr47qJgSSw'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2016-03-26',
      'funny': 0,
      'review_id': 'In9I9K6j_thJbDRvGr8p9A',
      'stars': 3,
      'text': "Loved the Delhi Sarvanna Bhavan so was here to give this place a try.. not sure if buffet time was worse time to try the food but my rating is for the buffet food.\n\nAmbience: 4/5 (on any other day would have given 5/5, only issue it's a little small place to have a buffet)\nService: n/a since it was a buffet I wouldn't rate the service coz there were a lot of people and food delays were natural\nQuality: 2.5/5 (food was average, not what I had expected and in comparison to what I have had)\nQuantity:5/5(since it was buffet)\nOverall, it was a 3/5 for me, deserts were too sweet food was cold. I will definitely give them a try but not on a buffet day",
      'useful': 0,
      'user_id': '0SNb69rzZWY0OQtBCNoqDg'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2015-04-21',
      'funny': 0,
      'review_id': 'pWd6pYD-NxHFimRJKS-HVg',
      'stars': 2,
      'text': 'Average! They could do so much better. No exceptional flat ours. Poor is were way too small. Only redeeming feature was the pappadums. \nService is attentive but we have had much better South Indian.',
      'useful': 0,
      'user_id': 'zcBBHrItGGRHMEOEt1sT_Q'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2015-07-25',
      'funny': 0,
      'review_id': '3lVfoZKsy1vgVyGshraC4A',
      'stars': 2,
      'text': 'I went here to have authentic south Indian food, however I was very disappointed.\nAlso the quality of food is low compared to their Scarborough location.\nIf you are looking for authentic South Indian food, I would not recommend this place.',
      'useful': 0,
      'user_id': 'R5_K6l9rSyYABGK9Bw2iEQ'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 1,
      'date': '2011-04-18',
      'funny': 0,
      'review_id': 'wN60MSvRif3ArcUX7wDkvw',
      'stars': 5,
      'text': "I love this place. The dosas are really good and portions are big. The place is very clean and service is the best I've had at an Indian restaurant. The waiters treat you like your at their home. Whenever I'm in town, I go here.",
      'useful': 2,
      'user_id': 'TCf3Qt9yyaU6kYERVTcBWw'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2016-01-17',
      'funny': 0,
      'review_id': 'VbomFRiOsKp5NEwIpb1rUw',
      'stars': 3,
      'text': "Worth a 3.5 star rating. We tried  the Rasam soup, Masala Dosa and Tomato & Onion Utthappam. It was all quite flavourful, though I found the soup to be a little too spicy. The mango lassi was quite nice and cooling after the soup. It was fairly busy on a Saturday night, but tables come available quickly and service is quick.  The furniture is a strange mix of marble style tables and old library chairs.  The floor was a little messy with bits of food, but otherwise the restaurant was clean. It's an interesting place; we'll be back to try some other options.",
      'useful': 0,
      'user_id': '9YAcyOPntf2HJszcWgkLow'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2015-09-14',
      'funny': 0,
      'review_id': 'K4yL9JmtDUDGQpiqdyihLA',
      'stars': 5,
      'text': "It's been a while since i have been to sarvana bhavan and boy had i been missing out. I have been going to another very famous South Indian restaurant which has forgotten how to serve their guests lately and the staff seems to be very arrogant lately. \nSarvana wasn't too busy at the moment of our visit on Sunday night at 9 and the food was delivered quickly. The dosas were delicious and crispy and sambhar was hot and just as i hoped. Staff has always been extremely polite at this place and i really enjoyed going back to a place that gets it right. \nEverybody in our group agreed that my favorite dish, paneer dosa, was the best they have ever tried. \nWill be going back there soon and telling all my friends about it too.",
      'useful': 0,
      'user_id': 'iTTq-wybXD0Ik_0g7E4sqQ'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2010-05-09',
      'funny': 1,
      'review_id': 'bMJUUbfsRDLKdXWw-rSF2Q',
      'stars': 5,
      'text': "Overall:  Worth the trek to Mississauga.  Great value, great taste, great people. \n\nRoom: Not fancy, spacious, clean\n\nService: Friendly, even avuncular (more materteral), everyone's an auntie, and frankly a bit concerned that i wanted a second coffee \n\nMenu: Thorough south indian menu, tough to decipher without guidance from someone who knows the cuisine\n\nPresentation: Traditional south indian restaurant, paper dosas are intimidatingly big but that's why you get them \n\nTaste:  Delivers on south indian vegetarian goodness.  \n\nValue: There's a reason this place is packed with Indians.  Tastes good and cheap!",
      'useful': 2,
      'user_id': '1NT8m7QfuocAGaSqwR905A'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2014-12-07',
      'funny': 0,
      'review_id': 'b3IVizd6WsfctbMJpRbmEA',
      'stars': 5,
      'text': 'Love the food.. We order thali last time, my god there was so much to eat.. Every bit of dish tastes heavenly..',
      'useful': 0,
      'user_id': '_a3wsODfjspKwa5hSPGxqQ'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2012-07-23',
      'funny': 0,
      'review_id': 'Jx0BjTJwT_5v3Nskm--oHg',
      'stars': 3,
      'text': "My mom took me here today. I've had random dosas here and there but am not much of an expert. Per my mom, the portions were excellent, the sauces/dips tasted good and were plentiful, and the dosa itself was very good. I thought it tasted quite yummy but spicy (my mom says that's the point!), and was fairly filling. Around $8 for a masala dosa, which like I said is a good sized meal.\n\nService was very prompt, and the lady serving us seemed pleasant. The restaurant was fairly full at lunchtime. You order at your table but pay at the counter.",
      'useful': 0,
      'user_id': 'Mxw0lh2sAXrnidgvA9QG4g'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2017-01-26',
      'funny': 0,
      'review_id': 't0va5YUHk53bR1KMaquL0g',
      'stars': 4,
      'text': "Came here last Saturday with my dad for some dosa.  We were pleasantly surprised with the texture of breads, dosa and the complex flavours of the bateta and chutneys.  The masala tea was also very good.\n\nThe channa could use a little bit of work though.\n\nI'll be back.",
      'useful': 0,
      'user_id': 'KJchQAfTQscgnQvDV38QIw'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2017-07-22',
      'funny': 0,
      'review_id': 'ZNjx--MiMZHKjyBs_Qp0_g',
      'stars': 1,
      'text': "Are you kidding? I waited in line for 45 minutes when they said it was going to be 15! We got in and turns out there were no potatoes. I mean I can't believe their service! Worst part is the management though. This isn't the first time this has happened looking at other reviews. If there was a way this would be getting a 0 star!!!",
      'useful': 0,
      'user_id': 'I5o9rHxGwIGc75aWpxxGzA'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2014-05-24',
      'funny': 0,
      'review_id': 'kP-zN7EuHdvuseLA8eySxA',
      'stars': 5,
      'text': 'Just had a great meal at the newly renovated restaurant. Dark wood and comfortable banquets. The food was excellent as always. Idli were fluffy and the vada were nice and crispy. The cauliflower gravy that comes with paratha was delicious and the service was prompt and pleasant. The best South Indian in the GTA!',
      'useful': 0,
      'user_id': 'Hkh7WyEuyUO4IXUz8TIygQ'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2014-05-24',
      'funny': 0,
      'review_id': 'Tw4pg5YmhcFCLFJxR7MPlg',
      'stars': 5,
      'text': "Delicious inexpensive South Indian food. I had the cauliflower dosa, not often available. Accompanied by sambar and 3 chutneys it was delectably filling for only $8. The new renovation has made dining in very pleasant. If you're a fan of South Indian, don't miss it.",
      'useful': 0,
      'user_id': 'HfkbAm7Z51q7LSOA-mUQuA'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2017-06-26',
      'funny': 0,
      'review_id': 'BAmsz_CslgqEVlNAM9QX3g',
      'stars': 1,
      'text': 'Check your bill! The restaurant adds 10% gratuity without clear disclosure. I added a 20% tip on top. In effect, I paid 30+% for service. And, adding insult to injury, this happened to be at the weeekend brunch buffet with minimal table service. The non-disclosure is deceitful.',
      'useful': 0,
      'user_id': 'jll5IjpfWOknurqUYxtsiA'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2016-09-25',
      'funny': 0,
      'review_id': 'WvFo4yScWd6G1sziajK1ew',
      'stars': 4,
      'text': 'Reasonably good food but again the downside with this place, just like most other Indian restaurants is the service. The servers are unsmiling but not as bad as in most other Indian places. Depending on the time of the day, be prepared to endure some heavily irritating and ill mannered crowd. The food is however quite delicious and very fresh. For this reason I will keep on going back.',
      'useful': 0,
      'user_id': 'QWzMD9MDZH2RgnvOpoKTgA'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2016-03-18',
      'funny': 0,
      'review_id': 'SmZqUEQUmSt2pp-MBFIl_w',
      'stars': 2,
      'text': "Poor service. Watered down sambhar! Cold idly and poori!! I have been coming here for past 2 years. There was never once when I had received good service. The server working there seems like he hates his job. He's been working there for as long as I dined there.\n\nThe dosa is average. However! The idly and poori always arrives at our table cold. The poori is so hard that it's barely edible. 50 % of the times, you will get good sambhar. But 50 % of the times you will get sambhar that is watered down and looks like water too.\n\nI really hope they improve their service and food. Don't serve the stale and hard poori to your customers!",
      'useful': 0,
      'user_id': 'E1hLHq6LZ6AoSJzW-oT4FQ'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2013-03-29',
      'funny': 0,
      'review_id': '0ctYpkSOWxRYnnnA7N7akQ',
      'stars': 1,
      'text': "I will never return to this place! They made us wait for 20 minutes even though there were rows of empty tables. Then when seated, again another 15 minute wait and I had to ask to be served. We had a pretty simple order-2 people, 4 items. They managed to mess it up. Food took forever to come and one item was missing. When I asked for the item the wait staff denied I had ordered it-their computer clearly confirmed otherwise. The manager came by and said 'sorry we can't handle the crowd.' Um...you're running a restaurant with a certain number of seats. What are you doing if you can't handle the customers? And to boot-they offered us a free dessert (which we didn't want) but didn't take a penny off the bill. \nThe food was mediocre. Stay far far away from this place. I'm sure there are better places for a dosa craving.",
      'useful': 1,
      'user_id': 'qO_WUTGQKRiIpyloaGoXDw'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2016-10-09',
      'funny': 0,
      'review_id': 'c3cV08FgOBsD13MDrSNr-Q',
      'stars': 3,
      'text': 'We went to this location during the dinner hour. The service was good and the servers are friendly. \n\nWe ordered a dosa and I went with South Indian thali as its very different than Punjabi thali. The food was average and good for the price and quantity point of view. \n\nThe only thing which threw away the stars was; while were eating the dosa, at the end we found a rubber band piece in the potato masala. I told the owner when I went to pay and he was very apologetic and stated we should have told him earlier and he would have made another one for us.',
      'useful': 1,
      'user_id': 'jnB_saJqNfOmVoCWquhAzg'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2016-07-01',
      'funny': 0,
      'review_id': 'VYyGjHN0Z98K8szZ-JlG_Q',
      'stars': 2,
      'text': "I have been here thrice...not because I am a fan...but because this place serves decent south Indian food...\nFellow yelpers who have ever tried at their outlet in Connaught Place, New Delhi...will call it below average but you can't expect same standard everywhere! Can you?\n\nService Wise: It's slow and when the restaurant is busy...expect slower service. I went here on this Ugadi. It was packed and they forgot my order. Two gentlemen on my table finished their thali and I got my tomato uttapam when they were done. \n\nThey have good servers...warm and soft spoken...but expect slow service!\n\nFood: Decent/ Average/ OK..Thalis are ok...so it the idli/wada combo and Uttapams. Its not at all an expensive place\n\nAmbiance: Please don't expect much",
      'useful': 0,
      'user_id': 'QAkz4WcqCjVmq6UGuZD49Q'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2016-09-11',
      'funny': 0,
      'review_id': 'j8f03XfX3AN8kXWJcdsI9A',
      'stars': 1,
      'text': 'We went here on Saturday evening and I wanted to order Kara dosa which they did not have. We ordered mix veg dosa and tomato onion uttapam and requested they make it in ghee which they refused.We had previously come here and they had made our dosas in ghee.The sambhar was cold and when we asked for another one that was cold too.Worst sarvanna bhavan in the world.I have been to Dubai and London locations which are much better than here and have more variety of dosas.',
      'useful': 0,
      'user_id': 'P6E338on5nClq_tAgeaTyw'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2013-06-02',
      'funny': 0,
      'review_id': 'i35liAtJ6XY1LG7ARWtQig',
      'stars': 4,
      'text': "Went for Saturday dinner with 4 ppl.  Very friendly and knowledgeable staff.  Extremely clean!!!  Love the hand washing station!!!  Great service and fantastic food, however only gave 4 out of 5 stars for two reasons:  (1) waiter forgot to bring our lassi (yet it was charged on the bill and we had to ask for it to be removed.. oh well, not too big a deal, it was an accident), and (2) the timing of bringing our dishes.  We ordered one appetizer to share, and four entrees.  One entree came first, several minutes later the appetizer came, and then several minutes later second entree, and several several minutes later the third entree, then much later the last entree came.  It wasn't ideal for us to be eating one at a time, lol.  This is a great restaurant to try if those two things don't bug you :)",
      'useful': 0,
      'user_id': 'tuEmVbQfHXG5BmD7edTLNA'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2017-05-21',
      'funny': 0,
      'review_id': '06cpVHnUQP6Ewusn88TlmQ',
      'stars': 5,
      'text': 'Consistent good quality, flavors and fast service. My favorite South Indian food option in GTA.',
      'useful': 0,
      'user_id': 't1GC374dOfL9FzuxBqYrzA'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2015-12-29',
      'funny': 0,
      'review_id': 'd8MUpsiXZaNJnxxNkR3Oig',
      'stars': 4,
      'text': 'Tasty food , nice clean place . prices little bit high for dosas but similar to some others in the area. Try some of the specialty dosas.',
      'useful': 0,
      'user_id': 'AsiS6Ho8XR3OjEh2XLnauA'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2013-05-27',
      'funny': 0,
      'review_id': 'xN20-TC9cunuQyzXlXraMw',
      'stars': 5,
      'text': 'Great place if you want to taste real South Indian food. Great customer service. The food was tasty, specially their Special Meal with different curries and rice. So far I have had very experience with this place after several visits.',
      'useful': 0,
      'user_id': 'ipGQ9MZC_7fFDeJmv35J1Q'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2016-09-02',
      'funny': 0,
      'review_id': 'KuQaicmFpzBvGPvBQ0RbmA',
      'stars': 4,
      'text': "I've been to both the Scarborough and Mississauga locations and I prefer the Mississauga one because of cleanliness and also they have washroom and hand-washing area tucked away in the corner. \n\nThis time I ordered the paper dosa, sweet lassi and the special madras coffee.  The dosa and lassi were great.  The coffee was awesome! I'm not a coffee drinker but I liked that fact that it wasn't strong (they can make it strong if you want it that way) and I asked for sugar on the side. It comes in a cute little stainless cup in a small bowl and it has a lot of froth on top.  All in all a tasty meal.  At the end (where the cashier sits) I had some of the saunf (fennel) with candy...its good for digestion and its nice to eat something sweet (and free) at the end of the meal.",
      'useful': 0,
      'user_id': '-yqTLzfd-L9war_Mm2s8Eg'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2015-05-13',
      'funny': 0,
      'review_id': 'ISh7m3FGLBFFsa_Iz-nOSQ',
      'stars': 4,
      'text': 'Saravana Bhawan has regained its touch after the last review though the Manager can certainly work on his attitude.',
      'useful': 0,
      'user_id': 'icqY1lAj3NvhLXkMEjdeRg'},
     {'business_id': 'TFDT0ueZ3b3uyyMzhJFHTA',
      'cool': 0,
      'date': '2013-08-18',
      'funny': 0,
      'review_id': 'ZuPJ7IPijglLoVd3X7o_4w',
      'stars': 1,
      'text': 'Today I went to sarvana bhava with my family at scarborough location this is one of the worst sevice I got I went with my infant baby (I had requested a seat in corner for my baby but the waiter was very rude he was more interested in accommodating other guest though place was relatively empty) and the service was pathetic. waited half an hour for tandoori paratha after finishing fist one and which never cameI wont recommend this place for anyone',
      'useful': 0,
      'user_id': 'ybSBMfvtIRXSlkOZ6sinaQ'},
     {'business_id': 'ShnXvNkJKsDIQaGJeM0L6w',
      'cool': 0,
      'date': '2013-12-24',
      'funny': 2,
      'review_id': 'ksAP7oFm2Eemm2WMyiXuWA',
      'stars': 5,
      'text': "It's inside the mall and the food is awesome! Probably the best authentic Chinese dinner you can have on Christmas Eve in North Carolina. Just don't peek in the kitchen. You're all good.",
      'useful': 1,
      'user_id': 'hj-o90h78ZHmxQ_VHC3M_Q'},
     {'business_id': 'ShnXvNkJKsDIQaGJeM0L6w',
      'cool': 0,
      'date': '2015-05-13',
      'funny': 0,
      'review_id': '76dFFvKoMDtfqAsu8O9V7A',
      'stars': 5,
      'text': 'A diamond in the rough... I get off the highway just to grab lunch from here. Remind me of the authentic Chinese take outs in NYC',
      'useful': 0,
      'user_id': '_NYfQjo7y9VPgUsJh3ch5Q'},
     {'business_id': 'ShnXvNkJKsDIQaGJeM0L6w',
      'cool': 0,
      'date': '2013-04-20',
      'funny': 2,
      'review_id': 'thIBUH_46y46e7jYtCR6Aw',
      'stars': 4,
      'text': 'It helps to have someone understand the language while in the Hong Kong BBQ, but they do have english menus that you can use on the counter.  It is pretty cheap for some good food.  Just ignore the pig hanging from the ceiling if you find that gross and also ignore the food rating as I survived my visit and I am sure you will yours.  Just make sure to wipe down your silverware when you get them.\n\nPersonally I really liked the BBQ pork which you can see hanging with all the meat in little strips; which, they cut up and serve with rice is really good.',
      'useful': 0,
      'user_id': 'kwyIPeGyXFpYVTJ5SoAuGQ'},
     {'business_id': 'ShnXvNkJKsDIQaGJeM0L6w',
      'cool': 1,
      'date': '2014-10-31',
      'funny': 4,
      'review_id': 'WWF2O94KLFL18o_fYceOHg',
      'stars': 4,
      'text': 'This is as authentic as you\'re going to get.\n\nI\'ll be straight up when I say that places like this can make me nervous.  The mall itself is sketchy...and it\'s in a sketchy part of town.  I make it this way every few weeks for some bánh mì at Le\'s, but came on a day that Le\'s was closed.  Dim sum?...or the place with ducks hanging in the window?  We opted for the ducks in the window and chose Hong Kong BBQ.\n\nWhen you enter Hong Kong BBQ...you can tell there\'s going to be a communication issue.  There is literally...not a single thing resembling "American" in this restaurant.  I knew I was going to like this place.  I did my best not to look back at the kitchen.  We all know what that is going to look like...and I\'d rather play dumb.  Dangit!  I looked.\n\nThe service was quick and the food was great!  We were even entertained with Chinese soccer on their small TV.\n\nI\'d recommend coming here if you\'re an adventurous person...otherwise...you\'ll never make it as far as the door...much less the table.',
      'useful': 0,
      'user_id': 'cx4-WfPgm1Jxurtd83I83g'},
     {'business_id': 'ShnXvNkJKsDIQaGJeM0L6w',
      'cool': 0,
      'date': '2014-03-30',
      'funny': 0,
      'review_id': 'DJJkwGxssiLmKH07Yal7iw',
      'stars': 4,
      'text': 'Pretty run of the mill mom & pop shop, roast pork & bbq pork is sold at $9 a lb. and what can I say the bbq is pretty on par with what I can get in Boston both crispy and flavorful respectfully. Great to know I can find some comfort food in Charlotte.',
      'useful': 1,
      'user_id': 'cZmPdFr5kBaJ-XGgt-2ARw'},
     {'business_id': 'ShnXvNkJKsDIQaGJeM0L6w',
      'cool': 0,
      'date': '2015-09-30',
      'funny': 0,
      'review_id': 'lHg3sPq5hFNM7IJAaJY7Bw',
      'stars': 3,
      'text': "Missing Hong Kong BBQ... found this on Yelp and was excited to try it.\n\nBut, the BBQ pork was extremely dry... was disappointed. However, the service was good and they had the ginger/green onion sauce which is a plus as most places either wouldn't have it or would charge you extra for it.",
      'useful': 1,
      'user_id': 'KLVRBakCWcDdUTcb3UMsZA'},
     {'business_id': 'ShnXvNkJKsDIQaGJeM0L6w',
      'cool': 0,
      'date': '2015-07-13',
      'funny': 0,
      'review_id': 'uM10SUTEKhukjqSUDQDKAw',
      'stars': 4,
      'text': "This is definitely a hole in the wall place. I was actually picking up banh mi sandwiches and saw this place. The duck and pork were prepared as well as the places I go to in ATL. I will definitely be stopping by this place again when I'm in the area.",
      'useful': 0,
      'user_id': 'cmCNYsCybXczlSL9O7uiRQ'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2005-03-04',
      'funny': 0,
      'review_id': '8AIs-KQ8JmSv3QCNRCikTQ',
      'stars': 4,
      'text': 'Incredible asian fusion, and an impressive sake list, if you like that sort of thing. Mmmmm... coconut-crusted mahi mahi.... Just start ordering and keep going until you are full.  But save room for the dessert sampler plate.',
      'useful': 1,
      'user_id': 'ElAQK5tbY7GkmoJHMqhDtQ'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2007-05-24',
      'funny': 0,
      'review_id': 'cKuPTIJjA78dATvnrcgiSw',
      'stars': 3,
      'text': 'Delicious food. \nGood service. \nBUT-- if you are allergic to shellfish BEWARE-- they do not label all the dishes that contain shellfish.  You might order, say, the tomato bisque or the black cod, only to find out later (the hard way) that they contain such items as shrimp paste or oyster sauce.',
      'useful': 2,
      'user_id': 'nkCQg16DbT_1P3NqVPjoig'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2009-08-11',
      'funny': 0,
      'review_id': 'm29WL4DJVCiwq9bGazxdEw',
      'stars': 4,
      'text': "I really love Kushi Bar, it's the kind of place that I wouldn't be embarrassed to take my old friends from Boston or New York to. The sweet potato fries are fantastic and the dipping sauce just adds fuel to my firey addiction to sriracha. For vegetarians, I recommend the house salad (kumquats!), kimchi tofu, mushrooms, curry veggie and rice bowl and wash it all down with the gin rickey.",
      'useful': 1,
      'user_id': 'CbwqJgXyoPFHCGkRFWt5Fg'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2008-01-07',
      'funny': 0,
      'review_id': '1ypC-siSWIKS0XcGv8s34Q',
      'stars': 5,
      'text': "I'm a little wary of Asian fusion restaurants, and I wasn't sure quite what to expect from Muramoto, but my dining partners (a chef and a personal chef) and I were thoroughly pleased with our light dinner on Saturday evening. \n\nWe had two rolls, one eel and avocado and another tuna, avocado and some other stuff - both were fantastic. After seeing someone else with it, we also ordered the calamari with slaw and chili paste and, by far the best dish of the evening, the black miso seared cod. I've been remembering that cod all weekend, it was so good. There were a lot of other things on the menu that I wanted to try as well. Maybe another time.\n\nActually, everything was really, really good. This was a place I would want to revisit if I lived in Madison. Although we didn't order anything off it, they did have a specials menu with at least three items, and the very extensive sake menu. \n\nContrary to what one of the other reviewers said, I think it would be really hard for the average person to spend $100 alone unless you just stuffed yourself silly. We ordered five dishes total and two small bottles of sparking sake to share among three women and the bill was $85. Actually I couldn't quite believe how reasonable the bill was when it came. \n\nBTW, make reservations. We weren't in a hurry, so we went and had a drink and came back, but the dining area is quite small, so if you have more than two people you'll definitely want to call ahead on a weekend evening.",
      'useful': 2,
      'user_id': 'itmb9uAolzQnmEBDlV24qQ'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2006-09-01',
      'funny': 0,
      'review_id': 'FhVvlurmgtZy6kkQO2xfEg',
      'stars': 5,
      'text': "Muramoto is a paradise if you are a sake lover. Their collection is enormous and I believed the owner claimed Muramoto has the best collection in midwest. If you are a sake lover and haven't been there, what  are you waiting for? and if you don't usually drink sake, you should still try the sake out (especially the ones list under fragrant) and you might change your view about sake forever.\n\nFor the food Muramoto is Asian fusion in tapas style. Although each plate might be only around $10, most of them are not filling (rolls are the exceptions) and at the end combining your sake bill the check might cost as much as $100 per person.",
      'useful': 0,
      'user_id': '-N1rDtpUIHpjrEujRyzVKA'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2008-02-28',
      'funny': 0,
      'review_id': '5bGJn27l6uqgpdsWV02s-A',
      'stars': 4,
      'text': "[NOTE:  this review is about Restaurant Muramoto, which used to be in the location currently occupied by Kushi Bar, but has since moved down the street 2 blocks.]\n\nIn a handful of visits since it opened a few years ago, I've found Muramoto to have generally outstanding food and often less-than-ideal service.  IMO, the former more than makes up for the latter, which is at its worst just a bit slow and unconcerned.  (Once, for example, we were two dishes into our meal before our first round of drinks arrived.)\n\nBut, ahh, the food.  This is probably the most creative food in Madison.  I describe it as pan-asian fusion in a tapas form factor.  There is some sushi, and some is even pretty good (I'm thinking of the signature duck mango avocado rolls).  That said, it's a mistake to think of Muramoto as a sushi place, though (there is a Muramoto sushi restaurant in the hilldale area ) -- my favorite dishes here have included the soy cod, a lamb curry, and a few amazing salads.  In my experience, the oft-changing menu contains little that isn't great -- an omnivore with an open-minded palate could safely close their eyes and just start pointing.\n\nCome with an appetite and prepare to spend a bit of change (dinner for 2 with drinks and dessert will exceed $100), and avoid the tables in the front near the door on cold days.",
      'useful': 0,
      'user_id': 'PkDnxdH_L9ADnsC-oGfSlQ'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2008-06-10',
      'funny': 0,
      'review_id': 'av2d1ll29hrtUYhZYJJN0g',
      'stars': 5,
      'text': 'Like Leslie S. said Muramoto has a somewhat limited menu but what is on that menu is outstanding.  I adore places that encourage tapas style/family dining and Muramoto\'s portions are what some may describe as small.  (Two to share two entrees is just enough food.) Additionally I appreciate that food is brought out as it is done being prepared and does not suffer death under a heat lamp.\nFor drinks I had Lychee Liquor and grapefruit juice which was mango-ey and so tasty. I\'d consider coming back just for that concoction alone.  We ordered the Tuna with Wasabi Flying Roe, Shrimp in Spicy Coconut Sauce, and the Pan Seared Scallop. (which was the favorite dish)  All of our food embodied unique flavors, and were demonstrations of  "Asian fusion" done right.  Our server was incredibly helpful with questions, and prompt without seeming as if she was trying to rush us out the door.  \nAlso prices were not that bad.  For 2 entrees, 1 roll, and drinks we exited the restaurant only about $60-ish poorer (including tip). I am definitely returning. I wish the restaurant was bigger and took reservations because I\'d rent out the whole place for graduation.',
      'useful': 1,
      'user_id': '5oM7gXmeX7s6T1_KlL30yQ'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2009-08-22',
      'funny': 0,
      'review_id': 'kAFyjie8UClchNTpIsM4TQ',
      'stars': 1,
      'text': 'Kushi bar will close at the end of August (may already be closed)\nhttp://77square.com/food/restaurants/story_462727',
      'useful': 0,
      'user_id': 'fJ_qSZH_akvWzHUPMqab1g'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2009-06-15',
      'funny': 0,
      'review_id': 'jXarz-fPsDFLSfwHym1gzQ',
      'stars': 4,
      'text': "Wandered in here looking for the restaurant and they told me to go down the street.  They have some food, but it's not the same as the full restaurant.  However, their shochu drinks are more extensive at this location, so I had a shochu shiso mojito.  It was okay--not my favorite, but not bad either.  Maybe I should have tried a drink that wasn't trying to be another drink.  \n\nAnyway, the place is really nice, the bartenders are really nice.  Well worth a visit.",
      'useful': 0,
      'user_id': 'seAX967Wk8qiVXOYjbregg'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2006-11-06',
      'funny': 0,
      'review_id': 'FQl1geZHUfSeUTwbXZRQ_w',
      'stars': 4,
      'text': "best meal i had since coming to Madison.(Been here 2 months) I would say that this place serves fusion Japanese Food. My recommendation is that this is the place to go with friends so that you can share dishes and try more stuff. Food quality is top notch and each dish really offers you a new experience. Only gripe about the place is that portions are SMALL and it's relatively pricey. expect to spend 20 plus per pax. A must try is their dessert sampler! Definitely going back : )",
      'useful': 0,
      'user_id': 'Kx9x_dAOMcQYiUu5-uUTdA'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2005-09-19',
      'funny': 0,
      'review_id': 'LFTVAZQ7jOQbF07x1bGpdA',
      'stars': 5,
      'text': "Although I regularly miss Dog Eat Dog around noon every week day, I'm pleased that Muramoto replaced it and gave us a place to blow our cash on Sunday nights. The chef's tasting is key. And use the bathroom with the sink that makes you want to get out your GI Joes and Barbies and play in the bubbles all night.",
      'useful': 0,
      'user_id': 'lV0TGQufw-draFAWiawdSQ'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2008-11-16',
      'funny': 0,
      'review_id': 'HyVFtiiJqbPKNQZ1QmZRBA',
      'stars': 3,
      'text': 'Please note: Pricey, gourmet Restaurant Muramoto has moved down the street, and Kushi Muramoto has become a sort of "Izakaya" style place. Think Japanese Tapas. Reviews prior to November at this location should be thought as applying to it\'s new location, just down the street. \n\nI was not impressed by the food at Kushi. Your order will come out quickly, because almost everything is fried, fried, fried. The exception is the Donburi -- something yummy on rice. Our Eel Donburi was solid, and decently priced for $8... but if you\'re a Midwesterner with a healthy appetite like me, don\'t think this will make a meal. \n\nI also have a complaint about the drinks. My Cosmo was delicious -- and also standard Cosmo size and price ($7). However, because they make all their mixed drinks here with sake instead of vodka, what you\'re getting really packs much less of a punch than what you\'d get elsewhere.\n\nThe pluses are this: The atmosphere hasn\'t change since this was the pricey Muramoto, So it\'s a groovy low-lighting place to hang. Also, it\'s really reasonably priced. $4 drafts. $1.50-$2 things on a stick, $5.50-$8 for Donburi. So you can go out and feel like you\'re getting the "Muramoto Experience" without bleeding yourself dry.\n\nHowever, when they said "fried Snickers are our only dessert and they\'re only 2 for a dollar," I was so sick of panko fried tidbits, I said "no" without a moment or regret.',
      'useful': 0,
      'user_id': 'JWgti_adPZywG9SKUbnB-w'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 1,
      'date': '2008-06-18',
      'funny': 2,
      'review_id': '1OWxam36s3CdkBLB1nozxQ',
      'stars': 1,
      'text': 'Oh my did this place not live up to the hype.  It was heinous how sloppy our rolls came out, my date\'s food came out a full 20 minutes after my own, and the black cod was incredibly salty.  The place looked "hip" but that\'s about as cool as it got.   If you are looking to make a fashion statement then by all means dress yourself up and head to Muramoto, if you are looking to enjoy your meal (be it tapas, sushi, real japanese or some fusion) look elsewhere.',
      'useful': 1,
      'user_id': 'BnLFYUzUnPYgDhazfQua-w'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 1,
      'date': '2008-09-08',
      'funny': 3,
      'review_id': 'Toh-YLCULBVXyWsgn1_pPg',
      'stars': 4,
      'text': "This place accommodated our party of 8 well-sauced gentlemen nicely. Our waitress was friendly and extremely helpful with our menu choices. Her recommendation of one roll and another fairly small plate was the perfect amount of food for us, as we had more beers to consume apres-dinner. \n\nI got the impression this is one of the nicer restaurants in Madison, because everyone was dressed up. Except us of course. In Chicago this would be a nicer-than-average sushi restaurant, but it's not on par with Japonais. Think more along the lines of Sushi Ra if the food at Sushi Ra were a lot better. Sort of a modern, almost trendy thing going on, with a pronounced wet bar. \n\nThe food was good, but to be perfectly honest, I have no business reviewing a sushi restaurant since I usually don't get any of the real shit. If it's crispy and fried or if the flavor of the fish is masked by jalepeno, avocado, or cream cheese, that's just fine with me. \n\nSo yeah, sorry I made you read this review for nothing.",
      'useful': 2,
      'user_id': 'IHFVljMsPhMc2-lW3UMNuw'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2007-09-22',
      'funny': 0,
      'review_id': 'ose5bGJUQo86HS2yHQjiXA',
      'stars': 5,
      'text': "ASIAN FUSION WITH INCREDIBLE SAKE MENU.\n\nAll info I've seen says 'tapas style', but I will tell you - these plates are not too small, and quite filling. Their other spot cross town is an incredible sushi restaurant with everything under the sun. Here, only have a few rolls on the menu (and they are incredible), but that is not why you come. Come for the tasty, hot dishes, and for the sake flights.\n\nEvery so often, you eat a meal that is soooo good, so perfect in almost every way, and the tastes are so unforgettable, that you want to get right up then and there, and do what I affectionately call, The Food Dance. This is food that makes your eyes roll back into your head with every bite, and leaves you wondering where the hell you have been eating all these years. This was, hands down, my saving grace in Madison. Presentation of the food is spectacular. Eating alone? Sit at the bar and stare at um-teen bottles of Japanese sake. Eating with a group? Even better, as the sushi rolls they do have there (soft-shell crab or avocado and mango roll w/ thin-sliced duck breast on top, drizzled with a duck reduction sauce, or a few others) are HUGE. Don't think California sushi, as each roll which costs $10 has about ten large slices, and you will find it hard to eat an entire one by yourself. These rolls would easily be double the price if this was in SF.  Look for Michael, the tall, bald-shaven, server/barman that knows everything about the menu, and can recommend many good sakes on the list. Try the sake flight (3 glasses) for a nice accompaniment to any meal. Now for the best part........you want small slices of some of the tastiest beef anywhere? Get the hangar steak. Not a large portion (meaning not a whole steak) but you get several slices of this beef, drizzled with a white miso-bleu cheese sauce that is to DIE FOR. Not a fan of bleu cheese? No worries, as the taste is made subtle by the combining with miso. But please, do yourself a favor - ORDER THIS DISH! Not a big beef fan? That's fine, as the Grilled Miso Black Cod is their signature dish. My suggestion? Start with some sake, and order yourself one sushi roll, and one hot dish. Take a few people, and everyone order a few different things, and share. And won't you be happy when you get out of there, after all this tasty food, and still you are not over the per diem. And if you are, maybe you shouldn't have had that third drink. Be advised - this place has a little wooden sign in the window, and it took me forever to find it. Do me one better, and take the address WITH YOU when you go!",
      'useful': 0,
      'user_id': 'gh_RvEPwTMj5OXp8j9uePw'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2005-05-03',
      'funny': 0,
      'review_id': 'VgPnYAOJdFF_vD0GaFUNag',
      'stars': 5,
      'text': "Who says the Midwest can't have tasty seafood?  Dim and intimate, Muramoto is easily one of my favorites places in Madison.  A varied menu with some fantastic Asian-inspired cuisine, this ain't your grandmother's Friday night fish fry.\n\nUpdate:  swing by on Sunday night to try the blind chef's tasting.  4 courses ending in dessert for $30.  Never disappointing.",
      'useful': 1,
      'user_id': 'HLaSqQMDVvlcFPGJL_kGCA'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2009-05-13',
      'funny': 0,
      'review_id': 'xsdVuL20EEzRR_MavYAIVg',
      'stars': 3,
      'text': 'I had my first visit since they turned into a tapas bar style spot. \nGot to say, they\'re still serving the best "imitation" of the real stuff - doesn\'t really fly by me. \n\nBut if you like that kind of thing, I suppose it\'s ok. Depending on the bartenders, some drinks are so underwhelming that it made me feel unnecessarily girly.  Oh well. \n\nOn the positive note, I think it is fun to try out many small plates. So 3 stars for the fun factors. The hype doesn\'t match the food, though.',
      'useful': 1,
      'user_id': 'fuoYjsuU7qy1RlCMEZ0xfA'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2007-09-09',
      'funny': 0,
      'review_id': 'N6uMLOhrhQUmkwU9Iv97NQ',
      'stars': 5,
      'text': "I am infatuated with my new love, Muramoto. At first, I was highly doubtful the state of Wisconsin could feature decent sushi but boy oh boy was I wrong. Mad props incredibly for the fact that they feature a deeeelicious Philly role with salmon and cream cheese. Heavenly! I enjoy the fact that they serve their edamame with big grains of salt to give it an extra kick. And last but certainly not least, I could drink a million of the Russian teas (a smooth mixture of kahlua and green tea vodka, which I have never tried or seen before this restaurant introduced me to it!) \n\nI end my Valentine to Muramoto with a shout out to its modern decor that made me feel like I was in big city again, and that it smells like fresh sushi on the inside, since most Japanese restaurants don't smell like anything! This is the first place of many in Madison that I have graced with 5 stars because I was that impressed!",
      'useful': 1,
      'user_id': 'esaDbON8NXJC4PWt_CheUg'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2006-12-12',
      'funny': 0,
      'review_id': 'XRzpHvkivRV9yHdjiNJ8ng',
      'stars': 4,
      'text': "There were only a handful of rolls available at $10 each that are very fulfilling.  I would recommend the soft shell crab roll.  For some reason the Duck, Mango and Avocado roll made my stomach really queasy.  I didn't try the sake because it was a Monday night, but I would come back for the oyster shot!  A mixture of oyster, raw quail egg and spices it was something very unique!  I had two before my meal.  Slurrrrrp!",
      'useful': 0,
      'user_id': 'MPozTU7MRVKYISO5prvzdg'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2008-04-13',
      'funny': 0,
      'review_id': 'WRvtog3sUIYxcYdJO-srYw',
      'stars': 5,
      'text': "Yep, I did it- 5 stars.\n\nAfter a few missed attempts at going to muramoto, we finally made it in.\n\nWe tried some sake to start it off, and the Wakatake was superb- at the suggestion of the waiter. Next were the oyster/quail egg shots. What a nice melding of flavors. This was followed by the corn/lemongrass chowder- big disappointment, but I'll let it slide.\n\nOn to the main courses- started with the tuna/mayo roll. This was a quite a big portion and really was a great sushi roll for those that like the bells and whistles. The hangar steak was tasty. The bleu cheese sauce was just perfect and made the steak that much better. Could have used just a tad more crust on the meat. Finally, we had the miso black cod. This is how fish should taste. It melted in your mouth, and was salty, but in a great way. \n\nTIP- go early or late (after 9) because they don't take reservations.",
      'useful': 0,
      'user_id': 'q91tQNboB-0HWtM8cXRjdg'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2008-11-10',
      'funny': 0,
      'review_id': 'RItK2tnwM2QMw1sg9T7hGQ',
      'stars': 5,
      'text': "Yum!  Muramoto is really the best place in Madison for Sushi.  Period.  The service is always great, not just that they are prompt and attentive, but they know the menu and the drink menu and can make intelligent recommendations that I am usually pleased with.  Muramoto's creative rolls are SO delicious, they are all different tasting and the only comparison to their taste comes from good sushi places in Chicago.  I get take out here at least once a week and dine here about once or twice a month.  If you want fantastic sushi at really a quite reasonable price, this is your place!",
      'useful': 0,
      'user_id': 'jtD2TF2Pr-bwT6RwPCAE4g'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2009-06-03',
      'funny': 0,
      'review_id': 'NDJPlFn9agAkfyo7mMB4Vw',
      'stars': 2,
      'text': "Just to be clear, this is a review of the new 'tapas-style' Muramoto.  Ok, with that out of the way....\n\nThe Space:  Just like the old Muramoto that was there, its a great space and feels like a 'big city' restaurant.  Dark, calm, cool, and the lighting is superb.  Great date place and it just feels sexy.\n\nThe Food:  I wanted to like it, I really did.  We ordered a wide variety of small plates and not ONE of them was good.  In fact, even they even managed to blow bacon-wrapped figs stuffed with blue cheese.  Now thats not easy to screw up.  The plates ranged from merely ok to bad (octopus 'balls' were like fried croquettes with elastic bands in the middle).  \n\nThe Drinks:  Drinks were good.  Good pours, good selection and the Japanese booze was interesting.  \n\nThe Price:  Here's why Kushi Bar received an extra star.  The place is cheap for what it is, especially the drinks.\n\nSo all in all, go for drinks, ambiance and to impress a date.  Maybe even order a small plate or two but don't expect it to taste as gourmet as it sounds on the menu...",
      'useful': 1,
      'user_id': '66e_wFUf4GsDT2DTZ3u-Xg'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 1,
      'date': '2008-08-12',
      'funny': 0,
      'review_id': 'csePl2jSG6teWWuGVys7cA',
      'stars': 5,
      'text': 'Muramoto has moved into its new space and its all good (great space, expanded menu, same great staff)...but, I strangely miss the old space. Maybe some therapy for this reviewer.',
      'useful': 1,
      'user_id': '6vlB2FYf2dGVWte9cViVcg'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2007-04-10',
      'funny': 0,
      'review_id': 'Z4wqE0K0w3lOHMY4jVBqwA',
      'stars': 5,
      'text': "I like this place.  The food and the service is great.\n\nSesame Crusted Big Eye Tuna with tahini sauce - I love it!  The sauce is so nutty and tasty and goes perfectly with the tuna and mixed greens.\n\nRock Crab Croquettes - it's their version of crab cakes but they deep fry the croquettes (4) and they arrive at your table piping hot.  They're delicious! - reminds me of my beloved dungeness crabs back home.   \n\nShrimp and Spicy Coconut Sauce with crushed peanuts - this is pretty good.  It's very similar to Thai food.\n\nHangar Steak with bean salad in bleu cheese & miso dressing - both times I've had it it's been a little tough and very chewy.  The bean salad (scattered around the steak on a long rectangular plate) drenched in the dressing is quite good.\n\nDESSERTS\nWe ordered the Sample Platter.  I'm glad we did because we were able to try some really interesting desserts.  This Platter is perfect for two people.\n\nYuzu (Japanese Citrus Fruit) Panna Cotta - a little more firm than normal panna cotta.  Didn't really taste the citrusy fruit.  But it was very good.  It kind of reminded me of a smoother version of cheesecake without the graham cracker crust.  It also had a drizzle of some berry compote.\n\nStrawberry Spring Roll w/Honey ginger ice cream - I really liked the spring roll.  It was nice and hot and I think there was some sweetened sticky rice in it as well.  I'm a big fan of ginger but for some reason I didn't like the ice cream.\n\nHoney Black Sesame Creme Brulee - this was not my favorite but it was interesting.  The creme wasn't super creamy but had a little texture to it.  The color of the creme was a dark charcoal grey and tasted like smokey sesame.  It's a clever idea but not something I would try again.\n\nKabocha (Japanese pumpkin/squash) Cheesecake - pumpkiny cheesecake with cinnamon or nutmeg with maple syrup lining the plate was my favorite.",
      'useful': 0,
      'user_id': '2yLD3esBneRzDI4Uh26q-A'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2006-10-22',
      'funny': 0,
      'review_id': 'GvvwtTL0KEqO9l8qNIgQ9w',
      'stars': 5,
      'text': 'I will never look at sushi the same way again!  I am a huge sushi addict, and this is above and beyond the best place to get your "fix" in Madison.  It even beats the places in DC that I\'ve tried, and they do a cool new take on  rolls.  Everything was fantastic, my date and I ate \'til we were stuffed...then had dessert on top of that (a delicious creme brulee).  At $10 a roll, these are more than worth the price..they are huge and come with their own individual sauces...no wasabi or soy needed.  We had a saki flight...not my thing, but good to try once I guess.  Service was great and I love the atmosphere.  I can\'t wait to go back!',
      'useful': 0,
      'user_id': '5Kbn1FjXmS7Jypz3_ybt_Q'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 1,
      'date': '2009-03-30',
      'funny': 0,
      'review_id': 'Rg9aSqytD0jCMEkvwpdY2g',
      'stars': 4,
      'text': "The food here is amazing.  Kushi is like Japanese tapas - all small plate dishes here.  I went here on a date and enjoyed everything we ordered.  It was pretty deserted on a Tuesday night, but I've seen more people there on the weekend.  The ambiance is loungy - and pretty dark, so it gives off a romantic vibe.  A bit expensive, but worth it.",
      'useful': 1,
      'user_id': '38V9LFE9gbmqdxACJ98F8g'},
     {'business_id': 'AfoBKJuGBJvQhc3By4K9Dg',
      'cool': 0,
      'date': '2008-03-11',
      'funny': 0,
      'review_id': 'pubTy2GEizrk1Es638wgIA',
      'stars': 4,
      'text': "Muramoto is a very good restaurant if you enjoy a limited selection of very good things.  The duck roll is outstanding as are the oyster shots, which are my favorite way to start dinner.\n\nCriticism-wise, there is a bit of a problem with seating in the wintertime, as there is quite a draft coming in from the door if you sit too close to it.  Moreover, you cannot make reservations at Muramoto.  This means of course that you cannot request a table further away from the door if you want to in advance (first come first serve).\n\nService is just fine, ambiance is lovely.  Come with a date, not your kids and definitely not in your red UW sweatshirt.  Dress to impress; this is a spot for grownups only.\n\n!!!!!!UPDATE!!!!!! \nDowntown Muramoto has moved to an new location, which is interestingly just down the road from their old location.  This solves the drafty chill issue, there are LOTS more seats, and you can now make rezzos.  On the downswing, it's missing something of its original, uniquely Madisonian charm.  It now has vaulted ceilings, adequate lighting, and at least one waitress of questionable moral character.  Plus they have an additional location by the Hilldale Mall which is yet another step down.  I think what I'm saying is that Muramoto sold out to the man and is now no longer actually hip or trendy.  But honestly, it's still quite nice and the food is very good.  The new Hilldale location does not have the oyster shots, but they do have a delicious octopus salad which knocks me sockies off.  You might as well keep going.",
      'useful': 0,
      'user_id': '237EL2sEpc-ZUEbo0HUYGQ'},
     {'business_id': 'ykSlMTRgyWzAmhP7ywc9gg',
      'cool': 0,
      'date': '2014-01-04',
      'funny': 1,
      'review_id': 'HrzzQj0KfR_MI68flznBLg',
      'stars': 5,
      'text': "Amazing place downtown Oakville that makes you feel very welcome and has a downtown Toronto feel.  It's open a bit later than other places in Oakville therefore it's great for the night owl crowd. I've been there at least 3 times and there is usually some party going on.  Later in the evening they play music and have the most interesting Japanese Anime or Sumo wrestling on a big screen.  \nThe food is authentic and high quality.  Their rolls are amazing.  \nBut what makes this restaurant such a hit is the personal staff and bartenders who make everyone feel like a regular.  They greet everyone in Japanese.  I'll be back!",
      'useful': 0,
      'user_id': 'tJO7401VdiZw8qlR7W52Bg'},
     {'business_id': 'ykSlMTRgyWzAmhP7ywc9gg',
      'cool': 1,
      'date': '2013-09-23',
      'funny': 0,
      'review_id': 'qIANnEo-oBuDld5xTYEUVg',
      'stars': 2,
      'text': "My GF and I ate at Fin's on Saturday night Sept 21, 2013.  We had the chef's special tasting box which had 9 different items. The honey topped eggplant was quite good but everything else was mundane at best.  There were rubbery little pieces of octopus with much chopped vegetables in a sauce that overwhelmed everything. There was spicy salmon which was not spicy and was overwhelmed by a massive amount of mayo.  The tuna was tasteless. Etc. In addition, the decor is primitive and the place is incredibly loud for a supposedly upscale restaurant. It is simply not a good restaurant and I would not return.",
      'useful': 1,
      'user_id': 'Z4uvXYqvOfVR32VGvNsLWw'},
     {'business_id': 'ykSlMTRgyWzAmhP7ywc9gg',
      'cool': 0,
      'date': '2013-09-25',
      'funny': 0,
      'review_id': 'xkVmcvD1A2qAMG75fC-miQ',
      'stars': 4,
      'text': "What a gem! I'm so happy I wandered into FIn! \n\nSince moving to Oakville I've been searching for as many Toronto-esque restaurants as possible so i can get my fill! I've found a few great places there, but no food (especially Asian) has actually been that great - until Fin! \n\nI went here on a Sunday evening, and was concerned it wouldn't be open, but it was until 10 PM. We were greeted in Japanese and seated immediately by our extremely friendly and attentive server. \n\nFood is a bit on the pricey side, but without there being much competition in terms of Izakaya in Oakville, I wasn't surprised, and it didn't break the bank. \n\nPersonal favorites were the selection of fish for three (see picture, so beautiful!) and the Eastern Style Caesar (who wouldn't love a drink with a crab leg in it!).\n\nOverall i was super impressed with the quality and variety of food, as well as the service. I couldn't recommend Fin enough to those starved for a bit of Asian food in Oakville!\n\nDisclaimer: If you're from Toronto and eat Izakaya all the time, this might be a touch pricey for you!",
      'useful': 1,
      'user_id': 'TUu4SppnJ1S3lQ_jTxMpOw'},
     {'business_id': 'ykSlMTRgyWzAmhP7ywc9gg',
      'cool': 0,
      'date': '2014-12-28',
      'funny': 0,
      'review_id': 'jXEDMeVE-UVzT1EKhrpt_A',
      'stars': 1,
      'text': 'warning. !!!!!!   This is not the same fin as before sold to new owner !   Super dry pull pork rice burger and stale sashimi salad and  gone bad chicken skewers were the 3 dishes ordered.    I gave 15% tip that the owner dared to ask for such food and service, smiled back at the owners and said "see you next time" NOT!',
      'useful': 0,
      'user_id': 'Cm1oxY-7c3lgiLDME21wfA'},
     {'business_id': 'ykSlMTRgyWzAmhP7ywc9gg',
      'cool': 0,
      'date': '2014-08-02',
      'funny': 0,
      'review_id': 'v2iNPLPlTLhkXjTwVnrplg',
      'stars': 2,
      'text': "Way too expensive for what you get. I don't mind paying more for good food but this place was not worth it. Was recommended by a friend after telling her how much I loved Guu but I definitely can't listen to her anymore! \n\nIt's a nice looking place and the service was good as well, but the food comes first and it definitely was lacking. Sashimi was on par with AYCE places, not what you expect when you pay 40 bucks for 16 pieces. \n\nOverall, not pleased with my food experience.",
      'useful': 0,
      'user_id': 'vKwJ1QZd9V4678Er_sBr0Q'},
     {'business_id': 'ykSlMTRgyWzAmhP7ywc9gg',
      'cool': 0,
      'date': '2013-10-04',
      'funny': 0,
      'review_id': 'JaAyQKuuxp1OOD-VQJEmSQ',
      'stars': 4,
      'text': 'Exceeded expectations. Ate early around 6.15 pm so was fairly quiet...2 others. Strangely asked if we had reservations...interesting! \nSauvignon Blanc from Chile was good and can be recommended. Cucumber Addiction tasty, avocado tempura very good, sashimi 3 kinds flawless, fresh raspberry point fresh oysters excellent value at $1.50 each. Shichimi duck salad also very good.  At $76 before HST & tip about right for the quality. Pleasant staff....will go again!',
      'useful': 1,
      'user_id': 'ON7__MUlZbxNn0a4T_09uw'},
     {'business_id': 'ykSlMTRgyWzAmhP7ywc9gg',
      'cool': 0,
      'date': '2013-07-08',
      'funny': 1,
      'review_id': 'u2n01D6o8TG9WLn2XODg6Q',
      'stars': 4,
      'text': 'Fairly new establishment on the Lakeshore strip in downtown Oakville.  A good thing to see with all the For Lease signs that have been cropping up lately.\n\nIf someone could translate what is yelled at patrons once they enter, it would be greatly appreciated.  I\'ll assume it is a greeting.  The friendly staff that helped us explained that the general restaurant experience is that of arriving, eating and then leaving to go enjoy something else, somewhere else.  The Izakaya experience is designed for you to eat, drink, socialize, etc. all under one roof.  Hence the loud and sometimes frantic atmosphere.  Pretty much a tapas menu.\n\nAfter perusing the Sake menu, my wife and I settled on some Sapporo, mine in the bath-sized glass.  It wasn\'t that the Sake menu wasn\'t impressive, but my wife doesn\'t love it, I have limited experience with the Gekkeikan brand and I just wasn\'t in the mood to take a chance on a 3 oz, $15-30 sake that I may or may not like.  Maybe if I\'d have been encouraged or had some sort of crash course I would have considered.\n\nAny way, food:\n\nIn absence of a bread basket, you are given what is called thin spaghetti, coated in salt and flavoured with a spicy curry.  A little strange a different as it is like eating spicy uncooked spaghetti.  \n\nStarted with the Cucumber Addiction - sliced pickled cucumber with sesame seed and oil.  A great clean-tasting starter.  All orders come out fast and furious, but staggered so as not to bung up the table too much.\n\nThe Pork Rib Kakuni was really good.  3 fall off the bone pork ribs that are slow braised and served on mashed potato with a spicy mustard for more flavour.  This was really tasty and a good portion of meat for only 3 pork ribs.\n\nThe Soy Bean Cream Crab Croquette is a mouthful.  Resembling arancini balls, but stuffed with crab, there 3 little guys are filling and have more substance to them than your average crab dish.\n\nThe Tuna Carpaccio is delicious with a savoury mixture coating the fish.  A little tricky to pick up with the chop sticks, as it is chopped rather finely, but worth the effort.\n\nThe Deep Fried Prawns are quite massive and heavy on the garlic, but the spicy sauce make \'em quite a lovely plate and pretty to look at once delivered.\n\nAs the evening progresses, more and more residents of Oakville filter in.  There is a lot of bumping and jockeying for room in this narrow little space.  Conversations can be heard from basically every table.  The fake-chested table were amazed when Barbie\'s shorter sister came in, no less than one month after giving birth and back to her fighting weight.  A quick cat walk, a twirl and they\'re sat beside us.  Yaaaaay.\n\nThis is where service fell apart.  We had one more order coming and we were contemplating another round of drinks and possibly dessert.  But with all the air-kissing going on and the place getting louder and packed, the person taking our orders came back to our table after 15-20 minutes, stating that he\'d forgotten to enter our last dish.  Another 20 minutes later and the special of the night Cod we\'d ordered arrived.  It was incredible and naturally buttery.  Any aggressive thoughts were forgotten instantaneously.\n\n\nFood for Thought:  if you come for the special catches of the day, come early.  4 of the 6 fish specials were sold out by 7:30.  And the Blowtorch Mackerel?  It may be delicious but smells of burnt arm hair.  Maybe its just me and my wacky palette.\n\nHopefully the "trendiness" wears off in the passing months.  The food is quite good and staff are helpful, sincere and very knowledgeable with the menu.  It is the clientelle that I could do without.',
      'useful': 1,
      'user_id': 'FTNaQZ3t0dsVWw1WZUQGFg'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 0,
      'date': '2015-11-06',
      'funny': 1,
      'review_id': 'f-bF8GvXvzjYAdG1cXmW0Q',
      'stars': 4,
      'text': 'My fiance and I really liked this place. Huge step up from Stonepepper Grill. Our server was very attentive and the food was quick and on point. Tried the bruschetta, chicken parm and cajun salmon. Every dish was good and the chicken parm was better than many spots in the area. The one negative was the hostesses attitudes. They seemed stressed and frustrated that they were busy at 7pm in a Friday night which should make them happy, right?',
      'useful': 1,
      'user_id': 'wqrXw4Vs-oW2WNZhWh3KKQ'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 0,
      'date': '2016-05-07',
      'funny': 0,
      'review_id': 'Z7wgXp98wYB57QdRY3HQ3w',
      'stars': 4,
      'text': "Wow. So surprised at the one and two star reviews!  We started with the most tender calamari. Although the marinara sauce was a bit bland, but a touch of salt made it just right. My husband had the veal with peppers and said it was so delicious and tender. The mashed potatoes were perfect. I had the salmon Diablo which was also delicious. Our salad was beautiful! Dressing was served on the salad and it was a nice amount. We ended our delicious meal with a piece of tiramisu. Our server Matt was right on!! Very pleasant and knowledgeable about the menu. Our appetizer, salad and entrees were timed perfectly. I love salad and did not mind that my entree was served while I was still eating it! No problem it let my dinner cool to just the right temp for me to eat it comfortably. \nI wonder sometimes if people just don't appreciate relaxing and taking time to eat a wonderful and beautifully prepared meal.  A wonderful atmosphere. So relaxing. The chairs are super comfortable too!!! We will certainly be back. \nGive it a try.  Don't  always go by the reviews. \nA bottle of Riesling, calamari app, two delicious entrees and dessert for $92! \nWell with it.",
      'useful': 0,
      'user_id': 'GYNnVehQeXjty0xH7-6Fhw'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 0,
      'date': '2015-09-10',
      'funny': 1,
      'review_id': '-k0u6krCe0vkJL-RKlEMtw',
      'stars': 4,
      'text': 'Came here trying a new place. Literally! Just opened two days ago and well worth it! Warmly welcomed at the front by a nice hostess, and we\'re told we could even have a seat in the bar. Since we were interested in the Pirates game, it worked out, they have a few TV\'s in the bar for such purposes. \nWe ordered our drinks; Prosecco & Malbec. Both by the glass. Our server gave us big glasses of water with out drinks so A+ there. \nThe bread came out first and it was awesome. Perfectly warmed, large slices, crunchy crusted Italian bread that tasted like it was made in house. Served with ambient temperature butter (another pet peeve of mine is ice cold butter) so A+\nThe salad came out next and was the first area where we\'d recommend a little change, maybe a little larger for sharing between two. But it was flavorful, cold, and dressed just right. The fresh cracked pepper added to it so overall: B\nNext our entrees: I had the Three-Cheese Lasagna my partner had the Gambellini Fra Diavolo (sorry if I misspelled it.) they came out at just the right temperature and at the same time. I tried a bite of his dish and the Diavolo was mild, but with a "back-of-the-throat" heat which is perfect. So: "A" for his dish. My lasagna was rather impressive, and I didn\'t realize the recipe was total cheese lasagna, but it was delicious. The only downside would be a personal preference of a little more herbs or spice in the cheese mixture. The tomato  sauce was slightly sweet and tasted of homemade. So well worth trying! \nOverall the reason behind the 4-star review is only for the slight downsides, but again, could just be my personal preference. \nI highly recommend coming here to try it out! Great ambience and extensive menu.',
      'useful': 0,
      'user_id': 'fttIbxIx4_zre4zOUGiqCw'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 0,
      'date': '2016-10-10',
      'funny': 0,
      'review_id': 'FzdhTetM0a7teVVtrjuNKg',
      'stars': 3,
      'text': 'The wine selection was limited but of good quality.\nThe appetizer Arancini were good but seemed to just be a little bland the tomato sauce helped but it too needed more garlic.\nMy salad had too much vinegar and was drowning in it. \nThe seafood Rollitini was ok but under seasoned.\nThe strawberry mascarpone cake, basically tiramisu with strawberries instead of chocolate, was just good not great. \n\nService was ok but given the lack of a crowd should it should have been better.\n\nIt was not the best but far better than a chain store.\nWould I go back probably yes at least one more time.\n\nSuggestions to the owner.\n1. Stop seasoning for old white people or at least offer to kick it up\n2. Go to a team waiting model to give more attention to customers. \n3. Make us customers feel special not by cupon for free wine but giving a shit, talk to us treat us like we are honored guests.\n4. Keep the seasonal menu going. \n5. Offer old fart seating at 4:00 to 5:00 "give cupon for buy one pay 1/2 price for second " you can offer the bland "Special " then.',
      'useful': 0,
      'user_id': 'g94Ml1oo2qQe_iGBJL8nAA'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 0,
      'date': '2016-08-22',
      'funny': 0,
      'review_id': 'XrKFPwrBPOwx8duWDLHFrg',
      'stars': 2,
      'text': "My husband and I went to Pazzo's for dinner during restaurant week.  We were both disappointed with our entrees.  We ordered off the restaurant week menu.  We both loved the lobster bisque.  My husband's crab ravioli was not warm.  My veal romano was hard and difficult to cut and eat.  My vegetable, carrots, were also not soft enough to chew. We enjoyed the ice cream for dessert.  The waitress was very friendly and attentive.  We may try Pazzo' s for lunch, but we wish we would have chosen a different restaurant for dinner.",
      'useful': 0,
      'user_id': 'qZawcXc_AtF1wmZky3KGAQ'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 0,
      'date': '2016-08-07',
      'funny': 0,
      'review_id': 'C0B5VXECVGinXiTB7qWAJw',
      'stars': 5,
      'text': "We loved our experience here tonight.  Kim was our waitress and she was great.  Our children have an allergenic disease and are very limited in what they can eat.  I also have Celiac's disease, so no gluten for me.  Our meals came out just right, they made special accommodations for the kids and even had gluten free breadsticks they could eat.  You have now become one of our new favorite restaurants.",
      'useful': 0,
      'user_id': 'isxSgwqvKnOqUQPmRrJfcg'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 1,
      'date': '2015-09-16',
      'funny': 0,
      'review_id': 'BP2mCfBp__gQESuDUNEXzg',
      'stars': 1,
      'text': "This restaurant looks lovely. There are beautiful paintings on the walls. The tables are set nicely. You are greeted kindly. The servers are very nice. The decor/environment and the service is the only reason I can give one star.  It was all downhill after that. \nI say all the time that of you can't get the basics right you can't get anything beyond the basics right. So we ordered two basic dishes-fettuccine Alfredo and Spaghetti & Meatballs. We chose to skip appetizers but a family style salad was brought to out table. It included a few cucumbers, a few grape tomatoes, some purple onions and a few croutons. There salad was cold but there was nothing memorable about it. We were also served 2 slices of bread. Again, nothing that stood out. \nThe entrees were brought out while we were still eating our salad. It was awkward. When we went to eat our entrees they were room temperature. The meatballs were over cooked, dry, and didn't seem to have any seasoning. The spaghetti sauce tasted like tomatoes and not much else. I was extremely disappointed. I also tasted the fettuccine. It too lacked flavor.  The fettuccine noodles were inconsistent. Some were cooked to the correct consistency but some were hard. \nWe skipped dessert because we didn't want to take the chance of eating anything else that lacked flavor. \nIt's safe to say that there won't be a second visit.",
      'useful': 5,
      'user_id': '5JVY32_bmTBfIGpCCsnAfw'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 0,
      'date': '2015-10-27',
      'funny': 1,
      'review_id': '0mx2az90kwNHaQLPtK8VDQ',
      'stars': 4,
      'text': "Great food!  Thoroughly impressed with bolognese, manicotti, and veal.  Meatballs were soft and plenty of flavor. Sauce was a little too sweet for my liking, but still good. Service needs tightened up, but they're still new.",
      'useful': 1,
      'user_id': 'RHFKZji1mnsCXffs1VO1jw'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 0,
      'date': '2015-10-09',
      'funny': 2,
      'review_id': 'GCf-MGbzQKNtqulgtpff3w',
      'stars': 2,
      'text': "Overpriced and flavroless.  This place tries really hard to seem fancy, but falls far short.\n\nThe bread was fantastic, but it was paired with ice cold, solid butter.  The salad, as others mentiones, had very little nearly-flavorless dressing.  \n\nI had the chicken marsala ($18).  The three chicken tenders that comprised it were tender.  The sauce, however, was a mess.  It was as if they splashed some marsala wine on the plate, and then dumped the chicken onto the plate, oil and all, straight from the sautee pan.  It was sweet, watery, spotted with oil, and devoid of any kind of seasoning.  Oh, it had a few sliced mushrooms, too.  I chose pasta with white sauce for the side.  Bad move.  It was pasta tossed in plain white sauce - no cheese, no garlic, not even a bit of salt.\n\nMy wife had the shrimp scampi ($22).  It came with 4 or 5 shrimp that were actually quite big.  They weren't overcooked.  The sauce, again, was a watery mess, a bunch of butter quickly thrown together with some white wine.  The whole thing was topped with *garlic powder*.  I don't think any of it was seasoned with salt or anything else.  She chose broccoli for her side.  Aside from the bread, the broccoli was the best part of the meal.\n\nCome here if you're looking to spend $20 a plate for food that's below even Olive Garden's standards.  The service here was slow and unenthusiastic, and I'm willing to bet that none of the cooks have ever actually tasted the food they're sending out.\n\nI only gave it two stars because the food did not make me sick.",
      'useful': 1,
      'user_id': 'N2Do46MzonACg_SZ18buvg'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 1,
      'date': '2015-10-11',
      'funny': 1,
      'review_id': 'qGBVKSilE8S6oNqEZcOKIw',
      'stars': 2,
      'text': 'The perfect word for this place is: bland. \n\nI had the eggplant Parmesan, which had too much breading & was bland.\n\nAnother person in our group had the Greek pizza, which was thin & crispy & bland. \n\nOn the bright side (I guess), the kids liked their chicken fingers because, hey, kids like their food bland!',
      'useful': 2,
      'user_id': 'Lq6VdEsdhqKwR0ouqlxpMg'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 0,
      'date': '2016-03-14',
      'funny': 0,
      'review_id': 'SJ-Kz-0E2-J814AczzWQ2w',
      'stars': 1,
      'text': "I can't comment on the food here because we didn't get that far. We were in a hurry on a Friday and there was an hour wait at Bonefish so we decided to try Pazzo because we wanted to since it opened. The parking lot was pretty empty and very empty inside. I guess now I know why. They seated us immediately and then nothing. We saw other waiters at other tables and people cleaning up and nobody even looked at us. It was like we were invisible. They weren't even busy. After almost 15 minutes we got up and left and my husband told the hostess we would not be coming back. I have never been ignored in a restaurant before but now, based on the reviews, I'm glad we didn't waste money on the food.",
      'useful': 0,
      'user_id': 'ufoQBirTWN_9LFM332PCpg'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 1,
      'date': '2015-10-03',
      'funny': 0,
      'review_id': 'we0UPFb8-YCdO3031Q6ZqQ',
      'stars': 1,
      'text': "Save your $22+/ plate & take yourself over to Giant Eagle's Café. \n\nIt took over 30 minutes for the food to arrive at our table. The portions were generous, however nothing was seasoned properly. Shrimp Scampi was covered in garlic powder sitting in a bowl filled with microwaved butter. The broccoli was the best part of my meal. They were cooked to perfection. The salad that's given lacked any dressing. The dressing was bland. The beer is over priced. $9 for any craft beer. I could see if they had a 10% DuClaw on draft for Pazzo's to have a reason to charge that much.\n\nThe service is depressing. Servers don't smile here or seem enthusiastic. Perhaps it's because they feel guilty for being the poor souls to serve below average food for such a sky-high price. \n\nThe decor is nice. That's it. Now save yourself money and go to Chick-fil-A next door.",
      'useful': 5,
      'user_id': 'ng647hYAjWC4XANaACiPXA'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 0,
      'date': '2016-04-30',
      'funny': 0,
      'review_id': '9LkhocX36Lq_vlH3N-Bp4w',
      'stars': 4,
      'text': "I don't understand the 2 star reviews. Lunch today was very good, especially the thin crust pizza and the wedding soup was superb.",
      'useful': 0,
      'user_id': 'TvwEz7v1xwRYenlDlypGcQ'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 0,
      'date': '2016-08-19',
      'funny': 0,
      'review_id': 'OVYO5UZmnPIE_1vIFljokg',
      'stars': 2,
      'text': "The wedding soup was so good! After eating that I was hopeful that the reviews on here weren't accurate... I'm sorry to say that they are true! I would come back for the soup but wouldn't eat dinner here again. We came for restaurant week and the menu looked amazing but the actual food was pretty poor.",
      'useful': 1,
      'user_id': 'q3ycagiLqUaRkOgEPf_gbw'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 0,
      'date': '2015-10-07',
      'funny': 0,
      'review_id': '9xhE5oSnN73ynKHc3nC6Bw',
      'stars': 1,
      'text': 'I completely agree with Kimberly said. BLAND CITY!!!  Hopefully my dogs will like the leftovers. But I doubt it.',
      'useful': 3,
      'user_id': 'jz9OdWuWs9cGewY-E4ASmw'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 1,
      'date': '2015-10-18',
      'funny': 2,
      'review_id': 'DXegHspbOy3oWnQyEyj3iA',
      'stars': 1,
      'text': 'So far all the reviews have nailed it- flavorless and cold food. Four of us sat down and ordered Cabernet. After a few minutes the waiter stopped by to have us remind him what beer we had ordered... We placed our order for food and he brought out a "family style" salad. Bad lettuce with a couple of poor cut onions slices. There was 4 of us and they served the table of 2 next to us the same salad. We asked for me and the snotty waiter slammed a bowl of just lettuce on the table a few minutes later. The funny part was they give you a gallon of dressing per person.. We decided that since it was the new place in town we would order a few apps so that we could try it out. But, using some caution we stuck to the basics -- Calamari and Fried Zucchini. Both were dripping grease and had zero flavor what so ever. Luckily I had lots of the Lemom Basil dressing left from my salad to give the Calamari the citrus it needed (never been served Calamari without a lemon to squeeze on top). The Zucchini was so soaked in grease none of us could eat it. They didn\'t clear off the table so at this point we are overloaded and there was no room for the bread that was brought to the table. Unfortunately we made room for it and it was the most stale bread I have ever had. Perhaps I was mistaken and they brough us croutons to go with our lettuce?\nI\'ll wrap up with the entrees. I had ordered the "lightly breaded" veal chop with arugala and they served a disgusting piece of veal with a heavy and burned breading. No seasoning was involved. I opted for the starch and I was hoping it would be a pasta but instead if was a bowl of red skinned potatoes that had been given a butter bath. I literally poured the butter from the top before I could even tell what they had served me. I could have used these potatoes to apply wallpaper the place--gross. My wife was served cold, undercooked pasta and shrimp without any seasoning at all. It was supposed to be "diablo" but zero spice . She had two bites and was done. One guest ordered a side of chicken to go with his pasta and 5 minutes after his pasta was brought out they showed up with a bowl of shredded chicken.. a literal WTF moment.. The last meal was salmon and just to wrap this up was an awful piece of fish with no flavor -- same butter drenched gross potatoes..\n\nWe were honest with the waiter when he asked how everything was but he was more interested in being condescending in between smoke breaks. The manager does not have the skill level to deal with this type of complaint. She tried to make it better by taking the chicken off of the bill and offering a free desert.. And after all of this our bill was almost $200.... what a joke. I hope the place that goes in here next is better -- shouldn\'t be long.',
      'useful': 5,
      'user_id': 'yYagusi0EEOIn2qILEDXZQ'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 0,
      'date': '2017-03-19',
      'funny': 1,
      'review_id': 'CWY2Taq2YxKLAydGJ4U-7Q',
      'stars': 2,
      'text': 'First time at this restaurant for my birthday dinner which was less than memorable. Our entrance less than a.warm greeting by a young blonde girl at the host stand. "Hi, just two of you?"not all that welcoming. Brought to our seat and was not informed of any daily specials. Was not presented a wine list but asked for one and got a not to impressive small list. Who has lambrusco on the menu? This place does. Opted for a mixed drink since wine choices were slim. Had the bruschetta. Appetizer. Nothing to write home about. Bland, no seasoning. Wife had pasta with salmon. Salmon was small tail end piece that was overcooked and dry. I had veal and peppers which was dredged in too much flour and not shaken off before sauteed. Asked for bread twice before i got three small slices that was told just came the oven, (cold). Asked for warm bread and finally got it as i was taking my last bite of veal. Also had to ask for water twice. Needless to say i had a subpar meal and terrible service. I usually give a place a second try but since both meals were average on a good day and no wine selection, we wont be back',
      'useful': 1,
      'user_id': 'VdUuNoGQfSXhgroDQ6i1XQ'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 0,
      'date': '2016-01-16',
      'funny': 0,
      'review_id': '9O3ET_tLiuBfPbV4UF9zdQ',
      'stars': 1,
      'text': "My wife and i went there for the first time on Saturday night, decided to sit at the bar. We waited for 5 to 10 minutes without being acknowledged by the bartenders, even though it wasn't very busy. When a bartender walked by us without even saying hello, we decided to get a table.\n\nA waiter named Brandon took care of us, but his service was mediocre. He had to be reminded that we had ordered a second drink, the food was just warm, and not very good. (Chicken asiago and veal marsala) He never asked us if everything was ok after the food was delivered, didn't ask us if we wanted dessert or coffee, and didn't even say thank you after he returned the check after the payment was made.\n\nOverall, we were very disappointed with our experience at pazzo. We will not return or recommend it to our friends. It was overpriced for the quality of the food and service. Olive garden would be a much better value.",
      'useful': 3,
      'user_id': 'skSf0o3ba3sBZ_3Olxp_4Q'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 0,
      'date': '2016-01-26',
      'funny': 1,
      'review_id': 'iBVABFfC5Yr0vvt6e7t4BA',
      'stars': 5,
      'text': 'We had a great experience for dinner a few days ago! The service was wonderful and our server was very knowlegeable about the menu. I am gluten free and my daughter is egg and dairy free and she was able to guide us to the best choices. The kitchen appears to be careful about food allergies and cross contaimination. From our zucchini appitizer, family style salad, entrees and wine, everything was beautifully presented and tasted great. The courses were well paced and we had plenty of time to enjoy the warm,remodeled space with tables with white tableclothes. We will be back!',
      'useful': 2,
      'user_id': 'SQL4HmnzESupIJ-dadx2wA'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 0,
      'date': '2017-03-17',
      'funny': 0,
      'review_id': 'ss4x7yFbFKIkAjnhnswBRQ',
      'stars': 2,
      'text': "The parking lot is always empty which isn't a good sign. \nBut still wanted to give it a try. \nNo hostess. Waited a few minutes before the bartender came out to seat us. \nMy menu was dirty.\nMy silverware was dirty.\nGlasses spotty. \nCome on guys. If you're slow and still can't keep things clean.....?\nFood was ok, but definately didn't live up to the prices. I am in the business and it makes me sad when a new restaurant doesn't do well. \nThese little details could easily be fixed.",
      'useful': 0,
      'user_id': '4fdx6AiHPvLQv2Beca69uA'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 1,
      'date': '2017-01-24',
      'funny': 1,
      'review_id': '0glKaP0H84XJxCORugfbxg',
      'stars': 1,
      'text': "The worst dining experience I have ever had.  The food was decent, although some people in my party were unable to eat their's as it wasn't cooked thoroughly.  \n\nIt took an hour and 20 minutes for us to get our drinks.  When they first started bringing out the drinks, the waitress mentioned she had not yet placed our dinner order, which we had given her an hour before.  Someone from our group searched out the manager and found him in a back corner of the restaurant hanging out (turns out he is also the chef).  He basically ignored them.\n\nThere was much confusion, wrong drinks were brought out, not all appetizers were brought out.  At 9:00 (we got there at 6:30) we couldn't find our waitress to get our checks.  Finally at 9:30 she appeared with all single checks.  Not all checks were there, and she basically just put them on the hostess stand for us to sort through ourselves.  A member of our party had to search out the manager, who said he was just about to come over and see how our meal was.  We had no signs of him until we had to search for him.  \n\nThis was by far, the worst dining experience I've ever had.  Anytime I drive by, the parking lot is empty.   I can't imagine it will be open long.  If only the chef could back up his arrogance with his culinary skills.",
      'useful': 2,
      'user_id': 'yRaaXNFgqnZesfMyw0GclA'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 1,
      'date': '2015-11-01',
      'funny': 0,
      'review_id': 'Kb9xyEn9W2A1GMDsxF1xmA',
      'stars': 1,
      'text': "I really don't have too much to say other than the fact that I sent my food back and decided to just grab something to eat somewhere else.  Four of us went on a Friday night and were seated almost immediately.  Ordered drinks and 23 minutes later they show up.  Dinner was bland and not even italian food...\n\nPesto should at least have pine nuts and basil right?  Mine was tortellini with olive oil and parsley.  The service was slow and even after we complained no one came over to say a word or find out what they could have done better. \n\nOh, and a kettle one martini was 12.75  For ketel one and some vermouth.  WOW.   There is no way that the mgmt cares about this place based on all the other reviews.",
      'useful': 2,
      'user_id': 'MCsNB80uwohCMB16vc0mTg'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 0,
      'date': '2016-01-08',
      'funny': 0,
      'review_id': 'A10kO8mkZgORSlC3R-kUbQ',
      'stars': 2,
      'text': "The food is very good. \nBut...... Bartender charged for a double. Should have been 6.50 but they charged 13.00. When we mentioned we didn't order a double, she said it was a computer glitch, but made no effort to reduce it. Asked for a manager, he offered to take off 4.00. Buyer beware!! They are sneaky and when they get caught, they are unapologetic.",
      'useful': 0,
      'user_id': '8JFEoPWoH32C8h-PjmobMA'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 0,
      'date': '2016-08-24',
      'funny': 0,
      'review_id': 'M57psk3w8Tykb9nrbQ7bPA',
      'stars': 1,
      'text': "If you're going to charge $16 for manicotti and a $2 up charge for the WAY too salty wedding soup, I expect more... I have no problem paying for good food and if this food had been amazing, I wouldn't even mention the cost, but it's far from that. I normally get manicotti at Italian restaurants my first time there and maybe if I'd done that, I'd like this place more, but I doubt it, since there was barely any sauce with my crab ravioli and I didn't like the flavor. I know many people like bell peppers, but I'm not a fan. If I order crab ravioli, I expect to be able to taste the crab - the consistency of the crab was also very strange - it was mushy. My dad's linguine with red clam sauce barely had any sauce and the sauce that was there was SUPER runny - it looked more like died white clam sauce than red clam sauce. At least they used real clams, but neither of us were impressed. I wish Stone Pepper Grill hadn't closed.\n\nI'm REALLY not a fan of the Olive Garden, since there are so many better Italian places, but I think the food is better there, you get more, and it's certainly not as pricey... Of course, there are many much BETTER Italian restaurants than that (like Pasta Too, etc.), but if I'm saying the Olive Garden is a better choice, you know the food was not very good.\n\nA couple more thing I forgot to mention in my review... I had to practically beg for parmesan cheese and what I got wasn't fresh or even nicely grated from a container... It was the same kind of parmesan I have at home (though probably not even Kraft... ha ha ha...) it was in a dish and they spooned it on for me. I didn't even get any to go with my soup, since I didn't want to ask for it then. No cheese was EVER offered.\n\nThey seem to turn their air conditioning completely off instead of down when the restaurant is closed, so it was INSANELY HOT in the restaurant until we were almost ready to leave (we went there for an early dinner, since I met my dad right after work to avoid traffic). Someone should explain to them that this actually costs them MORE money and it makes guests uncomfortable, so it's bad on all fronts. \n\nAVOID this place at all costs. Here's hoping they don't last long so a better restaurant can open there.",
      'useful': 1,
      'user_id': 'ZnIaEvJwrxlBUjimIpx2Qw'},
     {'business_id': 'FxLfqxdYPA6Z85PFKaqLrg',
      'cool': 0,
      'date': '2016-12-31',
      'funny': 0,
      'review_id': '5E6r13MMn3HLIsTZU-28dw',
      'stars': 1,
      'text': "Review is based on service and bar only - did not order food. \n\nPlace was empty on Friday night. First warning sign.  My wife and I sat at the bar to order drinks and contemplate dessert. It was ten minutes before our order was taken. Second sign. Drinks were poured, took five minutes to find a lime for my wife's drink. Third sign. \n\nWe decided to skip dessert. We could tell as we looked around, wait staff was poorly trained and disorganized. This place needs help. Beautiful interior, and like I said - did not try the food. But with the number of good Italian places in Pittsburgh, come on - need to up your game!",
      'useful': 1,
      'user_id': 'FSRzQRl8GlnRU78w4vwP3A'},
     {'business_id': 'bzUSbd9YLoK3egnTkXFd1Q',
      'cool': 0,
      'date': '2014-07-12',
      'funny': 0,
      'review_id': 'rP1fOEMJZTUgJ5V8vAvTYA',
      'stars': 5,
      'text': 'You\'re not going to find Nihari and Biryani at this spot. It\'s American food made with a "desi" twist. \n\nStepping into this place, it\'s an enigma wrapped in an enigma. Sholay isn\'t heavily advertised in your face so you don\'t know what to expect. \n\nThe restaurant name brings memories of the legendary movie and elite characters. Come on. Don\'t tell me you can\'t hear Amjad Khan\'s voice :) \n\nEnough of me being Basanti . The place has: a large poster of Shah Rukh Khan, a pic of the Dalai Lama , and generic scenery pics. As you walk in you are promptly greeted by two American white people. The instinct as a desi is to run away but please resist this urge. \n\nYou find out the staff, are owners and chefs who love desi food. The "white dude" who is their head chef, couldn\'t be more "desi" when it comes to spices and food. \n\nLoved the chicken wings and chicken kabob here. \n\nOnly con: they are only open till 7pm.',
      'useful': 0,
      'user_id': '00Z-tCcJVe9LVP3GakkqDQ'},
     {'business_id': 'bzUSbd9YLoK3egnTkXFd1Q',
      'cool': 0,
      'date': '2014-05-06',
      'funny': 0,
      'review_id': 'iafwT1fvb5QQ-fqz7y7bQg',
      'stars': 3,
      'text': 'So here\'s the thing. This isn\'t an Indian restaurant. No matter how you spin it. \n\nThe "naan" is actually pita. The kind you buy in stores. There are three or four Indian curries on the menu. \nWe got the chicken curry. The guy (there was only one person there, to cook and work the counter, but they weren\'t busy, so hey) said it was really good. If you\'re desi, you know what shan and all the boxed masalas taste like, and this definitely had that taste. \n\nNot saying it\'s a bad thing, but fresh masalas give a curry an aroma that this lacked. \n\nThe chicken shish kabob was bland, except for the red chili powder added liberally on top, and it wasn\'t actual shish kabob. \n\nThe best thing I had was the lamb burger. Oh my gosh. The lamb was super juicy. And the fries already had black pepper. Loved it. \n\nThe place is super duper clean. Even the bathroom. \n\nLike I said, there was one person to cook, serve, answer calls, and take orders. He told us the others were out for a catering. \n\nThis is in no way true Indian food. The shish kabobs aren\'t Indian shish kabobs, but rather Middle Eastern kabobs, the naan isn\'t true naan, either. \n\nDon\'t go to this place expecting awesome Indian food. But it is a decent place to go for a burger or the pizza.',
      'useful': 1,
      'user_id': 'CS4c5WBPG14mHUx2qPJdMw'},
     {'business_id': 'bzUSbd9YLoK3egnTkXFd1Q',
      'cool': 0,
      'date': '2014-03-07',
      'funny': 0,
      'review_id': 'k0pUou-q-TFAV0oIvT_oPg',
      'stars': 4,
      'text': "Not sure why my lamb kabob meal took so long to be ready, but I overheard something about the rice still cooking, and this was at peak lunch hour (noon)! My only complaint (almost 20 minutes for one to-go order with only 2 or 3 other customers at the time) was the waiting, but the employees were nice and made sure to let me know that I wasn't forgotten.  The shish kabobs were tender and savory, and their balsamic vinaigrette for my salad was the best I've ever had!!!  The saffron rice was also delicious even though it appeared to be the item that was delaying my order.   I should've called ahead, but next time, I want to try their lamb burgers...  Limited menu but top quality ingredients, yummm!\n\n3/12/14\nI returned there after work to pick up my lamb burger and side salad to go.  The ground lamb meat was indeed tasty, but I was disappointed by the smallish size compared to the bun.  I'll stick with the kabobs and rice in the future.  I was told by the nice man behind the counter that they'll be expanding the Indian portion of the menu in a few months so here's hoping they add samosas!",
      'useful': 0,
      'user_id': '1o8vB0WKHfV7omrYWB_OLg'},
     {'business_id': 'bzUSbd9YLoK3egnTkXFd1Q',
      'cool': 0,
      'date': '2014-04-08',
      'funny': 0,
      'review_id': 'dBWviP3LipdHhdwbpMjiTw',
      'stars': 5,
      'text': 'Their burgers are so juicy and flavorful. Also greatly priced, and did I mention the burgers! I only tried a beef burger but the chicken looked so good! Te service was great with friendly staff. Definitely must try again!',
      'useful': 0,
      'user_id': 'HRP5zLOrhXgLYAlaBt-HgA'},
     {'business_id': 'bzUSbd9YLoK3egnTkXFd1Q',
      'cool': 0,
      'date': '2014-03-28',
      'funny': 0,
      'review_id': 'e3IeKncWZa55LPadXaFyUw',
      'stars': 5,
      'text': 'Fresh food... exotic flavors.  Their tag line says it all!  I ordered the beef kabob on the recommendation of Phil, the owner, and I was not disappointed at all.  The halal meat was moist and spiced just right... absolutely delicious!  The basmati rice was also spiced in a subtle way that complemented the meat.  He brought me a sample of the rice pudding, and it was very good, along with the mango yogurt smoothie, which would be a refreshing drink come the hot days of summer.  I cannot wait to go back and try some of the other items that other Yelpers have raved about!',
      'useful': 1,
      'user_id': 'kmC8nIGdQeLUOBCgRDzS8g'},
     {'business_id': 'bzUSbd9YLoK3egnTkXFd1Q',
      'cool': 0,
      'date': '2014-05-11',
      'funny': 0,
      'review_id': 'Q0Ngh3nFEiBnX4kng09ElQ',
      'stars': 4,
      'text': "The friendliest worker ever works here. I think he is the owner, or just a very friendly man. Every time my boyfriend and I come here we get free stuff. Last time it was dessert, the other time it was mango lassi. The food is good and decent pricing. Sure, it's not the most glamorous presentations, but it is good for someone who is very hungry.\n\nThis place just opened up like a 3 months ago, so they are pretty eager to get customers in.",
      'useful': 0,
      'user_id': 'TyepNPbEnMCy1_IhsT0YAg'},
     {'business_id': 'bzUSbd9YLoK3egnTkXFd1Q',
      'cool': 0,
      'date': '2014-04-02',
      'funny': 0,
      'review_id': 'VAuszSrcSOnnTz16-9Jadg',
      'stars': 3,
      'text': 'I usually come to Afandi a lot and would have never noticed the Sholay restaurant sign if it wasn\'t for my boyfriend. I had heard about it through yelp but didn\'t expect to try it out anytime soon. \n\nI was taken back by the fact that the owner of the place was not Indian! He was VERY friendly though and so I give this place a 5 for customer service! After having asked if the meat was halal, the owner told us that everything was halal except for the beef pepperoni. I\'m very grateful that he not only verbally stated it, but has it written down on the menu as well. \n\nWe weren\'t really starving so we placed an order for halal wings & fries as well as the pizza naan (with chicken, cheese and pizza sauce). He asked us if we wanted him to make it spicy, and like most Indians and Pakistanis, OF COURSE we said yes. So he made added chillies as well. \n\nFast forward to the food - the halal wings were GREAT. Spicy, but a little too breaded. My bf noticed this as well, and was the first one to point it out. The fries came out extremely salty, but I finished the majority of them anyways. The pizza naan was good! Especially if you like chicken keema! The chillies the owner added made it spicy and quite scrumptious. \n\nThe best part of our trip to this place was definitely the MANGO LASSI. The owner gave us each a sample which led us to ordering one. The lassi was thick, unlike many other places you order a lassi, and you could taste the yogurt. YUM. \n\nThe owner also gave us a sample of the rice pudding, but I wasn\'t that big of a fan. I\'m not the biggest sugar addict but I just wasn\'t feeling it. \n\nI hope to come back for the lamb burger and the lassi next time! \n\nOverall, this isn\'t entirely "Sholay". If you are looking for Indian (as in"desi khana") food, go elsewhere. If you are looking for American styled Indian food, this is your place.',
      'useful': 1,
      'user_id': 'n9zwrWp8EeB9ODTitKZYsw'},
     {'business_id': 'bzUSbd9YLoK3egnTkXFd1Q',
      'cool': 0,
      'date': '2014-03-02',
      'funny': 0,
      'review_id': '2ri42IvMGvPDKZCVBhJmCQ',
      'stars': 2,
      'text': "First impression is that this place is very clean but a little confusing- you walk in and see a buffet-type set up, but it's not being utilized and looks like there's a lot of junk around the kitchen area. I was immediately greeted and handed a menu, which has American breakfast specialties on the front, and inside is a small kabob selection, naan pizzas, and burgers. Very odd combination of offerings.\n\nOrdered the chicken kabob platter to go, and was offered a breaded chicken wing to sample while I waited- they are trying out new menu items. The wing was huge and mostly composed of breading, which is not what I'm looking for at all when I'm ordering wings. Flavor was okay, but the wing itself was just weird.\n\nGot home with my food in a reasonable amount of time and it was nice and hot- the kabobs were delicious, and my BF snagged a few bites even after devouring his sandwich. The rice in particular was very good: I'm not one to rave about rice, but it was moist and flavorful, and had some white, some yellow, and some orange in it- likely a combination of white and saffron rice, with additional seasonings. The naan and salad were also fresh, and the portion was generous, and on par with the $10 total. Best part of the meal was the cilantro sauce for the kabobs- it was killer. Highly recommend, get it on anything.",
      'useful': 3,
      'user_id': 'bLbSNkLggFnqwNNzzq-Ijw'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2016-05-02',
      'funny': 0,
      'review_id': 'lpx51CmKp0qnKunRi0LI0Q',
      'stars': 5,
      'text': 'Absolutely #Loving this place is. Scottish Mussels w/garlic, shallots, cream wow! Pan fried cod w/ asparagus, languostinemushroom sauce on a bed of creamy mashed potatoes #ToDieFor #KidFriendly - so welcoming to kids even later night! Ps side salad so fresh and tasty lots of feta and the mushroom pate appetizer #FAbulous',
      'useful': 1,
      'user_id': 'NZsvF7oESXAsGEowhZextQ'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2017-05-16',
      'funny': 0,
      'review_id': 'Ymuvr4CXsBLOtP0MeHorFg',
      'stars': 5,
      'text': "Went here for dinner with my mom on a one night stay in Edinburgh. We shared the mussels in shallots and cream sauce and they were delicious. Very generous portion. Had to ask for bread to sop up the sauce. \nFor the main course I had the Roasted Seafood and my mother had the rump steak (can't remember the other name for it). Both were excellent. Steak was perfectly done and the seafood was fresh and plentiful. Definitely recommend it, just sorry this will likely be my only time there.",
      'useful': 0,
      'user_id': '2KVpu1RQVZZ4Uv62mT_hEg'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2016-12-26',
      'funny': 0,
      'review_id': 'S-5sTPnzhFksk4AY09P65Q',
      'stars': 5,
      'text': "Popped in to meet a friend for lunch.  Looked very ordinary on the outside, but the food was top notch.   My husband had the best steak he's had this trip here.  Cooked perfectly and delicious!!  I had a seafood curry that was also fantastic. Friendly, helpful waitstaff.  Great experience.",
      'useful': 0,
      'user_id': 'le2_8k4PYRnWCBo_nKGXdQ'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2010-09-24',
      'funny': 0,
      'review_id': 'iZC379_rhDxb4fs-a4XBSg',
      'stars': 4,
      'text': 'Mouth is watering at the thought, this was the best meal I have had in a very long time, staff were pleasant enough. Muscles were huge, so tasty and very well cooked, the basket of bread was replaced a few times. Will be back soon :)',
      'useful': 0,
      'user_id': '4vl8Drja0PETBmlNCFZbvQ'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2012-08-23',
      'funny': 0,
      'review_id': 'naSqWo1iwIm_UNvRLmoFqA',
      'stars': 4,
      'text': 'Altro ottimo ristorante situato a 20 metri dalla piazza Grassmarket,livello alto di cucina sia di pesce che di carne!\nOttime cozze e pesci arrosto,buonissime ostriche scozzesi e deliziose bistecche di angus!\nPer concludere dolci ben fatti preparati da loro!!!!\nServizio gentilissimo curato e giovanile!\nSaluti Luca!',
      'useful': 1,
      'user_id': 'JCG7J-NmLxgmJK1c4ZRKmQ'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2017-06-07',
      'funny': 0,
      'review_id': 'DgrPLxnT6DJyvhwLEJnrfg',
      'stars': 4,
      'text': "This place is really good, but a little pricy I think. Although they do have some drink specials and a pot of mussels for about 13-15 pounds. But once you start ordering appetizers or any of the other entrees, it can get pretty pricey pretty fast. I didn't think the mussels were anything out of the ordinary really -- and I was surprised to find cilantro all over mine (usually it's parsley when you order a shallot, butter, and white wine sauce). This wasn't on the menu so I had to try and pick the cilantro out. The one must-order here though I would say is the grilled scallops with butter appetizer. It's juicy, buttery, slightly sweet scallops on the shell -- and probably my favorite thing here.",
      'useful': 0,
      'user_id': 'BnIY7cXbUfnrEqoQdQc9Pg'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2015-08-08',
      'funny': 0,
      'review_id': 'txV-qcucLaZSESdUSJdQ5Q',
      'stars': 5,
      'text': 'Perfekt, man kann draußen sitzen und das direkt am gut frequentierten Grassmarkt.\nLecker Fisch und Fleischgerichte, also rein mit dem Surf & Turf',
      'useful': 0,
      'user_id': 'qePB-HLOeim-BDMG0QkfuA'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2015-06-28',
      'funny': 0,
      'review_id': 'v6taWbXFF0a0r096qR3RNw',
      'stars': 5,
      'text': 'I ordered mussels with ginger/lime sauce and they were the best I have ever eaten. For £12.50, the portion was huge (may be 30 or more) and the sauce was more like soup at the bottom. It was spicy and delicious.',
      'useful': 1,
      'user_id': 'K5h_G-hGAkskrhqfW6l4DQ'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2014-05-21',
      'funny': 0,
      'review_id': 'UXYFVn5e_YoCU2mmxBglsQ',
      'stars': 4,
      'text': "it's been a while since I've posted anything to yelp - but I noticed a distinct lack of available info during my travels to Scotland and Ireland, so here goes\n\nThis place is located at the (end of the) Grassmarket in Edinburgh - from what the guide books tell me, it would have had a great view of all the hangings (puir man got hangit) that took place there.\nThe inside is modest but nice and aside from the somewhat crabbit (maybe German?) bartender, the service was just fine.  We split a bucket of mussels (shallot cream white wine sauce) and I had an 8oz ribeye - the husband had something weird like pasta.  The mussels were the best I've had in my life, truly amazing, and the steak, skinny little thing that it was, was delicious also.  The quality of the local beef and seafood spoke volumes - no extravagant preparations or sauces required.\n\nHighly recommend.",
      'useful': 1,
      'user_id': 'YczurbalRifsqoV3OmWE7w'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 3,
      'date': '2015-03-24',
      'funny': 4,
      'review_id': 'lcyKTYsFF8311U4reDlDsQ',
      'stars': 4,
      'text': 'I went to The Mussel and Steak Bar after a date at the nearby BrewDog. The date was going well (oh la la) so he suggested stopping by this place for dinner. It was a Friday around 730 and we didn\'t have reservations, but we were able to be seated for a table for two. The manager was cheeky and ribbed my date a little bit for pretending to look in the reservation book to pretend to be someone else, which was very amusing. We were able to sit upstairs and I thought the place had lovely ambience. \n\nWe started with some fried goat cheese balls and garlic grilled mussels. Honestly, I\'m not a huge fan of mussels but I thought "When in Rome!" and tried them. They were really tasty, but I\'m not sure garlic was the best date choice! Anyway, for my main I had a special fried plaice with black pudding mash that was extremely tasty. My date had a steak that he enjoyed. Topped off with a nice glass of house white and I was a happy camper. I will definitely come back to The Mussel and Steak for another nice dinner in the future when I\'m in the area, which of course I will be because I do love BrewDog...',
      'useful': 3,
      'user_id': 'h7kFt1px7Z5A5e92X6aMvQ'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 2,
      'date': '2013-09-08',
      'funny': 0,
      'review_id': 'F75u7vq1dndChBrAX_eA2g',
      'stars': 4,
      'text': "We had a light dinner here, it was fantastic! Fresh seafood at it's finest. Started out with Arran Smoked Salmon with Potato Scones, Crème Fraîche and Chives, the salmon was so fresh, it melted in your mouth. We also split a kg of the Scottish Mussels, we chose Shallots, White Wine, Garlic & Cream sauce, don't forget to order some bread to dip into the pot and soak up the goodness. This was probably one of the best mussels pots I've had. \n\nCall and make a reservation if you can, we waited 45 minutes for a table (we went over to the Bow Bar to pass the time). If you can, sit outside, great people watching, right in the heart of Grassmarket.",
      'useful': 1,
      'user_id': 'SfY0jEFqBkrqq9bVi6P31Q'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 5,
      'date': '2013-11-28',
      'funny': 3,
      'review_id': 'K2NHGrPLPncaemIs43q-ng',
      'stars': 5,
      'text': "The best surf and turf I have ever had. \n\nI stumbled upon this place by accident during a day trip to Edinburgh. It was during Scotland's unexpected (and sorely missed) heatwave so we sat outside - a rare treat in our climate. We were the first diners when we sat down but by the time we left the place was full and there was a great vibe. \n\nNow to the food. Me and my dad both opted for the same thing - the surf and turf. This consists of a Rib Eye Steak served with half a kilogram of mussels cooked in your choice of sauce, crevettes and squid rings. Being a big fan of Asian flavours I opted for the chilli, ginger and lime sauce. We also ordered some buttered new potatoes on the side to add some carbohydrate to all the protein.\n\nWhen the dish arrived it was amazing. Perfectly cooked steak (medium-rare for me) sitting on a mountain of seafood - not many things could look so appealing. All the seafood was full of flavour, the sauce had just enough heat and the Corona with lime just added to an amazing meal.\n\nNext time I'm in Edinburgh I will definitely be making a return here (I hope the heatwave will too).",
      'useful': 6,
      'user_id': 'ivfKrqUuxZuc824Nce6jIw'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 1,
      'date': '2015-11-28',
      'funny': 0,
      'review_id': 'fAX6RzUALYAp5mgN5VSAfQ',
      'stars': 4,
      'text': "I went there for a casual lunch date with my SO. It is lovely. They have a pretty reasonable lunch and even dinner deal if you want to fine dine on a budget. \n\nI had the steak and it was great! So good and nicely flavored and cooked just right. The mussels are great! They have a Thai inspired flavor which is not bad if you want something different. \n\nI do recommend getting the French fries fries in beef jus. It's good and so much better than their plain potatoes.",
      'useful': 1,
      'user_id': 'eN2pCfT17EGokP9UWnoZUA'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 1,
      'date': '2015-07-05',
      'funny': 0,
      'review_id': '17bT7TRCa5_BpPDBvoue9A',
      'stars': 5,
      'text': "I'd recommend any seafood lover to go to the Mussels and Steaks Bar.  Excellent food, great service and located in a bustling part of town. The decor is basic but the great tasting fresh food and friendly service more then made up for it.  \n\nAll the food was fresh, vibrant and flavorful.  Our group tried the seafood platter, the special lobster dish, curry mussels and the surf and turf combinations.  No one had a complaint about any of the dishes.  The best part was the double fried chips in drippings.",
      'useful': 1,
      'user_id': 'Wv2jkAduauTiySavy81RvA'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2014-10-24',
      'funny': 0,
      'review_id': 'JpSrnl1OYhiX2iKKZ5CGLg',
      'stars': 4,
      'text': 'Heerlijk gegeten. In eerste instantie lijkt het wat klein maar boven zijn ook nog plaatsen. Het vlees is heerlijk en de mosselen ook. De wijnkaart is mooi en compleet en zelfs de whisky liefhebber kan hier uit de voeten. Een aanrader.',
      'useful': 1,
      'user_id': 'xz9ckxdXaVgypQjvfGJHiQ'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2007-08-30',
      'funny': 0,
      'review_id': 'cqDIsJfwh50sTSyW6ZBxJQ',
      'stars': 4,
      'text': 'This is a gem in an otherwise touristy Grassmarket. This place has fantastic food, Oysters, Fish, Steak, and plenty of other options. But the specialty has to be the mussels. Choose a mussel pot and have it in any of five ways to suit your palate and mood, just take your pick. Depending on how hungry you are choose a full three course, or share a divine starter and dessert. All this with great service, good price and fresh atmosphere.',
      'useful': 0,
      'user_id': 'h_yuvACgAPnBL2FXiCF4Yg'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2008-04-20',
      'funny': 0,
      'review_id': 'VTylazw9aG4M7IQbGMzz_g',
      'stars': 4,
      'text': "Fantastic seafood, with a wide menu selection. Staff are friendly, eager to please and not afraid to make an honest recommendation or suggestion. The restaurant is generally busy, mostly down to it's great reputation and location. The prices are reasonable given the quality and quantity. Also great at dealing with larger groups of dinners so a good place to book a special family meal",
      'useful': 0,
      'user_id': 'Dqji5fRROkLc3eNMMQOI_Q'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 2,
      'date': '2010-05-31',
      'funny': 0,
      'review_id': 'U7wIblaF05QkXCKbGVIfOg',
      'stars': 5,
      'text': 'I come here. A lot. Either with my mum or with friends (usually to celebrate a birthday) this has become a personal favourite eating place. \n\nFirstly, it does good food- whether you are into seafood or meat. You get to watch your food being cooked on a screen (which is slightly odd) and I love the layout of the ground floor- quite modern without over doing it. Look out for the fun facts about mussels in the toilets.\n\nThe service has always been super friendly at this place. Being in such a prime location I was expecting shocking prices but I was pleasantly surprised. They have amazing lunch deals and I have never walked out of this place hungry.\n\nSadly they only have one vegetarian option as far as mains courses go. Also, in the winter the upstairs floor tends to get really warm. \n\nAside of these hitches Mussels and Steak Bar is a good standard for me.',
      'useful': 1,
      'user_id': 'nhVvcPyRGLw466qHVbUjrA'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2013-11-24',
      'funny': 0,
      'review_id': 'SpdjTAUq_AgsuP9ZuMi8bA',
      'stars': 3,
      'text': 'Was expecting a lot from this place based on the reviews, but honestly we didnt love it. It was just OK. Service was quick and friendly. They were also happy to oblige my dairy intolerance, which was nice. Both the seafood and steak were good quality, but I guess they were just a bit boring. \n\nBoyfriend wasnt so keen on the strange warm, limp cabbage on the side of the steak. Although the steak was cooked as ordered (medium-rare) which was good. The queen scallop starter was OK, but the scallops were quite small and over-salted. Mussels were pretty good. \n\nOverall a decent meal, but not good enough to get us over to Grassmarket on a regular basis.',
      'useful': 0,
      'user_id': 'A_dC3FJOqvlsDEiDyMrzlA'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2011-08-17',
      'funny': 0,
      'review_id': 'jaH1ZzTCgPAUZ5ktpcNBfA',
      'stars': 5,
      'text': 'AMAZING! Went at tea time on Monday night and couldnt have left fuller or happier :) Waiting staff are very very attentive and friendly. I had smoked salmon salad for starters and mussels in white wine and garlic for main and it was superb. It is a wee bit more expensive than other places but I would say its definately worth it. Food served very fast as well.',
      'useful': 0,
      'user_id': 'vbq-811-_beGtp-SSQhC5Q'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 1,
      'date': '2010-04-14',
      'funny': 1,
      'review_id': 't0f4p1JLscASG2LZD13MCA',
      'stars': 5,
      'text': "Everybody in my family likes something different. My Dad likes Italian, my Mum French, myself Indian and my sister Chinese. The problems it causes when we each want our restaurant to provide something different. As a comprimise, we often decide that a restaurant serving British food is the best option - but then its time for the real battle to begin. Light and fresh sea food or good and hearty steaks?\n\nIt seems then that I should introduce them to Mussels and Steak Bar - which, as you would expect, serves Mussels and Steak.\n\nIts no secret in the restaurant that the produce is local (or at least Scottish) and the ingenious menu combines seafood and heavy meats in brilliantly tasty ways. The sauces used are outstanding, if a bit 'out there' and incredably mouthwatering (In fact my mouth is watering as I type).\n\nThe restaurant itself is simple and practicle - and despite the annoying big screen TV's - has a seaside town feel to it,matching the down to earth cooking style. The staff are friendly and helpful, while the bar offers a large range of drinks.\n\nSo there you have it, a restaurant which can cater for all appitites and in a mouthwateringly excellent way.",
      'useful': 1,
      'user_id': 'SxV1Jq7UANuSYpn42JXvOA'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2014-08-10',
      'funny': 0,
      'review_id': 'Aqur97gqGmgSTP92QQ4W6w',
      'stars': 4,
      'text': 'We were in for the Edinburgh Festival and saw our favorite tapas place replaced by Mussels and Steak. \n\nThe atmosphere looked a little cold and modern but the menu was interesting. \n\nIn a word, the mussels were terrific! We tried the bacon and cream. Plump mussels, huge pot, delicious.\n\nOur scallop starter was equally good. The steak was fine but next time we are just doing mussels. This is really their thing.\n\nNot five stars as they just opened (second location) and fumbled a bit with forgetfulness and delays.\n\nThe staff and owner were very friendly and apologetic. Highly recommended\n .',
      'useful': 1,
      'user_id': '7-ahDxB3qUg6nf3tvfG5Nw'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 1,
      'date': '2012-12-13',
      'funny': 2,
      'review_id': 'BBUGwYfheR4YJL_77_o55A',
      'stars': 4,
      'text': "I must say the name of this restaurant is straight to the point and those two things are exactly what my friend and I ordered. The steak was alright, I've definitely had better, but I will say that the mussels were pretty damn good. Very flavorful and piping hot when they arrived at the table. They weren't as good as the ones I had in Belgium, but they definitely weren't bad, and there was also plenty to share for two people. The restaurant itself isn't all that fancy but their service is. They make sure you're taken care of throughout the entire meal. \n\nOh, you must get the Banoffee Pie. What's better than Banana + Toffee + Graham Cracker Crust. Absolutely nothing. It's heaven in your mouth.",
      'useful': 3,
      'user_id': '3jjiY5D7oIlKeiCIZGHQew'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 1,
      'date': '2013-08-20',
      'funny': 0,
      'review_id': 'bIOQ7AOyi7BWGT_9buM0Zw',
      'stars': 5,
      'text': "I don't like mussels.  They are too strong and too fishy! Don't care for the smell and don't care for them in soups or stews.  Until now I would have told you that I'd rather eat rattlesnake.  Appearantly in Scottland, there is an entirely different species of mussels.\n\nWe stopped in at the Mussel and Steak Bar for dinner before walking up to the castle for the 10:30 Tattoo.  Our travel mate ordered the mussel starter.  It arrived with green sauce on it.  OMG, it smelled amazing.  So much so that I had to taste it.  The little mussels were tender and steamed to perfection with a light pesto sauce driselled over them.  It is such a great combination!\n\nI had the surf and turf for dinner.  The mussels were done in white wine and garlic and the ribeye steak was tender, juicey and cooked exactly as I ordered it. All four of us agreed that our meals were delightful.  The price was not out of line.  Service was spot on.",
      'useful': 1,
      'user_id': 'ZgMiIZC1sH6giDYVNpx_XA'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2013-10-05',
      'funny': 0,
      'review_id': 'gYeiI_21LzFWidjHbNkiyQ',
      'stars': 4,
      'text': "I went here recently for a colleague's leaving lunch. There were 8 of us in our group & it was quite busy. \n\nWe ordered from the lunch menu - 1 course £7.95, 2 courses £9.95 or 3 courses £11.95.\n\nMost of the group ordered starters & the haggis dish was the most popular. \n\nI saved myself for the 6oz rump steak which was awesome. The chips were chunky & crispy, just how I like them & the mushrooms were earthy & full of flavour. My steak was perfectly cooked, moist & melt in the mouth. \n\nA colleague had the seafood curry. It was full of enormous mussels & other fishy delights. It smelled amazing & he said it was the best fish curry he's had. The only problem being that it took so long to arrive he didn't have time to finish it before having to get back to work.\n\nUnfortunately the service was slow and I we all had to skip dessert & coffee as we needed to get back to the office (despite being there over 90 mins).\n\nA lovely lunch with great food & company. It was a shame the slow service meant it was a wee bit stressful. The front of house staff were lovely maybe they just need another pair of hands in the kitchen.",
      'useful': 0,
      'user_id': 'R9UNH0rb3a_glQlk4ylKCw'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2015-07-21',
      'funny': 0,
      'review_id': '5gXLsbta81wMFknMKdDBwQ',
      'stars': 4,
      'text': "On our last day in Edinburgh, I still hadn't tried Haggis.  And I really wanted to try it.  The waitresses we excited to watch me try it for the first time, they were chatty but not overly.  They were attentive, but not overly.  I had a good time, but next time I'd like to take my time and order more and stay longer.",
      'useful': 0,
      'user_id': 'bepOkIaK9763sq3S-zsGEQ'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2014-02-09',
      'funny': 0,
      'review_id': '2pofOFx9JttlOFs4hUoT7A',
      'stars': 5,
      'text': 'An excellent experience - all of the food was of a high quality, as was the service..\n\nWhether it was scallops, mussels, oysters or plaice, the seafood was first class and as fresh as the sea itself.\n\nSteaks were also delicious, both ribeye and sirloin.\n\nThe wine choices worked well with the food and the choice of whiskies suited our need to try something different. \n\nStaff were terrific. Attentive, friendly and showing real concern for our table.\n\nWe will be back.',
      'useful': 0,
      'user_id': 'HRzN56OMH60f2JdEVhI8QQ'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2017-07-18',
      'funny': 0,
      'review_id': 'lxNDA2vZqt7eZ0k1hg5hcg',
      'stars': 5,
      'text': 'I went here for the langoustines, and they did not disappoint.  They were nicely cooked and flavored with garlic butter.  The main course of mussels came with at least 30, and they were delicious as well (we had the chili seasoning). Others in our group had steaks that were nicely cooked (medium rare) and tasty--great chips, too. Our waiter was a bit casual with the service, but the other wait staff picked up the slack.  Overall, it was an excellent Edinburgh meal, with ingredients that featured the best that Scotland has to offer.',
      'useful': 0,
      'user_id': 'p3YrS5V34KBIa3cYutL_Dw'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2016-07-31',
      'funny': 0,
      'review_id': 'h0S3VvpTFURIKAecNyE5tQ',
      'stars': 5,
      'text': 'Very easy to find. Seafood place with reasonable price. We enjoyed both the mussel and steak :)',
      'useful': 0,
      'user_id': 'W9oNoeByolGx-11pPfPtOQ'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 1,
      'date': '2009-02-06',
      'funny': 1,
      'review_id': 'V5RxjppLE-M_nFgIF2xFnw',
      'stars': 5,
      'text': "One of the best spots for seafood in Edinburgh, and one of my favorite restaurants in the world. \n\nLocated just across the street from a longer strip of restaurants and pubs in the Grassmarket, it would be possible to pass the Mussels Steak Bar without giving it a second thought. That would be a huge mistake.\n\nI've made about 12 trips here for lunch and dinner, and every time I've always been greeted by an unbelievably friendly waiter or waitress. They start you off with some of the best crusty bread and softest whipped butter you'll ever taste, and they are happy to replenish it many times over.\n\nMy favorite way to start a meal is with a kilo pot of their mussels, which are all ridiculously fresh and delicious, particularly with the white wine, shallots, garlic and cream sauce. Both the steaks and the seafood pasta (with loads of mussels, scallops, prawns, and calamari) are excellent main courses, and their fries with 'beef dripping' are some of the best I've ever tasted. Their wine list is not huge but offers good selection at reasonable prices. \n\nThe meal can definitely run on the pricey end depending on the conversion rate (if you're an American) but I have never walked out of this place unsatisfied. Highly, highly recommended!",
      'useful': 3,
      'user_id': 'jcNeSuL83hbsHyFDnbfHvw'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2015-06-28',
      'funny': 0,
      'review_id': 'mbTXVJxfZlgI5UAfEputjg',
      'stars': 4,
      'text': 'Great steak, great mussels, and great cocktails. Fun vibe in a great neighborhood. Get the chips fried in beef drippings.',
      'useful': 0,
      'user_id': 'wTJhR3NdvpCB2ZAx_qRiqQ'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2016-03-20',
      'funny': 0,
      'review_id': '-kBwORytF9aJ2PPivv39Lw',
      'stars': 4,
      'text': "Service was excellent!! The server was very attentive & accommodating. No one rushed or hurried us out even though we were the last to leave that night. Make reservations in advance if possible. \nThe Mussels were not bad although I had to order bread as it didn't come with bread. The scallops & bacon appetizer was a delicious combination. Yum! The Surf & Turf lobster tail & braised beef was my favorite entree and a big portion. The sirloin steak w/carrots & cauliflower was great. However, the mashed potatoes were not that flavorful. \nDessert - the chocolate mouse was good but the chocolate hazelnut (special of the day) was amazing! Must order this if you get the chance.",
      'useful': 0,
      'user_id': 'N67MEx1K3shsYoR8nCAtbA'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 1,
      'date': '2015-03-22',
      'funny': 1,
      'review_id': 'Cj_Cu6xvzKgbViE1Btat8A',
      'stars': 5,
      'text': 'Came for a late lunch and enjoyed great food, friendly service, and an overall excellent value. I recommend it.',
      'useful': 1,
      'user_id': 'mdc0aTVned3fPyRb77a0bA'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2015-08-14',
      'funny': 0,
      'review_id': 'EuItnAFcC7BvskhO4GEscg',
      'stars': 5,
      'text': 'Our meal was stupendous! Got the surf and turf. Ribeye steak was juicy and cooked perfectly. The mussels were cooked in a whisky, bacon and cream sauce. OMG! Unbelievably delectable. Portion sizes were enormous. Smoked salmon duo for appetizer was just as good as the main course. Overall one of the best meals we had in Edinburgh.',
      'useful': 0,
      'user_id': 'Q1NOOUUhY-jQc2y5MZgdhw'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2014-06-07',
      'funny': 0,
      'review_id': 'IrchMqHAL3mZbp02GeDThg',
      'stars': 5,
      'text': 'Wow. This place was Sooooo good.  Oysters rib eye and cheesecake were all incredible. The side of mushrooms were also very good. The prices are great for what you get.',
      'useful': 0,
      'user_id': 'y6m-jBnQ-iRhKog8VThwJQ'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2013-10-02',
      'funny': 0,
      'review_id': 'qcTJVZ5R_-Ef0_tKnPKneg',
      'stars': 4,
      'text': 'Fantastic location but very small on the inside, can be a long wait.  The seafood was only slightly above average in freshness and quality, and small portions, so had to knock off one star.',
      'useful': 1,
      'user_id': 'WEO4AW75i86EiMEfH0mRfw'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 2,
      'date': '2011-04-05',
      'funny': 1,
      'review_id': 'N2YP16zgWMdueJ_3e2vTJg',
      'stars': 3,
      'text': "So you get mussels or you get steak.  We both got the steak.  It was our first day in Edinburgh and the restaurant was recommended by a friend.\n\nFood was okay.  My steak wasn't all that flavorful and the chips were a tad bland as well.  Also, my food arrived to the table warm, not hot.  Y'all know how I feel about that.....\n\nA little pricey but not really worth it.\n\nService was very good.",
      'useful': 3,
      'user_id': 'kwLjniiYa1H57LwQX2TGIQ'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2009-01-07',
      'funny': 0,
      'review_id': 'Zbzh9ihIEPJCX0QgEjyDag',
      'stars': 5,
      'text': 'Wonderful steak, wonderful mussels, great value!\n\n\n A great place for meat and fish lovers like me - I highly recommend.\n\n\n Ess',
      'useful': 0,
      'user_id': 'vQnR9L9HILjw4PgWB8q1cQ'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2010-02-27',
      'funny': 1,
      'review_id': '26Z04ZSvjBclBLxQ21WdAQ',
      'stars': 4,
      'text': 'The other reviews are  surprising. The first surprise is that a reviewer is surprised to find that Scotland, the north part of Great Britain (a well known island), has decent seafood. The second, and more disturbing, is the complaint that a restaurant with the name "Mussel and Steak Bar" has a fairly short menu. One might have thought, nay expected, that the menu would be mainly ...\n\nOn a positive note, what I can say is that they provide first class food. What they say they do they do well. The steaks, in my experience, are meltingly tender. The chips (US: french fries) cooked in beef dripping are crisp and moreish.\n\nThe restaurant is good value because not a  morsel will be left on the plates.',
      'useful': 2,
      'user_id': 'QyP254XCNlI7DvPsze5LQw'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2016-10-08',
      'funny': 0,
      'review_id': 'ldoZRpriyacPFVkvtWqGGQ',
      'stars': 4,
      'text': "Stopped for lunch and we weren't disappointed.   It was fantastic.  My wife had the mussel special and I had the hot seafood platter.  I watched them deliver a few of the Tomahawk specials and they looked incredible.  My only disappointment was that the hot seafood platter was mostly mussels and not very varied but still well done.   The outside look of this place does not represent the quality inside.  Service was great.",
      'useful': 0,
      'user_id': 'xXm3sSVnn4mPFrTbph78gA'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2016-08-12',
      'funny': 0,
      'review_id': 'dW3MIOh9a1_ACWhcexIUtA',
      'stars': 5,
      'text': 'Stumbled upon this gem. An easy wLk off the Royal mile and below the castle.\nThe mussels were tasty and the scallops were terrific.\nThe cost was reasonable and the service was excellent.',
      'useful': 0,
      'user_id': 'QPYLJEzB07huiRnhtfrf6A'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2017-04-13',
      'funny': 0,
      'review_id': 'd9kdJMmc_wOBh9Kv-9GK3g',
      'stars': 3,
      'text': "Came here with family for dinner on a Saturday night. We booked in advance, but the place wasn't terribly busy in the end. The atmosphere is quite formal and a bit quiet, rather than bustling and busy. I went for the red curry mussels, as did my husband - my parents got the traditional with shallots and cream. The servers gave us the wrong mussel pots which took us a while to figure out, as all the sauce collected at the bottom and wasn't really coating the mussels, so we honestly couldn't tell whose was whose initially. We had to take some of the huge portion of mussels out and put them to the side, so we had space to use our spoons to coat the remaining mussels in sauce. Once we managed that though, the mussels were very tasty and fresh. The sauces were mild, but added a nice subtle flavour to the dish. Sides were decent but basic - slightly soggy fries, steamed seasonal vegetables, nothing terribly exciting. The price can't be beat though, at £14 for an enormous meal.\n\nNot the most memorable of meals, but an enjoyable one. I think a loosening up of the atmosphere would go a long way.",
      'useful': 0,
      'user_id': 'yaBIVHDxaUBN2YHQ8-YiuQ'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2012-07-27',
      'funny': 0,
      'review_id': 'Wm4TI3EnVaC35DQyZ2WBnQ',
      'stars': 4,
      'text': 'Very tasty food, great beer and really friendly service!!',
      'useful': 1,
      'user_id': 'qKzcpJh5CAXgXzfwC5NMhg'},
     {'business_id': 'g_fGPZJlGeKGLgC-j-tj6w',
      'cool': 0,
      'date': '2013-07-04',
      'funny': 0,
      'review_id': 'HMRA6uHM6qT3dfcndY_a4Q',
      'stars': 4,
      'text': "I never fail to enjoy a good feed at the Mussel and Steak Bar. Pot of mussels with white wine and garlic sauce does the job perfectly. The ambience is pleasant, but it's better to sit downstairs as the upstairs bit feels a bit empty and lonely I think. \n\nFood is generally lovely. Staff are generally lovely. Place is generally lovely. Go!",
      'useful': 2,
      'user_id': 'fcMTpwfLS9F5DWTqlp8ktQ'},
     {'business_id': 'Jdqg97lLo_qJlCqU9RjIEA',
      'cool': 1,
      'date': '2012-09-08',
      'funny': 1,
      'review_id': 'Bz5kR052DviDGQGV_q8VUQ',
      'stars': 3,
      'text': "Dipped cone all the way! This location is tricky because it gets very busy in the evenings. Best route, don't go in the drive-thru, park your car and order inside. It's quicker and easier!",
      'useful': 1,
      'user_id': 'j5yfS1QjGwNLQ0h4_wDyxg'},
     {'business_id': 'Jdqg97lLo_qJlCqU9RjIEA',
      'cool': 1,
      'date': '2016-09-17',
      'funny': 1,
      'review_id': 'Sc2eLc1kTsehqraMhrT1zg',
      'stars': 5,
      'text': 'Always amazing food and service. Love their chicken Parm sandwich. Their blizzards are awesome too.',
      'useful': 0,
      'user_id': 'dk8-B6hKb1XOmYJKC4Ig3Q'},
     {'business_id': 'Jdqg97lLo_qJlCqU9RjIEA',
      'cool': 0,
      'date': '2016-07-21',
      'funny': 0,
      'review_id': 'Qpz73uZ60Gk7MSKbD6Y6_A',
      'stars': 1,
      'text': 'This place is literally the worst. Hopefully the got rid of the awful management, because the owner is actually sort of nice. I ordered ice cream and just got a sloppy mess and no apology.',
      'useful': 0,
      'user_id': '9RG87QSBaTmcbDu3ydQJ2g'},
     {'business_id': 'Jdqg97lLo_qJlCqU9RjIEA',
      'cool': 0,
      'date': '2017-01-28',
      'funny': 1,
      'review_id': 'NdjK8rttbevfoDwgXcL-eg',
      'stars': 3,
      'text': 'Very nice and new. Staff is always friendly but the drive thru is ridiculous. Two burgers and a diet soda and Im still sitting here 10 minutes after the fact? People have stuff to do. I could cook a meal at home in this time.',
      'useful': 0,
      'user_id': 'y6IogOYLjAs8Mksx62vnDQ'},
     {'business_id': 'Jdqg97lLo_qJlCqU9RjIEA',
      'cool': 0,
      'date': '2014-01-17',
      'funny': 0,
      'review_id': '2PoRWtHb6ulqr8AJv_QGhA',
      'stars': 3,
      'text': 'Nothing special about this place, their ice cream is ok. On busy nights drive thru is insane and impossible to turn left out of here. Prices are high too. The employees are really nice and it is a clean store.',
      'useful': 0,
      'user_id': 'VKwI11qTXOxHmlh2yr_2TQ'},
     {'business_id': 'Jdqg97lLo_qJlCqU9RjIEA',
      'cool': 0,
      'date': '2014-02-27',
      'funny': 0,
      'review_id': 'WlqmIvizN8A_oKNR-MPuPw',
      'stars': 4,
      'text': "I love this location! The staff is very friendly and helpful. Also, you just can't beat the $5 meal special.",
      'useful': 0,
      'user_id': 'sxb4v27_35Lp1yktej7ZEQ'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 1,
      'date': '2013-11-19',
      'funny': 0,
      'review_id': 'FsS5TUFPI8QJEE60-HR3dw',
      'stars': 2,
      'text': "Wished it was better..\nAfter watching man vs. food I decided to stop by, décor was not that homey and welcoming, and the neighborhood was bad, but nothing I haven't been around before.  The ribs were very fatty and grisly, it was disappointing and I didn't get enough sauce and when I asked for a little more they wanted to charge me, the coleslaw was awesome!  I noticed a hair in my food and it turned me off to the rest of it, so i threw it away , I wont be returning...\nsorry guys",
      'useful': 1,
      'user_id': 'bWh4k_cCuVt5GLVd33xIxg'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 1,
      'date': '2014-12-18',
      'funny': 0,
      'review_id': '7xGHiLP1vAaGmX6srC_XXw',
      'stars': 4,
      'text': "Decor and service leave much to be desired, but the food is worthy of 4.5 stars.\nI've eaten at both locations several times now over the past year, since moving from San Diego, and I think the food is very good.  The fried chicken and Polish Boy in particular are awesome.  The ribs are just average. good sauce, but not super tender and have not been smoked enough for my taste.",
      'useful': 0,
      'user_id': 'nQ4e81UdfczimYcIUtO3HA'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 1,
      'date': '2014-09-12',
      'funny': 0,
      'review_id': 'ZWlXWc9LHPLiOksrp-enyw',
      'stars': 5,
      'text': "My husband and I ate here tonight for the first time and it will NOT be our last! The fried chicken is best I've ever had  and my ppl are from South Carolina! His sauce is very very good! And the owner was very hospitable and spoke to us about the  history of this famous restaurant. They will be taping on Rachel Ray's show next week! Kudos y'all!",
      'useful': 0,
      'user_id': 'gJPa95ZRozMhiOqvENpspA'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 1,
      'date': '2012-02-28',
      'funny': 1,
      'review_id': 'KpRwKYyQ93ypyDSdA7IXfw',
      'stars': 2,
      'text': 'Don\'t believe the hype. Nooooo! \n\nIn the Cleveland area, there\'s an endless supply of small, unique eateries that take the time to create and serve unique, tasteful, hand-crafted items with flavor. Hot Sauce Williams has created a great commodity in his BBQ sauce, however, ruins the spirit of Cleveland goodies by serving it over generic, tasteless, frozen bag items. \n\nGranted, my review is based off a Polish Boy sandwich combo. I haven\'t tried anything else and after trying the combo, I most likely won\'t make my way back to try anything else. \n\nG Love sang about it, so it can\'t be that bad, can it? Can the creator of such gems as \'Cold Beverages\' and \'Milk and Cereal\' really steer me in the wrong direction? He did, however, I\'ll forgive him with a listening of \'Baby\'s Got Sauce\' (pun intended?). \n\nThe fries are actually store-bought Ore-Ida brand fries. I can remember being served those with dinner when I was a lad in the late 1980s. They are perfectly tasteless. The sauce drenches over them, adding taste, but it just seems incredibly lazy to take something I can easily buy in the store, increase the price, and serve it to customers.  \n\nThe buns are of exactly the same nature - walk in to any grocery store and purchase the cheapest hot dog buns they have - you\'d be getting exactly what is served. \n\nThe sausage wasn\'t bad, had some taste and was of average size (I\'d throw in a \'that\'s what she said,\' but that phrase is so overused, I\'ll let it be). \n\nIt isn\'t terrible, but to be honest, without the sauce, there\'s no way Hot Sauce Williams would still be in business after all these years. The cute nickname helps, I\'m guessing. I\'m guessing a lot of people wouldn\'t keep returning to a "Mr. Williams Barbecue Restaurant" - so I guess the gimmick is working. Plus, it\'d be totally cool of Williams decided to sponsor a MMA fighter, just so there\'d be a fighter with the \'Hot Sauce\' nickname. \n\nThe service is, well, non-existent. I asked for \'slaw on the side, and when I received my meal, my \'slaw was in the middle, which made the bun and accompanying fries soggy on the center and sides - they fell apart upon contact. The \'slaw is mostly liquid. I was reduced to eating a weird liquid mixture of Ore-Ida fries, juicy \'slaw, BBQ sauce and bread. \n\nSort of weird, they also give out cans of soda instead of fountain drinks. \n\nI\'d buy a bottle of the sauce and use that at home, but I really wouldn\'t suggest eating here unless you\'re interested in a bland, sloppy mess of stuff you can grab at the nearest Piggly Wiggly (which, for Northerners, is probably pretty far away, so I\'d have no clue why you\'d travel that far to grab those simple ingredients).',
      'useful': 5,
      'user_id': 'bAwfPH4lXNzgcYp9JFy6ow'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 3,
      'date': '2014-10-06',
      'funny': 6,
      'review_id': 'OZvrgp4vWBsYqIt3-YMSEw',
      'stars': 3,
      'text': "Don't believe the hype!\n\nAfter seeing this location on Man vs Food I had to check it out for myself. I have given this location two back to back chances and it was a disappointment on both occasions. \n\nThe place is defiantly a dive and not the cleanest. One our first visit on a Saturday night we ordered two 4 piece chicken dinners for my wife and I and one 2 piece chicken dinner for our daughter. We finally received our food after a 45 minute wait and the place was completely dead! The four piece chicken dinner comes with a breast, wing, thigh and drumstick, coleslaw, fries and BBQ sauce on the chicken & fries. Talk about a party in my mouth! The chicken was hot and juicy and cooked just right and seasoned to perfection! The fries and slaw were good also.\n\nThe employee's attitudes and customer service skills are in need of as much of a makeover as the dining space needs. The second time we went was the next day as we were on our way back home from Cleveland and got a 20 piece chicken to go. The girl at the cash register was on her phone the whole time I placed the order. I understand that it's a slow Sunday afternoon but, really?! She had a really foul attitude and just couldn't be bothered with taking orders. She was acting like actually pausing her phone conversation to take my order was a major inconvenience to her. \n\nIt took about 25 minutes to get our chicken the second time which is weird as I ordered a lot more chicken the second time. I would defiantly call ahead to place your order because if you wait on your food it WILL be a while.\n\nThe chicken is very good but, not worth the wait and dealing with the employee's bad attitudes. I'd much rather go to Zanzibar for their delicious honey fried chicken. Hell, I'll eeven take Church's Chicken instead!",
      'useful': 10,
      'user_id': 'BjtJ3VkMOxV2Lan037AFuw'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2015-10-07',
      'funny': 0,
      'review_id': 'LDln5I1QXIIinI74RmjuSQ',
      'stars': 1,
      'text': 'Sauce spicy and blan absolutely no flavor the decor screams dont eat here plus the entire staff is moody not to mention chicken wings are 1.50 a piece a true rip off!',
      'useful': 0,
      'user_id': 'ZfqrQVvCfnhlqma6bkEWJw'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 1,
      'date': '2016-06-24',
      'funny': 2,
      'review_id': '_Ob9cAp8dT0w1X5GsZP6vg',
      'stars': 5,
      'text': "Seti's truck wasn't around so we decided to head to a restaurant with polish boys that didn't have wheels. I'm really glad we did that - the combo of the polish...girl...was delicious!  It had the added pulled pork on it, the combo of flavors, perfect.  \n\nWhat the fam really loved was that sauce.  I'll probably never be in Cleveland again, but I'll dream of that sauce.  We got the fried chicken, stuck the sauce on the side...but I can see why people just have it drenching that chicken.  Glad we did get it on the side, though, since it was glorious to dip everything into it.  Almost ordered another fries just as a vehicle for the hot sauce.",
      'useful': 1,
      'user_id': 'AiQL1INckKUk1Bx2WD8ozw'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2015-04-14',
      'funny': 0,
      'review_id': 'g9VGCBuaZ079NtNNfmVPaA',
      'stars': 1,
      'text': "The location on 79th Carnegie, is a hot ghetto mess. I am ashamed of what the establishment has turned into. Maybe I'll try out superior or Lee rd.",
      'useful': 0,
      'user_id': 'J50BanBFNY4naDYAJRAZAA'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2011-04-19',
      'funny': 1,
      'review_id': '3luqqY_D_XQCj_JUOaCL2w',
      'stars': 2,
      'text': "Let me just start by saying that this place did not live up to the Man V. Food hype. I think I have been to better buffets before.\n\nTerrible customer service earned them one out five stars. Extremely rude and inconsiderate server. Took us ten minutes to place our order when at the front of the line. She decided to have side conversations while we were still ordering.\n\nFood: I ordered the Polish Boy and Dark Meat Chicken. \n\nPolish Boy: Polish Sausage, Coleslaw, Fries, and BBQ sauce on a bun. It was drenched in sauce, making the bun soggy. It was not bad but was not all that. The sauce is not very good for BBQ. It is runny like Open Pit and has a little zing to it. Adam Richman was high when he said this sauce was incredible.\n\nChicken: Very good chicken, dark meat. The breading was very crispy and the meat was juicy. I enjoyed this part of the meal.\n\nOverall, I would not return. No place should treat their guests with such arrogance and ineptness. I think I would rather take my money to Popeye's, its most comparable match.",
      'useful': 0,
      'user_id': 'mzexPXl2iuASVakpjODhrQ'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 1,
      'date': '2014-08-31',
      'funny': 0,
      'review_id': 'Tfkyj1e0-_D_t2uos9W5gg',
      'stars': 4,
      'text': "I love Hot Sauce Wiliams. Well, I love their food. Their service is very hit or miss.\n\nThough HSW is a bbq joint, I don't think it's necessarily their strongest point. The bbq sauce is really fantastic. The ribs and pulled pork are really good, but not really great. Granted, that's really comparing to at-home bbq using a smoker. When it comes to restaurants, you will be hard pressed to find better bbq in NE Ohio. And it's definitely better than other Ohio bbq chains like Carolina BBQ and City BBQ. \n\nWhere HSW really shines is their fried chicken and Po Boy and Po Girl. The fried chicken is easily some of the best fried chicken I've ever had - crisp skin, cooked to perfect doneness. But if you go to HSW, be sure to get a Po Boy or Po Girl. What's the exact difference? I don't remember. The better one is the one with a hot sausage in a bun topped with pork shoulder, french fries, coleslaw, and french fries (I think that's the po girl). I cannot eat it with my hands, because it falls apart. If you can eat it without utensils, then give yourself 100 extra Man Points (c). Unfortunately, I dont get to make it to HSW very often, so I've not been able to try their fried fish or several other foods.\n\nFor drinks, they have soda machines and HUGs (kool aid in a tiny barel). Last time I was there, they didn't have a soda machine, so no free refills, which is a bummer\n\nThough I love the food and their pink & blue color scheme outside, I think they only hire terrible people to work the registers. I've literally never eaten there and experienced a cashier that wasn't rude. They not only seemed to hate me on a personal level, but were offended by my audacity of ordering food there. With that being said, the cooks and individuals handing out the food have always been very sweet and friendly. Also, there was an employee my first visit there who helped me open a stuck bathroom door, and he was really great. I'm confused as to why the cashiers have been awful. Granted, I haven't been to HSW in well over a year since I moved out of state, so perhaps their cashiers are less butthurt now.\n\nIf I remember correctly, all locations are not places you want to be late and night by yourself with obvious fistfuls of cash (if ya catch my drift), so if you're an easily scared white person, go for lunch. Also, they used to only accept cash - not sure if that's changed.\n\nIf you're looking for a local restaurant and would like some soul food or bbq, I very highly recommend HSW. Also, the Carnegie location is right down the road from the Cleveland Clinic main campus. So, if you have an appointment with your cardiologist or nutritionist or whatever, you may as well stop by for some lunchtime fried perch and peach cobbler after your appointment.",
      'useful': 0,
      'user_id': 'nNtI1xRXcZs5R3sBEa8xqw'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2016-01-09',
      'funny': 0,
      'review_id': 'sDmMfrsRQtEPYx_X4mmpCw',
      'stars': 2,
      'text': "The food was okay. The wings and fries were not that fresh and very little chicken came on the bone. The sauce was ok. I like Open Pit Bar B Que's (Cleveland, OH) sauce better because it's sweeter. I tried this because it was on the travel channel. It really wasn't that good. I like Harold's Chicken Shack in Atlanta and Chicago way better when it comes to chicken and sauce.",
      'useful': 1,
      'user_id': 'GKXrTZecEsqWuACpRNyuxw'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 1,
      'date': '2015-06-07',
      'funny': 0,
      'review_id': 'DR0dmQGFGQQEgn_BvG3CCw',
      'stars': 4,
      'text': "Best fried chicken out there. Skip the hot sauce, it isn't hot, and is frankly pretty disgusting. The various sides on offer are mediocre. There is a high likelihood of there being no food in the restaurant, especially later in the day, so call before going. Seems like it's takeout only in spite of having a large restaurant space.",
      'useful': 0,
      'user_id': 'tNy6TYGSOEgtFxYPKM-jrw'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 1,
      'date': '2014-11-21',
      'funny': 0,
      'review_id': 'DySF6mE1hZ2GwzPA7sVdEQ',
      'stars': 5,
      'text': "1st time visitor. Portions were great and the personable service really gave it that home feel. Had the Rib Tip....my oh my, shut my mouth.  What was that devine creation?  Go get you some now! Good food + nice price = great atomphsere.  Enjoyed learning their history via the cook during her lunch break. You know if Samuel Jackson and Charlie Sheen dine there it's gotta be good.",
      'useful': 0,
      'user_id': '2jlimdYxRiZt0-OegVqvqQ'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2015-05-04',
      'funny': 0,
      'review_id': 'R6wkvCo8OtSQetFyhTYGsA',
      'stars': 3,
      'text': "Hot Sauce Williams has some pretty poor BBQ, mediocre fried chicken, and an odd sauce that they'll put on either one.  Of course then there's the polish boy. I'll explain.\n\nThis place has become a Cleveland landmark, probably more from visits of Anthony Bourdain and ManVFood than anything else.   There are a number of similar spots in Cleveland and you have to respect it for what it is.  It's not the kind of place that's trying to win competitions here.  It's more of a fast-food style cheap BBQ.\n\nOn our most recent visit we got a polish boy and some fried chicken.  A polish boy is a kielbasa, fries, coleslaw, and sauce on a bun.  It's an unrefined mess and it's pretty tasty.  In my opinion their chicken is so-so.\n\nThe service seems to be down to business.  I wouldn't describe my experiences as overly friendly.\n\nIf you're an out-of towner I suggest trying it for what it is.  The polish boy is uniquely Cleveland.  The neighborhood is characteristic for much of the east side of the city.  It may not be pretty but it is what it is.",
      'useful': 1,
      'user_id': 't8gR8NCD56bF1XcAjwgkEw'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 2,
      'date': '2011-07-16',
      'funny': 0,
      'review_id': 'g3KutFVxg68VecHvFzkANA',
      'stars': 4,
      'text': "Despite being slathered in their sauce, this fried chicken surprisingly maintains its crispiness and isn't the least bit overpowered by the flavors of the sauce. Actually, the sauce compliments the chicken so well that it had me asking for more- I even bought a bottle of their original sauce for the road since we were just visiting Cleveland.\nOf the two sides that were ordered, the mac n cheese tastes similar to a homemade baked dish packed with cheesy goodness. The fried okra was decent, but was coated in so much batter that I could barely taste the okra.\nIn addition to the good food, the service is also worth mentioning- everyone behind the counter was hospitable and made for a great first visit to Hot Sauce Williams",
      'useful': 0,
      'user_id': 'hVOJJaaKRdQvB__BMq5yDA'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2016-07-13',
      'funny': 0,
      'review_id': 'ijrFuMmsYw8vI--DjkAQtA',
      'stars': 1,
      'text': "I can't remember the last time I ate here because it has been that long. Well it will be my last... The menu on Yelp quotes one price so a slab dinner 19.95 when I pick up my food it's $29. The girl at the register rude even when you call in to order  tone very dry... Food over I should have saved myself the trouble and gas and just pulled out my own grill!",
      'useful': 1,
      'user_id': 'gSwBZSw7lem2IEB8jId4Gw'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 1,
      'date': '2012-10-15',
      'funny': 0,
      'review_id': 'RD9R88hIOdkfWvsxXs3Swg',
      'stars': 5,
      'text': "I'm going to keep my review short and sweet! \n\nPolish Boy and the wings, OFF THE CHAIN!\n\nNuff siad!",
      'useful': 0,
      'user_id': '2XuP-lUVBF09qPmCpZ8oQA'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2012-06-13',
      'funny': 0,
      'review_id': 'dMyrZA_EXD9y_UqK7-Hj0Q',
      'stars': 4,
      'text': 'Hot Sauce Williams is a Cleveland tradition! I will always stop here for the fries with sauce!',
      'useful': 0,
      'user_id': 'NGnS6xFCefAB8JsuuNIIYw'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2011-01-31',
      'funny': 1,
      'review_id': 'C_9CV4C722vQ_Atd9_LDRQ',
      'stars': 5,
      'text': "hot sauce williams is amazing, but it is certainly not a place to go for a low calorie meal. fried chicken, st. louis style ribs, mac n cheese, sweet potato pie, etc. it's all awesome. the fried chicken is done perfectly! my mouth waters just thinking of this place. its a simple place, you order from a counter and can either take your food to go or eat it there. all that matters is the food is killer!",
      'useful': 1,
      'user_id': 'igcnVpgAYC2EMP0sXZmuTQ'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 3,
      'date': '2013-02-02',
      'funny': 7,
      'review_id': 'RYHKUSUyPeaU38RytDtZww',
      'stars': 2,
      'text': "I've been going to Hot Sauce Williams for the past 15 years and this place is going down hill big time!  The service continues to get worse and the attitude of the workers is piss poor.  They ignore you and can be very rude as well.  Also, the place is very dirty and I would love to see the health inspection report.  \n\nI will admit there Sauce is some of the best I have ever had.  The wings are just not as good as they used too, but the Polish Boy is still very good.  The Mac n Cheese is very overrated and DO NOT ORDER THE RIBS.  They are like chowing down on a gristled piece of fat and that has the texture of rubber.  My friend tossed them away they were so bad.  \n\nVery sad to see an Cleveland Institution goes down the tubes like this and hope they can improve but that's the last they have seen of my business.  \n\nP.S-Don't even bother going through the drive thru unless you have 20 mins to spare, no joke.",
      'useful': 10,
      'user_id': 'kEEx6yEf81i9Wk5Ww1q6GA'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 3,
      'date': '2009-03-06',
      'funny': 3,
      'review_id': 'yUeBnTvqPNB8Ay_5PS3kWg',
      'stars': 2,
      'text': 'My girlfriend and I been dying to try this place for sometime, but never found the motivation to go until we saw the Anthony Bourdain episode when he was in Cleveland. We ordered the Family Meal (Full Slab, 8 Pcs of Chicken and 2 sides). When we ordered most of the sides were not available so our choices were limited.  I\'ve seen their slogan around town "Voted #1 for Ribs" so we were excited, but after taking our first bite into those saucy ribs, we looked at each other and said "meh" as we have had better. The chicken on the other hand, drenched in BBQ sauce, was interesting. It was saucy yet crispy so it was an interesting dish, we probably favor the chicken over the ribs. The family meal is large so make sure you have plenty of mouths to feed, as for us we\'ll need to hit the gym for the next several days to burn off the calories we just packed on.',
      'useful': 3,
      'user_id': 'Jyi0WJt0UfGdEg0grR38ZA'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2015-01-04',
      'funny': 0,
      'review_id': 'q0cL9c_IQPjaGFIA64COFA',
      'stars': 1,
      'text': "Worst food I've ever had the laziest service. We ended up throwing 35 dollars in the trash. If you like getting sick or experience all time lows check this place out!",
      'useful': 0,
      'user_id': 'RyixLoQmLZt5BWp6xJZ5uQ'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2015-04-16',
      'funny': 0,
      'review_id': 'tBtVsv9M__Zl4xqQ7xN9AQ',
      'stars': 4,
      'text': "Hot Sauce Williams is a Cleveland institution... Put in other words- you have to try this place at least once. \n\nThe sauce isn't for everyone. It's a vinegar based sauce with a little heat to it. I absolutely LOVE it but I know others who prefer a thicker sauce. I normally go with the 2 piece all white dinner with the sauce on everything. The fried chicken is some of the best in Cleveland. The fries, slaw and bread are all solid and for what you pay ($6.99 for my meal) you get a ton of food. The inside of the restaurant is very old school-- definitely not a place for suits and ties. It has a drive thru which is super convenient, especially considering the area it is in.",
      'useful': 2,
      'user_id': '3AC_liBwORA9w18CIrGxcQ'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2015-05-01',
      'funny': 3,
      'review_id': 'qr9hC5-YKLjMBBT-DJ8Vow',
      'stars': 2,
      'text': "My good friend, a native Clevelander, described this spot as a Cleveland staple. In town for a Cavs game & we gave it a try! \n\nMy BF & friend had the pork shoulder & it looked great. They loved it, I tried it & wished I ordered it. It's on white bread & looks a little mysterious but it's delish. I ordered the chicken ...I waited....& waited for this stupid chicken. My friends & BF were done eating by the time I got my chicken *sigh* not cool. Luckily I got my sides first which were mac n cheese (dry and tasteless) and greens (ok). I was so excited for this place & what a bummer. The chicken was juicy but had no flavor. I am a white girl that grew up in Pittsburgh, I'm not asking for much in terms of flavor but my God this was tasteless. \n\nLet's touch on the location, def off the beaten path. I felt as if I were at a fire hall. The decor reminded me of a low budget bridal shower. It's positively outdated which is fine, it's a dive. I wish the food made up for the detoured drive, long wait & weird set up.",
      'useful': 1,
      'user_id': 'II7eBcZJRLwnFtuS6BtVRg'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2011-10-07',
      'funny': 0,
      'review_id': '91fnUQiklT741ypgwrNw-w',
      'stars': 3,
      'text': "Pretty solid / cheap chow.\nBut not BBQ\nLine moved fairly slow but i didn't expect fast food.\nStaff was polite and playful\n\nI ordered the chicken/ribs combo. ($10)\n-The chicken was pretty tasty, but the sauce poured all over it made it instantly soggy.\n-the ribs are smoked/grilled then BAKED. that said they had good flavor and texture.\n-the sauce is just plain not good. tastes like sweetened tomato juice.\nit worked okay on the ribs but ruined the chicken.\n-mac n cheese is meh\n-greens are cooked sweet. not my style.\n\nDef work a quick meal if you are in the area, but don't go out of your way.",
      'useful': 0,
      'user_id': 'FDFJk0feR-RAGHkxRr_3Nw'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2014-08-11',
      'funny': 0,
      'review_id': '1cnIDaTYinT2BNOD8aqCdw',
      'stars': 2,
      'text': 'The polish girl was good overall but the sauce was a little too sweet for my taste. The fried chicken was also a plus.',
      'useful': 0,
      'user_id': 'FkqkyifiYxBaajtYvAL3kQ'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 1,
      'date': '2013-11-16',
      'funny': 0,
      'review_id': 'gClPOdeAxVdcJByX5W5ojw',
      'stars': 4,
      'text': 'I have been going here for at long time and its one of few places I can overlook bad service because I love the food. The wings are great, best wings in Cleveland but the best offering from here is the barbeque sauce. Better service and it would be 5 stars.',
      'useful': 0,
      'user_id': 'C1TC9xbL-rbN8UspYPgXZw'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 1,
      'date': '2014-05-18',
      'funny': 0,
      'review_id': '0uZXvWjKqpYBI8n-eNU8RA',
      'stars': 4,
      'text': "I like to do Man vs. Food tours whenever I visit a city for work, and while planning for Cleveland, Hot Sauce Williams and their Polish Boy definitely made it onto my list of must eats. Man oh man, it did not disappoint. That Polish Boy is AMAZING. I'm a huge fan of your typical hotdog with coleslaw, not so much hot sauce or putting fries on things. But I'm telling you, this thing is spot on! The hot sauce wasn't too hot, the fries melded together with everything else in a special and tasty way, and the slaw and kielbasa were delicious. You have to try it if you get the chance. No wonder it's a signature Cleveland food. \n\nI did do the drive thru (after reading other reviews this white girl didn't want to risk going inside by herself late on a Saturday night) but didn't have any issue with a long wait - maybe 3 minutes tops. Glad there were no harassing panhandlers around either. I was bummed they were out of banana pudding but the strawberry cheesecake was ok. \n\nBottom line: go get yourself a Polish Boy!",
      'useful': 0,
      'user_id': 'ZpiAtR4WReIbMy54vXNqtw'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2016-01-06',
      'funny': 0,
      'review_id': 'n_mIIi1mNSqyOGGNEI9Q8w',
      'stars': 1,
      'text': "All I can say is that both the food and the restaurant were completely disgusting. The pork shoulder tasted like it was two weeks old and reheated in a microwave. Their fries were cold and mushy. The cole slaw was the only edible thing they gave me. The whole meal was so bad I threw it in garbage after two bites. To go along with the terrible food the inside of the restaurant was dirty and had a strange smell. It's probably not necessary to say, but I will never set foot in Hot Sauce Williams again.",
      'useful': 0,
      'user_id': 'hx6_lI8_JlwtCoeVJId9ZQ'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 1,
      'date': '2010-03-15',
      'funny': 1,
      'review_id': 'h4XxiPjntIPhjFAFegQh9Q',
      'stars': 4,
      'text': 'Hot Sauce Williams takes the best BBQ in Cleveland in my option. The home made mac and cheese is astonishing. The chicken is drenched in sauce and you just have to saver the flavor. I love to lick my fingers clean to finish off my meal.',
      'useful': 1,
      'user_id': 'T1XUUXIEU5uj3NmdWAdIlA'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2017-05-27',
      'funny': 0,
      'review_id': 't_JpuqPSeeBpDpWRKb-zFg',
      'stars': 3,
      'text': "Painful to watch this slow disorganized mess try to handle just a few orders at once. Took 30 minutes to get our food and that was because we accepted other chicken pieces than what we ordered because they said it would take longer. Short story long: if you're in a hurry this isn't the place for you.\n\nOn a positive note; the ribs were very good even with a little too much of their hot sauce on them. But the ribs weren't incredibly meaty. The chicken was nicely fried, but they put a bit too much hot sauce on it. The fries and mac and cheese were good with the hot sauce. \n\nAn odd mix resulting in a 3 star experience",
      'useful': 0,
      'user_id': 'eO7Wam6CD5-2xE4tfqFFkw'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 3,
      'date': '2011-12-09',
      'funny': 1,
      'review_id': 'EM1aHX7DsSAKDoqYOocyTw',
      'stars': 4,
      'text': 'The chicken isn\'t what makes this place great.  It\'s good but it becomes something different entirely when you smother it in the signature sweet, juicy sauce.  My God, I\'d smother my limbs in this and eat my own arms.  It\'s that good.  Get that and some of the mac n cheese and you\'re in heart attack heaven.  But it\'s so good you won\'t care!  Also, all this crap about "Oh i might get shot" or "Don\'t venture here at night".  This is Cleveland, kids.  It\'s rough.  But I\'d rather venture into a rough neighborhood than spend my time kickin\' it at Applebee\'s in the safety and comfort of some designed suburb eating crap ass Jack Daniel glazed "riblets."  Do yourself a favor, venture out of your comfort zone and get some good chicken...and extra sauce  You\'ll be super happy you did.',
      'useful': 2,
      'user_id': 'jHX2qMpRIg-W32vJgi50lw'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 7,
      'date': '2010-05-19',
      'funny': 5,
      'review_id': 'QPa95SRzS7JrrCtENK9l2w',
      'stars': 4,
      'text': "This is probably going to be one of my favorite places that I stopped at on my cross-country trip...thanks yelpers for letting me know it's worth going to the shadier part of cleveland (which isn't THAT bad) to get my grub on.\n\nThe two things you have to get when you go to this joint are the macaroni and cheese (my husband says it's the best he ever had and he's a snob when it comes to fatty foods) and the fried chicken (this was so juicy and flavorful on this inside and deliciously crispy on the outside). OMG, seriously run there is you have the privilege of being in Cleveland. \n\nOh, and a few other pointers:\nCornbread was not good at all.\nI didn't try the coleslaw because it was super lathered in mayo.\nWe also had the pulled pork sandwich, which was really good too. The bbq sauce is flavorful, but not hot at all. I liked it on the pork, but I'm glad I didn't get the fried chicken drenched in the sauce - better on its own.",
      'useful': 7,
      'user_id': 'ZJCWUrxd2YzEj5yLshYn8A'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2013-02-19',
      'funny': 0,
      'review_id': 'yUo2luFZRp9_oqRE_6hvjA',
      'stars': 4,
      'text': 'I know others have been moaning about their service, but our experience last Sunday was great. \n\nWe were disappointed about the dining room being closed, but the drive through was open and the staff were very patient with our endless debate about what to order (our son can be finicky). They even chimed in with recommendations when they told us that Fried Okra was unavailable.\n\nThe food was incredible -- and really messy in the car as we scarfed it down on our way back to NYC. We had Polish Girls, Mac and Cheese, Black-Eyed Peas and a Pork Shoulder sandwich. The sauce is really special. \n\nWe strongly recommend it to everyone. Great food, great service, really cheap.',
      'useful': 1,
      'user_id': '-WzyS1gUd85rrgPbLKWnEw'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2011-03-08',
      'funny': 1,
      'review_id': '9aCTKXLxKTNBJEhpXUY-1g',
      'stars': 3,
      'text': "This is a funny place.  Being from Youngstown, and not Cleveland, I went here after my Cleveland Clinic appointment, because I saw this place on Man vs Food.  I went there for one reason only...the Polish Boy.\n\nThe service staff was really under-enthused to wait on me...even though I was only 1 of 4 people in the whole joint.  The 1st lady who I said hi to pointed me in the direction of the stack of paper menus, without saying anything back to me.  \n\nThe Polish Boy, was not on it.  But after asking the lady at the counter, they DO still have it available.  It's a kielbasa on a bun, topped with fries, coleslaw, and BBQ sauce.  The TV show also added pork shoulder to the sandwich, but for some reason, they didn't put it on mine.\n\nIt was awesome nonetheless.  \n\nThe place was a dive, doesn't look that clean, and it's located in a not-so-great side of town, and the service was lacking any personality.  However, the Polish Boy was GREAT.  I would definitely have this again.  I hear their other food is good too, but the PB is all I wanted.\n\n3 stars for the food.  0 stars for the service.\n\nLetter Grade:  B",
      'useful': 2,
      'user_id': 'ZEwpJOG8PFlHiFRdAZZ0iA'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 2,
      'date': '2011-08-18',
      'funny': 5,
      'review_id': 'VzP7FDWc9YwWihxR59rDpQ',
      'stars': 4,
      'text': 'I wont lie.. I was a little scared to enter this restaurant. I have always wanted to eat here but the location on Lee has a horrible reputation for .. well if haven\'t heard.. picture the scene from Animal house when they all walk into the bar where Otis is playing.. Nuf said. \nI had heard that Man Vs Food was at this location, so i figured i would give it a shot. Upon entering most places the food network has been you will notice that not only do the places raise their prices, but pictures and plaques are everywhere commemorating the event. Had you not seen the episode, you would never know they had been there.. Its as if they couldn\'t have cared less.. \nSo i asked the girl.. "Whats good"  and with out even hesitation she fires back "You gotta try the polish girl and you might as well have the fried chicken too.." With that type of certainty.. i had to reply "Well. OK!" She then asked me "You want our sauce on your chicken?" To which i asked " I dont know, do i?" She promptly said yes and ladled it on. For those who dont know.. A polish girl is a sausage (hot dog like) on a bun with french fries, pulled pork, coleslaw and their sauce.. Its about as good of a sandwich as you will ever taste. \nNow fellow Yelpers.. I have been known to exaggerate a little to heighten the effectiveness and entertainment value of a good story.. But i say to you today.... That is the best fried chicken that has ever passed my lips. The very wise and helpful woman who took my order was appreciated ever time i took a bite of that chicken. That sauce.. complimenting an already outstanding chicken breast.. I well up with tears just thinking about how unbelievably good that food was. \n\nOn another note.. Buy a F*cking mop already.. The place is absolutely deplorable. I needed rubber gloves and a hasmat suit to enter the place. Seriously... what is that 10 years of grime on the molding? Its absolutely disgusting. How you havent been sited is beyond me.. Perhaps people are so blinded by the food (as i was) that they don\'t notice or care. You would have had 5 stars, but i just couldn\'t get over how dirty it was..',
      'useful': 5,
      'user_id': 'VOUnXOlll52vwQcJK1tm8A'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 2,
      'date': '2012-06-14',
      'funny': 0,
      'review_id': '2FJ_j0d4jY1hA65v94vc0w',
      'stars': 5,
      'text': 'I had always seen this place growing up, but I never actually tried it until this past Christmas! The fried chicken was probably the best I have ever had; worth every calorie! I had the sweet potato fries and they were great and crispy.  Everyone was really nice; the cook even came out and took a photo with us when I told him that I was visiting and had always wanted to try the place! I will definitely be coming back!',
      'useful': 1,
      'user_id': '9BeV9g26iBVMkDPRWjW3AQ'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 1,
      'date': '2014-01-13',
      'funny': 0,
      'review_id': 'ZUysgXDo-0nv4ZAl2CE4WA',
      'stars': 4,
      'text': "Um, can I just say that I want another 15 orders? This stuff is so so delicious. It has had rave reviews from all sorts of TV shows and I had to try. So last night, after a full weekend of eating, I decided I could just put on some super loose eating pants and go and get my fill. \n\nWe ordered the rib tip meal and the thigh snack. The rib tips were well cooked and the SAUCE! That is the winning ingredient. The ribs themselves were a little bit tough at times, but that sauce is really what makes your mouth water. We also got the macaroni and cheese as a side and as a not usual fan of mac and cheese (I KNOW, I'm the devil for saying such ludicrous things) I wasn't expecting to finish the ENTIRE tub by myself. My boyfriend wished he acted faster, because he just got a small taste. \n\nWhat you have to be careful about when you go late at night is that they close the doors and you have to order in the drive through. The intercom doesn't work so you drive up to the window and order. They take about 5 to 8 minutes to get your order. During that time, there will be homeless people that approach you. If that's not your scene, you won't score some awesome ribs. The environment its in is not great, but the food will surely make you want to come back.",
      'useful': 3,
      'user_id': 'GefiDYsb2U_-3ujGnheSpQ'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 1,
      'date': '2013-05-20',
      'funny': 0,
      'review_id': 'nXVeJxHQ_-vBYIO61lNqUw',
      'stars': 5,
      'text': "Forget about the place and service (which were fine) and let's talk about the food!  The Polish Boy was good, but the fried chicken was AMAZING!!  And the sauce was killer!",
      'useful': 1,
      'user_id': '4p87ApJJWOgqfJ-DnVnVgA'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2014-09-26',
      'funny': 0,
      'review_id': '4wGo32rPfWCt7J6S61pdzg',
      'stars': 2,
      'text': "So torn. I love their food. We wanted a polish boy. Yelp says open until 1:30, it is 10:26 pm and drive thru dude says they're closed. What? If you're normal and visit during normal hours, I gave hsw 5 plus stars. But consistence is nice too.",
      'useful': 0,
      'user_id': 'r_dYnqAQf4_G5T0ekGFyyA'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 3,
      'date': '2014-07-22',
      'funny': 0,
      'review_id': '9gwDQXFyjNuZpY0FfZ4_Ag',
      'stars': 5,
      'text': 'The chicken and po boy was amazing. Possibly the best I ever had. Get it with the sauce loaded on the chicken and fries. Staff was super friendly. \n\nI will come back next time I am in the neighborhood I visited because it wAs a nice walk from the Cleveland clinic. Probably not the best idea at night but during the day it was great.',
      'useful': 1,
      'user_id': 'Sxjc78bgN4xMrIkzvnrUjw'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2015-07-31',
      'funny': 0,
      'review_id': 'hEeajeUzkW2wABDE4SXF2A',
      'stars': 3,
      'text': 'The food is good but the service is slow.  This means do not go there for a quick to go food.  I ordered a side of "mac and cheese" and fries.  The "mac and cheese" was already prepared but it took them 20 minutes to get my fries.  There was no sense of urgency amongst the entire staff, they just assumed everybody came to spend their entire day in the restaurant waiting for food.',
      'useful': 0,
      'user_id': '79Vv70bMS1JC9ASCg4dWqg'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 6,
      'date': '2013-09-15',
      'funny': 5,
      'review_id': 'jjj5tomYOQt1ZBVE67eOTA',
      'stars': 4,
      'text': "Hot Sauce Williams is definitely a guilty pleasure of mine :) My favorite dish from here is the fried chicken and ribs dinner. For sides, I get the sweet potato fries and mac and cheese. They give you sooooo much food for just under $12! \n\nThe servers work only behind the food stand, so you pretty much have to clean off the tables yourself. You may at first be deterred from this place by its lack of cleanliness, but the food is so worth it!  One thing to keep in mind, they don't provide free water, but a bottle of water is only $1. \n\nDefinitely worth checking out!",
      'useful': 8,
      'user_id': 'SaSXRzjHx3SMTNnW90PetA'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 1,
      'date': '2016-05-29',
      'funny': 0,
      'review_id': '79NLGMAGH8DVHcWvj1DIPg',
      'stars': 5,
      'text': "Loved this spot! I got the pork plate which is shoulder with slaw and bread covered in sauce. Came with fries that I had them add sauce to. Just amazing. I've had bbq so many ways and I appreciated how unique and delicious this spot was!",
      'useful': 1,
      'user_id': '8RegOEQ0s89jMHGqs12RYg'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2011-04-15',
      'funny': 1,
      'review_id': 'CR4r0GkyYxK5FP7EaFEOgg',
      'stars': 2,
      'text': 'I went here on my lunch break and I waited for 20 minutes just for a polish boy! It was pretty good but waiting for the very slow and rude staff was frustrating.',
      'useful': 1,
      'user_id': 'G5reVMkdwUc-souSaLdm_Q'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 1,
      'date': '2015-06-27',
      'funny': 0,
      'review_id': 'pHZqrQnoYFjAPpUZywf--g',
      'stars': 3,
      'text': "My son and I stopped here on our way home from vacation. I wanted a polish boy, but Seti's is closed on weekends. This place had a good rep for theirs. I had the polish girl (polish boy with pork shoulder). It was tasty, but didn't think its size justified its price. Paid over $9 for something noticeably smaller than a regular hot dog. Yeah, it had fries and a can of soda, but still.....The sauce was good, but don't expect it to be hot. We waited a long time in an empty restaurant for anyone to come up to the counter to take our order. She saw us, but kept wandering around not really doing anything. Seemed like they were out of everything. No chicken sandwich, no corn fritters, no fried chicken breasts, and a few other things the people behind us ordered. The woman taking our order acted like we were bothering her and that she was doing us a huge favor by doing her job. There was a cheap boom box blaring awful music you coudn't get away from no matter where you sat. I don't care one way or the other about decor, but if you do, don't go here. Table cloths look like cheap shower curtains a 4yr old picked out, and there is a cheap entertainment center in the dinning room. Who puts an entertainment center in the middle of a restaurant, lol? All in all, the food tasted good, but wasn't the best value, the service was crap, and the decor was laughable.",
      'useful': 0,
      'user_id': 'WtypKCf3rzL50b5WW8p-bw'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 3,
      'date': '2011-02-28',
      'funny': 3,
      'review_id': 'YJhSfekObpQ-rBPmCKaZMA',
      'stars': 4,
      'text': "If you don't like this place, chances are you don't know what good barbeque is.\n\nI had the rib tip dinner (Off menu btw). This is a good alternative to the full rib dinner. It is still a ton of food, but not quite as much! The sauce is delicious.\n\nAs of 27 Feb 2011, you get a free plastic Pepsi 'Courtesy Cup' if you buy any one of their sodas.\n\nYes it's dirty, informal and right in the middle of an extremely sketchy neighborhood. IMO Cleveland needs more places like this. There are way too many places that wish with all of their might to be located in Manhattan or LA. Guess what? You're in Cleveland. Now ditch the valet and the uppity atmosphere, and give me some food.",
      'useful': 3,
      'user_id': 'hWGZN0cma0MtPk0PpDHnhw'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 3,
      'date': '2012-06-25',
      'funny': 1,
      'review_id': 'zIB6nQ8EL8sQqlRDFUxJHg',
      'stars': 4,
      'text': "Where do I start....My family & I went here on Saturday night and ordered 2-4 piece chicken dinners & 1-2 piece chicken dinner.  We finally received our food after a 45 minute wait!!!  The chicken dinner comes with a breast, wing, thigh, drumstick coleslaw, fries and BBQ sauce on the chicken & fries. Talk about  heaven!!!! You can get the sauce on the side though if you're not too adventurous. \n\nNow the service, that's another story.  Let me just say they need a tutorial on  customer service.  I'm a person who believes in second chances, so we tried this place again on Sunday and the service was the same....The only thing that changed was the table cloths?.....The reason why I gave this place a 4 star is because of the food but the service could use a lot of upgrading!!!\n\nI would suggest that you call ahead to place your order because if you wait on your food, it's like watchin' paint dry!!",
      'useful': 3,
      'user_id': 'djXzQSkIoDGP-IdycN2zjg'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 4,
      'date': '2011-01-18',
      'funny': 4,
      'review_id': 'DM0Y5vKoIH8myhsm6tnvJg',
      'stars': 3,
      'text': 'oh God....I think this may be my first review where I add the tag line "for Cleveland"....I\'m so sorry that I have to use it here...\n\nSo, this is decent BBQ....for Cleveland.\n\nDecided to commemorate MLK Day and a former Yelper by patronizing Hot Sauce Williams.  We had both been wanting BBQ all day long and figured today was just a good a day as any to give it a shot.  Reading the previous reviews, I found it quite funny that we wound up getting the exact same thing as Amy T. and her dining partner!  Ribs, coleslaw and fries for me...4 pieces of fried chicken, mac and cheese and fries for my gluttonous dining partner ;o)\n\nBut, getting to the point of ordering was an ordeal in itself!  We walked in and there were no menus posted....looking around for a couple of minutes, I was almost ready to go up to the girl behind the counter and ask her "so, what do you think we should get"...but I was kind of afraid to do that because of the response I envisioned in my head *cringe*.  So, I was pretty thankful when we finally looked around and found some printed up menus of their offerings....but, in what kind of order did they put these things?  There was no rhyme or reason to half of the menu, very confusing.\n\nOrdering was an event in itself, I started ordering with one girl and then she just walked away in the middle of my order, and then another chick came up and kind of finished it....I mean, I agree with Stef C. on this one, would it kill you to put a smile on your face and actually act like you enjoy getting paid??\n\nPluses, it was cheap, and it was fast, we had our food in less than 5 minutes and were out the door to enjoy it at home.\n\nPortions were HUGE, holy crap you could feed a family of four on the two dinners we ordered, neither of us could finish our meals.  Sauce was decent, ribs were a little grisly, chicken was pretty good....but in all reality, eating BBQ in Texas and Georgia has ruined me.  HSW is decent BBQ, for Cleveland.....this place would be out of business so fast in any of the Southern states due to the attitude you receive behind the counter and for the simple fact that it\'s just not that great when compared to some of the great BBQ places around the country.  \n\nLet\'s just say I think my BBQ craving is over until summer comes around and all the Yelpers can get together and do a proper cookout.',
      'useful': 7,
      'user_id': 'RSAHZjr2yPZHZ-HhrgCrXQ'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 5,
      'date': '2012-01-10',
      'funny': 1,
      'review_id': 'IZpHJ73Ai6daF7VJTBfWqw',
      'stars': 5,
      'text': 'There is a lot of buzz about this place, especially given its feature on the Food Network. Having lived in New Orleans for some time, I was both curious and eager to give HSW a try. (New Orleans is famous for some of its fried/comfort foods.)\n\nThe Bottom Line: this place is rather magical. If you love fried food that is slathered in hot sauce, then this is your place. To be honest, the fried food offers a one-dimensional taste experience; however, the food presentation is beautiful...in an ordered-chaos sort of way.\n\nThere\'s a menu, but you pretty much just get whatever they have available (I.e., whatever is sitting in the warmers behind the counter). I tried the fried chicken and the famous Polish Boy sandwich -- a fried kielbasa topped with coleslaw, fries, and hot sauce. I also "sauced" everything, allowing the kind worker to paint all of my food with the ever-so-sweet-tangy-and-delicious HSW hot sauce. The food was hot, and amazingly, the chicken was as crispy/tasty as some of the famous fried chicken of New Orleans\' Willie Mae\'s Scotch House and Dooky Chase. (Huge success!)\n\nThe decor is the best part of HSW. The restaurant is appears to be a bit rundown, ostensibly dirty, and some of the windows are adorned with iron bars/mesh for safety purposes. While this may seem to be a negative, it is not. These details, coupled with the ambiance of the restaurant\'s neighborhood, offer the adventurous and hungry an authentic City of Cleveland dining experience.\n\nGrade: A',
      'useful': 4,
      'user_id': 'I1Jhth-yxtz4mty6C2sWqg'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2012-04-27',
      'funny': 0,
      'review_id': 'f6lZEx4g8qj-beIv-jPnGA',
      'stars': 3,
      'text': "Decided that I had to check this place out since I am a huge fan of Adam Richman and he featured it on Man vs Food. He also wrote a section on Cleveland in his book, 'America the Edible', and he name drops this spot. \n\nMy fiancee and I were not disappointed, it definitely hit the spot. I loved the sauce. Ryan tried the Polish Boy which basically a kielbasa, with sauce, and cole slaw piled on top. The french fries are smushed in on the side, but definitely a tasty experience overall. I heard that the wait on the ribs was 30 minutes so I opted for their fried chicken which was crunchy greasy goodness. For a fairly quick bite I definitely recommend checking this place out.",
      'useful': 0,
      'user_id': 'zEil7zukF_Xtxq7sqIbyfA'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2014-03-02',
      'funny': 0,
      'review_id': 'SGxZizLl1l_enqADtFZUnQ',
      'stars': 4,
      'text': "Came here on Saturday evening and there was a line about 10 people in front of me. I ordered a 3 pieces white dinner and was expected to get my food within 10-15 mins; however it took me about more than half hour to get the food. It's fair to say that efficiency was not their priority. Despite the service the food was absolutely amazing. Great value with great taste. The chicken breasts were tender and not too oily. The sauce was top natch.",
      'useful': 0,
      'user_id': '-GD1g6P9ntZOljA--y5w8Q'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2016-08-30',
      'funny': 0,
      'review_id': 'sCwDTs4YOoDOF7ikeOWLXQ',
      'stars': 4,
      'text': 'Solid BBQ spot- definitely recommend! \nPros: Reasonably priced and delicious. \nCons: Food was pre-prepared and clearly sitting out a while before we got there. \n\nThe taste and flavor are outstanding! I kept it simple and ordered Ribs and fries. Sauce had me like "mmmmmm mmhhhmmmm" and ribs were like "Ahhhhh yaaa ahhhhh." Great time had by all!',
      'useful': 0,
      'user_id': 'mG1SRElljRwoIMdKGX1Ifg'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2017-06-19',
      'funny': 0,
      'review_id': 'iyEMjTgHEHjfGOwhS73Azg',
      'stars': 2,
      'text': "This place was extremely unorganized. Orders came out slowly and at least half of it was wrong. The air conditioning must have been broken on our visit because it was over 90 degrees in the dining area. The food was pretty good and the only thing keeping this review from getting a one star. We ordered the rib tip sandwich but they came back 15 minutes later and said they were out so we ended up getting the pulled pork. While the food was good ( I would I say 8 out of 10) it wasn't nearly good enough to overcome the disappointment. We won't be back.",
      'useful': 0,
      'user_id': 'hurnfMzgL3Id73mvg_mFGQ'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2010-05-14',
      'funny': 1,
      'review_id': 'GjvWN7Yco0BKFSL7Alhwyw',
      'stars': 4,
      'text': 'Got the chicken and ribs dinner with green beans and fries. All delicious. Pretty much as close to being in the south as you can get in Cleveland.  Served in foam carry out trays, fancy. The only thing lacking was the service but really I expected that.',
      'useful': 0,
      'user_id': 'P-ImWYCmCtBOn8WpFX5eew'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 2,
      'date': '2016-08-30',
      'funny': 0,
      'review_id': 'zWM-bQ3h-o4ku6uYHRn2ww',
      'stars': 2,
      'text': "I decided to stop by here to try the Polish girl sandwich featured on Man vs. Food.  The sandwich was tasty but the service is terrible. \n\nLike others have said in there reviews, the service is extremely slow.  The waitresses walk slow, talk slow, and were inattentive.  It also took almost 30 minutes to get one polish girl and fish sandwich (because they ran out of the popcorn shrimp my friend wanted...) which is pretty slow to me.  Our waitress even disappeared during the middle of our service and it took us a extra 20 minutes to get our bill after we were done.\n\nThe restaurant is also an eyesore, the pink and light blue color scheme is even more offensive to the eyes in person.\n\nI won't ever come here again but if you plan to come here, order to go!",
      'useful': 2,
      'user_id': 'ZcI4bBPsmsnfmOZZ0tMbEA'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 1,
      'date': '2016-10-31',
      'funny': 0,
      'review_id': 'wVbQxPcDYSIK62jmQo-axA',
      'stars': 5,
      'text': 'I shamlessly admit to straining that tin foil that comes with my rib order like a rag to get more of that tasty bbq sauce.',
      'useful': 0,
      'user_id': 'iRrKGBtJpTW9PrT7kLe97Q'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 1,
      'date': '2012-12-17',
      'funny': 0,
      'review_id': 'gX5hhgPQlmS6hfEICV_jug',
      'stars': 4,
      'text': "This place is oh so tasty!!  The great thing about their chicken is that, you don't even need their hot sauce for it to taste so good.  With the hot sauce, the chicken goes from good to great.  The pieces of chicken are typically big, hot, and filling.  Truly hit the spot if you're wanting a taste of the south, up here in Northeast Ohio.\n\nThe only problems I had with this place was that they close the dining room in the afternoon on Sunday (luckily they have the drive thru open) and the sides aren't seasoned well enough.\n\nOverall, this place is must try place.",
      'useful': 2,
      'user_id': '_YkzpKm4TAUfx8vckSTDgA'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2015-08-13',
      'funny': 1,
      'review_id': 'iXkiIxr2V5QtdLLQlULQDA',
      'stars': 1,
      'text': 'Manager, please delete this review after owner of Yelp has read . . . Thank you.\n\nWell, let me tell you.  I love he food here!  However, I had a costly-to-owner and manager, and a terrible experience today (well, the 12th a Wednesday), and they need to know if they want to keep customers!!!  Who knows how many people just walk out because of the rude and slow staff - I\'m sure they do it all the time and just don\'t tell the boss.  Here is what I am trying to communicate, and I can find no other way.  Website non-existent.\n \nMy would-be communique:\nYou lost 3 customers in a row today, August 12, at about 4:30, including the 2 ahead of me and then me.  I would have spent around $20.  The two were mother and son, and who knows what a growing boy eats?!  \nThe 2 in front of me (no one else in line or the restaurant) waited and waited even to be acknowledged let alone waited on.  There were 3 employees - a man and 2 women -who saw clearly we were there.  The customers and I chatted pleasantly for a good 5 minutes.  They got tired of waiting - for anything.  They left.  I was now the only customer.  One would think that their seeing 2 walkouts would have prompted them to action!  \nThe 3 of them stood directly in front of me at the register!  They did everything to pretend I wasn\'t there.  Could they not have said "we\'ll be right with you"?  Totally ignored me and instead counted tickets and chit-chatted about.  There was plenty of food, so they were in no crunch.  I\'d have spent $20 - LOSS.  Mother and son, who knows? - LOSS.  I go to your place because it\'s usually great and I am a frequent Clinic visitor.\nFrankly, I would fire all three of these employees if I were you.  You\'ve got a great business, but who knows how many customers walk out.  RUDE EMPLOYEES might be chasing away more than you know.  \nSusan Bednar',
      'useful': 14,
      'user_id': 'FKZG5e-1CoMZ6kLUUVrqcQ'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2016-02-28',
      'funny': 1,
      'review_id': 'PUkpeN3cZBzEBg0LZEN_BQ',
      'stars': 2,
      'text': "Went late evening...door was locked this time. (Wasn't locked the last evening I went.) Parking lot full of cars that's not at restaurant. So I go to the drive thru, there's a piece of card board on the speaker that says nothing (very unprofessional.) Sat there a few minutes not knowing what to do. Went to the drive thru and placed order there. Prices were a lot more than the prices listed on line which is what I was prepared to pay. \n\nBetween the drive thru speaker and not keeping up on current prices on their website, seems like they just don't care. So for now on I don't care to patron. I'll take my business elsewhere to someplace that shows the customer they care and stay professional. I'm not going to keep patronizing places that doesn't have any sort of professionalism about them. There are tons of restaurants of there!",
      'useful': 0,
      'user_id': 'GBo2RbrSn9vdOrlQV8N8Sw'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2015-02-09',
      'funny': 0,
      'review_id': 'b1nJMO8UAfgmRBu1Xf846g',
      'stars': 3,
      'text': "I found the ribs here underwhelming, they tasted a little overcooked and tough.\n\nHowever, fried chicken is really good here. Not overly breaded, perfectly fried, and the chicken itself is warm and tender. Get the sauce on the side, since I'd imagine you are getting takeout and the last thing you want is the chicken to turn into a sloppy mess by the time you get home. \n\nWait times have been surprisingly long (on the order of 20 - 30 minutes) despite how there is almost no line. If you come here hungry and stare at the food while waiting, this will be torture. At least they give you so much food that it still amazes me to this day how they actually manage to fit it in the takeout box.\n\nEating here regularly is ill-advised, if nothing else because I would prefer to not require the services of the world famous (and strategically close) Cleveland Clinic cardiologists and cardiac surgeons.",
      'useful': 0,
      'user_id': '07TtxrRM0b9qUBTWM5wNiA'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 2,
      'date': '2016-09-20',
      'funny': 1,
      'review_id': 'OrP2zt67MSMWpaIkjA1y2A',
      'stars': 5,
      'text': 'By far the greatest place to visit. Been a fan for twenty years and until you come here you aint had real BBQ',
      'useful': 1,
      'user_id': 'OKe55-f2tvTWDDPHV5cZtA'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2015-12-12',
      'funny': 0,
      'review_id': 'zbkLv0HYXVTfgxnw-lt_tQ',
      'stars': 1,
      'text': "My fiancé and her friend got food poisoning last night. They ordered the beef ribs... Not fun and not good. I got the fried chicken which was great and i feel ok. Service is kind of shitty like other reviews have said, some people are just not positive. I guess i wouldn't be working at a BBQ place till 1:30 in the morning. I will NOT be going back. Food poisoning is a really bad thing, which means they aren't doing proper food handling.",
      'useful': 0,
      'user_id': '5SYWlymEoSc8brPP568I6A'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2012-09-29',
      'funny': 0,
      'review_id': 'wv2M1L3dhDSaauUl0mMeOg',
      'stars': 4,
      'text': "Delicious Fried Chicken! Though the name is Hot Sauce Williams Barbecue Restaurant, this is the place to get fried chicken not ribs. Their ribs are so so to me. The fried chicken, however, is heavenly. Their sauce is wonderful too but I always get it in a cup and put on the chicken sparingly cause the chicken is top notch without it. The crust is perfect. After eating chicken here Popeyes, Churches and the rest of those national franchises' chicken taste awful to me. Love their excellent down home side dishes too. Four stars and not five because of the ho hum customer service (sometimes) and poorly put together menu. A side note, no matter what time I come into this place the clientele is always around 60 percent Black and 40 Percent White, with more Whites eating in the restaurant and a higher percentage of Blacks taking out.",
      'useful': 0,
      'user_id': '9nw_A7HYmcIkZS5EcE3-ww'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 3,
      'date': '2014-07-17',
      'funny': 1,
      'review_id': '9R5D8tbFTUpuDi7xwCHl3A',
      'stars': 4,
      'text': "It seems they care about their food. Tried a polish boy (it's food, I swear). It had great taste but messier than a Hollywood divorce..with kids. Couldn't be happier with the hot sauce, chicken and ribs. Ton of food for a little money. No public restrooms, fake flowers on pink tablecloths in an old fast food joint. They still yell their orders to the back. So why are they on the travel channel? They actually care about their food.",
      'useful': 1,
      'user_id': '2n03U1y-9Lr3gl0CcMrHPA'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 1,
      'date': '2016-05-04',
      'funny': 1,
      'review_id': 'zMuK1UEb79fgPFJFsBI9NQ',
      'stars': 1,
      'text': 'WHAT HAPPENED??  A year ago I would recommend anyone visiting Cleveland to come here.  NOT AFTER TODAY!!  I used to LOVE going here!  The Polish boy was one my personal favorites.  Today, I was on my way home and just happened to come by Hot Sauce Williams so I pulled in.  I knew I would have to wait at the drive thru but ...WOW!  15 minutes??  I should have just parked my car and went in.  Drive Thru is a BAD IDEA at this establishment.  I pulled up to the speaker and a piece of cardboard was over the speaker and written in marker saying "STOP! Order here!" I ordered and I will come to find out that the prices on their outside menu are WRONG and MORE EXPENSIVE than listed! (I didn\'t know this but I will explain later). On top of that, it looks like the window covering the menu and prices has not been cleaned since 2012. I ordered the Polish Boy combo and small size fried okra.  I had a couple in front of me and waited approximately 5 minutes before I arrived at the window ( I made a mental note changing radio stations because when I arrived at the window it took longer than the entire song of Baltimora \'Tarzan Boy\' as I kept changing stations to pass times and ended back on the 80\'s channel to hear the end of that song and start a new one).  Do not think of this place as a typical \'fast food\' drive thru.  You order and you will WAIT but they don\'t tell you that which can be a good thing.  This can give you the impression that the food is fresh but to tell you the truth, it didn\'t seem like it was \'fresh\' in least bit.  In fact, it appeared that they took care of of whoever was inside BEFORE the drive thru customers because I actually watched 3 polish boys being made while I waited for my single polish boy because I guess the people inside took priority and they had only ONE person making food with 3 other people handling money and passing out beverages.  While watching someone else\'s food being prepared another employee came to the window and gave me my ridiculous total of $11.98!!  I gave her my money and at first I just simply thought the tax was high so I asked why my total was $11.98?  Apparently the prices listed on their dusty and dirty outside menu board are WRONG!  The RUDE employee didn\'t tell me anything.  She just "handed" me a receipt.  Instead of the okra being $2.75 it was $3.99!  I told the lady this and she just said, "It is?  I will have to change that."  She walked away and I NEVER saw her again. Well, what about you OVERCHARGING??  I was already upset waiting for food so I turned my car off to save gas until I arrived at the window. I was overcharged and the lady avoiding me now, had another employee hand me my beverage. Coke in a can. As I was saying my food was over charged, she closed the window!! WHAT!! This is how you deal with customers with issues!?!? You AVOID them! \n     Finally I received my food.  They had ANOTHER employee hand me the food and this worker was a male.  I was actually able to talk with him and say that I was overcharged. He was not helpful or even courteous in ANY way! He said, "I can\'t help you, I don\'t work with the menu." I kept telling him, this was WRONG, you can\'t overcharge customers and you can\'t treat people like this! Again, like a robot, "I can\'t help you with that."  I heard enough. "If you can\'t fix this or help and NONE of you are willing to help without me making a scene then I\'m DONE with ALL of you!" He replied again, "I can\'t help you" and shut the window then walked away while all the other employees avoided the window. \n     I\'M NEVER RETURNING TO THIS ESTABLISHMENT!! What happened?? The employees don\'t care! A year ago this was a decent place! They even have photos of Robin Williams (I miss him so much!) visiting there with his family.  What happened? I don\'t know and I WISH I could give this place more stars but this whole experience was enough for me to give them the LOWEST rating. I had a few good times here in the past with some memorable moments but it will ALWAYS be overlooked by this horrible treatment! THERE IS NO EXCUSE FOR THIS TREATMENT AND ATTITUDE! Even the food changed in my opinion. My Polish Boy was a regular sized hot dog with fries and coleslaw slapped on top of it and my fried okra was soggy.  Worst of the all, the sauce was different this time too. It was watered down and tasted old if that\'s possible. I WILL NOT RETURN.',
      'useful': 4,
      'user_id': 'JdYkvy8lIodJ0vwJ4jzYYA'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 1,
      'date': '2012-11-25',
      'funny': 0,
      'review_id': 'UlH1vcIaTGaoMNXeU8hKaA',
      'stars': 4,
      'text': 'Sweet merciful Fried Chicken and Polish Boys!\n\nIt is absolutely delicious food.',
      'useful': 0,
      'user_id': 'xWR-GSPPlt7taGuM1vk_0g'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2009-03-28',
      'funny': 1,
      'review_id': 'p-27femPsz2zvenUsCXh8A',
      'stars': 3,
      'text': "Good fried chicken and sides.  I had tried some of their ribs at the Taste of Cleveland last year though, and wasn't too impressed since the meat was a little tough and the sauce, just OK.",
      'useful': 0,
      'user_id': 'DK57YibC5ShBmqQl97CKog'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 2,
      'date': '2011-10-11',
      'funny': 2,
      'review_id': '8Mg5n8kUeCofQFa37EmNiw',
      'stars': 4,
      'text': 'Mrs. Doubtfire was here!\n\nNo really Robin Williams dined at this location.  \n\nThe Fried chicken is crunch in your mouth drippy kind of good!   My favorite meal is the polish girl.  it just screams "Eat Me!!"    polish dog piled high with pulled pork, crinkled fries drenched in slaw and oh so good tangy HSW sauce.   Oh how the sauce enhances the taste, Makes you want seconds\n\nNow when you eat your hot Polish Girl, you have to be gentle it tends to get sloppy.  I ease on into it.  I always end up with a mess on my hands and face.  wipe down needed after this Feast.  Sometimes I settle for the polish boy, a little less mess and take it all in kind of treat \n\n"Ease on into it"',
      'useful': 3,
      'user_id': 'IpReaQYuXO_nB07xnkHuNw'},
     {'business_id': 'tulUhFYMvBkYHsjmn30A9w',
      'cool': 0,
      'date': '2016-07-09',
      'funny': 0,
      'review_id': 'p9vooRGunX57oWrVAML3Jg',
      'stars': 2,
      'text': "I stopped in here today to get a chicken wing dinner for my wife.  After it took this young lady 10 minutes to take my my order.  She tells me its going to be a few minutes on the wings to cook.  So I had time to look around at all the celebrities to visit this staple in Cleveland.  I also had time to see how ran down and dirty this place was.  Dirty tables, floors, and walls.  Ceiling falling apart.  It's obvious they don't put money back into this dilapidated falling building.  30 minutes past I finally get my order.  The older lady gives me my order.  I asked don't I get a drink with my platter.  She told me it doesn't come with a drink.  I know the last time I was here I got the same thing and I got a drink with my food.  Matter fact this time I got coleslaw.  Last time I didn't.   Bottom line I'm all for supporting my community, but if you can't do better I'm taking my money elsewhere. But I'll give you another star because my food was good.",
      'useful': 0,
      'user_id': 'nPKNLcE_R_L_IVSX82VWXw'},
     {'business_id': 'cQSQi0YWTcpuw5WGJi9XMw',
      'cool': 0,
      'date': '2012-06-08',
      'funny': 0,
      'review_id': '_iw0IGlFZtPCqxP5xcsTIQ',
      'stars': 3,
      'text': 'Wie schon erwähnt, rustikales und eher altbackenes Ambiente, aber gemütlich. Bedienung ist sehr freundlich. Wir hatten Pizza, die recht passabel war. Insgesamt kein Laden zum Schick essen gehen, Aber für eine Pizza zwischendurch eine Empfehlung.',
      'useful': 0,
      'user_id': '-IbGzbx2qsGjJHm78fbj8w'},
     {'business_id': 'cQSQi0YWTcpuw5WGJi9XMw',
      'cool': 0,
      'date': '2011-10-19',
      'funny': 0,
      'review_id': 'k2WNspTH6Mazkz4dxeIf4g',
      'stars': 5,
      'text': 'Bodenständige italienische Küche, eher rustikales unedles Ambiente, sehr freundliche, aber manchmal schnarchige Kellner, wenn der Chef noch nicht da ist. Finde es liebenswert und gemütlich, gehe immer wieder gerne hin.',
      'useful': 0,
      'user_id': 'bVkNsrA6W0LwEIVIKhiV-w'},
     {'business_id': 'cQSQi0YWTcpuw5WGJi9XMw',
      'cool': 0,
      'date': '2011-01-10',
      'funny': 0,
      'review_id': 'BWCCewfflc2wI6AdPT8-Fg',
      'stars': 5,
      'text': 'Eine kleine und gemütliche Pizzeria mit leckeren Pizzen und Nudelgerichten. Gutes Preis-Leistungsverhältnis und freundlicher Service.',
      'useful': 0,
      'user_id': '7EpMV3KhqozTdG2qVE1-fw'},
     {'business_id': 'cQSQi0YWTcpuw5WGJi9XMw',
      'cool': 0,
      'date': '2013-01-15',
      'funny': 0,
      'review_id': '-R-OianhKSSuLiZ6mzR19g',
      'stars': 5,
      'text': 'Tolle italienische Küche in einer unscheinbaren Location in Stuttgart West. Sowohl Service als auch die Speisen sind super und das Preis-Leistungs Verhältnis ist mehr als gut! Gutschein wurde gerne angenommen und als Dankeschön gab es ein Grapa aufs Haus. Sehr zu Empfehlen!',
      'useful': 0,
      'user_id': '4xqaWaXGO0el_jmTkm3c0A'},
     {'business_id': 'cQSQi0YWTcpuw5WGJi9XMw',
      'cool': 2,
      'date': '2015-10-07',
      'funny': 0,
      'review_id': 'bokbBFumrarDfaQwMoR9Bg',
      'stars': 5,
      'text': 'Ein Vorurteil verwandelte sich bei dieser Pizzeria in eine absolute Überraschung. Wir sind schon öfter an dieser Pizzeria vorbeigegangen und ich habe über den Namen etwas die Nase gerümpft - Regenbogen, das klingt so nach schlechter 80er Jahre Taverne.\n\nDrinnen ist es dann im ersten Eindruck auch ein bisschen, wie bei einer kleinen Zeitreise - allerdings nicht in die 80er, sondern in die 90er, in die mich gerade die Lampen und die Einrichtung zurückversetzten. Was ich aber sehr charmant fand, waren die Theaterfotografien an den Wänden.\n\nBeim Service war ich am Anfang etwas verunsichert, weil der gute Herr wirklich finster gucken konnte, aber das verbesserte sich im Laufe des Abends merklich und jeder, der an unseren Tisch kam war immer höflich und professionell. Außerdem waren Getränke und Speisen immer unverzüglich an unserem Tisch, auch als es später am Abend sehr voll wurde - so lob ich mir das.\n\nDas Essen war dann ein wahres Gedicht, angefangen beim Brotkorb mit frisch gebackenen Pizzabrötchen bis hin zur Nachspeise, was wieder zeigt, dass man sich nicht von Äußerlichkeiten abschrecken lassen darf. \nAuf der gemischten Vorspeisenplatte fanden wir ein Gedicht an italienischer Salami und Schinken sowie exzellentem Käse. Mein Bruschetta war ein frisch gebackener halber Pizzafladen mit frischen Tomatenstückchen, die ordentlich mit Knoblauch garniert waren - sehr lecker!\nDie Hauptgerichte haben uns dann absolut überzeugt: Ich hatte frische Nudeln von der Tageskarte mit Steinpilzen. Die Nudeln waren einen winzigen Tacken zu al dente für meinen Geschmack, aber sie zogen noch etwas nach. Die Steinpilze in der Soße mit frischen Kräutern waren hervorragend. \nAber das absolute Highlight war die Pizza: wie in Italien. Hauchdünner, geschmacklich hervorragender, knuspriger, aber nicht trockener Teig mit leckerem Mozzarella, der ordentlich Fäden zieht und frischem Belag, der die anderen Zutaten nicht erstickt. Ich bin sonst von Pizza nicht so leicht zu beeindrucken, aber diese war einfach ein Gedicht.\nDer Tiramisu zum Nachtisch musste dann noch sein, obwohl wir eigentlich schon satt waren. Er hat den hervorragenden Eindruck des Essens noch unterstrichen.\n\nAuch der Wein war hervorragend und mit 4,50 Euro für 1/4 Liter auch nicht besonders teuer. Insgesamt war das Preis-Leistung-Verhältnis (9 Euro für die Pizza, 10,50 Euro für die Nudeln) für die Qualität absolut angebracht.\n\nEiner meiner Lieblingsitaliener in Stuttgart, wenn es mal eher gemütlich als schick sein soll.',
      'useful': 2,
      'user_id': 'natSObJ4-jEev6rJRta7jA'},
     {'business_id': 'cQSQi0YWTcpuw5WGJi9XMw',
      'cool': 0,
      'date': '2017-07-20',
      'funny': 0,
      'review_id': 'nEejFLrC78aaEeu981RpuQ',
      'stars': 4,
      'text': 'Erst war ich Nicht so überzeugt von der Pizzeria. Von außen eher unscheinbar. Als ich dann meine Pizza bekam war ich sehr überrascht. Die Pizza geschmacklich super. \nDie Bedienung immer freundlich und hilfsbereit. \n\nIch gehe auf jedenfalls wieder hin.',
      'useful': 0,
      'user_id': 'Okg3qvGtPrSXTQPrBU2-7w'},
     {'business_id': 'cQSQi0YWTcpuw5WGJi9XMw',
      'cool': 0,
      'date': '2013-04-10',
      'funny': 0,
      'review_id': 'O_ygKUhl7aRM1kAmShJgpw',
      'stars': 1,
      'text': 'Unser erster und letzter Besuch: Leider müssen wir von einem Besuch der Trattoria Regenbogen im St. Westen abraten. Unsere Erfahrungen von Heute:\n1. Aus dem Nebenzimmer zog Rauch zu uns rüber \n2. Wir kippten entsprechend das Fenster  kurze Zeit später stürmte der Kellner an uns vorbei und schloss es wieder\n3. In der Vorspeise Vitello Tonnato fand sich ein Stück Plastik (3cm!). Ich gab das Plastikstück zurück, erhielt aber weder eine Entschuldigung noch einen neuen Teller  lediglich ein Oh. Darüber hinaus war das Fleisch nicht durch und schmeckte nicht gut.\n4. Der Hugo war sehr wässrig und der Holunderblütensirup war ganz unten\n5. Nach dem Bezahlen empfahl ich dem Kellner, doch künftig bei solchen Dingen wie Plastik im Essen zumindest einen Grappa oder Cappuccino anzubieten. Die Antwort: Nein, das machen wir nicht  ist menschlich, kann doch passieren.',
      'useful': 0,
      'user_id': 'p9PvT7BXuD1rJ_jnlfRwTA'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2016-10-27',
      'funny': 0,
      'review_id': 'VwuwD_UMM4oNF98URBQKrQ',
      'stars': 5,
      'text': 'I can typically take or leave hotel restaurants as well, they are hotel restaurants.  Burnham is way different. \n\nOn our last AM in town we were hitting breakfast before hitting 71 to go home. Anthony was our server. Two omelets with wheat toast, fruit for me potatoes for my BFF. Coffee. Juice. Ice Tea. \n\nWhat SOLD US and has us on Yelp and then Trip Advisor was my ask for ice water after I downed my coffee. Anthony turned around grabbed a glass, held it up and checked to make sure it was clean.  We went crazy - that NEVER happens but at the Burnham. When we praised him he said, "I\'m not gonna give out no dirty glass." LMAO!!!!!!!!!!!!!!!\n\nWhen he stopped by to pick up our plates he so proudly said, "I love picking up dirty clean plates!" BAWHAHAHAHAHA! Just absolutely too cute.  Lots of professionalism. Lots of personality.  Quite the impression. \n\nThe food was perfect. The omelets were not in excess. Enough eggs and stuffing of our choice to balance it all out.  What an experience.  We hit so many breakfast places in town I\'m sorry we didn\'t do this more than once. Thank you Anthony!',
      'useful': 2,
      'user_id': 'tb7q_TIKu_pha2UFyneMmA'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2017-05-03',
      'funny': 0,
      'review_id': 'ImIAoiDt-SUdzPG6K78sYg',
      'stars': 2,
      'text': 'The waiter was poorly trained and seemed unhappy with his job. Complained that Lebron was a bad tipper. Interrupted our business conversations.  Didnt give me a menu until i asked. We had bought 2 bottles of wine and he did not come back to refill.  Frankly I was a bit embarrassed hosting a business meeting there.  Food was unremarkable as well.',
      'useful': 0,
      'user_id': '53tYed3KAuIwwPb7ZIj2zQ'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 1,
      'date': '2017-04-15',
      'funny': 1,
      'review_id': 'j-bvLvbgh4fIEcfFKRDHrQ',
      'stars': 4,
      'text': 'I came to Cleveland to watch the cavaliers, raptors game the restaurant at the hotel was the burnham restaurant, we were all trying to figure out where to eat and we pulled the menu up on our phones and we decided this place. The service was some of the best I\'ve had, My server Jeff was a very genuine guy, I\'ve been to many restaurants where servers are nice, but it seems like it\'s a forced attitude, but I could tell Jeff even outside of work acted the exact same and was a joy to be around, The food was amazing too we tried a bit of everything and had no complaints about anything. Some of the food we had was the "soul bowl" which was pork belly on top of mac and cheese, the chicken wings confit, hangar steak, Buffalo chicken pizza, and a burger. I would get it all again, but the steak and the pork belly was my absolute 2 favourites, you have to try it out for yourself to see why I\'m talking about!',
      'useful': 1,
      'user_id': 'rCUvlj4TSETvtAWxVlZPwA'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 1,
      'date': '2016-08-07',
      'funny': 1,
      'review_id': '_lfkSTQ62kH8po5Z0T57og',
      'stars': 5,
      'text': "Came for my birthday dinner.  The hotel is new and very beautifully designed, and we had read good reviews of the Burnham.  Did not disappoint!  I recommend the small plates--so much choice!  I had the peiroghi, the fried chicken confit, and the table shared the cheese creme with mushroom.  H had the oyster spaghetti.  Excellent service--not pushy, not absent.  Crowd is good--dates/couples, friends/family meet up.  I don't think this is a kid place--definitely an adult vibe.  I'd like to stay down here and make it a whole night away in my own city.",
      'useful': 1,
      'user_id': '6-RXoqxa9L3FfBwkPLrtbw'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2017-07-21',
      'funny': 0,
      'review_id': 'W1Bx4udLb6n1nXyXR1EwkQ',
      'stars': 2,
      'text': "Food is ok but expensive.  SERVICE IS TERRIBLE.  The hotel tries to do a good job and wants to provide a great experience,  unfortunately,  the restaurant staff have poor attitudes don't really understand what good service means.",
      'useful': 0,
      'user_id': 'WESo_kDiMzvbKOAGJE_QWQ'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2016-11-11',
      'funny': 0,
      'review_id': 'KvlVTV0apy4iA2uYeGGXyQ',
      'stars': 5,
      'text': "Fantastic restaurant in the Hilton Cleveland! We ordered salads and they were as good as the entrées. Huge offset bowls with enough fresh, unique ingredients (fresh, whole anchovies, pickled eggs, fresh shaved cheese) to make you think twice about ordering additional items from the menu...nah, there's more to explore! You have to try the French toast...yes, anytime of day! It's not what you think. Shredded chicken, a sauce and season combo that takes your taste buds on a roller coaster, and soft & slight crunch in the toast peeking from beneath. One of my all-time favorites!!",
      'useful': 0,
      'user_id': 'Ty8k7Wj3kXMcCvOHsu-ptQ'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 3,
      'date': '2017-04-02',
      'funny': 4,
      'review_id': 'I5Z_P4SftkiI3YEpauiI8g',
      'stars': 4,
      'text': "Overall impressions: swanky lounge in a swanky hotel serving up new takes on old favorites. Staff are still growing into their roles, but friendly. Good for dressy dates, business meetings, and hopefully drinks on the patio when the weather is warmer! \n\nIt's a shame that this shiny new part of Cleveland doesn't get the traffic it deserves after the boom that was Summer 2016. Getting here is about a 10 min walk from the RTA station, but at night, when there's hardly anyone around, and your phone light is shining brighter than the streetlights... The empty buildings can be pretty eerie. Buuut, I guess I can't complain about having my pick of the reservation times during Downtown Restaurant Week. \n\nThe Burnham can be entered directly from the side of the Hilton building, and my friend and I were pretty impressed by the upscale nautical wall deco. Everything still looked as though the plastic wrap had just been taken off. We were shown in short order to our table, a half-booth ordeal. Normally I'm all about the booth tables, but our booties sank a little too deep into the plush seat and made the table an awkward height to lean your arms on. Which, if you have just above can't-take-you-ANYWHERE etiquette like me, you tend to do quite often. Anyways. We were brought some complimentary bread and olive oil while looking through our menus, but the server nearly dropped the bread on the table instead of our plates, and the bread was quite plainly not fresh. A rough start. \n\nOur hearts and stomachs dropped a little when we saw our first appetizer (not part of the Prix Fixe menu), five tiny Garlic & Onion Beignets. This was definitely not Southern hospitality. They were pretty good, but gone too soon, including the pickle garnish. Each of us then chose different salads for the first course and were pleasantly surprised by the unique ingredients. No plain Caesar from the grab bag - mine had a crunchy mix of greens, pickles, sunchoke, and a light dressing. Main course involved a Swordfish Schnitzel for me, and a Sous Vide Fried Chicken for her. You heard right. Schnitzel and sous vide, chicken and swordfish? Did they draw these combos out of a hat? Maybe, but they work surprisingly well together - both meats were tender, the accompanying sides (radish + mashed potatoes for me) appropriately balancing. I could probably have done with less salt overall, but know that you'll be leaving full after an entree. The desserts were cute trays of Petit Fours, including cake pops and macarons. Nothing mind-boggling, but easy to pack up and eat on the go. \n\nLike others have said, service was fairly slow. It took a while for our dishes to come and for water to be refilled even though we were one of probably five tables that evening. The staff is definitely rough around the edges compared to the establishment (I mean, have you SEEN the bathrooms? Go, just go outside across the hallway, and enjoy the gilded glory.). It's not a place you should expect to grab a quick bite - although I think I spotted a kid eating chicken tenders a few tables over. But food-wise, we felt we got what we paid $30 (+ an appetizer) for. The menu, the staff, the booty-sucking seats, it all feels a little experimental, but I think we got some positive results out of the evening.",
      'useful': 4,
      'user_id': '5ZrFDFPJe8Qq3Zgue315oA'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 1,
      'date': '2016-10-03',
      'funny': 1,
      'review_id': '9Gql5KxYL7CNFQ-8p7A28A',
      'stars': 5,
      'text': 'What a refreshing experience! Beginning with a friendly greeting and seating, followed by a delicious breakfast buffet satisfying every imaginable taste desire, and all made better by the most polite and attentive server we have experienced in a long time. Craig was a pure delight! His contribution to our dining experience was as important as the food. Craig needs to be cloned.',
      'useful': 1,
      'user_id': 'o6PwFjn4WZ9Axal6zdGDrw'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 1,
      'date': '2016-09-08',
      'funny': 1,
      'review_id': 'II-azWny-UcJ4qRbfLi99w',
      'stars': 4,
      'text': 'Nice upscale take on Cajun. Nice place for a rich dinner after a day of conference smarts. Our favorites were actually the appetizers: pierogies and the divine goat cheese creme brulee! Hanger steak was delicious and perfectly cooked, if a little salty.',
      'useful': 1,
      'user_id': 'J9mHpteG5XVio6NqcwHPCg'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 1,
      'date': '2016-09-08',
      'funny': 1,
      'review_id': '2vzhoc7sLbHgg8Sgw9JLAg',
      'stars': 3,
      'text': "Ate here when I arrived in the evening on a business trip. It was a strange restaurant. There was a large buffet that appeared to just be supplies (not for customers). \n\nThe waitress couldn't answer any of our questions. My colleague and I both have fairly typical food restrictions (I don't eat meat; he's Celiac/gluten-intolerant) but the waitress didn't know what would be suitable for either of us. The menu did not indicate some key ingredients, either (like that they use bacon fat on the green bean sides). \n\nShe asked the chef and he suggested a salad for my colleague and a pizza for me. The pizza a 6-inch flatbread. It was ok, but felt like an appetizer. \n\nOverall, it was fine for the lazy evening after a long day of travel, but I wouldn't go again if there were other choices.",
      'useful': 2,
      'user_id': 'wAZ4HzPS8rzla5xCYkI-Pg'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2016-09-28',
      'funny': 0,
      'review_id': 'XLe5jIvTJ1sNWK1eUFGL0w',
      'stars': 5,
      'text': 'We stayed at the hotel and decided to check out this restaurant. All of us were very impressed with the food and staff. The hanger steak I had was amazing. Their version of the carrot cake was nothing like I have ever had but it was great. If I ever go back to cleveland I will make sure I return.',
      'useful': 0,
      'user_id': 'Rrgf1xScox-hRIIaUHdKYA'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2017-06-18',
      'funny': 0,
      'review_id': 'DafT7bpKy17ZDVAMEaxuYQ',
      'stars': 4,
      'text': "The Cleveland Plain Dealer gave this restaurant an excellent rating, so I made reservations.  After that I read other reviews, some of which were negative, so I dined a there tonight fearful that I had made a mistake.  I was delighted by the delicious food and excellent service!  I highly recommend it and hope that others will try it, as it was pretty empty on a Saturday night, although this could have been caused by upcoming Fathers' Day.\nVisit the bar on the 32nd floor for a wonderful view of the city.\nI will return and recommend it to my friends.",
      'useful': 0,
      'user_id': 'EEI1wjfuZIFshDl1ue9N8w'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2016-06-19',
      'funny': 0,
      'review_id': 'tSfH2QO3VaqzSSoptwGURA',
      'stars': 3,
      'text': "Attention to detail on favorite comfort foods such as the Shrimp and Grits and Fried Chicken with charred cabbage results in lively fresh takes on tradition. However the staff was not up to par, the hostesses could not pull up our reservation, waitress got our dessert order wrong, forgot what my entree was immediately after I ordered and there was that discussion on what Prosecco is. That being said, this is a newly opened restaurant and there is clearly a lot of field training going on. Just wish the staff was better prepared. Rating would be higher except for the service.\nWe'll be back in a few months after the kinks are worked out as the food is so worth it.",
      'useful': 0,
      'user_id': 'UGszBvq3yL7XyV-fqZbarg'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2017-04-02',
      'funny': 0,
      'review_id': 'WHGIOp9WXdmdQfK0J75ZGA',
      'stars': 5,
      'text': "I could go on for hours about how wonderful this place and its staff are. I had the best pancakes I've ever eaten and a wonderful mimosa. We came in at noon (lunch time), and the staff happily accommodated our dying need for some breakfast food. This is a must for brunch.",
      'useful': 0,
      'user_id': 'VtGnZApQr-QwwgfU5HiE2w'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2016-09-27',
      'funny': 0,
      'review_id': 'jEJ51P_kl0tAQeHv6vJ0xw',
      'stars': 4,
      'text': 'I stopped in here a few weeks ago when my friend and I were trying to go to the rooftop bar, but it was closed for a private event and we were starving. It was pretty empty while we were in here, so we got a seat right away. Our server was great - very attentive and our food was super timely. I ordered the Southwest Shrimp Pizza, which was delicious. I wish I could have taken it home with me, but I was heading out for a night on the town, so I had nowhere to stash it. This place is great for an anniversary dinner, lunch with a friend or colleague, or a place to celebrate a family occasion.',
      'useful': 0,
      'user_id': 'TjXqTqYOXAs8VKo_otj3lw'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2017-07-01',
      'funny': 0,
      'review_id': 'xmuXY_woGKKXCDE51fqOJw',
      'stars': 4,
      'text': 'Excellent service & food.  I had the grilled salmon & my husband had the muffuletta sandwich.  Yum!',
      'useful': 0,
      'user_id': 'rXuLYU7imms3QUpp6faaoA'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2017-01-22',
      'funny': 0,
      'review_id': 'e6nYTVFvKUpu5DHM9iFkOw',
      'stars': 1,
      'text': 'Terrible. Had reservations. Not busy. Hostess was clueless. Left us at restaurant bar and kept seating people with no reservations. Talked to hostess, did not care. Never seated us. Wont be back.',
      'useful': 1,
      'user_id': 'w1CYD3yWvajHzAbyNEjKuQ'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2017-06-14',
      'funny': 0,
      'review_id': 'xrM4pyq8aZp2FSI0w2QlzQ',
      'stars': 3,
      'text': "I came here last night while on a business trip. I was not too impressed. The food was decent but the service not so much. We sat on the outdoor terrace and the waitress forgot about us. For appetizers we ordered the calamari which was light and delicious. We also had the shrimp and gnocchi (only came with 3 pieces of shrimp) and the wings which were good but nothing special. I ordered the fried chicken confit which was good but not as flavorful as I had expected. It's conveniently located in the hotel but next time I'll venture out into the city.",
      'useful': 0,
      'user_id': 'Lr9fJ0wAVC5ATCIKntYydQ'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2016-06-18',
      'funny': 0,
      'review_id': 'J5YmEo_KAGxKF0qHPkDAAg',
      'stars': 5,
      'text': "Chose to visit the new Hilton Downtown and new restaurant for a birthday.  Chose this restaurant because our party had been to most restaurants in Cleveland and the Burnham had only been open 16 days when we ate there.  \n\nOurs was a party of four and had no time constraints so we decided to do a couple courses of splitting the small plates.  After our incredible waiter Joseph walked us through some of his favorite things and some of the background for some of the food, we chose our first course as Octopus salad, soft shell crab special, chilaquiles, shrimp and grits, and fried rice.  All of these dishes were as advertised and devoured by our party.  The octopus salad was some of the most tender, tasty octopus we have had anywhere in the country let alone Cleveland.  The chilaquiles were like a mexican pancake with pulled pork and rock shrimp baked in.  The shrimp and grits were great.\n\nOur second course we chose soul bowl (mac and cheese with pork belly and collard greens), boudin and mash (blood sausage and mashed potatoes), confit wings, and the five cheese pizza.  While I don't like collard greens, the mac and cheese and huge pieces of pork belly was good.  The blood sausage and mashed potatoes were a british/irish delicacy and delight.  The mashed potatoes and gravy were great.  The confit wings were a bit salty but they literally fell off the bone and were served with a green sauce that I can't quite figure out what it was.  The pizza was great topped with tomato and basil and quite flavorful.\n\nOur party had a couple bottles of wine from the extensive wine list and some of their cocktails (Sacre Bleu and French 75).  The cocktails were all made from scratch and delicious.\n\nThe staff at this restaurant were incredible with multiple people from servers, hosts, back kitchen staff, and managers all making their way to our table to wish one a Happy Birthday and welcome us to the restaurant.  The new hotel is beautiful inside.  Wish the Bar 32 (bar and patio on the 32nd floor) was open but its still not finished.\n\nOur server Joseph made the night especially nice with his knowledge and hospitality.\n\nWould like to try breakfast/brunch next time maybe but we will definitely by back.",
      'useful': 1,
      'user_id': 'BgI4-bHddSlRkp7UbJkMaw'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2017-04-11',
      'funny': 0,
      'review_id': 'h3zHv4zMD6Zk2lNcUOw_-A',
      'stars': 1,
      'text': 'If you want to eat at a restaurant that strives to exactly satisfy the average palate of the average American diner, this is the place for you. The menu and the wine list will have exactly all the regular, normal choices that the average eater will not be offended by. If you want a memorable meal, go elsewhere.',
      'useful': 0,
      'user_id': 'wQSS08uB5vwIFS-ZM4BkdQ'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2017-07-17',
      'funny': 0,
      'review_id': 'JmnlfM2kHumv4doQVMVCtA',
      'stars': 5,
      'text': "I'm not sure why urban farmer has better review than this place. The food was really good for a restaurant that is maintained by the hotel. I've tried 4 different meal and dessert and nothing is bad.. I'm thinking that the review is low due to price but I can assure you that the quality and freshness of the  ingredients is fantastic.. u get what u paid for. The service is not bad... just a relax environment in which u can enjoy your dinner.. I'm giving this a 4.5 star due to price but if u are looking for a quality and taste, this is definitely a must try in Cleveland..\n\nI recommend any seafood entree.. salmon, trout and walleye..",
      'useful': 0,
      'user_id': 'Y-XAUpt0XbA8NBxPYTz85A'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 1,
      'date': '2016-08-15',
      'funny': 1,
      'review_id': 'BPjU8lYAIF_lIXyb1fytfQ',
      'stars': 4,
      'text': "I came here for a late solo dinner, got in on a business flight around 10 PM.  I had two small plates, the shitake perogies and the goat cheese creme brulee because they sounded the most intriguing to me.  B  The goat cheese was rich if not particularly flavorful, went very nicely with the crostini that came along with it.  The perogies were a definite hit however, savory with tart sauerkraut and a nice mashed potato side.\n\nThe bartender was very friendly and helpful.  A nice selection of beers on draft.  Overall a very acceptable hotel restaurant, can't comment on the entree options but overall this is a place worth a visit if you want something simple and are staying at the Hilton.",
      'useful': 1,
      'user_id': '_tHSAp8CgI14if0rnD2aHw'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2017-03-23',
      'funny': 0,
      'review_id': 'cETj4CWuBK9BOcevvqyj3A',
      'stars': 2,
      'text': 'Slow service and okay food.\nOnly ate here because too cold to walk anywhere. Waiter spent most of the time chatting with another table while I needed service.  Will not eat here again.',
      'useful': 0,
      'user_id': 'biymfENdbU2ECcgRdRhROw'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2016-08-29',
      'funny': 0,
      'review_id': 'tDAYPNunvDQICwZDLnEMdw',
      'stars': 2,
      'text': "Came here for lunch with two others and did not have a great experience. It was pretty busy upon arrival, however we made a reservation so we were seated within 5 minutes of arriving. It was a nice day so we were asked if we'd like outdoor seating, which we agreed to. \n\nService was extremely poor. We had a reservation for 12:15, we were sat at 12:20 and we did not get our food until 1:20. It took 20 minutes for our server to ask us what we wanted to drink, and another 5-10 minutes for my soda to arrive. We weren't told what the soup of the day was, we weren't told about any specials and we weren't even asked if we'd dined with them before (this place just opened after all). All in all, our server took 30 minutes just to take our order.\n\n25-30 minutes later our food arrived. My friends got the field green salads (which they actually enjoyed, hence the two stars). I ordered the buttermilk chicken sandwich, which I did not enjoy at all. The breading on the chicken was bland and tasteless and to be honest, when I cut my chicken open with my knife (this is an open-faced sandwich by the way, something the menu doesn't say) it looked pretty gross in the middle. My fries were simple shoestring fries that tasted fine.\n\nThe poor service in combination with the bland tasting food will be the reason I likely will not return. With all the great restaurants available in Cleveland, this one is not worth going to (at least for lunch).",
      'useful': 1,
      'user_id': 'd2LnPFw9ClyAH_N4-NCt9A'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2016-11-12',
      'funny': 0,
      'review_id': 'AVBow8SUmRV8kWSm42TKVg',
      'stars': 2,
      'text': "This was not the best experience. The food was decent and the service was terrible. Waited entirely too long for basic things like drinks and to place our order.. mind you, the place was no where near full. Our waitress could have cared less about anything, and it showed. \n\nWon't be going back ever. Not worth it.",
      'useful': 3,
      'user_id': 'tYPLcU_P6-j_PDLQ21nnvg'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2016-07-10',
      'funny': 0,
      'review_id': 'e5xyJVOcHfnK_TfxoD-0YQ',
      'stars': 5,
      'text': "Really enjoyed our night sitting at the bar at The Burnham. Rob the bartender/mixologist (not to sound too hipster but this guy is a professional) was very knowledgeable, funny and a great steward for Cleveland. He gave us tips on where to spend our weekend and he did not disappoint!\n\nOn to the food: Husbandface was obsessed with the pierogies, I was obsessed with the French toast small plate. I'm going to try to recreate it at home and will probably fail a million times it was so good.\n\nDrinks: Thumbs up. Let Rob know what your tastes are and trust him to make you your personality in a glass. Because that pretty much happened.\n\nIt was our first time in Cleveland and we had to force ourselves to NOT go back to the Burnham for every meal just so we could branch out!",
      'useful': 0,
      'user_id': 'JMX2dHmI_xUQ1bBElfnzNg'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2016-09-17',
      'funny': 0,
      'review_id': 'w_SJN1pAGI05_3--iQRPsw',
      'stars': 5,
      'text': "Had the fried chicken confit. I swear on earth I've never had anything better. It was the perfect size and the cheese dumplings on the side were perfect.",
      'useful': 1,
      'user_id': 'SWsIATHHegIdilD3HAgfUg'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2017-07-14',
      'funny': 0,
      'review_id': 'wcNkOZrQ7EJkBVENNKYeEA',
      'stars': 2,
      'text': 'Food is pretty good for a Hotel. Much better places in downtown, especially for dinner. Service was terrible.  Hilton needs to up service in the restaurant.\nMy wife and I were meeting people in the terrific bar on the 32nd floor. I showed up late an had trouble getting someone so i could order a drink.\n\nDecide to have dinner in the Burnham.  It was 8:30 on a week night. Almost no one in the place. Our server may have also been the bartender.  She paid almost no attention to us. \nBad service can certainly ruin a good meal.',
      'useful': 0,
      'user_id': '2tO6W40AXedq6EWtOhBAww'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 1,
      'date': '2017-05-22',
      'funny': 1,
      'review_id': 'pMQLU95oQJhth_LlF2mE3Q',
      'stars': 3,
      'text': 'Food was good. Had to wait a very long time for eggs and sausage. Team seemed to be short-staffed in the kitchen. The wait impacted my overall experience. I expected more efficiency from a 4-star hotel.',
      'useful': 1,
      'user_id': 'DMoO7HEd7K1Pddkz2BtQmQ'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 1,
      'date': '2016-09-07',
      'funny': 1,
      'review_id': 'E66PNQ0qcgkP_XKYtLl67w',
      'stars': 5,
      'text': 'I love this place! Sherri the bartender was awesome and Chef Ryan brought the food directly to me! What a friendly environment I had the buffalo pizza and Caesar salad my first night and the muffalatta sandwich the next day. Everything was fresh and tasty!!',
      'useful': 1,
      'user_id': 'jWUJsIt9FPwNYsl3ckJSoA'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2017-07-09',
      'funny': 0,
      'review_id': '5ZRvCtvBdANoufZF02fSig',
      'stars': 4,
      'text': "Hotel restaurants are always hit or miss but this one was a nice experience. The ambiance is upbeat and lively -- an open kitchen will also help with that -- and the prices were fair, if not better than expected. My Cotes du Rhone Rosé was $9.50 a glass and the dinner specials on a Saturday night included a super affordable (and pleasantly portioned) Sous Vide Chicken plate. Nothing crazy but just a nice simple home cooked meal while traveling -- something we all crave. \n\nThe Sous Vide chicken was well prepared, maybe just a touch over done, and the creamy mash and broccolini were nice accompaniments. While the broccolini was a tad under-salted, the mashed potatoes were a bit over-salted so I ate them together and it was perfect. I can handle that. I wasn't however, a fan of the BBQ sauce that adorned the chicken. Thankfully, it was mostly served on the side so the dish was still a great choice. \n\nJohnny was a very pleasant server and we worked up a nice rapport by the end. His recommendations were genuine and he steered me toward a dish that I would enjoy - one thing I appreciate in a server! His push for the featured Mitchell's Banana Pudding ice cream was a win, especially with his flexibility in swapping out the vanilla ice cream on the chocolate angel food cake dessert dish. Yum!\n\nAll in all, I think the restaurant offers good value, a fun vibe, and it's awesome that you can go for pre- or post-dinner drinks at the 32 Bar (rooftop)!\n\nFinal thing: I loved their dining chairs. Upholstered and so comfortable. It may sound silly but chairs make a huge difference! I felt relaxed the whole time.",
      'useful': 0,
      'user_id': 'yK4njGSBfFpZhDuTxNpnLg'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2016-08-29',
      'funny': 0,
      'review_id': 'TlqanzBRxIDcTVD1R7cSJw',
      'stars': 3,
      'text': "We came in after watching a sunset up at Bar 32 on a Friday evening. I had read all the Burnham reviews and looked at the menu on line. It looked very promising. Unfortunately, I wasn't blown away. \n\nWe had a reservation which was not required. 40-60% full during the peak dining hour. Service while attentive was only hotel restaurant style, Lacking the type of service professionals found in fine dining establishments. The menu was very well composed and interesting. It offered some good choices. But the preparation and presentation was just ok. The prices were not outrageous. \n\nNo way it gets 5 stars. So not a destination place in my book. If I have to meet someone staying in the hotel, I am happy to have a meal there. But no more Burnham date nights for me.",
      'useful': 0,
      'user_id': 'LBcuPALypeSiLalEZa5JbA'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 1,
      'date': '2016-08-03',
      'funny': 1,
      'review_id': '_xE_ahMSmcELTNHq61CyrA',
      'stars': 5,
      'text': "I'm always a little skeptical of hotel restaurants/bars, but this one impressed me. Everything was delicious and service was great. The atmosphere is pretty decent, but the patio wasn't open yet when I was there, so I suspect that it's great. The tapas are really unique, but my husband and I opted for entrees. I did see the table next to us, ordered tapas, and they looked and smelled amazing, and we're really generous portions. My husband thought his pork entree was smaller than he wanted, but my fish was perfection. I can't wait to go back for girls' night to order a bunch of small plates!",
      'useful': 1,
      'user_id': 'J2ypbaPCF34BJ9IY7W0PLA'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 1,
      'date': '2017-01-29',
      'funny': 2,
      'review_id': 'MNhhAQe9H_wiWRulvkSgxw',
      'stars': 1,
      'text': 'I wanted to give this place one star before ever stepping foot into the restaurant... \n\nI called on a Monday to make a reservation for 12 people. I was told by the gentleman who answered the phone that he needed manager approval for that many people, and put me on hold. I waited a few minutes, and when he returned he informed me that he couldn\'t find a manager but would take down my information and someone would call me back. I waited 24 hours and didn\'t hear from anyone, so I called back. I explained that I had called the day before and needed to speak to a manager, and was placed on hold for ELEVEN MINUTES before I finally hung up. I called back for a third time and immediately asked to speak to a manager, by which the person responded rudely "well can I take a message, we are a little busy right now." I said "no, you may not take a message, I need to speak with a manager immediately." I was placed on hold AGAIN, this time for more than five minutes. I FINALLY got a manager, Franc, on the phone who didn\'t even bother apologizing for my troubles. \n\nOn Wednesday I get a call from the restaurant (two days after my first attempt to make a reservation) telling me they couldn\'t accommodate my party. I explained to them that the manager had confirmed just yesterday, and after being placed on hold YET AGAIN, they confirmed that indeed they could accommodate me. I mean honestly. FIGURE YOUR SHIT OUT. \n\nIf it were up to me, I would not have even given this restaurant my business after that experience, but the group had already collectively agreed to eat here.\n\nFast forward to Saturday... Our server was nice enough but his "helpers" were more attentive than he was, we waited forever for our drinks and dinner, one persons meal never made it out, and the food was just okay.\n\nOverall, the experience was awful and there are so many other restaurants in Cleveland with better food and service. I will not be returning to the Burnham.',
      'useful': 2,
      'user_id': 'uO21Oi5nmXCeEQJOm28lQA'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2016-09-27',
      'funny': 0,
      'review_id': 'RwsIfYNZL_BCI-38VLxOwA',
      'stars': 4,
      'text': 'The fact that this place has 3 dollar signs is not accurate to me. I had a chicken confit small plate, which was more than filling, and it was about $11. I was really craving fried chicken and that meal hit the spot with hints of exotic undertones! Really nice, modern, and quiet ambiance for a late dinner and the staff is readily available to suggest what to eat. My co-worker and I enjoyed the experience. \n\nOnly thing I would say is that the menu needs an interpreter basically. When you first look at it you read something like bread pudding but then underneath it states there are gizzards in it. Not typically what you would think, right? Fact of the matter is, after our waitress explained the take on bread pudding and french toast that they utilized, it was much easier to make a decision. My co-worker got a french toast type plate which was a savory take on a pulled pork dish and she thoroughly enjoyed it.',
      'useful': 1,
      'user_id': 'KloWGEOPweXiqyhcg1ulKA'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2016-08-29',
      'funny': 0,
      'review_id': 'uQNlt_KWDXMnLG3DNZNW9g',
      'stars': 5,
      'text': 'I was at The Burnham for the first time this past weekend.  It was my friends birthday and we had a party of 6.  The service was great!  Our server was super nice, kept our drinks full and brought out our food in a timely manner.  I got the salmon and it was delicious!  The skin was super crispy and it was a large portion.  Everyone at the table was pleased with their dishes as well.  The staff was very welcoming.  I would highly suggest going here.',
      'useful': 0,
      'user_id': 'NkYR1Ejn4nQ1Mm5zwZYIPw'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 0,
      'date': '2016-09-13',
      'funny': 0,
      'review_id': 'y-gnP-jAxUK7srlxuAX9sA',
      'stars': 3,
      'text': "I went here for dinner a few weeks ago. My friend and I were planning to do apps and drinks at the rooftop bar, but it was closed for a private event so we decided to head downstairs for dinner. A lot of things on the menu sounded really good to me, but I went with my usual go to and ordered the salmon. It was pretty good, but was just super salty so not the best I've ever had. My friend ordered the Southwest shrimp pizza, which was amazing! We were one of only 3 or 4 tables in the restaurant so service was great. The prices were about average with most restaurants downtown so nothing too shocking.",
      'useful': 0,
      'user_id': 'M_qwexE6ciWB4tJ_T1S6hQ'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 1,
      'date': '2016-06-20',
      'funny': 1,
      'review_id': '6H_D6GT4Q7NMMdhOdUkEAg',
      'stars': 5,
      'text': "The Burnham just opened on June 1st and we thought we'd try something new. Normally, new restaurants mess up something so our expectations were not high. Let me say this...Burnham was phenomenal and exceeded all of our expectations by leaps and bounds. I cannot wait to go back!\n\nThe decor is very modern, yet warm and inviting. The service was top notch and very friendly. We were seated next to the bar so had the chance to interact with their bartender who made the experience even better. Super friendly guy, very knowledgeable mixologist--everything he made was great. After the first round, we were letting him just give us whatever he wanted. He makes great mocktails too.\n\nFood..OMG. I want to go back the very next day and repeat the entire experience. We ordered several small plates to start with: calamari, pierogis, and octopus. Everything was delicious and the preparations were creative and unlike what we've had at other restaurants. We especially loved the sauce on the pierogis and the octopus was so good we wound up getting another order. For dinner, we tried Oxtail, Burnham Burger, and Shrimp & Grits. Once again, everything was top notch...our plates were clean very quickly. Even the dishes we were on the fence about ordering we wound up raving about. \n\nThe Burnham is a great addition to Cleveland downtown - we'll be back!!",
      'useful': 3,
      'user_id': 'jpdigJmHkSdTy6QzB74wdw'},
     {'business_id': '82aapoEF7sBcxrrP9AWiEw',
      'cool': 1,
      'date': '2016-09-13',
      'funny': 1,
      'review_id': 'Wx7rFTMZmYktoQ3d3orseA',
      'stars': 4,
      'text': 'The brunch buffet had a beautiful display and was dutifully maintained. The eggs were a strange, jiggly texture and bland. The French toast was great, but the syrup was flavorless. Excellent tasting and perfectly portioned smoothie shots. Not all of the items were clearly labeled so my vegan friend was upset from consuming meat in the potato hash. I recommend signs for each prepared item with symbols for meat, dairy, soy, seafood, and vegan items. \n\nThank you for handling our large group with great service and attention.',
      'useful': 1,
      'user_id': 'Kw5RHGNWA3ByBPzMv5oaRg'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2013-08-30',
      'funny': 0,
      'review_id': 'yopVrikncv_54eR44R9sHQ',
      'stars': 4,
      'text': 'Great variety and the best dog in town. Try the rice crispy marshmallow shake/flurry thing. Ridiculous!',
      'useful': 0,
      'user_id': 'Q9nZLjnQ_0RiWfNjCRt9sQ'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2013-12-09',
      'funny': 0,
      'review_id': 'fQn5gA-jXYHefi2Kxx9rcg',
      'stars': 1,
      'text': 'This was my first visit and I was definitely NOT impressed. The service was quick, but not necessarily competent and the food was overpriced. *On a side note, if you order root beer with your combo, be prepared to receive a 12oz drink instead of the 16oz that supposedly comes with it.',
      'useful': 1,
      'user_id': '9Vu2eIk0HXGAD2h1Vr81RA'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 3,
      'date': '2012-06-19',
      'funny': 3,
      'review_id': 'HzreYQD3_VOa-MvleCQhEw',
      'stars': 4,
      'text': 'Went to their grand opening on the June 17th and despite the fact that they were slammed I cannot say enough about the customer service and hustle the entire Retro Dog team exhibited.  The wait was long due to the insane amount of guests but have every expectation that their service will be no less than amazing going forward.  The atmosphere is bright and upbeat with an updated diner feel.  The original artwork speaks for itself.\n\nI tried the Retro Dog as I am a HUGE fan of Coney dogs.  The sauce was different and sweeter than the "traditional" Coney sauce but a good surprising spin on an old original.  Toppings were ample and the natural casing hot dog cooked to perfection.  I ordered the Coney fries to go with and I may have overdosed on the Coney aspect of things and think they might ne better paired with a different dog.  Delicious all the same.\n\nThe custard cone with the salted chocolate dip and caramel was ridiculous.  Although full from my meal, I still couldn\'t help but finish.  The salted chocolate adds the perfect savory to the caramel\'s and custard\'s sweet.  Retro Dog is definitely worth a visit and I plan on returning multiple times to try the entire menu.',
      'useful': 6,
      'user_id': 'MBf09zdFcOEnYA84aENzXA'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2015-10-27',
      'funny': 0,
      'review_id': 'E73McNvR0dT-l-7mlsh_7A',
      'stars': 3,
      'text': "Good food. Way over priced as everything is extra. Tried copying Teds Hotdogs in Western New York. I'll wait for a road trip back home to get my dogs at a much more reasonable price.",
      'useful': 2,
      'user_id': 'cA7hAhlsXIotA4wRLdAZBQ'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2015-11-18',
      'funny': 0,
      'review_id': '6D0JvHSBurLYbC_o_drcJA',
      'stars': 4,
      'text': "This is my regular lunch stop when I am in Stow. The food is fantastic. There isn't an item on the menu I haven't had and haven't enjoyed. \n\nCome for the food, but stay for the shakes.  If that isn't your thing go for the root beer on tap. It's the best in town. \n\nThe monthly food specials keep the menu interesting. \n\nThe absolute best aspect of Retro Dog is the staff. I haven't met a better crew anywhere else. \n\nIn short, this place receives two lion fists up from Voltron.",
      'useful': 0,
      'user_id': 'UCnk2nOq6cvtjboMWkhO9A'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 1,
      'date': '2014-03-28',
      'funny': 0,
      'review_id': '3Gvujhjh6KuKSI8_tQvKFw',
      'stars': 4,
      'text': "If you are looking for a fun little place to go every once in a while, go here. They have all sorts of good quality hot dogs and hamburgers; including veggie style ones. The drinks and milkshakes are awesome, too. The ambiance is very cheesy, but it works. I want to write more about this place, but there really isn't a lot to say. It's a hot dog place...\n\nYou'll enjoy!",
      'useful': 1,
      'user_id': '5NAdHDoMQ63PYn3CVFC5BQ'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2013-01-16',
      'funny': 0,
      'review_id': 'DHidMDAlWr0o68QjpQLWCg',
      'stars': 4,
      'text': "Love this place! My kids ask to go every week! I don't eat red meat, so of course when my husband suggested a hot dog place I reluctantly went! I was pleasantly surprised by the menu! A veggie dog, chicken sandwich and veggie burger! I've had the veggie dog and chicken sandwich, both were great! I'm trying the veggie burger next! Oooh, and the milkshakes, so yummy. Oh and the veggie cheese fries, so good! Even if you like meat the veggie options will satisfy...just has my husband the meat eater! \nNot only is the food great, the service is absolutely amazing too! So friendly!",
      'useful': 0,
      'user_id': 'LMDhAdECX5diWgClBjU5CA'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2014-07-15',
      'funny': 0,
      'review_id': '2MnVp-o1vsKm0e4OtjKzBg',
      'stars': 4,
      'text': "Fun place and something a little different.  Hot dog and burger options abound.  The onion rings and fries are greasy and delicious - dip them both in whatever magical sauce comes with the onion rings.  Chili and root beer are also good   This is not a place for people who are afraid of calories - they have a token side salad and a black bean burger if you want to feign healthy eating while scarfing fries. Ordering is kind of a pain especially if there's a line since there's not a lot of room to stand in line - you order at the counter; then the order taker takes your money and gets your drinks; you sit down; and they bring the food out to you sometimes having to call out by name.  They have car service outside, why not have wait service inside?",
      'useful': 0,
      'user_id': 'yT1A5dsWOfkOb2-H76JQIQ'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2015-12-06',
      'funny': 0,
      'review_id': 'SU2WI-MJ7xnXZjPEqvuNRw',
      'stars': 3,
      'text': "I don't know how they stay in business.  There's never anyone there.  \nThe dogs and milkshakes are good.  It can't hold a candle to Swenson's, but it's closer to where I love, so it's still a contender for me.",
      'useful': 0,
      'user_id': 'R_43l6k5NWPy7eRv6cjYEQ'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 1,
      'date': '2015-05-15',
      'funny': 1,
      'review_id': 'xdq5wPhRrjLnoW2_6ye5PQ',
      'stars': 3,
      'text': 'Not what I was expecting. The "retro" style was only apparent by the colors of the restaurant. 1990s music was playing. \nThe food was nothing to write home about. The same veggie hotdog can be bought from the local grocer. The vegetarian coney sauce was nothing fantastic.\nFries are not fresh cut. Not especially delicious.\nOn a positive note, my mud pie shake was delicious.\nFriendly staff.\n2 dogs+2 fries+2 shakes dine in= $20.81 \nOverall, it was mediocre. Not worth the bill.',
      'useful': 2,
      'user_id': 'yDirjgppPDkY2Bv60ok6bA'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 1,
      'date': '2016-06-26',
      'funny': 1,
      'review_id': 'rCjA0LCkBnvyz-56SMVxUw',
      'stars': 5,
      'text': "Retro Root Beer in a FROSTY MUG!\nSuper creamy, real Frozen Custard! \n\nBurgers, Dogs and Fries with an overwhelming array of topping choices. \nWe visited on a Sunday evening, and ate inside. \nEverything was fresh and delicious. Service was spot-on!\n\nThe Onion Rings were a highlight for me... Made fresh to order, tender inside, light and crisp fried tempura batter outside. I'm my opinion... They could NOT be better!\nHighly recommend that you give RETRO DOG a try!",
      'useful': 1,
      'user_id': '6Zdd9wX-fif8ocrgv21TKw'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 1,
      'date': '2016-06-27',
      'funny': 1,
      'review_id': 'KJjIQhaE5Ws6ujjRvUq9DA',
      'stars': 4,
      'text': '1st trip here .. my husband had 2 dogs w/yellow mustard .. I had 1 w/chili + yellow mustard + 1 with chili + yellow mustard .. we both enjoyed our dogs .. but the stand out was the battered onion rings .. totally delicious .. wish we had gotten a large order to split but w/2 dogs each we wanted to pace ourselves .. root beer float & black cherry soda were the chosen beverages .. we will be back .. not only for more onion rings but perhaps to try a burger next time .. a bit of a drive from our home in Cleveland but a decent road trip to treat ourselves every so often .. far exceeded our expectations & now there is no need to go to Sonic ever again .. thank God!',
      'useful': 1,
      'user_id': '9PmUzfxz3vamMaTzGSesyg'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2017-05-06',
      'funny': 0,
      'review_id': 'zRhusCsWxEentZn1sI_o4Q',
      'stars': 5,
      'text': 'This review is written by my 6 year old son. "The ice cream is really good and icy here. The twist is my favorite. Also, the grilled cheese is pretty cheesy!"',
      'useful': 0,
      'user_id': 'ew4RSciwl1z4GBlFiL_jxA'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2016-01-08',
      'funny': 0,
      'review_id': 'ExBaH0kM9OqRKCrAresYEg',
      'stars': 3,
      'text': "I appreciate what they're trying to do here. They went all-in on the retro drive-in look and unique variations on the dog-and-burger menu. \n\nI've tried three different dogs and one burger; fries and rings. The meat and fixings are fine -- not spectacular, but OK. On my most recent visit, the poppy seed bun for the Chicago dog was distinctly stale. (I've run into a similar problem at some other places, where they don't sell many Chicago dogs so the special poppy seed buns sit around for far too long.)\n\nI'll echo what at least one other commenter said: The sodas are bizarrely small portions. They boast about their own root beer. In house, it comes in what looks like a nice-sized chilled glass mug, but turns out to be largely ice. I got a to-go version one, and it was a tiny cup, again with the ice -- and filled to the brim so that spilling was inevitable.",
      'useful': 0,
      'user_id': 'wRzcrtxZj7MfzQnxrsqqfg'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2015-08-23',
      'funny': 0,
      'review_id': 'X2sysdEU1OCSmb94fYQvGQ',
      'stars': 3,
      'text': 'i\'m generously giving RETRO Dog 3 stars because it\'s a cool concept and a cool looking place. there was really nothing wrong with our food, it just wasn\'t awesome and there was a HUGE issue with the place we didn\'t notice until we sat down (we ate inside).\n\nit was Danielle\'s idea for lunch (because she thinks i have the palette of a 4 year old, i guess) and we drove out after making our final payment for our September vacation (which was on the way). \n\nDanielle ordered a Memphis Belle Chicken sandwich and i ordered a Carolina dog and a White Hot dog. we got these as combos so that we could share an order of fries and an order of onion rings and each came with a soda. \n\nDanielle said her sandwich was good but was served very hot (which i\'d like to think is the idea, right?). the White Hot that i ordered (a hot dog made of veal and pork - which looks "white" when cooked) was good enough, but not terribly flavorful. it could have used a spicier mustard or maybe just more than what was on it (which honestly looked like enough). i didn\'t care for the Carolina dog (a Coney w/ coleslaw and yellow mustard). i didn\'t like the coney sauce or the coleslaw. in fact, i removed the wiener and ate it separately - making it considerably more tolerable. the fries and the onion rings were good, though maybe the rings were a little overcooked. i thought they tested good anyway. the sodas were really small and the entire cup was filled with ice so that you took about 3 pulls off the straw and the Coke or Diet Coke within was gone. they were kind enough to refill, though. \n\nour biggest complaint was that the place was overrun by flies. they were everywhere. i\'m not some dandy that can\'t handle a fly here and there, but there were like 8 per table just hanging out flying around and on the food and just buzzing our faces while we tried to eat. \n\nassuming we go back (and i would like to try the Kraut Dog as well as the Big Sal sandwich and maybe others) we\'ll probably order from our car or we\'ll get it to go and dine without the flies.',
      'useful': 0,
      'user_id': 'aRzBh8fCzIRpMwijyN4sLw'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2016-04-09',
      'funny': 0,
      'review_id': 'joEoPyM5UFau13jXIfP-nw',
      'stars': 4,
      'text': "My friends and I walked to Retro dog after school. We assumed it would be cheaper than wing warehouse. The staff was helpful and friendly. My friends and I got milkshakes and burgers.\nI got a milkshake and cheese fry and my friend, who I payed for, got a burger and milkshake. I didn't expect the total to come as $20 but we enjoyed our time there and the food was good.",
      'useful': 1,
      'user_id': '3W-1dd1S6FylTU4QIFv-_Q'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2013-09-19',
      'funny': 0,
      'review_id': 'pSlId5PfiBAJZpHAEFu-qg',
      'stars': 1,
      'text': 'I really was looking forward to a great hot dog and was suprised the retro dog was so tasteless and boring.  I was surprised when I saw the lady sweeping the floor put  the broom down and take the French frie with her bare hands and put my order of fries in the fryer.',
      'useful': 0,
      'user_id': '7sOJ9P1dcHC-8PwPpRjrFw'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2016-06-02',
      'funny': 0,
      'review_id': '-gbBG4UXcK_cAqO1O8NQJg',
      'stars': 1,
      'text': 'Went for lunch with co-workers. I had the Chicago dog and onion rings with a drink. Dog was excellent, rings not so much. Incredibly greasy - even too much for rings. Unfortunately my co-worker had a disasterous Retro Burger. There was a 9" or so brown, curly hair found in the MIDDLE of her burger. \n\nWe observed a female employee with shoulder-length, brown, curly hair; not tied back or in a hair net, cooking the food. This is gross, unsanitary, and bad restaurant management.\n\nAccording to www.fda.gov - retail food protection - Chapter 2-402.11: " (A) Food employees shall wear hair restraints such as hats, hair coverings or nets, beard restraints, and clothing that covers body hair, that are designed and worn to effectively keep their hair from contacting exposed FOOD; clean equipment, utensils, and linens; and unwrapped single-service and single-use articles. \n\nIt was asked that the burger be re-made, and yet again, the same female employee was cooking as before. The practice of employee hygiene is a basic essential of running a restaurant. Shame on the owner for not being aware of day-to-day operations with his/her employee(s). \n\nI will not be going back, nor recommending this establishment.',
      'useful': 1,
      'user_id': 'kbubuLPhcLYAN1x6zNFObg'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2016-02-13',
      'funny': 0,
      'review_id': 'TyrcA5E6m4Bgeg1_I5GH2A',
      'stars': 5,
      'text': "New drive in - classic style! Hosted my son's 8th birthday here and the staff and food is amazing. They brew their own pop, root beer and beer as well. Lots of special events and great burgers and dogs.",
      'useful': 0,
      'user_id': 'RNUV7k_Ng4KdasXQMsOo6w'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2015-11-13',
      'funny': 0,
      'review_id': 'OwhOCgrFw8kI2mFKPNm3ww',
      'stars': 2,
      'text': "Others seem to enjoy the restaurant, so maybe you will enjoy it, but...\n\nMy wife's hotdog was warm rather than hot. It was also very small compared to the bun. My cheeseburger was the same way.\n\nThe root beer was close, but too sweet for both out tastes.\n\nI think that they spent too much effort on the decor and TVs and not enough on the food.",
      'useful': 0,
      'user_id': 'krWkRW88UtKbsQMrPFzgkQ'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2013-01-29',
      'funny': 1,
      'review_id': 'skTMN0wbSn-munCNyOBKEA',
      'stars': 5,
      'text': "Run, don't walk, to Retro Dog. Had lunch there yesterday. Great food and very nice and attentive people. Had dogs and delicious fries, very crispy and hot! Dogs had a nice snap and the buns were perfect. They have so much more than the dogs, too. Lots of choices. Don't let the exterior fool you, it's not just drive up. There is nice seating inside for at least 40. Get out to Retro Dog and support a local business with very good food and wonderful people!",
      'useful': 1,
      'user_id': '93cD5Ev-PhA1Hut7wU0ctw'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2016-08-17',
      'funny': 0,
      'review_id': 'pIs1EOdGrunBWO00KLOpZA',
      'stars': 5,
      'text': 'Love this place, cool decor, great menu and friendly staff.  We go out of our way just to eat here.  Most likely one of the rare instances where a hot dog place also offers great burgers, shakes and even a veggie option!',
      'useful': 0,
      'user_id': 'qKaGM8XXsNDiUbAMoY2X3w'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2015-10-21',
      'funny': 0,
      'review_id': 'XJ3061Gi1XkQl69oKLhdkw',
      'stars': 3,
      'text': "Nothing fancy at all.  I was expecting something like Chicago dogs.  Pretty small also considering it's a hot dog place",
      'useful': 1,
      'user_id': 'oDoZe4gYUCb7ylHuhWFUow'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2013-05-01',
      'funny': 2,
      'review_id': 'B_fkii_XiE-DYaGaM_mfqA',
      'stars': 1,
      'text': "This is the part of being a food adventurer that bites you in the ass.  This has got to be the 2nd worse place I have ever been too.  There dogs are no better than what you buy at the store, so save yourself time and money and just go to the grocery store.  For 4.29 I got a foot long dog that was saturated in crap!  I couldn't find the dog at first and had to get a spoon and scrape everything off.  There were more topping than dog.  If you don't like toasted buns, don't go here.  I HATE toasted hot dog buns WITH A PASSION and they do it here, without asking if that's what you want.  What I make at home was also more tastier.\n\nI also asked for a strawberry shake, because, really.... how can you mess up a strawberry shake.  Well, they did.  It wasn't that cold.  It was lumpy with what you might call frozen strawberries but it wasn't cold or thick at all.  Almost as if they let ice cream thaw, then scooped it into a glass and let it sit for awhile.  EWWWWWWW!\n\nSave yourself trouble and got to McDonalds if you're really in a hurry and can't go to the grocery store.\n\nThe staff is ok, but not on my list of nicest people in the world.  I got the feeling there were more concerned with sales than pleasing people.  Small parking and one car runner ran into me on the way in.  \nDON'T GO HERE!",
      'useful': 2,
      'user_id': '_MjgqPR1pvDnZZ6wCwabBQ'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 1,
      'date': '2016-03-06',
      'funny': 1,
      'review_id': 'ermuBFPmiXeA7ld1j9VlRg',
      'stars': 5,
      'text': "Drove by after a long run and we were curious... No expectations...and were super happy after! We were pretty hungry, order a Carolina dog and Chicago dog (yum yum).  The all beef dogs are excellent and they have a good choice of hot dogs types.  The Buckeye shake was delightful, but we love PB and chocolate so you can't go wrong.  The fries were basic.  Would stop back if ever in the area again!",
      'useful': 1,
      'user_id': 'jLpQhFMglWJVqOQmAeDVRw'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2012-07-16',
      'funny': 0,
      'review_id': 'aihG-xrXpflyAdgcJx5maQ',
      'stars': 1,
      'text': "1st try we sat in parking lot with lights on for 15 minutes.  No service, no one even looked our way.  The only people waited on were the ones under the canopy.  No signs anywhere that they only have service under awning.  Swenson's services their whole lot!\n\n2nd visit I got a spot under the canopy.  Waitress came to car and wanted me to order.  No menu boards, no menu, I ordered 2 dogs and Onion Rings.  Ordered 1 all beef and 1 natural with the same topping and an order of onion rings to go.  Both dogs barely warm and 1 had the correct toppings, the other one had I don't know what on it, some sort of hot relish maybe?  Onions rings were the greasiest I've ever had!  Great concept that failed to deliver!!",
      'useful': 0,
      'user_id': '9HrVYsOmI13357tWpD18Jw'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 1,
      'date': '2015-04-18',
      'funny': 1,
      'review_id': 'Z-IvmnW-mBo0tLt5WdaCLA',
      'stars': 4,
      'text': "Best Chicago Dogs around. Vienna hot dogs would be better, but these are the best we've had in the area! Kind of pricey, but cheaper than driving to Chicago! ;)",
      'useful': 1,
      'user_id': '_tX6-jgQkcimvIYzj0ycUw'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 3,
      'date': '2016-04-27',
      'funny': 4,
      'review_id': 'V7RP6vuDEakRfbKev0Tk3A',
      'stars': 3,
      'text': 'I love a good hotdog. I was out with my son and we came across this place. It looks like a new place trying to look old. So, the name is perfect. The layout reminded me a lot like sonic and the drive ins back when I was really young. The interior is really nice and I thought they did a great job of decorating.\n\nI ordered the Banzai dog and my son ordered the White hot dog. We also decided to get a large order of fries to split and my son got a root beer. We went there about a week ago and my son is still talking about how magnificent the root beer was. He also loved the hot dog, but said it might have been a little salty. I thought that the banzai dog was also fantastic. The banzai sauce is really what makes it. You could put that on just about anything and make it taste good. Plus, its covered with bacon.. \n\nWhenever I order fries and I see that they are fresh cut fries, my brain automatically thinks Malt Vinegar. I grabbed one of the containers and took it back to our seat. We were seated along the windows so when the light hit the container, something caught my eye. It looked like there were 2-3 ants floating in the bottom of the container. I took the container back to the counter and gave it to the person there and showed them, told them I thought they were ants. Then I noticed all of the vinegar containers had this substance in them. I went back to my seat and a few minutes later the manager shows up with a pan, in which she had poured out the vinegar in to show me they were not ants. She explained that it was a fatty spongy substance that collects in vinegar containers. I told her that I have several different vinegar containers at home, some that I have had for 2+ years and I have never seen this substance. I She said it is common and found it on the internet. So of course I asked the obvious question, "How long has that vinegar been in those containers that it would have this fatty spongy substance?" She told me that they empty them every week and refill them. Now, I know I look like I just fell off the yam wagon, but I find it hard to believe that your vinegar forms that in a week and mine hasn\'t in over 2 years. My ten-year-old was able to ascertain that you were full of crap so needless to say, your excuse needs work. I wasn\'t asking for a free meal, just say "Geeze, that\'s gross.. let me get you a new one." \n\nRegardless of the vinegar incident and the manager clearly trying to cover her ass, I thought the place was great. I will probably return again if I am in the area, but I think I will skip the fries and vinegar.',
      'useful': 3,
      'user_id': 'VOUnXOlll52vwQcJK1tm8A'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2017-04-23',
      'funny': 0,
      'review_id': 'tHDgCqXk5X7g2zxSwX9H6A',
      'stars': 5,
      'text': 'Talk about living the old car hop idea.  Retro Dog has amazing food and awesome service.  Their milk shakes are amazing.  The flavor combinations are always spot on.  You can go in or have the car hop come out and bring you your food.  Love it.',
      'useful': 0,
      'user_id': 'IkLWlkJek2e4It8Xt6vIwQ'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2012-07-29',
      'funny': 0,
      'review_id': 'WN0XEZ6pI8_jJAHea7jkBg',
      'stars': 5,
      'text': 'Delicious natural casing dogs with a variety of toppings, great milkshakes, awesome design. Very happy to have this new addition to the scene.\n\nEDIT: After many return visits I can say the quality is consistent and the staff is always very friendly and competent. It has become my favorite "fast food" stop. Oh, and try the Retro Burger, it is, dare I say, better than a Swenson\'s Galley Boy. I know, blasphemy, but true.',
      'useful': 0,
      'user_id': 'zypGaZmq7QhyV6TZfQrayg'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2016-11-22',
      'funny': 0,
      'review_id': 'fCmGRcztKpZjOx-utDF6TA',
      'stars': 4,
      'text': 'Nice variety of dogs and burgers. Place has tasty vegetarian alternatives. Come alone, with friends, family or just you and the kids.',
      'useful': 0,
      'user_id': '79Vv70bMS1JC9ASCg4dWqg'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2014-10-31',
      'funny': 0,
      'review_id': 'YyueVBhLcEbkTqwS-Mgd6g',
      'stars': 1,
      'text': "I was the only car at curbside, yet it took 5 minutes for someone to bring me a menu.  I paid $22 plus for a cheeseburger, veggie dog, two orders of fries (which were very small) and 2 shakes.  The cheeseburger was not edible.  It was fatty and there was so much salt on it that I had to throw it out.  While good, the fries were in a really small cup.  I could pay the same price and get a large overflowing cup at Five Guys.  The only good thing from the whole experience was the free trade coffee shake.  Won't be back for the food.",
      'useful': 0,
      'user_id': 'o1AatBW0bh_rl5AkPEDUpw'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 1,
      'date': '2015-01-31',
      'funny': 1,
      'review_id': '5GqgskX_c1xjciY7UVbXSw',
      'stars': 5,
      'text': "Great local joint with an old fashion car hop service or eat in there dining area bar stools and tables.\n\nThey have great dogs, burgers made to perfection with all kinds of toppings.  The dogs are great love the Chicago dog best in the area.  The shakes and ice cream can't go wrong and try the old fashion root beer float.",
      'useful': 1,
      'user_id': 'VAmKvQNamoQi4uMqkNmMJw'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2012-07-26',
      'funny': 1,
      'review_id': '5PlM8Xo0v0Kmu4jNn64UyQ',
      'stars': 1,
      'text': "This place was gross! The best thing on the hot dog was the mustard I had Conney sauce and Cole slaw and they were no good. And on top of it the Server that came to the car was rude. They didn't say hello...how are you...nothing! They just handed a menu thru the window and said turn on your lights when you're ready to order. Then when we were done they never said how was everything...nothing! Even tho they seen my brothers hot dog only had one bite missing from it because it was so nasty! For a $4 dog and service like that this place doesn't even deserve a 1/4 of a star!!",
      'useful': 0,
      'user_id': 'SHqPluE5dTLsKctxBi_RZA'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2015-07-21',
      'funny': 1,
      'review_id': 'DSqAdScruiejUwPCQSDmyg',
      'stars': 5,
      'text': 'Really great food! A nice variety of hot dogs, toppings, and sauces. They also have burgers and fries.  We ordered a footlong dog, a natural casing dog with bacon, and the home on the range fries. Amazing! The fries were a bit spicer than expected, so we "had" to order ice cream. Darn. :)\n\nOverall, a really great little place. I\'ll be back soon.',
      'useful': 2,
      'user_id': 'JMe2ZbikZVxRr6zEMIys1A'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2015-05-26',
      'funny': 0,
      'review_id': 'jO6eZE2mi_WxVIP26DRInQ',
      'stars': 4,
      'text': "If you want to relive your misspent youth, or get a flavor of where to go after the sock hop ends, this is the place. Tasty, tasty hot dogs, hamburgers, and the usual sides, plus milkshakes and other desserts. Pull up and order from your car, or sit outside or inside. I often get a hankerin' for a coney dog in the summer, and I really enjoyed the Retro Dog. Hot dog was perfect and a nice soft bun to keep the coney sauce to hot dog ratio just right. The Carolina dog had a flavorful complement of coleslaw for a mellow blend of dog and vegs. Got cheesy bacon fries, which have some heat due to the spicy fries coating. Onion rings were also good. They are also open late when there are events going on at Blossom.",
      'useful': 0,
      'user_id': 'vJipE2XUFxM1u29ZjLNItg'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2013-08-20',
      'funny': 0,
      'review_id': 'PhBKEEu83SAetEomaT7Jxw',
      'stars': 1,
      'text': "Heard this place was terrible from a family friend & wish I would've taken his opinion to heart. My fiance & I were scouting Retro Dog out for possibly having their food truck for our wedding reception but they instantly failed. \n\nI am a french fries queen! I can tell you who has the best fresh cut, frozen, sweet potato, etc fries around and Retro Dog is not one of them. They claim to be fresh cut but they tasted a little frozen. My fries were not raw but were definitely not cooked long enough (no crisp, just limp d*ck). Sadly, bathing them in a small cup of salt & malt vinegar didn't save the day. \n\nThen my fiance ordered their Chicago & White dog but said they weren't very good & he could've made better at home. He also ordered their tempura onion rings- F**king disgusting!!!! The onions inside were cut way too thick & therefore weren't cooked all the way.  The batter tasted like idk what- a really bad knock off of Arby's curly fries &, again, was undercooked & greasy. Even if I had just gotten off of a crappy 12 hour shift at work with no break and an hour long drive home I wouldn't eat their food. Less than 15 minutes later I got heart burn so...\n\n\nAs for their service- not bad, but their Root beer float was the only good thing that I had. But i must admit, I love the concept & the decor- just not the food!",
      'useful': 1,
      'user_id': '51munej7rdZuB7KfPofhaA'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2016-08-01',
      'funny': 0,
      'review_id': 'CjkPcxbRsWEIY09-t9aFOQ',
      'stars': 2,
      'text': "Soooooo disappointed.  As others had mentioned the onion rings are horrible...just rings of grease. The actual hot dog and burger were tasty...but bread and stuff not so much. I've had better root beer...odd after taste. We gave you a try...there is potential.",
      'useful': 0,
      'user_id': 'JMXhCBeOEpB4Tg7ey0ZYJw'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2013-08-17',
      'funny': 0,
      'review_id': 'ksvW7VPBlSnNNL9YVc4CSg',
      'stars': 5,
      'text': 'Good hot dogs. Good root beer. Not a chain. Good times.',
      'useful': 0,
      'user_id': 'oV7mgTSAg1l26ieQ82yyVQ'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2012-12-09',
      'funny': 0,
      'review_id': 'vdQw1CMDiiDkt9kM6YqoEw',
      'stars': 5,
      'text': "I was very impressed with RETRO Dog. As an ovo-lacto vegetarian (I eat eggs and dairy products), I am never sure if somewhere that specializes in hot dogs will have any options for me, but RETRO Dog has not one but three vegetarian entrees--a veggie dog, a veggie Italian sausage, and a black bean burger. They also offer veggie chili, veggie coney sauce, and veggie cheese fries.\n\nI had a veggie dog with veggie coney sauce and cole slaw and a side of onion rings. The dog had a pleasant texture and flavor and the toppings were scrumptious. I have tried quite a few veggie dogs in my day and this was definitely one of the best. The onion rings were unlike any other rings I have tried. The batter had an interesting flavor and was not crumbly like many other rings, but it was crisp and delicious. I also enjoyed the zesty sauce that accompanied the rings.\n\nI chose the locally-brewed root beer for my beverage and I was not disappointed. The flavor was mellow and not overly sweet and it was served in a fancy chilled mug. \n\nMy dining companions ordered a veggie Italian sausage, a RETRO Dog, a Red Hot, and a couple other meat dogs and everyone was satisfied. We also happened to go on a buy one, get one free hot dog day, which was a pleasant surprise for the pocket book. \n\nWe chose to eat inside, rather than utilizing the Swenson's-style carports. They looked fine, if that's your thing. The atmosphere inside is a fun mix of retro kitsch and contemporary. The chairs are surprisingly comfortable and the service was efficient and friendly. \n\nI highly recommend RETRO Dog. I will definitely be returning to sample the rest of the vegetarian delights and specialty sodas.",
      'useful': 2,
      'user_id': 'ak9ijP24Lk3DvXKRx3IM5Q'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 2,
      'date': '2012-07-14',
      'funny': 1,
      'review_id': 'b6dami4YHihHprxXxHnLUQ',
      'stars': 5,
      'text': "Here to tell you I think this place is off the hook!  I was excited to try it & was not disappointed.  At first I thought they were a little pricey, but put things in perspective.  The ingrediants are of higher quality than you would think.\n\nWent on a Sunday afternoon & it wasn't crowded at all.  Four of us sat at the counter and talked to the friendly staff members the whole time.  I ordered the Chicago dog & really loved it!  Boyfriend got the Carolina dog & he loved it!  The boys got the Coney dogs and wanted more.  Nom, Nom, Nom...\n\nFew things worth noting... I am a designer and I was digging everythig about this place!  The layout, the lighting, the surface materials, the graphic design, etc.  I can really appreciate all of the nice, clean details this place offers.  I even took a pic of the rest room!  A+ to the design team!\n\nThey even have nice, thick, soft napkins which were like a luxury!  I want to work here just so I can get a red pair of Converse! Seriously,  I love their uniforms, very cool.  If they sold their shirts I would buy one!  Love the artwork & colors.  Again, well done & kudos to the artist!",
      'useful': 2,
      'user_id': 'NJqmAzerpbAIC0-VBclwgA'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2012-06-25',
      'funny': 0,
      'review_id': 'MN1aB9zV3Gzf0brixIOOpQ',
      'stars': 2,
      'text': "(6-18-12) My daughter had a Noon appointment not far away - pulled in at 11:05am figuring a hot dog joint should be able to get us in and out within 30 minutes. Boy was i wrong I had to FLAG DOWN a server @ 11:40 just to take back the menu since i knew we wouldn't have time to order and eat. I sat there with my lights on from 11:20 until 11:40 waiting for someone to take my order. Several servers looked like they were afraid to talk to people as they run back and forth looking at the ground. At one point i counted 6 cars under the canopy with lights on and no one around. One employee stood in the front of the building (apparently on break) just looking at all the cars with lights on. MANAGER NOWHERE TO BE FOUND. I will try again as they are new and hopefully everyone will be better trained.",
      'useful': 0,
      'user_id': 'jVx2YOXofzQJFL1pXVlVjg'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 1,
      'date': '2012-09-04',
      'funny': 0,
      'review_id': 'Fjzz9mrJOJYcgzSKK4hwFw',
      'stars': 4,
      'text': 'Overall fantastic place. The decor is very cool and the food is good. Even if you are not a fan of hot dogs, they do have burgers (including a black bean burger!) and chicken options. I had the original retro dog- diced onions, chili, and cheese. It was quite yummy. There was a hint of cinnamon in the chili that I could have done without but it was still delicious! My combo also came with a side of home-style fries which were surprisingly crunchy as well as tasty. \n\nRetro dog also offers a variety of retro sodas- cream soda, black cherry, root beer and so on. The root beer is actually their own recipe and made locally. It was a sweeter take on root beer than most places, so if you like sweet pop, you will definitely be a fan :) plus it comes in a frosted mug, just like all root beer should!\n\nCustomer service was fine, seems they have worked out any kinks they may have had since opening earlier this summer. I know I will be going back!',
      'useful': 1,
      'user_id': '0GxKcZqhtzBbhHsTpcv1cg'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2015-06-25',
      'funny': 0,
      'review_id': 'V9m8P5v5j1oR3a_7yV4oyw',
      'stars': 2,
      'text': "I recently dined here and was very disappointed. I have dined at many different hot dog establishments and this one was overpriced in my opinion. The fries were over seasoned, making them very salty and inedible. I enjoyed the snap of the hot dog, but didn't find it to be a hot dog I would throughly enjoy eating again. I also had a float, it was ok but I feel it was overpriced. 2 fries, 1 hot dog, 1 foot long and 2 floats equated to almost $30. Sorry to say, I won't be coming back.",
      'useful': 0,
      'user_id': 'IL_vLMXtXw7RvzMB2N5h3w'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2014-12-30',
      'funny': 0,
      'review_id': 'x9VRH2EiErU6nRR17Lhmsw',
      'stars': 5,
      'text': "Loved it since it opened. Everything's great but be sure to ask for your dog ungrilled if you don't like the charring. I like it like that but my kids don't.",
      'useful': 0,
      'user_id': 'Q1hfot_E_H6xwmBb99JjIA'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2015-06-20',
      'funny': 0,
      'review_id': 'ISqcpzX8fEXxZIBrltu8pg',
      'stars': 4,
      'text': 'This place is good.  The hot dogs are excellent and the burgers are good.  They seem like decent people and are friendly.\n\nMy biggest complaint  is that it is a bit on the expensive side.  We go here fairly regularly.',
      'useful': 0,
      'user_id': 'S6gAeiubzQ7FJOdMvipsog'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 1,
      'date': '2015-03-23',
      'funny': 1,
      'review_id': 'U7HAqwM-zm5lj03y3FXt1w',
      'stars': 3,
      'text': 'I have to rate my visit as a 3 this evening (unfortunately). And really that\'s pushing it. The hotdogs were excellent; however, the fries were over salted and oily, the onion rings were dripping in grease. Very little onion and way too much batter. Please, please change your oil! The root beer floats - well let\'s break that down; root beer was great but the custard ice-cream seemed to have ice chips in it. Not creamy, but "ice milky". Lastly Retro Dog should have "Retro" music. Stick with the dogs and you may enjoy your visit.',
      'useful': 2,
      'user_id': '2AKGU47f2SXSoJeW10ACXA'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2013-07-15',
      'funny': 0,
      'review_id': 'Nl7Txb3LDY4MadHSHOYb7Q',
      'stars': 4,
      'text': "I didn't get a hot dog, but I did get the ice cream sandwich.  I would give just the ice cream sandwich five stars!  The butter pecan and caramel cookie ice cream sandwich was recommended to me by the cashier.  She did not lead me astray.  Two huge cookies with ice cream (which was at least 1.5 inches thick).  I would go back just for that delightful treat.  \nI also got the chocolate malt, which was very good, but not great.  \nEverything else looked decent in the pictures, but I did not partake.",
      'useful': 0,
      'user_id': '7jyNxI6SqoTfiu6IPDTc_A'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2017-04-23',
      'funny': 0,
      'review_id': '2u5l4skrEctKOZfhqvJHYw',
      'stars': 3,
      'text': "Retro Dog is a cool place to stop if you are out in the Cuyahoga Valley or at Blossom and feel like a snack or some ice cream.  Kind of separate from anything else, you will see a large take out with parking out front where you can have the food brought out to your car or go in to the cool dining area.  The decor is bright and in your face in a good way, kind of a very modern expression of a 50's diner.  The menu consists of hot dogs, sandwiches, sides and plenty of ice cream based desserts.  \n\nI have had the veggie retro dog, the fries, onion rings and a couple of milk shakes, overall the food is good, not great.  The veggie retro dog is a pretty nice veggie hot dog with veggie coney sauce, mustard and onions on a wheat bun.  The mild hot dog has a nice snap but it could have used more of the toppings, especially the coney sauce, which is a mild veggie chili.  The french fries are fresh cut, but were not terribly crisp.  The onion rings missed the mark with the fried batter overpowering the flavor of the onions and they seemed a bit overcooked.\n\nThe 2 milkshakes I tried were the best items I had at Retrodog; prepared fresh with soft serve ice cream.  The flavor selection is extensive and they have some unique flavors for you to try.  The chocolate banana was a nice balance of the banana milk shake with hot fudge; the salted caramel was a nice balance of sweet with a hint of salt.  Overall good shakes, especially using soft serve ice cream as the base.\n\nRetro Dog is probably not worth a special trip or the place for a full meal, but if you are in the area and need a snack it is worth a try.",
      'useful': 0,
      'user_id': 'dD5vTDPzv_79m8_zZhc9hA'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2014-09-16',
      'funny': 0,
      'review_id': 'iq7sszeuGeX_XlEu8ESXbA',
      'stars': 2,
      'text': 'The Red Hot was very soggy. My bun fell apart. The Chicago Dog has good flavor, but the wiener was nothing to write home about. Overpriced and flies all over.',
      'useful': 0,
      'user_id': 'sVt7x4X4nFcr2uTfmZ8GkA'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2012-06-25',
      'funny': 0,
      'review_id': 'Z9HivQIQpAijL2ZccSI6Rg',
      'stars': 2,
      'text': "Great service, OK food.  It seemed like everything was just a little off.  I had the Retro Dog and onion rings and the hubs got the Carolina Dog with fries.  The hot dog itself was really good... Mellow taste, nice snap... but the coney sauce was a turn off.  Very sweet, not much spice.  My husband said his Carolina dog was good, but he said he would try something else next time.  The onion rings really had me excited because they looked perfect... they weren't.  The spices in the tempura overwhelmed any onion flavor that might have been there and they were SO GREASY!  I ate one and a half rings and had to stop.  The fries were just OK too and needed to have salt added to them.  They had a really nice natural potato flavor but they tasted like they'd been sitting around for a while, which seems implausible considering how busy they were.  We also ordered the root beer and the flavor was not unpleasant, but it was very flat.  Considering how very sweet it is, it could have used more carbonation.  \n\nI would like to return to Retro Dog and try a dog with just mustard.  Of course, we also need to try the custard!",
      'useful': 1,
      'user_id': 'MZVSvbnghsV8uDM-TnPTKA'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2014-02-27',
      'funny': 0,
      'review_id': '_Hc2ABjlPWUxOGtvhKFTTA',
      'stars': 3,
      'text': "I really wanted this place to be amazing. The food is just okay. Nothing too spectacular. I didn't find a WOW factor. I do feel like it is REALLY expensive. For my boyfriend, daughter and I, we can drop $25 easily and thats not getting a lot of food. At all. I do find the service to be hit or miss too.",
      'useful': 2,
      'user_id': 'zrEx0FVoPnk23Ny-uiOQSw'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2012-08-06',
      'funny': 0,
      'review_id': 'UDJEIoZ1sNCFNz1BTrp5FQ',
      'stars': 3,
      'text': "This was our second time here.  The first time we came the service was slow and they kept forgetting parts of our order.  It wasn't even busy.  The coney sauce wasn't very good on our first visit so this time ordered our stuff without it which was much better.  The service this time was better but we also ate inside too.  Their custard and fries were great both visits!",
      'useful': 1,
      'user_id': 'JwQUSRWaxQjsXqTCKjkmbQ'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2016-08-03',
      'funny': 0,
      'review_id': 'WDNZOLZ8xqD34XUeRBqwSw',
      'stars': 5,
      'text': 'Nice little "drive-in" diner. It\'s literally in the middle of nowhere but worth the trip. We went on the behest of my brother in law, and couldn\'t have been more satisfied. Actually ordered a couple hot dogs and the wife got a burger. The meat was seasoned well (pause) and the dogs were grilled perfectly. Ordered the loaded fries (tip: you can go for the regular size portion if sharing with another person, it\'s plenty!) and was stuffed the rest of the day and we ate there at 5PM. Staff was super friendly and helpful with our order.\n\nWill definitely be back. To get more food, because why not?!',
      'useful': 0,
      'user_id': '-B95zZfxg16diBJvZIx-1Q'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 1,
      'date': '2013-04-06',
      'funny': 4,
      'review_id': 'TzLCnpe2XT_iJzkPE2th3Q',
      'stars': 5,
      'text': 'BING BANG BOOM BABY! This place is baller. From the Chicken Sandwich to the Hot Dog to the kick Ass Shakes. I love this place! Gets me feeling craZy!!!! Killer stop after cruising the valley on your mid 90s hog.\n\nSeedable-',
      'useful': 2,
      'user_id': 'gpV4qU5swBe2Eo7rRDEzjQ'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2016-02-07',
      'funny': 0,
      'review_id': 'IV_XpEcTssyXZz6QQHDlfw',
      'stars': 1,
      'text': 'Cool LOOKING place but about $25 for 2 chili dogs 2 fries and some sodas was WAYYYYY TOOO MUCH MONEY and while it was a cool looking place....I would NOT go back only because of the sky high prices.',
      'useful': 0,
      'user_id': 'uYUGdTQLXGtfngSOqmnJMg'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2017-05-11',
      'funny': 0,
      'review_id': '2AgUAZmfayZAqVOhrJ15Cg',
      'stars': 1,
      'text': 'Owner could not even take the time to address concerns with management and bullying of employees. Terrible treatment, service, and management. Will NEVER get my business again.',
      'useful': 0,
      'user_id': 'vjX5ncgL6WWf3hxWu5mrTA'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2017-02-15',
      'funny': 0,
      'review_id': 'nnnr3NU06dUbHPHYhH6K7g',
      'stars': 3,
      'text': "Went there on a Monday. Placed looked like a slow night.  The selections were interesting and the fries Tasty.  Got two of there special dogs(Chicago and Bonzai). Buns were very stale and i think the dog was on the griddle a little too long. If I'm in the neighborhood I might give them a try gain but nothing I would go out of my way for. On a positive note the Employees working were very attentive and nice.",
      'useful': 0,
      'user_id': '7YqIOPRpr_ldE9xwfMZvDw'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 2,
      'date': '2015-04-28',
      'funny': 1,
      'review_id': 'Oq3fB2xci6V6jugCa0Pt4w',
      'stars': 5,
      'text': "What can I say about the food. It's awesome. Great family atmosphere. The onion rings are amazing and even the chicken sandwich is good. But skip that, grab a dog.",
      'useful': 1,
      'user_id': 'iNUH_vfvhzF7qxg9k0ofsQ'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2014-12-27',
      'funny': 2,
      'review_id': 'TTMT3YcZ1me82C-Azw7PBQ',
      'stars': 2,
      'text': "It's an unusually warm day, almost 60 degrees, and I thought you need to go eat at a drive-in and take in the balmy day. I've been to Retro Dog once, not long after it opened, and I wasn't impressed but I thought give it another shot. This was going to be a 3-star review until after I finished eating, turned on my lights, waited 15 minutes, then had to CALL them to come out and take my money. The young man really was giving good curb service until the end. He was very apologetic, but come on, he saw my lights on as did another young man. They took care of 3 other cars and ignored me. Anyway, as to the food, it was just ok and pricey for the quality, $10.55 for a bacon cheeseburger, small fries, and small Coke. Nothing I couldn't have made at home better. No next time for Retro Dog.",
      'useful': 1,
      'user_id': 'hDSeawZ10uiqU8lZk6knUg'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 2,
      'date': '2013-09-11',
      'funny': 0,
      'review_id': '5uiEKMwuH4BNPKPncNCRvg',
      'stars': 4,
      'text': 'Even though I planned my weekend in the Akron area around burger joints, beer and baseball, I couldn\'t pass up a hot dog place so close to my hotel.  You know, just for a snack.\n\nFor some reason, I was feeling adventurous that day.  I went against my instincts and, at the recommendation of Liana E, I went with the veggie dog.  I got mine topped simply with just the special mayo mustard.  I also went, of course, with a cup of the locally brewed root beer.  I debated the growler when I saw them on the counter, but it seemed a bit pricey (I think it was $15 for the growler full of root beer).\n\nI took my cup and dog to go, but really just sat outside at one of the tables since it was such a nice day.  I unwrapped my dog and was surprised at it\'s size - bigger than I expected.  This was going to be more than a snack.  I took a bite.  And instantly my brain thought, "This doesn\'t taste like a hot dog."  Then I remembered, duh, I ordered the veggie dog.  The taste was not unpleasant to my meat-loving sensibilities.  And the special mayo mustard, which I think was just mayo mixed with dijon mustard, had a nice flavor to it.  Though I\'m not sure it was the best condiment for the veggie dog.  The root beer was aces.  I still regret not splurging on that growler.\n\nI\'ve vowed to return to the area and I\'d like Retro Dog to be part of it.  If I enjoyed the veggie dog so much, I can only imagine how much I\'d like the meat hot dogs.  I also saw some amazing looking onion rings I\'d like to give a try as well as some tasty looking desserts.  Seems like my kind of place.',
      'useful': 0,
      'user_id': 'WoKCLSctS7G2547xKcED-Q'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2015-07-12',
      'funny': 0,
      'review_id': '6IjSwXOZYw08x7XVqASkYA',
      'stars': 2,
      'text': "I am a vegetarian. The veggie dog was not that good. It tasted a bit...old or dry maybe. The veggie chili was also a bit dry and very salty. There were lentils in it, but there just wasn't enough moisture to make it that enjoyable or enough cheese to save it.",
      'useful': 0,
      'user_id': 'oU7r51yitCVUFApwy0RP9w'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2012-07-09',
      'funny': 0,
      'review_id': 'cJvPR9c9f9aOYAqc3W5LGQ',
      'stars': 4,
      'text': 'Great concept.  Food is good-fun menu and terrific custard desserts.  Drive-in service as well as indoor/outdoor seating.  Firepit for cooler nights.\nPlan on visiting many more times.   Onion rings are fantastic, hot dogs are good, chicken sandwiches are great!',
      'useful': 0,
      'user_id': 'mDfSB8UtFoKxHkrtBcikhg'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 2,
      'date': '2015-05-19',
      'funny': 1,
      'review_id': '626Ty7htUS6jl4HdsbFQMw',
      'stars': 4,
      'text': "Eaten here a handful of times. The hot dogs are good. They have nice snap when you bite into them. Burger decent. Nice to go to when your going to or coming back from Blossom. Root beer is good too. Worth a try. Use the coupons you get. I think the prices could be a bit better. But I'm cheap so I don't like to pay a lot for a hot dog. It's just a hot dog... But worth the try to see if you like it. Eat inside. Too messy to eat in your car. But it's cool to eat in your car so if you don't care then go for it.",
      'useful': 1,
      'user_id': 'RN0r5vFDP0DcmL7rmKD45Q'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2012-07-04',
      'funny': 1,
      'review_id': 'Jt0eh-6av_BQbatsrFngbw',
      'stars': 4,
      'text': "I've been to Retro Dog several times including before, on and after the grand opening on June 17th. I'm not sure what people expect from a brand new concept restaurant in this area that's not even a franchise. I, myself, being a reasonable and realistic person who also manages people for a living, was prepared for the imperfections in service. The great part is there's already significant progress being made. \n\nThe food is really good. I've had the retro dog, the white hot, and kraut dog (so far, my fav), and am starting to have some fun building my own. The bean burger is very interesting and tasty! The veggie dog is also really good. I love that they have healthier, vegetarian options. But the ultimate is the Retro Dip. This is a cone of amazing deliciousness. Who knew I'd look forward to the very bottom of the cone as much as the top?",
      'useful': 2,
      'user_id': 'HR1A5U1Rg-IWwryuJN38KQ'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2014-08-19',
      'funny': 0,
      'review_id': 'YteUSdhWSsMjuFPxAiWwVw',
      'stars': 5,
      'text': "Randomly found this place after a concert at Blossom and overall I enjoyed it.\n\nService is quick and food is out quickly as well. I like the fact that they bake their own buns, definitely creates a different experience.\n\nPersonally, I'm a chili cheese dog man and was sad to see none on the menu. Never had tried a coney dog prior, instinctively I ordered one as well as a Chicago dog. As far as taste goes, it was definitely tasty, just not in the sense that I'm use to. The Chicago dog was excellent and PACKED with overflowing toppings and tasted great!\n\nMy SO got a coney dog as well and pretty much shared the same experience. She's a chili cheese dog person too. I know.. we're meant to be together haha.\n\nFries were fresh and their onion rings were unique and intimidating but definitely delicious! Loved every bite.\n\nAnd last but not least, their MILKSHAKES. Thick and flavorful, just like a traditional milkshake should be. Chocolatey goodness!!!!\n\nI'd definitely come back again!",
      'useful': 0,
      'user_id': 'EQGF7Eyt65qhcCMxExDDNA'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 1,
      'date': '2015-06-04',
      'funny': 1,
      'review_id': 'DlGANFJ2Lfu6K-mdpRN2MQ',
      'stars': 5,
      'text': 'Traveling through town, we were looking for an inexpensive, quick, unique place for lunch. This is certainly it.  Wonderful hot dogs and great frozen custard. The atmosphere is great but the highlight had to be the onion rings. Wonderful batter, crispy and delicious. Would certainly recommend this place. Our family of 4 ate lunch for about $34 without custard.',
      'useful': 1,
      'user_id': 'dQ4jp8ThQXvX8CK1cC5FMQ'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 2,
      'date': '2015-06-06',
      'funny': 2,
      'review_id': 'sXLp0ZarK3D9Xh1Nh6Q5Pg',
      'stars': 5,
      'text': "I just loved this place! Neat atmosphere and the entire staff was friendly and helpful! I got the Hula Burger and BLT Fries - loved the hula sauce and burger! The fries were very good too - but next time I'd get the fries with hula sauce on them. \n\nThey have very cool ice cream options too! One of cups has a sundae at the bottom and a milk shake poured over it!\n\nI can't wait to take my family here! These kind of burger/hot dog joints are my new favorite places to find!",
      'useful': 4,
      'user_id': 'PmgqNO0-5Y3e3UoR61TD7w'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2017-03-03',
      'funny': 0,
      'review_id': 'Bok7L4o0gJtIb5u_O-EU4Q',
      'stars': 2,
      'text': 'Huge disappointment. Great concept, great atmosphere. However, food was less than. Hot dog and fries were overcooked to the point of being burnt. The onion rings were the only good part of our meal. It was sad as our expectations were high.',
      'useful': 1,
      'user_id': 'Jn4HQu010ypleAfuoI_ejA'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 1,
      'date': '2015-07-05',
      'funny': 0,
      'review_id': '76-P_Of30jjOzioj7_H-2g',
      'stars': 5,
      'text': "I have never had a negative experience here! Quick service, food is ready quickly, servers are sweet. Cream soda is addicting! The hot dogs and burgers are delish! We have been coming here since they opened. Only thing we have never done is go inside to eat. It's always so fun eating in the car with the kids!! You should eat here!!",
      'useful': 0,
      'user_id': '0GVyC7DxFsgnG8SeMPHPIA'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2016-11-15',
      'funny': 0,
      'review_id': 'g5mWOnIXc317bZ6W7Mci1A',
      'stars': 1,
      'text': "When I drove by I had high hopes.  Not so much after stopping in.   The dogs were just ok.  I got the White Hot and a regular with Coney sauce.  The wife got a Kraut dog.  We each had an order of fries.  Mine were the garlic Parmesan hers were the rosemary.   We couldn't tell the difference between the two.  There was not enough salt/flavoring on either of them to tell the difference.  The Coney sauce tasted burnt.  The Kraut dog was just okay.  The white hot was just fine. And last of all the prices were very high.  We wont be back which is a shame.  The decor is great. The service was fine.    The place could be great it just needs some work and better quality control and 15 to 20% lower pricing.",
      'useful': 0,
      'user_id': 'T9yWXszhzEKDVaNH09aa_w'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2016-04-03',
      'funny': 0,
      'review_id': 't0kYA8OAJmlIy8QYIIndhg',
      'stars': 4,
      'text': "very good tasting hot dogs, a lot of options for toppings. a little bit more expensive than you'd want just for hot dogs, but the atmosphere is awesome and there's a lot of menu choices. I would recommend.",
      'useful': 0,
      'user_id': 'Q2xIT2AWVPDnoknkghaDhg'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2015-07-20',
      'funny': 0,
      'review_id': '_rJzi4ycLF3t0VscjJkqlA',
      'stars': 3,
      'text': "I normally like retro spots like these. This one didn't really wow me. The food is extremely greasy, kind of flavorless and they cover all the ice cream in chocolate syrup. The service was pretty speedy and I've never seen it jam packed so you can get in and out quickly. Decent place for a quick bite or if you live in the neighborhood and want to walk up it would be a nice little walk.",
      'useful': 0,
      'user_id': 'gTE9uptqKvCDP9EZ8o7CmA'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2012-08-13',
      'funny': 0,
      'review_id': 'wrB1_kYGWqSBKII_rMO45g',
      'stars': 2,
      'text': 'The beef hot dog tasted like a regular pork hot dog. No"snap". Very disappointing. The fries were good. I won\'t be back.',
      'useful': 0,
      'user_id': '5hBsrup-f9YpY86FjhIBkg'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2017-06-20',
      'funny': 0,
      'review_id': 'nK4xEEC7B868dGKxRdzFUw',
      'stars': 5,
      'text': 'Chicago dog was delicious.  My husband had a double cheeseburger was probably the best cheeseburger Ive had in while.',
      'useful': 0,
      'user_id': '6q3szv1RW1iJ9jdDhHo9Eg'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2016-04-06',
      'funny': 0,
      'review_id': 'e40w_6A0Nqw73GBsOBO0Vg',
      'stars': 5,
      'text': "My significant other and I have been frequenting this establishment since it opened. We could probably walk there in less than ten minutes. He loves the natural casing hot dogs and I love the Retro burger. Both are an incredible quality at a great price. They also have some of the best seasoned fries around. I throughly enjoy all of their frozen treats as well. They make a frozen custard marble that is amazing, with any topic you can think of, plus their milkshake selection is seriously impressive. I dig the curb side dining experience, and Retrodog is doing it much better than any of the other competitors around. The staff is always sweet, quick and you can tell that they enjoy working there. Great customer service. Oh yeah, and they make homemade root beer that is very good and unique. It's a must try. I'll never go back to a Swensons/Skyway/A&W ever again. The quality of food alone makes them stand out above the competition. IMO they are the number one burger/hotdog/fries experience in the Akron area.",
      'useful': 0,
      'user_id': 'ZXL747B5h5_VJQ9L0CQYYg'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 1,
      'date': '2014-04-15',
      'funny': 1,
      'review_id': '33-dltZ7zXzFO7p-XxfqZA',
      'stars': 5,
      'text': "This place is great! The white hot (veal/pork) dog is amazing! The shakes are amazing and their hula ghost burger BLEW MY MIND! made with a ghost chili sauce with bacon, grilled onion and pineapple makes for a burger that's out of this world!",
      'useful': 1,
      'user_id': 'dEle3NbNjl48EsZjIoVbIw'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2014-07-25',
      'funny': 0,
      'review_id': 'ZCKHHymksJ9iMOte-yxF8Q',
      'stars': 5,
      'text': 'Seriously. Banzai burger. Best I have ever ever had.  Everything at this old school/new school place...reasonable prices and tasty service...',
      'useful': 1,
      'user_id': 'DZChyIi8X3LMVLi86793VQ'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 1,
      'date': '2015-05-12',
      'funny': 1,
      'review_id': 'SWQdtCMugPomTdq_iAqEUg',
      'stars': 5,
      'text': 'I tried this place today and it was really good! I ordered the retro veggie dog and cheese fries. The retro dog was really good it came with chili, mustard and onions. Being an all vegetarian meal I was pretty pleased. The cheese fries were fantastic! Crispy and the cheese was yum! We will be back.',
      'useful': 1,
      'user_id': 'Q_QsCiaeQ5T_CfmidxEM7Q'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2013-07-01',
      'funny': 0,
      'review_id': '5yYeifSxcVQPJESPlE6Drg',
      'stars': 3,
      'text': "This was a nice change of pace from our typical road trip meal stops, but it was nothing that blew my socks off. The dog was fairly tasteless (I'd rather have my store-bought Hebrew National any day) and while the toppings were decent, it was nothing that made me say WOW, I WANT ANOTHER. I tried the home brew root beer and again, it was decent, but I like Barqs better. The best part of the meal was the fries. So, an A-OK stop and if I lived close I'd definitely try other menu options...but as a one-off stop, it didn't thrill me as much as I'd hoped.",
      'useful': 2,
      'user_id': 'VAExcxlYFVzCgLuPmertmw'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2015-07-28',
      'funny': 0,
      'review_id': 'Hfhr1C53RyT9PoDVJonE8Q',
      'stars': 2,
      'text': "Save your money. Onion rings were dripping in grease! Hamburger had no flavor. My third time here it's good in theory, just not in execution.",
      'useful': 0,
      'user_id': 'bxtUw-buCnWybRbjc8cYYw'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 1,
      'date': '2014-09-15',
      'funny': 0,
      'review_id': 'gANC6mBUOLKHqqRjaYMgQQ',
      'stars': 5,
      'text': "Was down in the area for some hiking and remembered seeing some good things about this place. Retro Dog is really in the middle of nowhere which explains why it wasn't so busy but easy enough to get to once you know what streets to take. The car hoppers came out very quick, gave us the extensive menu, and came back out just as quickly when we were ready to order. The shakes were DELICIOUS and the hot dogs were very good too. Good choice of bun as well. The fries with cheese sauce and bacon were also very good.\n\nOverall, my wife and I definitely enjoyed it. We put it right up next to Swenson's as one of our favorite places to go for food. Very good!",
      'useful': 2,
      'user_id': 'Y9oXy1NMExRpK3ac1mYCIw'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 1,
      'date': '2016-11-17',
      'funny': 1,
      'review_id': 'ZB02COirMJvKCOehmzEePQ',
      'stars': 4,
      'text': 'Did a Hula burger, it was different but pretty good. Onion rings are awesome! Coleslaw was not all that good.',
      'useful': 1,
      'user_id': 'LJqJ8N0WCFtD2qQ7aiK5Lw'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2015-01-06',
      'funny': 0,
      'review_id': '2pEd1p3bv5NhZIV7FeUw-g',
      'stars': 5,
      'text': "Man do I love Retro Dog!! This awesome mom and pop business does not disappoint.    \n\nIt's a hot dog place, so of course the dogs are wonderful (my favorite is the Carolina) but that doesn't mean you should shy away from the burgers! Whenever I'm really hungry I get a Hula burger. Provolone cheese, lettuce, grilled onions, bacon, and a pineapple slice all doused in this incredible spicy hula sauce. Yum!\n\nDon't forget to save room for custard! Not ice cream, but delicious, creamy custard. There is not a single thing on the desert menu that I wouldn't get. \n\nThe service is very friendly and helpful! They genuinely seem to be very happy at this establishment. Not the usual disgruntled employees. The managers are just as wonderful! Always making sure you're happy, and that your visit is a good one.\n\nThis place is unlike any other restaurant I've been to, and I always recommend it to anyone who is ever going through that area. I love retro dog!",
      'useful': 0,
      'user_id': 'ORfiVdUir9AHHLuV_lA32g'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 2,
      'date': '2012-07-02',
      'funny': 1,
      'review_id': 'fDPEUWEylLnHQjGTb88o1g',
      'stars': 4,
      'text': "I'm surprised to see so many negative reviews! I went for the grand opening and while we waited for our food it was so worth the wait. I'm a hot dog snob, expect to pay more for it and was not disappointed. The natural casing dog was perfectly crisped in the outside, flavorful on the inside. I ordered the Kraut Dog, kraut, pickles, mustard, delicious! We shared the chili fries that really hit the spot and finished it off with their house root beer. I'll definitely be going back to try the rest if the menu soon... Especially the chili dog.",
      'useful': 2,
      'user_id': 'wtLB2NwR7r7hsoI5JMytow'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2015-03-27',
      'funny': 0,
      'review_id': '0ATspk6mvqL5M6DoIjFuyg',
      'stars': 1,
      'text': 'We were here on a weekday afternoon.  I had a white hot dog and my companions had Chicago dogs.  Mine was acceptable.  Their hot dogs were not great.  If you strip away the bun and the toppings, the hot dogs should be amazing.  They were very flat.  There is a huge difference between cheap hot dogs and flavorful hot dogs and these were generic at best.  The hand cut fries were decent but not amazing.  The onion rings were awful, full of grease, and well done.  The dip for the onion rings would be better if it was a full flavor mustard.  The service was quick.  The prices and portions were acceptable until we ate the food.  We were seriously disappointed.',
      'useful': 0,
      'user_id': 'jBEOFvUNNGrG3AE0GmR-lg'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2012-08-03',
      'funny': 0,
      'review_id': '5nXykjVXwjIYHHhnyus_AA',
      'stars': 5,
      'text': "This place was great. I don't understand the poor reviews. One of the best hot dogs I've ever had. Milkshake was solid as well.\n\nAnd the poor waitress braved a monsoon to take our order quickly. \n\nCan't wait to go back.",
      'useful': 1,
      'user_id': 'H0lxx708beW0UyNBYE8hBg'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2013-07-10',
      'funny': 0,
      'review_id': 'Fn6CK4sjJt4PL9sYGcLuKw',
      'stars': 4,
      'text': 'Excellent local restaurant for grilled American fare. The selection of burgers, hot dogs, shakes and fried foods is just incredible. There are also vegetation options (dogs and burgers) which is greatly appreciated! The place is clean and neat, and the service is quick and modernized (they use iPhones to ring you out). One minor issue - the hot dogs and burgers are great, but the buns themselves are either too big or too soggy.  The food is incredible, but a better bun makes a even better sandwich.',
      'useful': 0,
      'user_id': 'H4_YPiEqsGYiUWWHx63h9Q'},
     {'business_id': 'PjQngP_7m7PA8K1WUHdXqg',
      'cool': 0,
      'date': '2015-10-29',
      'funny': 0,
      'review_id': 'UXnMFX2z5-SX3uZoYqzlIg',
      'stars': 5,
      'text': "I can't believe some of the complaints on here! Retro dog is awesome plain and simple, if you don't like something you picked out build your own meal! They offer that. Any who, never have had a bad meal, in fact we drive from North Royalton/Strongsville just to have there food! There hot dogs &&& burgers are to die for, seriously! The prices I don't mind at all either because I work at a family establishment and times are tough and so are soaring prices in the food industry! We love you retro dog!",
      'useful': 0,
      'user_id': '3-Bby8p3bmiaoAL8oVI9RQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-04-10',
      'funny': 2,
      'review_id': '5QLj2PYisso1Mjf8ePUAsg',
      'stars': 1,
      'text': "Harold and Kumar traveled how far for these dumb burgers...\n\nTotally not worth the hype. My ex stoner roommate was obsessed with White Castle and used to buy the frozen ones.. So when I tried these for the 1st time I couldn't believe how it tastes exactly the same (yuk)\n\nPeople were buying them by the case when we walked in, total disappointment. People in the east coast got it all wrong these soggy sliders are crappy. Another reason the west coast rules.",
      'useful': 1,
      'user_id': 'sreN9pXwVsTu1RMY9WnLpQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2017-06-25',
      'funny': 0,
      'review_id': '1x5xGqyKg9vtfrIXKVd6uw',
      'stars': 1,
      'text': "Need to be a teenager, drunk, or stoned. It took 15 minutes for a single to go original slider. Greasy, heavy on sautéed onion, very thin patty with very little texture... I'll take a McDonald's hamburger any day. Looked better in the movies.",
      'useful': 0,
      'user_id': '_H0ydYpHUo6fzAZq2EphnQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-02-28',
      'funny': 0,
      'review_id': 'MZ8DnofZvbnespruAOT4Pg',
      'stars': 2,
      'text': "This is seriously overrated!! They are really just burgers mini sized. My fiance scarfed down 4 mini burgers. Their fries were gross. It was not salted. I ordered their chicken rings and it was really just chicken nuggets in a different shape. AND it's expensive!",
      'useful': 0,
      'user_id': 'yV1PN-KG7vMKV9xyM4JToQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-05-07',
      'funny': 1,
      'review_id': 'aOc2urRuLBAol6y10lwvbw',
      'stars': 2,
      'text': "Set your expectations really low when coming here and you might not hate yourself too much for spending your well earned dollars here.  \n\nThere was nothing memorable about having them.  We came here for the novelty of trying it out just once for the sake of Harold and Kumar Goes to White Castle.\n\nI'll give it an extra star because I think these things would make great drunken munchie meals, but then again, so would everything else.",
      'useful': 0,
      'user_id': 'xlZrg40Tmk9xPlp7diyyXQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-05-28',
      'funny': 0,
      'review_id': '1CGsM6-g0gP2tP430jDriA',
      'stars': 5,
      'text': "Yes, there is now a White Castle located on the strip in Las Vegas. Everyone who wants to try them without going to the Midwest/East coast can now have a taste of the well known slider. I have enjoyed White Castle before in Chicago and found that they are quite tasty and I can put a few back. My cousin always orders them with extra pickles.  I should have remembered that when eating them in Vegas.  I forgot that tidbit, but they were still really good. I really wish they had a location in Denver.  That would make my year.  The best 2 things about the one on the strip, they are open 24 hours and they sell beer.  What can make this place better? Well, it is also attached to a casino! The place was hoping and cranking out burgers at a rapid rate.  If you haven't tried White Castle, now is your time.",
      'useful': 0,
      'user_id': '8a0GHVSVs3rSRtNIIR4rlw'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-11-20',
      'funny': 2,
      'review_id': '-4lz4jSlOeGyIbmFqbdPxQ',
      'stars': 5,
      'text': "Got to get em hot fresh off the grill! No you can't take these Lil babies home. They get cold quick. They don't sit on your belly all day. It's a reason people in Saint Louis call them sliders. Lol Yes my husband is from Saint Louis so I have a degree in the fine art of White Castle. You order these when their is a long line. You sit and eat them. I enjoy them most when I have been dancing all night and the club is closed and it's 3am. And all my friends meet up to have these. Omg the fries are great with cheese. And the breakfast sandwich is devine. I give them five stars!",
      'useful': 0,
      'user_id': '6ZC-0LfOAGwaFc5XPke74w'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-05-26',
      'funny': 0,
      'review_id': 'yn6Pxvx-R5X3LUrdxVSq2g',
      'stars': 5,
      'text': 'Thoroughly enjoyed the Midwest sliders. Freshly made and tasty. Brought back memories from Chicago late night slider runs.  They have onion "chip" rather than full onion rings. Think onion rings cut into pieces. \nAnd being Las Vegas, this White Castle serves beer - keep that buzz going! \nFriendly staff and a dining room cleaning crew that was constantly keeping the place clean.',
      'useful': 0,
      'user_id': 'p30VbeNZC6mzRUTL6lg69w'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-06-22',
      'funny': 1,
      'review_id': '9VlUxstf22VBfFPIcZM6eg',
      'stars': 4,
      'text': 'Very small restaurant capitalizing off the brand\'s nostalgia and charm. They sell their signature miniature "slider" sized burgers at an overpriced rate. This seems to be the selling gimmick for White Castle, luring in drunk gamblers or curious tourists to stop in. \n\nThe restaurant itself is attached to the casino of the Venetian I believe. It is located on the Strip, with a one door entry point. A long line of hungry customers fills the restaurant, although it moves rather quickly. There are only 5-6 tables, with 2 larger bar-stool type tables for seating. As you reach closer to the register, you are welcomed with a better view to the behind-the-scenes look of White Castle. Rapid hands swiftly assemble buns amongst grilled onions and juicy meat patties, exchanging between different work stations to finally end up placed in a singular order, ready for the customer. \n\nA large glass window display showcases some of the White Castle souvenirs that are available to purchase to take home. A $10 coffee cup or a $25 T-shirt? Not for me but maybe for others. \n\nLast, but certainly not least, was the food. If you\'re familiar with the case and recognition of White Castle, then eating at the actual location only provides a slightly more fresh experience than the frozen, store-bought burgers. The buns are much softer and are squishy to the touch. The taste and aroma of the grilled onions complement the meat patty. It\'s White Castle, there isn\'t much too glorify about it or expect. \n\nMy experience was pleasant, but to each their own. For $8.99 you can get 4 Cheese Sliders, a small fry and a small drink. The price is consistent with the glamorization of the branding. I ate each slider in about 2 bites and was still hungry afterwards, but didn\'t want to face waiting in a long line again. \n\nIf you are nearby in the area, take a look!',
      'useful': 0,
      'user_id': 'D3GeQgpvMLwZCCHHdvyqaA'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2017-05-11',
      'funny': 0,
      'review_id': 'rNhtH_D8FOF79b7u5e4png',
      'stars': 4,
      'text': "More of a nostalgia thing for me growing up in Minneapolis and eating them as a kid.  Price seemed always right and always fresh and delicious.  That thing they do with the onions is over the top.  Not as good a burger as I remember as a kid but maybe I have eaten too many celebrity chef burgers and acquired a different taste.  Hard to park to get there but if you're in the neighborhood pick up a few and enjoy.  Fast and friendly service and smells great.",
      'useful': 0,
      'user_id': 'WM7MFrRP-7YFuGBAJ6quRQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 1,
      'date': '2015-01-28',
      'funny': 3,
      'review_id': '7j_gZ0Q_itkSL2gLS1SrQg',
      'stars': 2,
      'text': "Ok so I'm not from the east coast so I've tried many burgers in Vegas and I was pretty disappointed with these. \n\nFood: \nCheese sliders- 2/5 I ordered the cheese sliders. The burgers were small and pricey. I didn't like the smell of them and I didn't like how it tasted like a burger from any other franchised restaurant. Nothing stood out for me. \n\nChicken rings- 1/5 Seriously don't get these. They are crispy but flavorless and super thin. Honestly thought McDonalds chicken nuggets were better. \n\nFrench fries- 2/5 Fries were a bit soggy. But these were my favorite out of all the stuff I ordered. \n\nService is GOOD! 4/5:\nExpect a long wait. I waited in line for more than an hour but I got my food in less than 5 mins! They are friendly and efficient!! They try to make the wait bearable by talking to you, which I thought was neat. \n\nOverall if your a person who likes a burger with flavor and who has ate many flavorful burgers, White Castle isn't for you. I was quite disappointed but many other people love the simplicity of their food. This joint just wasn't for me.",
      'useful': 3,
      'user_id': 'X1Cy1xsDawDiH0o9dmDIXA'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-10-05',
      'funny': 0,
      'review_id': 'tk0HH_JjPKbTGDldkzFDtQ',
      'stars': 4,
      'text': 'The wait is long, but definitely worth it! I wish there was one in So. Cal so I can have it more often than twice a year. The service is definitely quick and accommodating.',
      'useful': 0,
      'user_id': 'su07frN5-r51qWeyQGY1kw'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2017-03-31',
      'funny': 1,
      'review_id': 'zd7Y4SNEVxs3DiNME-R-ow',
      'stars': 2,
      'text': 'I really don\'t get the whole hype about this place. I mean it\'s a bit different in the way they serve their burgers but I felt I couldn\'t really enjoy it entirely because they serve them in bite-size pattys. \n\nI was hoping it would of felt like a meal but honestly it felt like a mid-day snack before lunch. Fries are good but I don\'t think I will be coming back to try it again. Unless someone can convince me that there is a must try item on their menu. I\'ll stick to regular in n out. Service here was horrible. The one thing that did bug me was when our order was called. My friend was already heading to pick it up and the lady begins to speak on the Mic. "Number ???, you will find your number if you have  looked in your receipt on the bottom part of it."  And I mean she said it with an attitude as if we didn\'t know it was there. So unnecessary and uncalled for.',
      'useful': 0,
      'user_id': 'AZBqmjJSCuGFvx5OTbd7Kg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-05-30',
      'funny': 0,
      'review_id': 'vbmLnOY92VKHKqGXkLeHHg',
      'stars': 3,
      'text': "Got the #1 - 4 Regular burger pack w fries and drink $7.99. \nI wish I got the cheeseburger it has more flavor the regular was a bit dry. But nevertheless it was fluffy buns, grilled onions and tasty. I realized there is no real sauce in these burgers but grilled onions gave it flavor. I've tried the frozen ones it's similar but these were more fluffy buns and softer meat patty the frozen ones were drier.",
      'useful': 0,
      'user_id': 'UcXu-RZbJdjRhgFbfV-gXw'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-06-21',
      'funny': 0,
      'review_id': 'zLA_57ge8X1oUXY2rEUHzQ',
      'stars': 1,
      'text': 'This place is literally the worst place I have ever been. (Service-wise). The food is relatively standard-ware. I came and waited in line with 3 of my friends on an afternoon, and waited about 20 minutes in line to place my order. I understand that places get busy, so it was acceptable.\n\nHowever, after about 30 minutes of waiting for our orders, my friends all got their orders, but I still had not yet received mine. I inquired several times about my order, and where it was in the queue. My friends finished eating, and we kept waiting around.\n\nEach time I asked, the person handing out the food would promptly dismiss me, and even the manager bluntly and rudely told me they serve hundreds of burgers a day, and it would come out soon enough. \n\nAlso note - my order number was 374, but they kept on calling order 274 out. I asked if they perhaps had my order ready, but they told me no, they had Order 274 ready, and that it was not my order.\n\nNow, about 1.5 hours after entering the restaurant, and an hour after placing my order, I demanded that they finally look inside the bag, and the manager who had been extremely rude finally conceded that they had mislabeled my order.\n\nInstead of offering anything to help make it better, they only offered me a brownie, which means that they valued wasting about an extra hour of my time at ~2 dollars.\n\nIf they had merely checked the queue the first time I asked, no problems would have occurred, and we could have resolved all the issues. Instead, this Whitecastle decides to run as if the customer is always wrong, and will stonewall and lie to you about your order.\n\nOnce a mistake is also made, they will make a measly effort to fix their problems.\n\nI strongly recommend you go elsewhere for a burger fix in Vegas.',
      'useful': 3,
      'user_id': 'MbEV-LcSKSGcIevJJXw1zg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-02-16',
      'funny': 0,
      'review_id': 'DBrpeGYkGcDn211STRX1WQ',
      'stars': 3,
      'text': 'Im from NY and loved it there, but this was not as I remembered.  I bought a crave case, all the burgers were warm and dry like eating toast.  They all had some kind of freezer burn marks on the edges of the patty.  I remembered it being steamy and delicious, I hope they improve because I love WC!',
      'useful': 0,
      'user_id': 'EOdrV0tRwq7KBcXwVC5u0Q'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-02-14',
      'funny': 0,
      'review_id': '9Jl0EiHnHLkYqeMPRQSYGQ',
      'stars': 4,
      'text': "Listen folks....White Castle is not for everyone.  If you're curious about these little cooked to order sliders just try one or two at the most. I definitely don't recommend a full meal.  Vegas location means Vegas prices. It may not be worth it to you. Now for all the lovers & fans of this tasty little spot..... be prepared for greatness! Its always super busy here which is assurance that everything is fresh, hot, & ready to go. They ppl here a great system to get you in & out. You can even see them cooked the burgers, grill the onions, & steam the buns while placing your order....but i must warn you...be prepared to pay Vegas prices. Hope this helps!",
      'useful': 1,
      'user_id': 'XW6znlSnLivnkhePlw8jkQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-01-24',
      'funny': 0,
      'review_id': 'ZQlYABRnCFo1yyBVvlDjrw',
      'stars': 4,
      'text': "I've always wanted to try this place BC people always says it's good. Well they were right. I love the fish and chicken breast burgers.",
      'useful': 0,
      'user_id': 'zrqxV5G0bUIKvGObnMSNcA'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-02-20',
      'funny': 0,
      'review_id': 'Z9TEswpvicHTKFdCiOM4Ug',
      'stars': 1,
      'text': "white castle is a horrible  place to work! my son n law worked his first shift and was locked in a freezer multiple  times,made fun of because Edward and Francisco didn't  like his name,they refused to teach to teach him his job even though they were suppose  to train him, and he didn't  get a lunch or breaks until the two boys got off shift and an actual manager was working and gave him a lunch break his last he of work! I will most definitely  be contacting an attorney! go back to the east coast white castle we don't  need companies  like yours here if your going to treat people poorly! remember  bullying  is not tolerated!",
      'useful': 0,
      'user_id': 'INNbJHteG02dS4VqoXoqKw'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-02-01',
      'funny': 0,
      'review_id': 'L78YGuZQsdiQf5O3OywalQ',
      'stars': 4,
      'text': "First of all this opened 15 hours ago. The wait was not that long considering I ordered a crate w/cheese (100 sliders w/cheese). The manager maintained eye contact with everyone on line (less than 20 people at 530am). I seen over 400 burgers leave in the 15 minutes I was there. The place was clean and I could not smell smoke (it's inside a casino).",
      'useful': 0,
      'user_id': 'EK4B8i9FJfw539XnWGAV0g'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-06-03',
      'funny': 0,
      'review_id': 'htOqpm9U7I1E_F2_89LA-w',
      'stars': 3,
      'text': "We have eaten the White Castle freezer pack sliders and I was curious to see the most western White Castle location in Las Vegas. None in California. The sliders are almost identical except for 2 things: these ones contain a pickle and the bun is more moist, some were soggy. I got super indigestion after eating 4 of these guys. It was nice to try once in person but that's it.",
      'useful': 0,
      'user_id': 'eO1ZCFWJCZcETr9dfSydRg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-06-20',
      'funny': 0,
      'review_id': 'PKYWQwRAgqqQu7s-amFglA',
      'stars': 4,
      'text': 'Cute little burgers. Good ratio of onions and pickles. Kind of wish it had already come with ketchup or mayo inside. I guess this place is really only for those who know what to expect. If not, kind of setting yourself up for failure.',
      'useful': 0,
      'user_id': 'FS_TP9EWK5a3rX0xByXW6w'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-06-24',
      'funny': 0,
      'review_id': 'YXIwgxUdY2ZsvXQx45RVtw',
      'stars': 3,
      'text': 'We finally tried White Castle. The line was unbelievably long. The service was friendly and quick in helping. The hamburger was different then any burger I have eaten. Their very simple and tiny. The burger is super thin and small  which includes a pickle and a few chopped onions. The bun is super soft but delicious. The combination of cheap and small makes you want to come back for more. Unfortunately you will need to order quite a few in order to feel full. Sometimes simple is better. There is no glamour to these burgers. The restaurant is full of people with happy smiles. Unfortunately the location needs to be cleaned more often after people eat at the tables.',
      'useful': 1,
      'user_id': '1wz1YTNiw9FJye3FRhj1AQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-05-17',
      'funny': 0,
      'review_id': 'wR9bPZ7PfsmxKeBM7t-oZA',
      'stars': 5,
      'text': "I like this place. The workers have great customer service and their services are fast. I seen a couple complaints that this place is similar to the frozen stuff.  Um.. yeah... shouldnt it be??  If i owned a restaurant and decided to sell my food to the grocery stores, i would want it to taste like it was the same as dining in.  \nLas Vegas is a tourists city and I'm very grateful for restaurants from around the world to land here.  It's very welcoming for those that moved out here and miss what they grew up with.",
      'useful': 0,
      'user_id': 'ee9YcVVzAuRLeQnY8ctDUg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-02-04',
      'funny': 0,
      'review_id': 'PrT4llvkHw6UDw6mboL6Kw',
      'stars': 4,
      'text': 'I agree with most people that White Castle is a bit overrated.  The fries are akin to the ones you got in grade school cafeterias.  The sliders are tasty.\n\nThe service was good.  We came by at 11am on a Monday and ordered a 10 slider pack.  There was only one couple in front of us but we got our food within a minute after ordering.',
      'useful': 0,
      'user_id': 'gL3uaudejZFlVPbNeDib2w'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-02-08',
      'funny': 0,
      'review_id': 'Zkq4gXrQ8xKcnFrEf9giLg',
      'stars': 4,
      'text': "Oh my...\n\nAs a person who has at times struggled with weight I have to say that going to the Las Vegas White Castle location was the biggest boost of self confidence.\n\nThe average circumference of the people that were there to eat was over the morbidly obese range...one guy had a gut that was drooping over his shorts.  And I saw a few order the 100 burger box...I actually couldn't help myself but stare several times...I felt instant guilt and wondered how many children in the Philippines were hungry\n\nI liked the burgers but after seeing what can't be unseen...I won't be going back and I'll be heading to church to pray and to the gym to run",
      'useful': 1,
      'user_id': 'KqTnIUfwUto4Ka74JnzE8A'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 6,
      'date': '2016-11-22',
      'funny': 6,
      'review_id': 'yJwTxilbGkfjcfcwnToYog',
      'stars': 4,
      'text': "My mom use to buy these babies at the grocery store all the time but I finally got to try this place out!  I loved it!! I am sure people love it when they are drunk as well but I was sober when I ate these!  The ones at the market do not have pickles in them but these do.  The taste so good with the cheese!!  Their fries are pretty good too as well as their fried onion things.  Kinda like onion rings.  I would definitely come back!  Too bad they don't sell it in singles.\n\nNOTE:\n-Extra charge for cheese (30 cents each)",
      'useful': 6,
      'user_id': 'IxrkMwtW1emHp7cifyq40A'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2017-06-01',
      'funny': 0,
      'review_id': 'moir5eYkWaMtIVvdc9lWjA',
      'stars': 2,
      'text': 'So, I came here the last week I was living in Vegas hoping that it would be as amazing as Harold and Kumar show in their movie about how amazing White Castle is.  I literally went on a journey from Reno to Las Vegas, then finally got to White Castle with high hopes.\nYea there was a long line (its Vegas, there are long lines everywhere and long wait times in Vegas on the strip).\nWe got our food quickly, but oh my gawd it was disgusting.  I am so disappointed that the burgers were watery, like they were frozen and not cooked all the way.\nThe sliders filled me up quickly, but it was not satisfying. Definitely really expensive and not worth the price. \nDont expect to have that amazing experience there, unless you are super drunk or high and dont care what you put in your mouth. Fries were ok.',
      'useful': 1,
      'user_id': 'IG3VDDgJOlvdU_K1YZtFUg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-06-20',
      'funny': 0,
      'review_id': 'lspsuJKW4OEgErfxhzl6Ig',
      'stars': 5,
      'text': 'This place is great for a late night drink food run. The line can be long but is worth it. If you do come with a large group order the large pack.',
      'useful': 1,
      'user_id': 'Bv_3sQjWYucEY3LTnHOSkw'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 2,
      'date': '2015-06-18',
      'funny': 2,
      'review_id': 'q94LMwM0jRkV7o60ISDftA',
      'stars': 1,
      'text': 'There is nothing special about this place. Read the Yelp Reviews and trust them! Do not come here! Harold and Kumar promoted this place like no other so everyone had to come try them, I was one of those people.\n\nThe sliders are worse than the ones sold in the frozen foods section. It is small and will fit into the size of your palm. The bread was slightly warm and soggy. The patty was dry. The small pickle was lying naked alongside small speckles of onions. It was not tasty at all. I had to squeeze a full bag of ketchup on it to give it life. For a dollar something each, I would rather walk myself to a McDonalds and order something off the dollar menu.\n\nMy boyfriend and I decided to come here because it was 5am and he was starving. Shake Shack and Rasising Canes were both closed so White Castle it was. The next morning, my boyfriend and I woke up with bad tummyaches. Luckily, our hotel had two bathrooms....',
      'useful': 2,
      'user_id': 'g_4zySOI_HisK5rG_2dw6Q'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 1,
      'date': '2017-01-13',
      'funny': 1,
      'review_id': 'TwNnTZubPbYjUBWGnBa3vQ',
      'stars': 3,
      'text': "White Castles are not a common sight in Los Angeles county.  Friends say that they were once plentiful, but I understand that they have retreated to the comfort of the east coast. Save this one. This White Castle is all alone out here in Vegas. So, it was on my list of things to see and experience in Las Vegas. I'd never had one. Not even from the grocery store. I'd even seen all three of the Harold & Kumar movies about them. I was super excited. Well done White Castle!  \n    So, I parked my ride back at the LINQ self parking, and hiked down this way to get to the Treasure Island for the Pirate ship battles and the Marvel thingy. It was quite of a hike to thie point. Several times I pondered giving up and taking the Monorail. But I persevered, and this was a well deserved stopping point. The Las Vegas strip is deceptively long.  \n     As I entered there was a bit of a line at the counter when I walked in. The sounds of the nearby casino jingled and buzzed and tempted through a huge opening in one wall. I assume the White Castle is open at all hours to collect the hungry gamblers. Which is probably alot, as the smell of burgers wafted across my nose as I looked up at the great big menu to figure out what I wanted. Looking at the items, the prices ranged within your usual 'fast food prices'.\n     There was a range of options. Meat, other Meat. Fake Meat. No Meat. All the glutens, none of the glutens. Even Vegetarian and probably Vegan options. I saw a sign for a double cheeseburger combo (little teeny ones) and decided that's what I wanted. I wound my way through the line and got to a register, and ordered one from the nice lady behind the counter.\n    I was given a cup and a receipt. I took both to find a place to sit. Easier said than done. This place was pretty popular. Eventually I found one at a high common table. I set my camera bag and phone on the table corner to mark it as being taken. As I went over to get an iced tea, I heard my number called. I altered my trip back to my table, and veered back over towards the counter to pick up my tray of food. I was excited, I was eager. I was going to try White Castle burgers. I reached into my bag and withdrew the three small boxes inside. \n    Hrm. These three tiny things are supposed to fill up me? Seriously? Hrm. Ok. After the yelp photo, I carefully pulled out the miniscule burgers, doing my best to save the buns from ripping. I excitedly took a bite of one. It was okay? I was not as awed and amazed as Harold and Kumar were. I wasn't overwhelmed and taken away on a meaty dream sequence of awesomeness. I was just kinda sorta whelmed.\n    Even the iced tea was that fountain done stuff. Not brewed iced tea. Meh. I'd have rather been eating a Carl's Jr. Big Carl. But. I'd finally experienced it. Checked it off my list. That's part of the excitement of life I suppose. To get out there and try new things. Sometimes they're just not as great as the commercials.",
      'useful': 2,
      'user_id': '43WiflwfTbmjizIfxvQ_jw'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-05-08',
      'funny': 0,
      'review_id': 'iacbQ03b5IQM1WgaAS89oQ',
      'stars': 5,
      'text': 'I turned the corner and there it was--HOLY SQUARE BURGERS, ITS WHITE CASTLE!  Been living in Washington for almost 20 years now and when I saw that blue and white--ITS WHITE CASTLE!  I know when I walk through the pearly gates, right next to the NY Pizza place will be a White Castle!',
      'useful': 0,
      'user_id': '4kM0hJIETOdBa2lQOnO8tQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-04-30',
      'funny': 0,
      'review_id': 'GLaeGjCSIO0krPal3-DW-A',
      'stars': 2,
      'text': 'No. It\'s over rated because of its reputation but the food is not good. A single slider was 2.50 and that\'s as much as a burger at in-n-out. This is a "only choice" type of eat. Don\'t eat here if you have other options. Not to mention it\'s the only one in Vegas and for locals the parking and ordering and dealing with all those tourists is too ridiculous.',
      'useful': 0,
      'user_id': 'Krqu0QDgPX_hF8jt5MB_qQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-11-25',
      'funny': 0,
      'review_id': 'FtO1FykFid19U03zPxbobg',
      'stars': 2,
      'text': "I expected better from here. The sliders weren't that tasty and became a chore to eat after a while. The chicken rings tasted like frozen dino nuggets (not even as good as Dino nuggets). I also had the vanilla milkshake, which was pretty good. It tasted like an In N Out shake, but sweeter. The fries weren't that bad either, I would eat them again. Lastly, the onion chips were probably the best out of the whole meal; they were really crispy and delicious. \n\nOverall, I wouldn't come back here by choice but it wasn't a HORRIBLE experience. Their staff was friendly and service was quick, impressive since it's pretty full..being on the strip and all. All and all, i say come and try it out for yourself. It just didn't meet my expectations.",
      'useful': 0,
      'user_id': 'idtBBgts7weFgfoJLyBb3Q'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-02-01',
      'funny': 2,
      'review_id': 'D4cPi_H1zetA_T5UYnRNRQ',
      'stars': 1,
      'text': "I went there yesterday line was long, that wasn't the problem. The problem was there was no communication or organization. They were selling milkshakes and didn't even have them ready me and several other people had to wait a little too long for the rude manager. The lady screaming out the orders was very unorganized and rude. They need a better way of putting the food out when it's ready. And the worst part was when I get home I find out that one of the bags was burnt, it was like somebody left it next to a warmer and was like oops let me just put the bag where they can't see it. Really? You are gonna sell burnt burgers and bags? \nThe only positive thing was I did get my 30 burgers quickly and before other people, even though it was BURNT!! Seriously I never had so many problems in one place.",
      'useful': 6,
      'user_id': 'u7oVDq2EOOufFn8mDlDQgg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-10-14',
      'funny': 0,
      'review_id': 'be8sRmiSK7d2LO5k3BR4Xg',
      'stars': 5,
      'text': "First time trying this burger joint and it didn't disappoint us.  For a family of 3, it was me, my wife & our 3 year old, i ordered the 4 cheeseburger sliders combo and extra 2 cheeseburger slider. I thought it was enough for us, but I had to order a few more lol.  For me its best to eatitwhen it is still warm, everything just melts in ur mouth, and im craving for it right now again while writing this review.\n\nI ordered a $5 chocolate shake too and that was also good!  Their service is really fast and food quality is great, that is why for me, they deserve a 5-star rating.  At the time that we were there, there was a good amount of people.  When I paid for my food they gave me a number, i went to go get my drink, a minute or two after, they called my number right away, so i was very satisfied.\n\nIts a small dining area, so It might be a little hard for you if you come here when there is a lot of people.\n\nWhen I come back to Vegas, I would definitely come back here.",
      'useful': 0,
      'user_id': '_YaE_5VKIlE3IK_fg9sYhg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-09-11',
      'funny': 1,
      'review_id': 'AGilPZcTb1pdxgZ1abLmMg',
      'stars': 4,
      'text': "11 years! That's how long it has been since Harold and Kumar made such a fuss about these tiny burgers, and how long I have been waiting to find out if it was worth it.  Spoiler: It kind of is.\n\nThe meat is really nothing special here.  It looks preformed from frozen (I could be wrong, but just my observation) and it very thin.  What makes this joint stand out is everything else in addition to the location.  The burgers grill on a flat top coated with onions to soak in flavor.  They are simply made with a couple leftover onions and a pickle, plus cheese if you pay extra.  No frills needed, or really requested for that matter.  The other options I tried were pretty basic type fast food, so nothing special I would go back for.\n\nThis location is 24/7 right on the Vegas strip and whether it was noon after a night of drinking, or 4am during a night of drinking - this place is gold! 4 bite burgers that are pretty cheap, freshly made in front of you and aren't super heavy.  I chowed down on 5 of them easily feeling satisfied without feeling sick the next day or really even feeling regret from eating anything overly greasy or poor quality.\n\nIn a regular fast food setting it would be just that to me.  A quick bite when I don't really care about what I'm eating the way most fast food is.  On the Vegas strip however, this is probably my new favorite stop during the night or before I get some sleep!",
      'useful': 0,
      'user_id': '23cjtl3AWJtxKL3O78-Djw'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2017-06-11',
      'funny': 0,
      'review_id': 'mz0Os467GFecQktCYaq14w',
      'stars': 5,
      'text': "I've always wanted to try a White Castle and this is the only place in western US.\n\nThe burgers are more like small sliders, so order a few of them. They come with pickles and grilled onions, but they are very tasty!\n\nI also ordered 'onion chips'; Its like onion rings but in French fry form. They are really good.\n\nThe service was quick, the prices were reasonable (considering it's on the strip). I would definitely come back.",
      'useful': 0,
      'user_id': 'm4e2fs8BObN0Cd69stfo9A'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-07-10',
      'funny': 0,
      'review_id': 'GwCc7bSP2zWO0d-zFHfteQ',
      'stars': 4,
      'text': 'I\'ve been wanting to try this place WAY before Harold & Kumar made it cool. I was pretty excited to hear that one opened up in Vegas. I\'ve had friends who\'ve gone to New York & eat it & they basically told me "What you see, is what you get." \n\nTrue to word. I wasn\'t disappointed though. The line was ridiculous but it went by fast. We ordered almost everything on the menu & it was done pretty quick. The cheeseburgers were pretty good, the fries were lame, the onion chips WERE DELISH, the chicken rings were okay.\n\nI mean, its a fast food place - I don\'t know what people were really expecting.. lol.',
      'useful': 1,
      'user_id': 'G6Ni0mCoEaZ18ql5Sc2DZA'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-02-17',
      'funny': 0,
      'review_id': 'La9r_Z9rGf29pWChpP8nkQ',
      'stars': 3,
      'text': "Tiny burgers (eh hem, really they're SLIDERS). Good flavor mainly attributed to the grilled onions but these things are small. The patty is paper thin. Makes me wonder if I'm ingesting beef or any kind of real meat for that matter. I'm not a big girl, but I swear I could have killed like 20 burgers (eh hem, really they're SLIDERS) if our group didn't already have reservations for Gordon Ramsay's steakhouse. I was there for Superbowl and this location had just opened so the wait was pretty ridiculous over the weekend but we waited in line in the late afternoon so our wait wasn't as bad.  We waited about 30 minutes and thankfully not in the Vegas heat.",
      'useful': 1,
      'user_id': 'mFM9e6geCc0_nX7QqMnpYQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-08-13',
      'funny': 0,
      'review_id': 'nJWkFWJQE-gdEaAF7gAGMA',
      'stars': 1,
      'text': "OMFG.... I had low expectations for this place AND it still managed to underwhelm me. I've never had White Castle being from NorCal there isn't any close by. Ordered the #1 combo: 4 sliders, fries, drink. I couldn't even finish all the sliders... didn't want to ruin my appetite for a redemption dinner. \n\nIf you never had White Castle, don't worry you're not missing out on anything. Not even worth a try!",
      'useful': 0,
      'user_id': 'U3sCAJc4_E3MGC1jGi19gQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-09-22',
      'funny': 0,
      'review_id': 'FiPt3dIVh1Gpavb_RqW4AQ',
      'stars': 1,
      'text': "If I could give White Castle a negative star I would. I can't believe this place exists. We had a gift card and wanted to try it out. We ordered 40 mini sliders, fries and onion rings. The meat didn't even look or taste like meat, it was horrible. Everything was soggy and not eatable it felt like I was eating something foreign. The best part about the meal was the Powerade. Very upsetting.",
      'useful': 0,
      'user_id': 'LBnjYRrpTKgtGkW0ea7Ecg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-08-21',
      'funny': 0,
      'review_id': 'tPqW3CRFc8cETcqJ-X1WAg',
      'stars': 1,
      'text': "I'm so glad we don't have these in Niagara Falls. These were the greasiest blandest fast food burgers iv'e ever had and they weren't cheap for what you got.",
      'useful': 0,
      'user_id': 'V0NPlek_N9sSPeexVkri_w'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 5,
      'date': '2016-07-09',
      'funny': 4,
      'review_id': 'Zf0jX5faD8ZSkHGJK9MDCA',
      'stars': 3,
      'text': "The hype was there. I think it was overly hyped but I get it a little. \n\nWe parked at the Venetian and walked outside by where Sephora was at and its tucked in a little casino. But you will be able to see the White Castle sign. \n\nWhen we got there the lines were not as busy like we hoped it would be. It took me a long time to come just because the hype made it impossible to come. Thank god we waited though.\n\nWe ordered 20 like cheeseburgers sliders. They also has beer for you beer drinkers. They are like I expected little tiny cheeseburgers with 1 pickle Alice and some onions. For some reason it did taste good. But it taste like how the frozen one would taste at a grocery store. \n\nAlthough those sliders were tiny I did get start to get full off 3. \n\nI probably won't be returning here unless one of my tourist family members want to try it out. It wasn't a horrible experience I just feel like there's nothing special about White Castle. Maybe it's better on the East Coast!",
      'useful': 5,
      'user_id': 'pxzs-Dy2hXTis-PuNCV37Q'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2017-02-20',
      'funny': 0,
      'review_id': 'zsJzPkqlJfc3ynlyNu8z5Q',
      'stars': 5,
      'text': "We drive up from the Phoenix area for White Castle. Husband is a WC freak! I grew up in No Cal, so I only knew from the movies. ;) He says they are far better than the microwave / grocery store heat and eat. We did a drive across the good ol' USA last summer (from coast to coast) and I honestly can't tell the difference between Las Vegas and St Louis. Which is a good thing. They taste the same here as they did on the East Coast.",
      'useful': 0,
      'user_id': 'Ov-0pRQxz5fTTElFE-WtTQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 4,
      'date': '2015-01-29',
      'funny': 6,
      'review_id': '5xAxkEmNVGmv2TmfkiZFvA',
      'stars': 3,
      'text': 'For starters...for all the people giving it low stars because of the wait they had to endure yesterday (this review is being written the day after the Grand Opening), you\'re all morons.  Seriously.  What in the world did you expect?  \n\nAs far as taste goes, White Castle is not for some.  I can\'t hate on people not liking it if it\'s their first time.  I\'m also well aware there are FAR more better burgers available, but sometimes you just want a White Castle that is cooked unlike any other burger.  \n\nThe burger tasted like I expected, and the cashier, who I\'m sure wanted to rip her own head off, was a sweetheart.  Being the lines were so crazy, I asked her if anybody in her team had walked out yet, and she said "we\'re soldiers!".  I loved her attitude, and I wish I had remembered her name.  My gripe, and why it\'s 3 stars, is with their pricing.  I know it\'s a strip joint, but there are other chains that charge their regular prices on the strip.  A few steps down is a Chipotle and a McDonald\'s that charges the same as any other one of their locations.  I don\'t care that rents are higher on the strip and it\'s their only location.  Whenever they come into town...locals CRAM THEIR FOOD TRUCKS for their burgers.  They could have just as easily chosen an off-strip location and made a killing off of us.  I split a Crave Pack that came with 10 hamburgers (we paid extra for cheese), 20 chicken rings, and 2 regular fries.  Large sodas ($2.99) were separate.  The bill came out to $38.88.  I\'m sorry, but $19.50 for 5 cheeseburgers, 10 chicken rings, fries, and a soda is lame.  For $20, I can come up with 50 other places I rather eat at.  \n\nAlso, they offer a combo that comes with 3 cheeseburgers and regular fries for 8.77...WITHOUT A SODA.  That\'s just wrong considering fountain soda usually has the highest markup of any of their items.  They could\'ve easily included the soda.  \n\nI won\'t be going back, but I\'m hoping that they\'ll open up some off-strip locations and possibly offer the pricing that other White Castles are used to offering...because at the end of the day...this is White Castle, not Five Guys (or any other burger joint of your choice).',
      'useful': 21,
      'user_id': 'FH3GgmOeAgXST37jG1-uWg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-06-22',
      'funny': 0,
      'review_id': 'SgPGp92-OERIN6NljU6Cag',
      'stars': 5,
      'text': 'College at Ohio State and these things were a regular food item for us. I haven\'t had a "fresh" one in 15 years! These were the highlight of my trip to Vegas. Are they kinda crappy? Yes, but if you\'re from Ohio you get it. You know why we love them. The atmosphere isn\'t so great, homeless walking up to you asking for money is a bit of a downer when you\'re relishing your first bite in over a century but it\'s Vegas, so whatever.',
      'useful': 0,
      'user_id': 'mdmdnlHLlZUvAE3-J5omCg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-07-14',
      'funny': 1,
      'review_id': '9F3toH7hsnof2rlxWiTnUw',
      'stars': 1,
      'text': "This place shouldn't receive any stars. My husband and I were looking forward to their great food. When we finally had a chance to go, the service was horrible and the food was worse than hospital food. The burgers meat were tasteless and the fries were soggy. Their soda machine had post-its on every other drink being out of order. Plus, we both ended up getting a stomach ache and the runs that night.",
      'useful': 0,
      'user_id': 'DaOeYwxwtwDtiKKstYj3lQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 2,
      'date': '2015-08-14',
      'funny': 1,
      'review_id': '3N37dagJx9bF9IyfzGDj-Q',
      'stars': 1,
      'text': "Exactly like the frozen, eating them\nAt 5am in Vegas didn't even improve the taste. Have know idea what hArold or kumar were all hyped about. Shitz wack!!!!",
      'useful': 0,
      'user_id': '6kT39C2bXusgDjODDQiJ_Q'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-07-20',
      'funny': 0,
      'review_id': 'npKblLYAg3RtkjVmyxXvnw',
      'stars': 5,
      'text': 'I never heard of White Castle till I watched Harold and Kumar. It never dawned to me till I seen it while in Vegas. The line gets really long (which is normal on the Strip). They are small portions so get a few.  It is kinda pricey but I thought it was really good. I would choose White Castle over Innout any day now..',
      'useful': 0,
      'user_id': '0LgC7YzAQXBdbQZl0wgUnQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2017-06-02',
      'funny': 0,
      'review_id': '1Hr9A4ckCvg3ufpMTKBcqw',
      'stars': 1,
      'text': "Soooo disappointed! Harold and Kumar don't know what a good burger tastes like... sliders tasted like it was microwaved I threw it away I couldn't take another bite. Just thinking about those nasty onions makes my stomach turn ... BLAH!",
      'useful': 0,
      'user_id': 'hFFB48ybb4uSmh_CCeTljg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-08-21',
      'funny': 0,
      'review_id': 'r_Fc95bp1Dj57wARz6heaw',
      'stars': 3,
      'text': "It's White Castle and since there none in California and you need your White Castle craving you might as well stand in line and eat your steamed sliders.   It's located at a smaller casino called Casino Royale which is sandwiched between the Venetian and Harrahs.   Some people love these sliders, I'm not a big fan but my boyfriend loves them even bought a 10 pack to bring back to home.   They sell packs of 10, 30 and even 100.",
      'useful': 2,
      'user_id': 'WoE8rLUh8Pu76lWw8soohQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 1,
      'date': '2016-12-02',
      'funny': 1,
      'review_id': 'aJZVFuOwomHjlCW6HbdWCQ',
      'stars': 3,
      'text': "A east coast comfort food that reminds you how sometimes, you need some cheap food to fulfill your needs. It just so happens Vegas isn't a place to fulfill your needs. If your drunk, gambled all your money, desperate and hungry. This is the spot. Other than that if your in the west coast and desperate for some white castle fast food drop on by like I did.",
      'useful': 1,
      'user_id': 'rkBWP3B-OF-M3BXlUhdc5g'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2017-03-02',
      'funny': 0,
      'review_id': 'UzFBld958aVBwInqd4HLTg',
      'stars': 3,
      'text': "The only times I've ever been here is when I / my friends have had a lot to drink.  So from that aspect, its very good drunk food.  The staff is for the most part very patient which I appreciate b/c I know they deal with jerks all the time.\n\nMy major complain is the cleanliness of the sitting area which is very small, it can get pretty gross in there and had it been cleaner I would have given it an additional star.",
      'useful': 0,
      'user_id': 'J2ZTcQhg6_RJzOIiF_F3-A'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-08-28',
      'funny': 0,
      'review_id': 'DTFAOD1e2Gbj-Bm6O8wIwg',
      'stars': 1,
      'text': "Shit, for the price we paid I assumed that when I ate it, i'd taste the damn meat instead it felt like I was eating only the bun. Their meat is as flat and thin as the cheese..... I enjoyed more seeing the options of drinks than I did eating the meal... Fries were not salted and tables were dirty and filled with a lot of mess from previous people.",
      'useful': 0,
      'user_id': 'QRtupETQOddXEUo9DYfaOw'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-07-11',
      'funny': 0,
      'review_id': 'cfgtXN2T-ULSaFgt72H4bQ',
      'stars': 2,
      'text': "Right next to Harrah's along the strip.\n\nSure, you don't have to go to the East Coast for this. On the other hand, you're better off getting something else.\n\nThe sliders were alright. Ended up getting the 20-pack with two fries. Add cheese to them and it's a little extra. Also ordered some mozzarella sticks. Total came to just under $40. If you buy the slider packs, it would be cheaper than buying 2 or 3 meals that come with maybe 3 sliders and a drink and fries. Up to you what you wanna do.\n\nIt's good to try. But I'm probably going to skip this and find another 24-hour place for something quick.",
      'useful': 0,
      'user_id': '5elDILcQojTsX7GtU_N1Og'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 1,
      'date': '2015-02-05',
      'funny': 0,
      'review_id': 'J0navf62QHjt9IQybSMdHg',
      'stars': 5,
      'text': "White Castle burgers give you the munchies without smoking pot. I grew up with White Castle burgers and I love them. They are unique. They do not taste like any other burgers you may have eaten. They are 'steam' grilled and this gives them a unique flavor and texture. \n\nIf you go to White Castle with any kind of preconceived notions of what a burger is supposed to taste like, you will be disappointed. This is where the negative reviews come from; people who are going to White Castle for the first time who are expecting something different. All I can tell these people is that, unless you've grown up with White Castle burgers, they are an acquired taste. They are the caviar of fast food burgers.\n\nYou may not like White Castle burgers the first time you try them. But, after you eat them few times, you'll wake up at 3 a.m. some night and find yourself craving them. Those of us who grew up with them understand this. As I said, White Castle burgers give you the munchies without smoking pot.",
      'useful': 0,
      'user_id': 'V57ZLKz0hKIz3H4gb0sCGg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-12-15',
      'funny': 0,
      'review_id': 'YnAumXxywnI5q6GJnANEkg',
      'stars': 5,
      'text': "Oh I love you White Castle! The most amazing place. Only in Vegas can you get a 6 pack of burgers and beer! Sweet combo! I'll be back very soon!",
      'useful': 0,
      'user_id': 'vqoKrj-g8gcgpNMWZTZ0pg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 2,
      'date': '2017-02-26',
      'funny': 2,
      'review_id': 'W8ewdzpjHAxtnhmu8E2EVg',
      'stars': 4,
      'text': "So you're drunk on the Strip and you want that delicious but poor food choice. Yep White Castle fits the bill. The sliders are little bites of heaven which do a great job coating your alcohol laden belly. Being originally from Detroit there was a White Castle on every corner so it wasn't an urgent run to try them when they arrived on the Strip. Being drunk and at Harrah's seeing the White Castle was like a beacon calling us over. The place was packed as we weren't the only ones with the same brilliant idea. The cashiers do a pretty efficient job filling all the orders as quick as possible so expect to wait 15 minutes on a busy weekend night. 4 cheeseburgers, fries, and a pop (dammit I'm from Michigan and it's POP not soda) will set you back $8.99 but the extra space it soaks up for more alcohol consumption is priceless. Thank you White Castle!",
      'useful': 2,
      'user_id': 'ziiMYejVJAjbkRjSeAzrbQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-02-08',
      'funny': 0,
      'review_id': 'I2NzE9YXkK7BRa_7bgtq1Q',
      'stars': 3,
      'text': "So I finally came to see what the hype was. I have had these before on the East Coast and knew what to expect. It was nothing to get excited about but these are pretty good sliders. The staff is amazing about getting everyone through the line and the food comes out quick. So mad props to the employees and their good service. I give it three stars because it just isn't a big deal. Be sure to order double what you think you can eat. Otherwise you'll be hungry. I enjoy it but not a frequent stop for me.",
      'useful': 1,
      'user_id': 'DuKiq3MPeLXESTZXRhJPHw'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 4,
      'date': '2016-04-21',
      'funny': 5,
      'review_id': 'jV-geojy75UXZoMJMlbjzQ',
      'stars': 4,
      'text': 'Well, well, well-- another burger joint in Vegas. And guess what? It\'s affordable, especially after an intense session at the slots or tables! Or if you had \'\'too much fun\'\' at the clubs.\n\nLocation:\nThis burger hot spot is located near the Linq, across from Mirage-- it\'s literally next to "Margaritaville\'\' or whatever. But definitely easy to spot (look for the castle structure sign).\n\nInterior:\nThis is a typical fast food joint. Need I say more?\n\nFood:\nThe burgers were pretty small. Being a Hefty Hannah that I am (inside and out), 5 burgers & fries would probably fill me up. I kid you not, I could probably fit 2-3 burgers in the palm of my hand. Compared to Shake Shack & In n Out -- this place is.... different. If you\'re into small & soft burgers-- this is the place for you. Also, I really like how the cheese is always melted in the burger (that\'s total perfection). However, the squiggly fries were "eh." Same with the chicken rings-- I could pass on both and stick with the CHEESEBURGERS. Oh, and be sure to ask for the sauces with your burgers-- it definitely makes for an exciting twist. Between the other burger spots, I would choose this place because it\'s one of those, hashtag cheap eats -- and bragging rights of "OMG, I ate 10 burgers!"\n\nService: \nIt was alright. Nothing to complain about.\n\nTLDR:\nThis place is great for cheap burgers-- comparable to McDonalds, but better. in flavor and MELTED CHEESE! Burgers are small, so order A LOT (5-6 per person).\n\n\nTurn Ons:\n- Cheap\n- Melted cheese\n- Super small KAWAIIIII burgers\n- Sauces\n- Opens 24/7 (I think...).\n\nTurn Offs:\n- Fries & Chicken rings (ew)',
      'useful': 4,
      'user_id': '1cb8lvA4fluHn2j4Y5k45g'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-10-18',
      'funny': 0,
      'review_id': 'aT_P9fCo9L-jmVSs9J4kHw',
      'stars': 3,
      'text': "I was very skeptical to try this spot because of all the bad comments I got from my friends before coming here. But I liked the ones that you buy from the freezer so that was enough for me to give it a shot anyway. It wasn't as bad as I thought it would be, but it wasn't great either. Surprisingly there's always a line specially past midnight. I'll have to be drunk to eat here.",
      'useful': 0,
      'user_id': 'tlTuLJOkeWga6vV3H6wnIg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-03-19',
      'funny': 0,
      'review_id': 'kHlEtBlwSvoxNea0EMQX6w',
      'stars': 2,
      'text': "White Castle is Whites Castle. Always sounds like a good idea when you have had a few. And it's good at first. Then come regret and gastric distress. But that's what you do. But when it's super busy as this location, and it takes forever and it filled with people even drunk than you. Probably way drunken than you it's just not worth it.",
      'useful': 0,
      'user_id': 'aT_uTTaEJtWoJcqnSspUNg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-04-02',
      'funny': 0,
      'review_id': 'n-s5ldtAJp-8Stj5qjnJxw',
      'stars': 5,
      'text': "I'm a native new yorker and this couldn't be better I gave missed this, onion rings look different but it all remains me of home . thank you",
      'useful': 0,
      'user_id': 'H-hgpOaBKD6JNfDCfqfBRw'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-06-29',
      'funny': 0,
      'review_id': 'JztXi1Oc-PDTO6A1U4_x-g',
      'stars': 3,
      'text': 'Went here on a busy Saturday night and there was ONE PERSON taking orders and over ten people on the assembly line - extremely poor management of man power.  There were workers goofing off or just blankly staring off into space.  I felt bad for the guy taking orders.  \n\nFood wise: sliders were OK, nothing special and definitely over hyped by the movie Harold and Kumar.',
      'useful': 0,
      'user_id': 'VCsfdBcaXAfmH58i8-CgPA'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2017-06-03',
      'funny': 0,
      'review_id': 'Y2ww9W3vjepRjihHqnOkxA',
      'stars': 4,
      'text': 'After years of eating the frozen burgers from the grocery store, I finally got to try them in person in a real location. I\'ll be honest, these are not the best burgers I\'ve ever had - nor are they best sliders either. But hey, they\'re tasty and they get the job done. They\'re pricey but that\'s common in Vegas, but I must say the sliders with cheese are delicious. Their chicken rings, yes "RINGS" are delicious as well. So if you find yourself nearby while walking the strip and need either a small snack or full meal, try it. \nIt\'s not the best burger in town, but it\'s a tasty for what it is.',
      'useful': 0,
      'user_id': 'xsTc0-jryDgAE_HuOUA2Ng'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-05-08',
      'funny': 0,
      'review_id': 'sPdArgGQSCEhHGZ3OPcYew',
      'stars': 3,
      'text': 'The Las Vegas restaurant is in a tight location. It\'s a small place in which the restrooms are located next door in the casino. Although they seem to have a sufficient number of tables. \nOrders are placed at the counter and your number is called when the order is ready for pick up. At the time of my visit the order line moved slowly because only 1 of 3 registers were open. The order was filled quickly even before I reached my table. \n\nBeing from Chicago what can you say about White Castle other than "It\'s an Institution". But in Las Vegas it becomes another choice in the growing Burger business. Glad they are here because fresh is better than frozen and I won\'t have to satisfy the crave when visiting Chicago. Buy them by the sack.\n \nOne difference here is that they do not have onion rings. They have onion chips. These are basically rings that have been cut into pieces. Just as good as the rings. Just place several on the slider.',
      'useful': 2,
      'user_id': 'ywEB0MB_2U11NIGPs35pew'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-03-02',
      'funny': 0,
      'review_id': 'NIROBiHn7KT3frfdJJyoxg',
      'stars': 4,
      'text': 'Ok if you are DRUNK, HUNGOVER, OR GONNA BE.  These rock.  If not they are kinda grease pits that sit in your gut.  My friends had issues with the onions I did not.',
      'useful': 0,
      'user_id': 'TPKUUkAOlklihNGSOLSYGw'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 7,
      'date': '2015-01-26',
      'funny': 5,
      'review_id': 'M44jYdh9LEuBasiWUafcUw',
      'stars': 5,
      'text': 'My bf is from the east coast and was beyond excited when news broke of a White Castle coming to Vegas. Being from California my heart belongs to in n out. That being said, their sliders are pretty good for the price. The sliders are on a soft pillowy bun with grilled onions and pickles with the option of cheese.\n\nWe received a text 3 days before the official opening date stating that White Castle was giving away free food. Within minutes we were out the door and made the 20 minute drive to the strip.\n\nWhen we got there, there was caution tape preventing us to enter the restaurant from the casino, but we were able to see the entire restaurant. We saw kitchen staff making sliders and contractors making finishing touches. We figured that they would give away all the sliders they were practicing to make for the grand opening. After being told that they were only giving them to casino employees we decided to hang out for 45 minutes hoping for the chance they would have extra for us. We even offer to pay for them.\n\nThe patience paid off and we went home with 14 sliders and fries for free!\n\nThe customer service was amazing. Curious casino patrons would peek into the White Castle and sure enough an employee would come over to answer questions and encourage them to come back in a few days when they officially open. \n\nThey open Tue, Jan. 27th at 2pm!',
      'useful': 8,
      'user_id': '5950WejSl2xLIa1bY1Du3Q'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2017-05-31',
      'funny': 1,
      'review_id': 'm-o_gjEyaRERXCK_krSbiA',
      'stars': 1,
      'text': 'We came Sunday 5/28/17 at 4:30 pm on our way home. The fries were the best part of the meal but let me tell you, nothing special about the burgers. We have had White Castle in New York and my husband was excited to try it here in Vegas. \n\nWe BOTH left (my husband and I) with FOOD POISONING and we BOTH ended up in urgent care on Memorial Day and still too sick to return to work today! Right now is my second attempt to eat since eating there Sunday. \n\nI will NEVER eat there again! I have been burping up those nasty onions for two days! Ughhh! \n\nI will stick to the regular White Castle on the east coast. Lesson learned.....',
      'useful': 1,
      'user_id': 'OcV4T78c_v0E9wgNH2whjg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 2,
      'date': '2016-02-22',
      'funny': 1,
      'review_id': 'T-p5xYc1dFdq_mKlHsoZsg',
      'stars': 4,
      'text': "This was my first White Castle experience. \nI had tried the frozen ones before and didn't care for them and was assured by many that I had to try the real thing.  \n\nI was excited that White Castle had opened in Vegas since it was more likely that I would make it to Vegas than the east coast.  Final the Vegas trip was booked and I was excited to try out White Castle.  \n\nFlash forward to Sunday morning.  The breakfast debate begins and me and my home skillet decide on White Castle.  Since its Vegas it's open 24/7 so why not!  White Castle is located on the strip in the Casino Royale.  The store front is located on the strip and easy to find.  \n\nAs you walk in it feels like any fast food restaurant.  Ordering was quick, easy and friendly.  We ordered the 2 person combo with 10 sliders (5 with cheese, 5 original), a fry, an onion chip and two soda's for a little over $20.  Our order was up pretty quick.  They do require you show your receipt when you pick up your order.  I'm sure this is a safety measure as I can imagine as day turns to night and the party animals are on the loose there has been previous burger burglaries.  \n\nThe moment has arrived the tiny little boxes of meat onions and bread is before me!  I take my first bite and.....it's alright.  I figure it's a east coast thing.  I'll stick to my my west coast In n' Out.  The restaurant itself was good, I can't blame them for my taste buds being born and raised on the west coast. \n\nMoral of the story.....try it, you might like it, even if I didn't!",
      'useful': 1,
      'user_id': 'LU43O0dJV6vbyp2cll2JPg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-09-08',
      'funny': 0,
      'review_id': 'JzEytxYVxPQQ8hllzHnGSA',
      'stars': 2,
      'text': 'Nahhh dude, you must be outta your mind because this was not worth the west coast adventure...',
      'useful': 0,
      'user_id': 'ZvMODEburzB32dJTm7T9Ow'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-11-05',
      'funny': 0,
      'review_id': '9xLPhqutFRsOG8M4mpxBUg',
      'stars': 4,
      'text': "They are what they are - greasy little piles of goodness.  The pickle and cardboard box adds the missing pieces that the frozen ones can not provide.  If you grew up with them then you know what I'm talking about.  For first timers they probably won't be to your liking and that's fine too.   Cut down on the fat by removing the bottom bun.  It soaks up the majority of the fat.",
      'useful': 0,
      'user_id': 'i5wO6E011YdvZvqgYb9M3A'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2017-04-25',
      'funny': 0,
      'review_id': '4IMVoNFHmCx9pyO4cWn7jg',
      'stars': 1,
      'text': 'Terrible food. Chicken sandwich was just a piece of fried chicken between two buns. Absolutely nothing else on it.\nDouble burger was better but still unsatisfying. Garbage',
      'useful': 0,
      'user_id': '6u49z3pjNVKs1O4rBmKXvA'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-06-06',
      'funny': 0,
      'review_id': 'lE3nnM8qCX4fglUQT0lgVw',
      'stars': 2,
      'text': "Go to the Mcdonald's just a few minutes away if you want a cheeseburger.  The sliders were pretty gross but the mozzarella sticks and fries were great. This place is overhyped for the taste and quality.",
      'useful': 0,
      'user_id': 'kRaJ_RuZqje3j5wOJe7R4w'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-02-17',
      'funny': 0,
      'review_id': '7jeXytiPbcTTMOo3VBU_Yg',
      'stars': 1,
      'text': 'Many years ago, I visited a White Castle back east (once) so I knew what to expect, but we decided to give this place a try just because it\'s new to Vegas.\n\nBefore I get to the food, I have to mention that the two cashiers were both visibly annoyed at the customers in front of us, and they treated the couple two spots ahead of us in a completely unacceptable manner, simply because they were new to the menu and unsure how to order, making an annoyed face and screaming "NEXT" to get to the customer behind them. I was sure to place my order as quickly as possible (we ordered the #2 cheeseburger, fries, and drink), so at least the cashier appeared to merely tolerate me as I paid. This place has only been open a few weeks and the employees at the front are already burnt out. What\'s up with that? With so much unemployment in Vegas, couldn\'t they have found a few cashiers who are a bit friendlier?\n\nAs for the food - well, it\'s White Castle. It is what it is. Tiny, thin burger in a plain bun, with a bit of cheese and onion. There\'s a burger in there, but can anyone really taste it? I estimate the ratio of bun to burger is about 8:1, so you\'ll have to eat a huge amount of bun before you get much burger in your tummy! The fries were just bleh.\n\nAbout the location - I\'m not sure where to park to get into this place. There is no parking in front, so we ended up parking in the Venetian and walking back to the Strip, but surely there\'s a spot closer than that? Also, there were TWO security guards working this location, which only amplified the negative vibes in this place. And our table was damaged enough that it made eating uncomfortable - such a shame at such a new location.\n\nNot recommended.',
      'useful': 0,
      'user_id': 'tJTv3DUwksO8FKYi48wmqQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-02-06',
      'funny': 2,
      'review_id': 'HhZwGpr3hpJYDyyRmeUgYg',
      'stars': 4,
      'text': 'I first had the honor of noshing on these edible miracles, while attending the U or MN (Twin Cities), beginning in 1969 (Sliders were 12¢/each, then).  I had a "dry spell" (living W of the Mississippi River) until I visited NYC in 1976.  Since then, none.\n\nNow, it\'s like a prayer has been answered in my favor, and I can once again re-live my college days, even though I am now 65 and can\'t down the 20 I used to put away at a single visit in the 1970s.\n\nI have a friend in Meadview AZ who\'s more "addicted" to them than I am, so if he has HIS way, I\'m sure he\'ll be traveling the 218 round-trip miles every weekend, and crying when that\'s not possible.\n\nWhile I\'m glad they have finally opened an outlet in the Las Vegas Valley, I only have one question regarding that:\n\nWHY didn\'t you open one here, in the 1970s?',
      'useful': 0,
      'user_id': 'EMqsN7DTFnR_VywY1SEc5w'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2017-03-24',
      'funny': 0,
      'review_id': 'pR8Gq04HYJdENFetmqB4wA',
      'stars': 3,
      'text': 'Like everyone else, I was super excited to try this place after watching the movie "Harold and Kumar go to White Castle". Sad to say that I didn\'t enjoy the food very much. Ordered each item from the menu and the only good thing was the little chicken sandwich. Services was good and fast though.',
      'useful': 0,
      'user_id': 'gTdmn2YZE8hEZek2cI5K2w'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-10-10',
      'funny': 0,
      'review_id': '6ZjELVX-kHkEzTVqV503tQ',
      'stars': 2,
      'text': 'Nothing to go out of your way for. The sliders honestly sucked. Cool that it was open late but not worth the wait. Onion fries had really no onion',
      'useful': 0,
      'user_id': 'GWHLnhRBlGBUmpNG-WiOOg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-02-09',
      'funny': 2,
      'review_id': '2vDaUAYOEgn2mnXJEO7_kg',
      'stars': 1,
      'text': "This is not much of a burger. The meat, if you can find it is so small, you can't taste it. Although I pulled it out and tried just the meat. It was flavorless. The buns are wet. Not moist...WET. Even though you watch them cook the burger, it presents like a frozen, microwaved something you by in the frozen section at the supermarket.\n\nFries? Yes, they have fries. Again it's like they microwave them. That's just how it is presented. The fries, crinkle cut, are barely cooked, soggy, and served Luke warm.\n\nI know this place is popular on the east coast, but from what I can tell, I am not sure why.",
      'useful': 0,
      'user_id': 'ZtSzW0n2gSqWRAmQ_stkXQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-06-16',
      'funny': 0,
      'review_id': '4Mw-gScvCjdf9VoRXTV4fw',
      'stars': 2,
      'text': 'I only expected better because we have these in St. Louis.  It took them forever to get our order prepared and then it was wrong...so disappointing.  once they got it right it was good.  However, the wait was way too long and I wanted it right the first time.',
      'useful': 0,
      'user_id': '_d74r_wGLwE4E8MxRKk-FQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-09-26',
      'funny': 0,
      'review_id': 'JJvuprjoGcUTvZIBZgaAiw',
      'stars': 5,
      'text': "I'm not sure why the bad reviews. I mean it's not the best burger I ever had,but it is pretty tasty. Doesn't taste anything like the microwaved version of the chain. For what it is it does the job. It's a little pricey but I would eat it again next time I'm in town.",
      'useful': 0,
      'user_id': '3EmymNRS1lgvIIt__PxhZw'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-10-09',
      'funny': 1,
      'review_id': 'qStu3fXgajhkfRhNXX83yw',
      'stars': 1,
      'text': 'I was in live for an hour last night when finally got to cashier to make my order I got 2 regular no onion and no cheese and  10 jalapeño no cheese- none where jalapeño and all cheese - all burgers ended up in the trash - 1 AM you know I was hungry ! Lost my appetite when saw wrong order -I try to call the vegas number and no way to be transfer to manager no one picks up. Left 5 messages no one bother to call back !!\n\nNo excuse to take bad order !!\n\nHire none experienced young kids to work !\n\nBad bad and bad !',
      'useful': 0,
      'user_id': 'okwnuHs_LG62gq-QC0rpMw'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 3,
      'date': '2016-12-26',
      'funny': 4,
      'review_id': 'uxf9EId3tZFRtA7VMoBe_w',
      'stars': 5,
      'text': "I finally made it to White Castle! Located on the busy strip of Las Vegas, the only White Castle on the West Coast(ish). This small shop is located next to the Harrah's on the east side of a small casino. As you may already know, parking on the strip (if you are coming from off the strip) is a pain in the butt, but if you are coming just to have White Castle I would choose a hotel/casino near the Harrah's.\n\nThe restaurant is pretty small and narrow. Once you walk in there are a few tables including one really long table with seats on both sides of the table. The counter is in the back of the restaurant. The menu is pretty basic; there are a few combos that include the sliders, fries and a drink, sliders that are being sold by themselves and also options to order large amounts of sliders! \n\nWe ordered a regular combo, but added a few more sliders to our order. They came out quick and hot... and they tasted just like I imagined a fresh White Castle Slider would taste like... my childhood nostalgia. I easily packed away like 7 sliders without blinking, but boy do you pay for it later.\n\nI have friends that have visited the Midwest and told me that White Castle was delicious, but gives you gnarly gas... and boy does it! You have been warned.",
      'useful': 3,
      'user_id': 'knoQSph_vnj9SiugRB1BpA'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2017-01-12',
      'funny': 0,
      'review_id': 'T0vDAcf6bDpmvGI-XvwIsw',
      'stars': 3,
      'text': "I grew up on the west coast, so White Castle is a relatively new phenomenon for me. I've been here at least 3 times, and I have to say that it tastes the best after a long night out.  This is located in the Casino Royale casino, in between Venetian and Harrah's.  This past time we went, we tried 1 of almost every slider available. Here's my personal ranking from best to worst:\nDouble cheese slider \nOriginal slider w/ egg and cheese \nCheese slider \nFish slider and veggie slider\nChicken ring slider and chicken breast slider\nI hope that makes sense. I honestly couldn't tell the difference between the two chicken sliders. The fries are a must here, especially when they're freshly fried. Also, don't forget to ask for all your sauces at the counter when you pick up your food! The food usually comes out pretty quickly, so don't be discouraged by long lines!",
      'useful': 1,
      'user_id': 'dY6GK465j-kLlrXalgdsNA'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 7,
      'date': '2015-08-05',
      'funny': 10,
      'review_id': '93wATH9XsUWe3hzc0uY-gw',
      'stars': 3,
      'text': 'OK, let\'s calm down here, people. It\'s just White Castle. Is this really something to get excited about?\n\nIf you happen to be in the area and you want to try it for the first time, be my guest. I would skip the fries if I were you. The onion chips are a much better bet.\n\nSorry, vegetarians, but there are no veggie sliders to be had here. Rise up and be heard!\n(Tip: they\'re made by Dr. Praeger\'s and they\'re available in fine grocery stores everywhere.)\n\nNo matter what you eat or don\'t eat, you probably have dozens of better options that you could walk to from here.\n\n"The Crave Is A Powerful Thing"? Well, then I guess it doesn\'t matter what I tell you.\n\nBon appétit.',
      'useful': 7,
      'user_id': 'rvkVh-FNvg2fYSgHjG4mkQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-01-07',
      'funny': 1,
      'review_id': '9_M4HICgAvNyZVqd2ltTug',
      'stars': 2,
      'text': "I seriously don't understand the hype.\n\nI went on an adventure to just to get here, and I was fairly disappointed. The line wrapped around and almost reached the dining tables. The wait just to order took about 30 minutes, and the wait for the food was about an additional 20.\n\nI ordered a bunch of cheese sliders and original sliders to bring back to the hotel for my group of friends. Needless to say, my first White Castle slider will probably be my last White Castle slider. There was nothing special about it, and I definitely was not impressed. The patty was incredibly thin and I don't remember it tasting much like anything.\n\nI will consider trying the one on the east coast, but I will not be returning to this location.",
      'useful': 0,
      'user_id': 'U2f_VCU5GrfOZlZHsJjetg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-05-08',
      'funny': 0,
      'review_id': 'AiV8qr-bpLsqKwFnsGgkRw',
      'stars': 5,
      'text': "Am I gonna wait 45min for 4 mini burgers and fries after the club..? Hell ya! So I'm not from the Midwest and I didn't grow up on these, but I gotta say, these small hamburgers are damn good. Every time I come into Vegas I always get them after I've been out. The line can get long so be prepared to wait. These little patties are like basically sliders with thin meat and chopped up onions, catchup and mustard and a pickle. They are pan fried so they do retain a little more fat and grease which is why they taste so good. I do love their fries cuz they are extra crispy.",
      'useful': 0,
      'user_id': 'SvwD0zPT9So6cqgEsAhmWg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 1,
      'date': '2015-04-29',
      'funny': 1,
      'review_id': 'b-gDuSChBO5chsDbV9cpVw',
      'stars': 1,
      'text': "If not for the movie, I would have never tried this. I regret it. White castle is straight disgusting. Its expensive for what it is too. Don't waste your time... McDonald's is head and shoulders better. Hell, so is Burger King and that's saying a lot!",
      'useful': 1,
      'user_id': 'Ktd04SKOw89dE2DPOCj2wQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-04-24',
      'funny': 0,
      'review_id': 'WY-eLmQKz0_ACYnXhnEwDg',
      'stars': 5,
      'text': 'Ahhhh a familiar taste! This is the only location in Vegas! I am glad they have one from where I from (St Louis) they have them everywhere! I have read some of the reviews and I must laugh at some, this place is not a five star burger place, they are called "Belly bombers" for a reason! They a quick snack food and great late late at night or early am, or beating a rough night before. I LOVE white castles and will continue to go back they are great and I am glad to have them in town!!!',
      'useful': 1,
      'user_id': 'KCD6daAp293FoOUoTjT_YA'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-07-10',
      'funny': 0,
      'review_id': 'xJdw-F7mm7RgMX6Y6H8fAQ',
      'stars': 2,
      'text': "Me personally I didn't like it all it taste weird to me and it smelled like a dead animal n my opinion I will not return to any because it's just not for my taste buds the staff was friendly and very knowledgable about the menu and suggestive selling... The lines can be long but that's expected need more setting...it was pretty clean \n\nOverall I rated it 2 stars because of staff....",
      'useful': 0,
      'user_id': '5ILqkAedM_FjHp0NiDZg-g'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 1,
      'date': '2016-02-05',
      'funny': 1,
      'review_id': 'nt3j8SAGcZLi2zTdr7BbFQ',
      'stars': 5,
      'text': "I had only ever heard legends of White Castle burgers whether it was through the Harold and Kumar movies or the frozen versions at Albertsons. I was pretty excited to learn that White Castle was fairly new to Vegas so the hype was intense. The line is crazy and the menu is like nothing i've seen before. You basically order for the group or small orders for 1 person. Obviously we went for the 20 burger combo with fries and the chicken rings. Seeing how the burgers are made is mesmerizing. The kitchen is pretty open compared to other fast food restaurants where large machines cover the process. White Castle however lays everything out for you to admire and salivate while you order. They cover the grill with the chopped onions and then place the patties on top. It is definitely a sight to see. 20 minutes later, we got out food which was well worth the wait. Those little sliders are so delicious and fresh. You almost feel bad as you count how many you eat from the 20 burger combo. The onions are probably my favorite part. The fries are really nothing special and i would not order them again. The chicken rings are pretty good but are pretty basic tasting compared to Burger King's. I ate 6 sliders by the way...",
      'useful': 1,
      'user_id': '1PQYGqxBCMpAIkmFJi9orw'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-09-08',
      'funny': 0,
      'review_id': 'hXxxaRuh-OU3EdOcyth6EQ',
      'stars': 1,
      'text': "This iconic bigger chain popularized mini slider burgers.  I found them fairly tasteless, skimpy, and unsatisfying.  Going in I had very low expectations of the burgers but even then I was not satisfied with the product.  I wouldn't recommend this to anyone and I will definitely not be going back.",
      'useful': 0,
      'user_id': '7taWMcaCMQ1F2qVP1W3jCw'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 1,
      'date': '2016-02-29',
      'funny': 1,
      'review_id': 'z1lVUD9zpDxtGInbSdyfzA',
      'stars': 3,
      'text': "The atmosphere is very loud and crowded joint. The casino is next to the restaurant so you can hear people gambling and making noise. I instantly noticed this place was popular due to the large line, so I was intrigued to see what all the fuzz was about.\n\nOrdered items: Combo #2 which came with 4 cheese sliders, some fries and a drink. The burgers were nothing special. They were good, but I will definitely not go out of my way to get these burgers. I would say 4 sliders are equivalent to 1 full size burger. \n\nI was told this place is the east coasts in-N- out. In my opinion, it doesn't come close. The concept is interesting with the sliders, but the overall quality of the burger Isn't great. \n\nOverall, it was an interesting experience. The line was too long and the burgers were average. I would recommend you only try if the line are short and you're very hungry.",
      'useful': 1,
      'user_id': 'fxcyt4lx9XDgTPoAwW7roQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-02-02',
      'funny': 0,
      'review_id': '3bMbB5NfWz0VngFx4lsrgw',
      'stars': 1,
      'text': 'WHITE CASTLE --- Not worth the wait--\n I was so excited to have a white castle open in LV like so many east coasters and went today. What a disappointment!! First timers are going to wonder what all the fuss is about . The burgers were lacking flavor, not hot and you had to look for the onions - and as we all know that gives this little (but expensive) burger the flavor and steam needed to make that bun so special. Neither were there. The bun top actually made it a bit dry since there was NO ketchup and cold! As a local, I will buy the frozen ones and not make the trip to the strip for this $1.29 miss . At least when heated they possess the flavor we remember even adding your own ketchup and pickle',
      'useful': 0,
      'user_id': 'vC3BGwGpaYgDT9rUdk9_SA'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-11-19',
      'funny': 0,
      'review_id': 'jOTRB2LWSDTijelIKY2RgA',
      'stars': 3,
      'text': "Came here last week for lunch. Finally made my way here 11 years after the movie. I didn't have too much expectation coming here. Had the regular beef sliders, chicken sliders, and cheese fries. The beef I found was pretty thin, but the toppings and the fluffy bun def made it taste above average. The chicken lacked toppings and was a tad dry. The cheese fries are good, but you could probably make your own easily at home. Overall, it was a mediocre experience for me. At the end of the day, it's fast food. So if you just want something relatively quick and cheap, this could be one of your options.",
      'useful': 0,
      'user_id': 'mAynnMvVJVdHLOFkmLyftg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-06-13',
      'funny': 0,
      'review_id': 'HWyrbAXyFCzQlNP8_NlnMQ',
      'stars': 3,
      'text': "The only white castle burger I've ever had prior has been the microwaveable frozen kind. A last minute decision to eat something reasonably priced lead me here. So here's my verdict: \n\nCheeseburger- 2.5 stars \nJalapeño cheeseburger- 3.75 stars. The white cheese is a nice change of pace from American cheese, adds a slight kick and adds an extra oomph to the overall burger compared to the regular cheese. \nFish burger- 3 stars \nCrinkle cut fries- 3 stars. Just average. \nOnion chips- 3.5 stars. Good breading, nicely fried. Have had better but not too shabby for fast food. \n\n7 burgers, one drink, and onion chips came out to about $19. \n\nI'll be back for the onion rings and the jalapeño cheeseburger.",
      'useful': 2,
      'user_id': 'B1829_hxXSEpDPEDJtYeIw'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-02-06',
      'funny': 0,
      'review_id': 'sgzd9F0317i6pv0uN_9fMQ',
      'stars': 2,
      'text': 'Overpriced and halfassed\n\nI am a NJ native and have had white castle for over 15 years.  This place is like a cheap imitation while charging almost double for it.  \n\nFirst issue is the fact they don\'t add ketchup like they do at all white castles.  I had to tear all of my burgers apart to add ketchup which being sliders was cumbersome.  Even the guy I complained to who is from NJ said " we don\'t do that, even though all of the other ones do it".\n\nSecond issue was the quality.  The burgers weren\'t fully cooked, since the patties are so thin, this is very hard to screw up.\n\nThird issue was the fries.  They were much thinner and far less quantity than the normal white castles.\n\nFourth issue was the very loud intercom, they yell almost every 15 seconds your order numbers.  This interrupts the dining experience and hurt my ears.\n\nIf you want the real white castle experience I highly suggest you skip this horrible excuse for a white castle and go to an established one.',
      'useful': 0,
      'user_id': 'IsR8VRaHhY9u3baj9m8SbQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-11-19',
      'funny': 0,
      'review_id': 'QO-ihFNkUhphPVWGuQQXjw',
      'stars': 5,
      'text': "I have only been here twice at this location. We don't live far but both times we have gone there, food was very good :-) Fast service!",
      'useful': 0,
      'user_id': 'n0r_0HkCFaC175vXtIVcaQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 3,
      'date': '2015-12-03',
      'funny': 2,
      'review_id': 'vAFBR0AAcXGY2xZcr98nfw',
      'stars': 3,
      'text': "I wish this place lived up to the hype. I mean Harold and kumar had a whole adventure trying to get to this place! Anyway, we came in at around 6pm on a Wednesday and was pleasantly surprised that there wasn't a long line. We ordered sliders with the American cheese and jalapeño cheese, fries, chicken rings, onion chips, and a drink. We also got the brownie and cheesecake dipped in fudge for dessert. We wanted to try everything on the menu basically lol the food came out pretty fast. It really fell flat on the food. Everything was really bland. There's no seasoning whatsoever on anything. The sliders were kinda soggy. The onion chips had such a thick coating with no flavor. The best thing we had was probably the brownie for dessert lol it's cool to try if you're curious since it's quite famous. Just don't expect a lot so you won't get disappointed.",
      'useful': 3,
      'user_id': 'lnMz0MZIAmCbclLJf7W9zQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-07-07',
      'funny': 2,
      'review_id': 'jIfFYL2NbnROQJ6b55dKEQ',
      'stars': 1,
      'text': "So we were vacationing in Vegas and got super excited when we saw a White Castle on the strip, living in the West Coast we've never tried it and heard how amazing these burgers are so we wanted to see what the whole fuzz was about.  Oh man we expected too much those burgers were so gross it tasted like something you can feed to a dog, after the first bite I wanted to puke.  It even got us sick ugh I seriously don't think it's real meat I don't know the quality was horrible. McDonald burgers even taste better,  better yet stick to In N Out , best burgers ever and taste fresh unlike here , NEVER AGAIN , SAVE YOUR MONEY PLZ",
      'useful': 0,
      'user_id': 'f3-grYq_3uPKxMwY2xj48Q'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-08-08',
      'funny': 0,
      'review_id': 'p0aRZ-27_uncdcFYNKinXA',
      'stars': 4,
      'text': 'If you never tried it, must experience it. Great little burgers that will fill you up. 10 should be enough for a person. Good place for late night in Vegas. Opens 24/7.  Line isnt so long. Worth to try and wait for. Great for hangovers after the club.',
      'useful': 0,
      'user_id': 'J-MhhFKS_QFCjbIWJZGMQg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 4,
      'date': '2015-08-29',
      'funny': 3,
      'review_id': 'VadQuxkx2DbT3kY9rIewbw',
      'stars': 2,
      'text': "White Castle was in Las Vegas as a food truck during the Las Vegas Foodie Fest previous years and the line for it was un-fricken-believable. Reportedly, people waited long hours for the infamous tiny sliders. I didn't wait for it personally, but I did see the extreme amount of people in line as I was exploring other (and better) food options. And now it's here not as a mobile restaurant, but an actual restaurant, hidden away in the Casino Royale. \n\nTo find it, you need to walk a little bit, so make time for that. I don't mind playing tourist in my own city every now and then, but lemme tell ya, for White Castle? Not worth it!\n\nThe little sliders are, at best, just cute. The meat was dry and it didn't help that the regular sliders were like 90% buns and meat. There were onions and pickles on it but it seemed like 2 little tiny pieces of diced onions, and a pickle the size of a dime. I prefer the cheese sliders over the regular ones.\n\nAbout 5-6 sliders can fill me up really good on a hungry day, so order accordingly. The way they fill up the beer with the magnet is pretty cool! :) Oh, and perks, 10% off for locals with a local ID!\n\nI'd say this place isn't really worth the hassle to get... maybe a good place to stop by when you're playing tourist in your own city, but that's about it.",
      'useful': 3,
      'user_id': 'os-sDQoh-hCt2Lg76z6ZUA'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2016-05-29',
      'funny': 0,
      'review_id': 'sAiUJ4B8YGhuS4oT00Ynxg',
      'stars': 5,
      'text': "For a super busy fast food joint on the strip they are getting it done. Waited less than 5 minutes for my food, they clean off the tables as soon as anyone stands up. Ask for the dusseldorf mustard or you won't get it.",
      'useful': 0,
      'user_id': 'KJ5gfIAWhE-IZ2DEqgS1nQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2017-05-30',
      'funny': 0,
      'review_id': 'DcANETIMG1TO9d17wAfoiw',
      'stars': 5,
      'text': 'Hi Mike my friend darzell MC joy applied there he is just waiting for you guys to call him',
      'useful': 0,
      'user_id': 'bEoDzX21nzrpynYVuK8GAQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 0,
      'date': '2015-06-18',
      'funny': 0,
      'review_id': 'bJJuIFmT52A-Ye6kZ54I6g',
      'stars': 2,
      'text': 'Growing up back east we would go to white castle and get the burgers by the dozen. Of course that was many years ago and you could get them for around 25 cents, that\'s really showing how long ago that was. So I was glad to see one open here on the strip. So I packed up the wife (who never had one) and off we went. After a short walk and a short line we ordered. Shocked by the prices, but everything costs more today. Told the wife how good these were while growing up. I guess all things change from what I remember. As soon as I pulled the burger from the package it seemed the burger was way thinner than I remembered and after the first bite I was disappointed on the taste. Guess this is off our list on coming back along with the "shake shack" also. How disappointed this was from the memories of my youth. Pay a little more and find one of the other burger places nearby so not to be disappointed.',
      'useful': 0,
      'user_id': 'WSJJF_FyRz0osgBQgndlDg'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 1,
      'date': '2015-02-20',
      'funny': 0,
      'review_id': 'RqUY5MgE9FnlcRyaU0Gq8g',
      'stars': 5,
      'text': 'I have always wanted to try White Castle; I have only had the frozen burgers.  Needless to say when I saw a White Castle on the strip in Las Vegas I was ecstatic.  The line to order moved fast, the line to pick up food was super slow but who cares its White Castle.  The burgers are small and delicious, the bread was warm and soft.  It was the burger slider.',
      'useful': 2,
      'user_id': 'O-m_9oMnS45NrnxQ6JlNEA'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 1,
      'date': '2016-02-10',
      'funny': 0,
      'review_id': 'Vrm9mffNiLflnkY33ixRvg',
      'stars': 3,
      'text': 'Harold and Kumar sure make it seem like this is the place to be. For many years, I have been wanting to try this hoping that I would be blown away from the taste. Since they do not have this chain down south, where I stay, I had not been able to taste it.  Until one day, I was driving by the vegas strip and noticed it tucked deep in the sidelines of the Casinos.  After some tricky parking, and a long trek I was finally at the register ordering the cheese slider meal.  At first glance, everything seemed good, heck it sure did make for a great instagram post, which is mandatory for me on all food I eat.  After the first bite, I was left unimpressed.  I would easily rank Fatburger and In and out above this chain and even Mcdonalds.  It was definitely great to finally try it but would not go out of my way to do it again.',
      'useful': 2,
      'user_id': 'Ot-ODtUth8kGGujm_AscdQ'},
     {'business_id': 'dKdApYVFDSNYsNOso6NYlA',
      'cool': 1,
      'date': '2017-04-07',
      'funny': 0,
      'review_id': 'zthpcSQxPti3iGEL70Gq1w',
      'stars': 5,
      'text': "When I was living in Vegas, they didn't have a White Castle yet. This one is fairly new. I wanna say maybe a year now? \n\nAnyways, if you're in Vegas, White Castle is a MUST if you're on the strip. Especially when you're drunk. Their burgers are like sliders, small but they're less than $2! \n\nThey are delicious! \n\nThey are located near Fat Tuesday and Harrah's hotel. \n\nIf you aren't in any of the states that has a White Castle, Target sells frozen boxes of White Castle sliders. It won't be the same but it's close!",
      'useful': 0,
      'user_id': 'o1zif09fuJv8f9DJ7Zl73w'},
     ...]





```python
restaurant_data = [x for x in business_data if 'Restaurants' in x['categories']]

```


### Getting global averages



```python
user_total = [x['average_stars'] for x in user_data]
global_user_average = sum(user_total)/len(user_total)
print (global_user_average)
```


    3.7118346541447185




```python
restaurant_total = [x['stars'] for x in restaurant_data]
global_restaurant_average = sum(restaurant_total)/len(restaurant_total)
print (global_restaurant_average)
```


    3.461104760428574




```python

reviews_total = [x['stars'] for x in restaurant_reviews]
global_review_average = sum(reviews_total)/len(reviews_total)
print (global_review_average)
```


    3.702161161664101


### Getting restaurant and user deviations



```python
user_dict = {}
for item in user_data:
    user_id = item['user_id']
    user_dict[user_id] = item
```




```python
user_deviations = {}
user_deviations_2 = {}
for item in user_data:
    user_id = item['user_id']
    user_deviations[user_id] = item['average_stars'] - global_user_average
    user_deviations_2[user_id] = item['average_stars'] - global_review_average
```




```python
restaurant_deviations = {}
restaurant_deviations_2 = {}
for item in restaurant_data:
    rest_id = item['business_id']
    restaurant_deviations[rest_id] = item['stars'] - global_restaurant_average
    restaurant_deviations_2[rest_id] = item['stars'] - global_review_average
```


### Making Baseline Model



```python
def baseline(user_id, business_id):
    pred = global_review_average + user_deviations[user_id] + restaurant_deviations[business_id]
    return int(round(pred))
def baseline2(user_id, business_id):
    pred = global_review_average + user_deviations_2[user_id] + restaurant_deviations_2[business_id]
    return int(round(pred))
```





```python
from pandas.io.json import json_normalize
import pandas as pd

```




```python
restaurant_dict = {}
for item in restaurant_data:
    restaurant_id = item['business_id']
    restaurant_dict[restaurant_id] = item


```


#### We use 100,000 datapoints for our train and test sets



```python
import copy
data_array = (np.random.choice(restaurant_reviews, size = 100000))
data_set = list(data_array)
df = pd.DataFrame(data_set)
```




```python
from collections import Counter
all_categories = []
for r in restaurant_data:
    #print (r['categories'])
    if 'Restaurants' in r['categories']:
        for c in r['categories']:
            all_categories.append(c)
counts = list (Counter(all_categories).items())
counts.sort(key=lambda x: x[1], reverse = True)
most_popular = [x[0] for x in counts[:150]]

```




```python
expanded_reviews = copy.deepcopy(data_array)
len(expanded_reviews)
```





    100000





```python

for review in expanded_reviews:
    #print (review)
    restaurant = review['business_id']
    user = review['user_id']
    restaurant_info = restaurant_dict[restaurant]
    #print (restaurant_info)
    user_info = user_dict[user]
    for attribute in restaurant_info:
        #print (attribute)
        if attribute not in ['is_open', 'latitude','longitude','name','business_id',
                             'neighborhood','address','city','postal_code','hours']:
            if attribute == 'categories':
                for c in most_popular:
                    if c in restaurant_info[attribute]:
                        review['R_' +  c] = 1
                    else:
                        review['R_' +  c] = 0
            else:         
                review['R_' + attribute] = restaurant_info[attribute]
    for attribute in user_info:
        if attribute not in ['user_id','name']:   
            if attribute == 'friends':
                review['U_friends'] = len(user_info[attribute])
            elif attribute == 'yelping_since':
                review['U_yelping_since'] = user_info[attribute][:4]
            elif attribute == 'elite':
                if user_info[attribute]:
                    review['U_elite'] = True
                else:
                    review['U_elite'] = False        
            else:
                review['U_' + attribute] = user_info[attribute]
        

```




```python
pd.set_option('display.max_columns', None)
```




```python
flatframe = json_normalize(expanded_reviews)
```




```python
flatframe.shape
```





    (100000, 263)





```python
flatframe = flatframe.drop(['text','useful','funny','cool','date'], axis=1)
flatframe['U_years_yelping'] = [2015 - int(x) for x in flatframe['U_yelping_since']]
flatframe.drop(['U_yelping_since'],axis = 1, inplace = True)
flatframe.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>R_Active Life</th>
      <th>R_Afghan</th>
      <th>R_African</th>
      <th>R_American (New)</th>
      <th>R_American (Traditional)</th>
      <th>R_Arcades</th>
      <th>R_Arts &amp; Entertainment</th>
      <th>R_Asian Fusion</th>
      <th>R_Bagels</th>
      <th>R_Bakeries</th>
      <th>R_Barbeque</th>
      <th>R_Bars</th>
      <th>R_Beer</th>
      <th>R_Beer Bar</th>
      <th>R_Beer Garden</th>
      <th>R_Bistros</th>
      <th>R_Brasseries</th>
      <th>R_Brazilian</th>
      <th>R_Breakfast &amp; Brunch</th>
      <th>R_Breweries</th>
      <th>R_British</th>
      <th>R_Bubble Tea</th>
      <th>R_Buffets</th>
      <th>R_Burgers</th>
      <th>R_Butcher</th>
      <th>R_Cafes</th>
      <th>R_Cajun/Creole</th>
      <th>R_Canadian (New)</th>
      <th>R_Cantonese</th>
      <th>R_Caribbean</th>
      <th>R_Casinos</th>
      <th>R_Caterers</th>
      <th>R_Cheesesteaks</th>
      <th>R_Chicken Shop</th>
      <th>R_Chicken Wings</th>
      <th>R_Chinese</th>
      <th>R_Cocktail Bars</th>
      <th>R_Coffee &amp; Tea</th>
      <th>R_Comfort Food</th>
      <th>R_Convenience Stores</th>
      <th>R_Creperies</th>
      <th>R_Cuban</th>
      <th>R_Dance Clubs</th>
      <th>R_Delicatessen</th>
      <th>R_Delis</th>
      <th>R_Desserts</th>
      <th>R_Dim Sum</th>
      <th>R_Diners</th>
      <th>R_Dive Bars</th>
      <th>R_Do-It-Yourself Food</th>
      <th>R_Donuts</th>
      <th>R_Ethiopian</th>
      <th>R_Ethnic Food</th>
      <th>R_Event Planning &amp; Services</th>
      <th>R_Falafel</th>
      <th>R_Fast Food</th>
      <th>R_Filipino</th>
      <th>R_Fish &amp; Chips</th>
      <th>R_Food</th>
      <th>R_Food Court</th>
      <th>R_Food Delivery Services</th>
      <th>R_Food Stands</th>
      <th>R_Food Trucks</th>
      <th>R_French</th>
      <th>R_Gastropubs</th>
      <th>R_German</th>
      <th>R_Gluten-Free</th>
      <th>R_Greek</th>
      <th>R_Grocery</th>
      <th>R_Halal</th>
      <th>R_Hawaiian</th>
      <th>R_Health Markets</th>
      <th>R_Hookah Bars</th>
      <th>R_Hot Dogs</th>
      <th>R_Hot Pot</th>
      <th>R_Hotels</th>
      <th>R_Hotels &amp; Travel</th>
      <th>R_Ice Cream &amp; Frozen Yogurt</th>
      <th>R_Imported Food</th>
      <th>R_Indian</th>
      <th>R_International</th>
      <th>R_Internet Cafes</th>
      <th>R_Irish</th>
      <th>R_Italian</th>
      <th>R_Japanese</th>
      <th>R_Juice Bars &amp; Smoothies</th>
      <th>R_Karaoke</th>
      <th>R_Kebab</th>
      <th>R_Korean</th>
      <th>R_Kosher</th>
      <th>R_Latin American</th>
      <th>R_Lebanese</th>
      <th>R_Local Flavor</th>
      <th>R_Lounges</th>
      <th>R_Malaysian</th>
      <th>R_Meat Shops</th>
      <th>R_Mediterranean</th>
      <th>R_Mexican</th>
      <th>R_Middle Eastern</th>
      <th>R_Modern European</th>
      <th>R_Music Venues</th>
      <th>R_Nightlife</th>
      <th>R_Noodles</th>
      <th>R_Pakistani</th>
      <th>R_Pan Asian</th>
      <th>R_Party &amp; Event Planning</th>
      <th>R_Patisserie/Cake Shop</th>
      <th>R_Persian/Iranian</th>
      <th>R_Peruvian</th>
      <th>R_Pizza</th>
      <th>R_Poke</th>
      <th>R_Polish</th>
      <th>R_Portuguese</th>
      <th>R_Poutineries</th>
      <th>R_Pubs</th>
      <th>R_Ramen</th>
      <th>R_Restaurants</th>
      <th>R_Salad</th>
      <th>R_Sandwiches</th>
      <th>R_Scottish</th>
      <th>R_Seafood</th>
      <th>R_Seafood Markets</th>
      <th>R_Shopping</th>
      <th>R_Soul Food</th>
      <th>R_Soup</th>
      <th>R_Southern</th>
      <th>R_Spanish</th>
      <th>R_Specialty Food</th>
      <th>R_Sports Bars</th>
      <th>R_Steakhouses</th>
      <th>R_Street Vendors</th>
      <th>R_Sushi Bars</th>
      <th>R_Swabian</th>
      <th>R_Szechuan</th>
      <th>R_Tacos</th>
      <th>R_Taiwanese</th>
      <th>R_Tapas Bars</th>
      <th>R_Tapas/Small Plates</th>
      <th>R_Tea Rooms</th>
      <th>R_Tex-Mex</th>
      <th>R_Thai</th>
      <th>R_Turkish</th>
      <th>R_Vegan</th>
      <th>R_Vegetarian</th>
      <th>R_Venues &amp; Event Spaces</th>
      <th>R_Vietnamese</th>
      <th>R_Waffles</th>
      <th>R_Wine &amp; Spirits</th>
      <th>R_Wine Bars</th>
      <th>R_Wraps</th>
      <th>R_attributes.AcceptsInsurance</th>
      <th>R_attributes.AgesAllowed</th>
      <th>R_attributes.Alcohol</th>
      <th>R_attributes.Ambience.casual</th>
      <th>R_attributes.Ambience.classy</th>
      <th>R_attributes.Ambience.divey</th>
      <th>R_attributes.Ambience.hipster</th>
      <th>R_attributes.Ambience.intimate</th>
      <th>R_attributes.Ambience.romantic</th>
      <th>R_attributes.Ambience.touristy</th>
      <th>R_attributes.Ambience.trendy</th>
      <th>R_attributes.Ambience.upscale</th>
      <th>R_attributes.BYOB</th>
      <th>R_attributes.BYOBCorkage</th>
      <th>R_attributes.BestNights.friday</th>
      <th>R_attributes.BestNights.monday</th>
      <th>R_attributes.BestNights.saturday</th>
      <th>R_attributes.BestNights.sunday</th>
      <th>R_attributes.BestNights.thursday</th>
      <th>R_attributes.BestNights.tuesday</th>
      <th>R_attributes.BestNights.wednesday</th>
      <th>R_attributes.BikeParking</th>
      <th>R_attributes.BusinessAcceptsBitcoin</th>
      <th>R_attributes.BusinessAcceptsCreditCards</th>
      <th>R_attributes.BusinessParking.garage</th>
      <th>R_attributes.BusinessParking.lot</th>
      <th>R_attributes.BusinessParking.street</th>
      <th>R_attributes.BusinessParking.valet</th>
      <th>R_attributes.BusinessParking.validated</th>
      <th>R_attributes.ByAppointmentOnly</th>
      <th>R_attributes.Caters</th>
      <th>R_attributes.CoatCheck</th>
      <th>R_attributes.Corkage</th>
      <th>R_attributes.DietaryRestrictions.dairy-free</th>
      <th>R_attributes.DietaryRestrictions.gluten-free</th>
      <th>R_attributes.DietaryRestrictions.halal</th>
      <th>R_attributes.DietaryRestrictions.kosher</th>
      <th>R_attributes.DietaryRestrictions.soy-free</th>
      <th>R_attributes.DietaryRestrictions.vegan</th>
      <th>R_attributes.DietaryRestrictions.vegetarian</th>
      <th>R_attributes.DogsAllowed</th>
      <th>R_attributes.DriveThru</th>
      <th>R_attributes.GoodForDancing</th>
      <th>R_attributes.GoodForKids</th>
      <th>R_attributes.GoodForMeal.breakfast</th>
      <th>R_attributes.GoodForMeal.brunch</th>
      <th>R_attributes.GoodForMeal.dessert</th>
      <th>R_attributes.GoodForMeal.dinner</th>
      <th>R_attributes.GoodForMeal.latenight</th>
      <th>R_attributes.GoodForMeal.lunch</th>
      <th>R_attributes.HairSpecializesIn.africanamerican</th>
      <th>R_attributes.HairSpecializesIn.asian</th>
      <th>R_attributes.HairSpecializesIn.coloring</th>
      <th>R_attributes.HairSpecializesIn.curly</th>
      <th>R_attributes.HairSpecializesIn.extensions</th>
      <th>R_attributes.HairSpecializesIn.kids</th>
      <th>R_attributes.HairSpecializesIn.perms</th>
      <th>R_attributes.HairSpecializesIn.straightperms</th>
      <th>R_attributes.HappyHour</th>
      <th>R_attributes.HasTV</th>
      <th>R_attributes.Music.background_music</th>
      <th>R_attributes.Music.dj</th>
      <th>R_attributes.Music.jukebox</th>
      <th>R_attributes.Music.karaoke</th>
      <th>R_attributes.Music.live</th>
      <th>R_attributes.Music.no_music</th>
      <th>R_attributes.Music.video</th>
      <th>R_attributes.NoiseLevel</th>
      <th>R_attributes.Open24Hours</th>
      <th>R_attributes.OutdoorSeating</th>
      <th>R_attributes.RestaurantsAttire</th>
      <th>R_attributes.RestaurantsCounterService</th>
      <th>R_attributes.RestaurantsDelivery</th>
      <th>R_attributes.RestaurantsGoodForGroups</th>
      <th>R_attributes.RestaurantsPriceRange2</th>
      <th>R_attributes.RestaurantsReservations</th>
      <th>R_attributes.RestaurantsTableService</th>
      <th>R_attributes.RestaurantsTakeOut</th>
      <th>R_attributes.Smoking</th>
      <th>R_attributes.WheelchairAccessible</th>
      <th>R_attributes.WiFi</th>
      <th>R_review_count</th>
      <th>R_stars</th>
      <th>R_state</th>
      <th>U_average_stars</th>
      <th>U_compliment_cool</th>
      <th>U_compliment_cute</th>
      <th>U_compliment_funny</th>
      <th>U_compliment_hot</th>
      <th>U_compliment_list</th>
      <th>U_compliment_more</th>
      <th>U_compliment_note</th>
      <th>U_compliment_photos</th>
      <th>U_compliment_plain</th>
      <th>U_compliment_profile</th>
      <th>U_compliment_writer</th>
      <th>U_cool</th>
      <th>U_elite</th>
      <th>U_fans</th>
      <th>U_friends</th>
      <th>U_funny</th>
      <th>U_review_count</th>
      <th>U_useful</th>
      <th>business_id</th>
      <th>review_id</th>
      <th>stars</th>
      <th>user_id</th>
      <th>U_years_yelping</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>full_bar</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>average</td>
      <td>NaN</td>
      <td>True</td>
      <td>casual</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>2.0</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>NaN</td>
      <td>True</td>
      <td>free</td>
      <td>143</td>
      <td>3.0</td>
      <td>AZ</td>
      <td>3.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>Dj8MfG5-UGbjp9GxaNnGxQ</td>
      <td>0dpWElUaORFtcitj9RvoSQ</td>
      <td>1</td>
      <td>TErRhmdZPpK03khnD7LIxw</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>beer_and_wine</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>average</td>
      <td>NaN</td>
      <td>False</td>
      <td>casual</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>2.0</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>NaN</td>
      <td>False</td>
      <td>no</td>
      <td>110</td>
      <td>3.0</td>
      <td>ON</td>
      <td>3.23</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>64</td>
      <td>10</td>
      <td>OZF7EM-W-2-V0LQOYSYhBA</td>
      <td>f8x6Z7TBICCqrib25W9WTQ</td>
      <td>2</td>
      <td>qpujBDnG3HIys1duA_SJow</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>full_bar</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>average</td>
      <td>NaN</td>
      <td>True</td>
      <td>casual</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>2.0</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>no</td>
      <td>True</td>
      <td>free</td>
      <td>268</td>
      <td>3.5</td>
      <td>NC</td>
      <td>4.10</td>
      <td>11</td>
      <td>0</td>
      <td>11</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>7</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>True</td>
      <td>10</td>
      <td>186</td>
      <td>0</td>
      <td>70</td>
      <td>4</td>
      <td>Abm-1HRGPyevBtnC3Rpxag</td>
      <td>HjBmGAPXlTjwLdXXRYv-3Q</td>
      <td>3</td>
      <td>fRrTNhHxKrBXZyncyoSsCA</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>full_bar</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>average</td>
      <td>NaN</td>
      <td>False</td>
      <td>casual</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>2.0</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>no</td>
      <td>53</td>
      <td>2.5</td>
      <td>NV</td>
      <td>4.64</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>1</td>
      <td>21</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>G9Cz2lUnLxv1cj6fnv73XA</td>
      <td>dU0I4JeEaJ8SF6LTc9yzew</td>
      <td>4</td>
      <td>yGYYcXTlQQriY6moVkXgzA</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>none</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>average</td>
      <td>NaN</td>
      <td>True</td>
      <td>casual</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>1.0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>True</td>
      <td>no</td>
      <td>1145</td>
      <td>4.5</td>
      <td>NV</td>
      <td>3.06</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>False</td>
      <td>1</td>
      <td>39</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>EnCIojgP5KTr1leaysFE3A</td>
      <td>kMlBePvm7Fj152T3wmUaGg</td>
      <td>5</td>
      <td>xvzstIJ3Crmlpf9GJG4csw</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>





```python
flatframe_v2 = flatframe.drop(['business_id', 'review_id', 'user_id'], axis = 1)
```




```python
#one hot encode state
flatframe_v3 = pd.get_dummies(flatframe_v2, columns = ['R_state', 
                                                       'R_attributes.Alcohol', 'R_attributes.AgesAllowed', 'R_attributes.RestaurantsAttire',
                                                      'R_attributes.RestaurantsPriceRange2','R_attributes.Smoking',
                                                      'R_attributes.WiFi', 'R_attributes.NoiseLevel','R_attributes.BYOBCorkage'])
flatframe_v3.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>R_Active Life</th>
      <th>R_Afghan</th>
      <th>R_African</th>
      <th>R_American (New)</th>
      <th>R_American (Traditional)</th>
      <th>R_Arcades</th>
      <th>R_Arts &amp; Entertainment</th>
      <th>R_Asian Fusion</th>
      <th>R_Bagels</th>
      <th>R_Bakeries</th>
      <th>R_Barbeque</th>
      <th>R_Bars</th>
      <th>R_Beer</th>
      <th>R_Beer Bar</th>
      <th>R_Beer Garden</th>
      <th>R_Bistros</th>
      <th>R_Brasseries</th>
      <th>R_Brazilian</th>
      <th>R_Breakfast &amp; Brunch</th>
      <th>R_Breweries</th>
      <th>R_British</th>
      <th>R_Bubble Tea</th>
      <th>R_Buffets</th>
      <th>R_Burgers</th>
      <th>R_Butcher</th>
      <th>R_Cafes</th>
      <th>R_Cajun/Creole</th>
      <th>R_Canadian (New)</th>
      <th>R_Cantonese</th>
      <th>R_Caribbean</th>
      <th>R_Casinos</th>
      <th>R_Caterers</th>
      <th>R_Cheesesteaks</th>
      <th>R_Chicken Shop</th>
      <th>R_Chicken Wings</th>
      <th>R_Chinese</th>
      <th>R_Cocktail Bars</th>
      <th>R_Coffee &amp; Tea</th>
      <th>R_Comfort Food</th>
      <th>R_Convenience Stores</th>
      <th>R_Creperies</th>
      <th>R_Cuban</th>
      <th>R_Dance Clubs</th>
      <th>R_Delicatessen</th>
      <th>R_Delis</th>
      <th>R_Desserts</th>
      <th>R_Dim Sum</th>
      <th>R_Diners</th>
      <th>R_Dive Bars</th>
      <th>R_Do-It-Yourself Food</th>
      <th>R_Donuts</th>
      <th>R_Ethiopian</th>
      <th>R_Ethnic Food</th>
      <th>R_Event Planning &amp; Services</th>
      <th>R_Falafel</th>
      <th>R_Fast Food</th>
      <th>R_Filipino</th>
      <th>R_Fish &amp; Chips</th>
      <th>R_Food</th>
      <th>R_Food Court</th>
      <th>R_Food Delivery Services</th>
      <th>R_Food Stands</th>
      <th>R_Food Trucks</th>
      <th>R_French</th>
      <th>R_Gastropubs</th>
      <th>R_German</th>
      <th>R_Gluten-Free</th>
      <th>R_Greek</th>
      <th>R_Grocery</th>
      <th>R_Halal</th>
      <th>R_Hawaiian</th>
      <th>R_Health Markets</th>
      <th>R_Hookah Bars</th>
      <th>R_Hot Dogs</th>
      <th>R_Hot Pot</th>
      <th>R_Hotels</th>
      <th>R_Hotels &amp; Travel</th>
      <th>R_Ice Cream &amp; Frozen Yogurt</th>
      <th>R_Imported Food</th>
      <th>R_Indian</th>
      <th>R_International</th>
      <th>R_Internet Cafes</th>
      <th>R_Irish</th>
      <th>R_Italian</th>
      <th>R_Japanese</th>
      <th>R_Juice Bars &amp; Smoothies</th>
      <th>R_Karaoke</th>
      <th>R_Kebab</th>
      <th>R_Korean</th>
      <th>R_Kosher</th>
      <th>R_Latin American</th>
      <th>R_Lebanese</th>
      <th>R_Local Flavor</th>
      <th>R_Lounges</th>
      <th>R_Malaysian</th>
      <th>R_Meat Shops</th>
      <th>R_Mediterranean</th>
      <th>R_Mexican</th>
      <th>R_Middle Eastern</th>
      <th>R_Modern European</th>
      <th>R_Music Venues</th>
      <th>R_Nightlife</th>
      <th>R_Noodles</th>
      <th>R_Pakistani</th>
      <th>R_Pan Asian</th>
      <th>R_Party &amp; Event Planning</th>
      <th>R_Patisserie/Cake Shop</th>
      <th>R_Persian/Iranian</th>
      <th>R_Peruvian</th>
      <th>R_Pizza</th>
      <th>R_Poke</th>
      <th>R_Polish</th>
      <th>R_Portuguese</th>
      <th>R_Poutineries</th>
      <th>R_Pubs</th>
      <th>R_Ramen</th>
      <th>R_Restaurants</th>
      <th>R_Salad</th>
      <th>R_Sandwiches</th>
      <th>R_Scottish</th>
      <th>R_Seafood</th>
      <th>R_Seafood Markets</th>
      <th>R_Shopping</th>
      <th>R_Soul Food</th>
      <th>R_Soup</th>
      <th>R_Southern</th>
      <th>R_Spanish</th>
      <th>R_Specialty Food</th>
      <th>R_Sports Bars</th>
      <th>R_Steakhouses</th>
      <th>R_Street Vendors</th>
      <th>R_Sushi Bars</th>
      <th>R_Swabian</th>
      <th>R_Szechuan</th>
      <th>R_Tacos</th>
      <th>R_Taiwanese</th>
      <th>R_Tapas Bars</th>
      <th>R_Tapas/Small Plates</th>
      <th>R_Tea Rooms</th>
      <th>R_Tex-Mex</th>
      <th>R_Thai</th>
      <th>R_Turkish</th>
      <th>R_Vegan</th>
      <th>R_Vegetarian</th>
      <th>R_Venues &amp; Event Spaces</th>
      <th>R_Vietnamese</th>
      <th>R_Waffles</th>
      <th>R_Wine &amp; Spirits</th>
      <th>R_Wine Bars</th>
      <th>R_Wraps</th>
      <th>R_attributes.AcceptsInsurance</th>
      <th>R_attributes.Ambience.casual</th>
      <th>R_attributes.Ambience.classy</th>
      <th>R_attributes.Ambience.divey</th>
      <th>R_attributes.Ambience.hipster</th>
      <th>R_attributes.Ambience.intimate</th>
      <th>R_attributes.Ambience.romantic</th>
      <th>R_attributes.Ambience.touristy</th>
      <th>R_attributes.Ambience.trendy</th>
      <th>R_attributes.Ambience.upscale</th>
      <th>R_attributes.BYOB</th>
      <th>R_attributes.BestNights.friday</th>
      <th>R_attributes.BestNights.monday</th>
      <th>R_attributes.BestNights.saturday</th>
      <th>R_attributes.BestNights.sunday</th>
      <th>R_attributes.BestNights.thursday</th>
      <th>R_attributes.BestNights.tuesday</th>
      <th>R_attributes.BestNights.wednesday</th>
      <th>R_attributes.BikeParking</th>
      <th>R_attributes.BusinessAcceptsBitcoin</th>
      <th>R_attributes.BusinessAcceptsCreditCards</th>
      <th>R_attributes.BusinessParking.garage</th>
      <th>R_attributes.BusinessParking.lot</th>
      <th>R_attributes.BusinessParking.street</th>
      <th>R_attributes.BusinessParking.valet</th>
      <th>R_attributes.BusinessParking.validated</th>
      <th>R_attributes.ByAppointmentOnly</th>
      <th>R_attributes.Caters</th>
      <th>R_attributes.CoatCheck</th>
      <th>R_attributes.Corkage</th>
      <th>R_attributes.DietaryRestrictions.dairy-free</th>
      <th>R_attributes.DietaryRestrictions.gluten-free</th>
      <th>R_attributes.DietaryRestrictions.halal</th>
      <th>R_attributes.DietaryRestrictions.kosher</th>
      <th>R_attributes.DietaryRestrictions.soy-free</th>
      <th>R_attributes.DietaryRestrictions.vegan</th>
      <th>R_attributes.DietaryRestrictions.vegetarian</th>
      <th>R_attributes.DogsAllowed</th>
      <th>R_attributes.DriveThru</th>
      <th>R_attributes.GoodForDancing</th>
      <th>R_attributes.GoodForKids</th>
      <th>R_attributes.GoodForMeal.breakfast</th>
      <th>R_attributes.GoodForMeal.brunch</th>
      <th>R_attributes.GoodForMeal.dessert</th>
      <th>R_attributes.GoodForMeal.dinner</th>
      <th>R_attributes.GoodForMeal.latenight</th>
      <th>R_attributes.GoodForMeal.lunch</th>
      <th>R_attributes.HairSpecializesIn.africanamerican</th>
      <th>R_attributes.HairSpecializesIn.asian</th>
      <th>R_attributes.HairSpecializesIn.coloring</th>
      <th>R_attributes.HairSpecializesIn.curly</th>
      <th>R_attributes.HairSpecializesIn.extensions</th>
      <th>R_attributes.HairSpecializesIn.kids</th>
      <th>R_attributes.HairSpecializesIn.perms</th>
      <th>R_attributes.HairSpecializesIn.straightperms</th>
      <th>R_attributes.HappyHour</th>
      <th>R_attributes.HasTV</th>
      <th>R_attributes.Music.background_music</th>
      <th>R_attributes.Music.dj</th>
      <th>R_attributes.Music.jukebox</th>
      <th>R_attributes.Music.karaoke</th>
      <th>R_attributes.Music.live</th>
      <th>R_attributes.Music.no_music</th>
      <th>R_attributes.Music.video</th>
      <th>R_attributes.Open24Hours</th>
      <th>R_attributes.OutdoorSeating</th>
      <th>R_attributes.RestaurantsCounterService</th>
      <th>R_attributes.RestaurantsDelivery</th>
      <th>R_attributes.RestaurantsGoodForGroups</th>
      <th>R_attributes.RestaurantsReservations</th>
      <th>R_attributes.RestaurantsTableService</th>
      <th>R_attributes.RestaurantsTakeOut</th>
      <th>R_attributes.WheelchairAccessible</th>
      <th>R_review_count</th>
      <th>R_stars</th>
      <th>U_average_stars</th>
      <th>U_compliment_cool</th>
      <th>U_compliment_cute</th>
      <th>U_compliment_funny</th>
      <th>U_compliment_hot</th>
      <th>U_compliment_list</th>
      <th>U_compliment_more</th>
      <th>U_compliment_note</th>
      <th>U_compliment_photos</th>
      <th>U_compliment_plain</th>
      <th>U_compliment_profile</th>
      <th>U_compliment_writer</th>
      <th>U_cool</th>
      <th>U_elite</th>
      <th>U_fans</th>
      <th>U_friends</th>
      <th>U_funny</th>
      <th>U_review_count</th>
      <th>U_useful</th>
      <th>stars</th>
      <th>U_years_yelping</th>
      <th>R_state_01</th>
      <th>R_state_AZ</th>
      <th>R_state_BW</th>
      <th>R_state_C</th>
      <th>R_state_EDH</th>
      <th>R_state_ELN</th>
      <th>R_state_FIF</th>
      <th>R_state_HLD</th>
      <th>R_state_IL</th>
      <th>R_state_MLN</th>
      <th>R_state_NC</th>
      <th>R_state_NI</th>
      <th>R_state_NV</th>
      <th>R_state_NY</th>
      <th>R_state_NYK</th>
      <th>R_state_OH</th>
      <th>R_state_ON</th>
      <th>R_state_PA</th>
      <th>R_state_PKN</th>
      <th>R_state_QC</th>
      <th>R_state_RCC</th>
      <th>R_state_SC</th>
      <th>R_state_WA</th>
      <th>R_state_WI</th>
      <th>R_state_WLN</th>
      <th>R_attributes.Alcohol_beer_and_wine</th>
      <th>R_attributes.Alcohol_full_bar</th>
      <th>R_attributes.Alcohol_none</th>
      <th>R_attributes.AgesAllowed_18plus</th>
      <th>R_attributes.AgesAllowed_19plus</th>
      <th>R_attributes.AgesAllowed_21plus</th>
      <th>R_attributes.AgesAllowed_allages</th>
      <th>R_attributes.RestaurantsAttire_casual</th>
      <th>R_attributes.RestaurantsAttire_dressy</th>
      <th>R_attributes.RestaurantsAttire_formal</th>
      <th>R_attributes.RestaurantsPriceRange2_1.0</th>
      <th>R_attributes.RestaurantsPriceRange2_2.0</th>
      <th>R_attributes.RestaurantsPriceRange2_3.0</th>
      <th>R_attributes.RestaurantsPriceRange2_4.0</th>
      <th>R_attributes.Smoking_no</th>
      <th>R_attributes.Smoking_outdoor</th>
      <th>R_attributes.Smoking_yes</th>
      <th>R_attributes.WiFi_free</th>
      <th>R_attributes.WiFi_no</th>
      <th>R_attributes.WiFi_paid</th>
      <th>R_attributes.NoiseLevel_average</th>
      <th>R_attributes.NoiseLevel_loud</th>
      <th>R_attributes.NoiseLevel_quiet</th>
      <th>R_attributes.NoiseLevel_very_loud</th>
      <th>R_attributes.BYOBCorkage_no</th>
      <th>R_attributes.BYOBCorkage_yes_corkage</th>
      <th>R_attributes.BYOBCorkage_yes_free</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>143</td>
      <td>3.0</td>
      <td>3.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>110</td>
      <td>3.0</td>
      <td>3.23</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>64</td>
      <td>10</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>268</td>
      <td>3.5</td>
      <td>4.10</td>
      <td>11</td>
      <td>0</td>
      <td>11</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>7</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>True</td>
      <td>10</td>
      <td>186</td>
      <td>0</td>
      <td>70</td>
      <td>4</td>
      <td>3</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>NaN</td>
      <td>53</td>
      <td>2.5</td>
      <td>4.64</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>1</td>
      <td>21</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>True</td>
      <td>NaN</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>1145</td>
      <td>4.5</td>
      <td>3.06</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>False</td>
      <td>1</td>
      <td>39</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>





```python
nan_count = {}
for column in flatframe_v3:
     nan_count[column] = flatframe_v3[column].isnull().sum()
```




```python
nan_sorted = sorted(nan_count.items(), key=lambda x: x[1], reverse = True) 
drop_nans = [x[0] for x in nan_sorted if x[1] > 50000]

```




```python
flatframe_v3 = flatframe_v3.drop(drop_nans, axis = 1)
flatframe_v4 = flatframe_v3.fillna(flatframe_v3.mean())
flatframe_v4.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>R_Active Life</th>
      <th>R_Afghan</th>
      <th>R_African</th>
      <th>R_American (New)</th>
      <th>R_American (Traditional)</th>
      <th>R_Arcades</th>
      <th>R_Arts &amp; Entertainment</th>
      <th>R_Asian Fusion</th>
      <th>R_Bagels</th>
      <th>R_Bakeries</th>
      <th>R_Barbeque</th>
      <th>R_Bars</th>
      <th>R_Beer</th>
      <th>R_Beer Bar</th>
      <th>R_Beer Garden</th>
      <th>R_Bistros</th>
      <th>R_Brasseries</th>
      <th>R_Brazilian</th>
      <th>R_Breakfast &amp; Brunch</th>
      <th>R_Breweries</th>
      <th>R_British</th>
      <th>R_Bubble Tea</th>
      <th>R_Buffets</th>
      <th>R_Burgers</th>
      <th>R_Butcher</th>
      <th>R_Cafes</th>
      <th>R_Cajun/Creole</th>
      <th>R_Canadian (New)</th>
      <th>R_Cantonese</th>
      <th>R_Caribbean</th>
      <th>R_Casinos</th>
      <th>R_Caterers</th>
      <th>R_Cheesesteaks</th>
      <th>R_Chicken Shop</th>
      <th>R_Chicken Wings</th>
      <th>R_Chinese</th>
      <th>R_Cocktail Bars</th>
      <th>R_Coffee &amp; Tea</th>
      <th>R_Comfort Food</th>
      <th>R_Convenience Stores</th>
      <th>R_Creperies</th>
      <th>R_Cuban</th>
      <th>R_Dance Clubs</th>
      <th>R_Delicatessen</th>
      <th>R_Delis</th>
      <th>R_Desserts</th>
      <th>R_Dim Sum</th>
      <th>R_Diners</th>
      <th>R_Dive Bars</th>
      <th>R_Do-It-Yourself Food</th>
      <th>R_Donuts</th>
      <th>R_Ethiopian</th>
      <th>R_Ethnic Food</th>
      <th>R_Event Planning &amp; Services</th>
      <th>R_Falafel</th>
      <th>R_Fast Food</th>
      <th>R_Filipino</th>
      <th>R_Fish &amp; Chips</th>
      <th>R_Food</th>
      <th>R_Food Court</th>
      <th>R_Food Delivery Services</th>
      <th>R_Food Stands</th>
      <th>R_Food Trucks</th>
      <th>R_French</th>
      <th>R_Gastropubs</th>
      <th>R_German</th>
      <th>R_Gluten-Free</th>
      <th>R_Greek</th>
      <th>R_Grocery</th>
      <th>R_Halal</th>
      <th>R_Hawaiian</th>
      <th>R_Health Markets</th>
      <th>R_Hookah Bars</th>
      <th>R_Hot Dogs</th>
      <th>R_Hot Pot</th>
      <th>R_Hotels</th>
      <th>R_Hotels &amp; Travel</th>
      <th>R_Ice Cream &amp; Frozen Yogurt</th>
      <th>R_Imported Food</th>
      <th>R_Indian</th>
      <th>R_International</th>
      <th>R_Internet Cafes</th>
      <th>R_Irish</th>
      <th>R_Italian</th>
      <th>R_Japanese</th>
      <th>R_Juice Bars &amp; Smoothies</th>
      <th>R_Karaoke</th>
      <th>R_Kebab</th>
      <th>R_Korean</th>
      <th>R_Kosher</th>
      <th>R_Latin American</th>
      <th>R_Lebanese</th>
      <th>R_Local Flavor</th>
      <th>R_Lounges</th>
      <th>R_Malaysian</th>
      <th>R_Meat Shops</th>
      <th>R_Mediterranean</th>
      <th>R_Mexican</th>
      <th>R_Middle Eastern</th>
      <th>R_Modern European</th>
      <th>R_Music Venues</th>
      <th>R_Nightlife</th>
      <th>R_Noodles</th>
      <th>R_Pakistani</th>
      <th>R_Pan Asian</th>
      <th>R_Party &amp; Event Planning</th>
      <th>R_Patisserie/Cake Shop</th>
      <th>R_Persian/Iranian</th>
      <th>R_Peruvian</th>
      <th>R_Pizza</th>
      <th>R_Poke</th>
      <th>R_Polish</th>
      <th>R_Portuguese</th>
      <th>R_Poutineries</th>
      <th>R_Pubs</th>
      <th>R_Ramen</th>
      <th>R_Restaurants</th>
      <th>R_Salad</th>
      <th>R_Sandwiches</th>
      <th>R_Scottish</th>
      <th>R_Seafood</th>
      <th>R_Seafood Markets</th>
      <th>R_Shopping</th>
      <th>R_Soul Food</th>
      <th>R_Soup</th>
      <th>R_Southern</th>
      <th>R_Spanish</th>
      <th>R_Specialty Food</th>
      <th>R_Sports Bars</th>
      <th>R_Steakhouses</th>
      <th>R_Street Vendors</th>
      <th>R_Sushi Bars</th>
      <th>R_Swabian</th>
      <th>R_Szechuan</th>
      <th>R_Tacos</th>
      <th>R_Taiwanese</th>
      <th>R_Tapas Bars</th>
      <th>R_Tapas/Small Plates</th>
      <th>R_Tea Rooms</th>
      <th>R_Tex-Mex</th>
      <th>R_Thai</th>
      <th>R_Turkish</th>
      <th>R_Vegan</th>
      <th>R_Vegetarian</th>
      <th>R_Venues &amp; Event Spaces</th>
      <th>R_Vietnamese</th>
      <th>R_Waffles</th>
      <th>R_Wine &amp; Spirits</th>
      <th>R_Wine Bars</th>
      <th>R_Wraps</th>
      <th>R_attributes.Ambience.casual</th>
      <th>R_attributes.Ambience.classy</th>
      <th>R_attributes.Ambience.divey</th>
      <th>R_attributes.Ambience.hipster</th>
      <th>R_attributes.Ambience.intimate</th>
      <th>R_attributes.Ambience.romantic</th>
      <th>R_attributes.Ambience.touristy</th>
      <th>R_attributes.Ambience.trendy</th>
      <th>R_attributes.Ambience.upscale</th>
      <th>R_attributes.BikeParking</th>
      <th>R_attributes.BusinessAcceptsCreditCards</th>
      <th>R_attributes.BusinessParking.garage</th>
      <th>R_attributes.BusinessParking.lot</th>
      <th>R_attributes.BusinessParking.street</th>
      <th>R_attributes.BusinessParking.valet</th>
      <th>R_attributes.BusinessParking.validated</th>
      <th>R_attributes.Caters</th>
      <th>R_attributes.GoodForKids</th>
      <th>R_attributes.GoodForMeal.breakfast</th>
      <th>R_attributes.GoodForMeal.brunch</th>
      <th>R_attributes.GoodForMeal.dessert</th>
      <th>R_attributes.GoodForMeal.dinner</th>
      <th>R_attributes.GoodForMeal.latenight</th>
      <th>R_attributes.GoodForMeal.lunch</th>
      <th>R_attributes.HasTV</th>
      <th>R_attributes.OutdoorSeating</th>
      <th>R_attributes.RestaurantsDelivery</th>
      <th>R_attributes.RestaurantsGoodForGroups</th>
      <th>R_attributes.RestaurantsReservations</th>
      <th>R_attributes.RestaurantsTableService</th>
      <th>R_attributes.RestaurantsTakeOut</th>
      <th>R_attributes.WheelchairAccessible</th>
      <th>R_review_count</th>
      <th>R_stars</th>
      <th>U_average_stars</th>
      <th>U_compliment_cool</th>
      <th>U_compliment_cute</th>
      <th>U_compliment_funny</th>
      <th>U_compliment_hot</th>
      <th>U_compliment_list</th>
      <th>U_compliment_more</th>
      <th>U_compliment_note</th>
      <th>U_compliment_photos</th>
      <th>U_compliment_plain</th>
      <th>U_compliment_profile</th>
      <th>U_compliment_writer</th>
      <th>U_cool</th>
      <th>U_elite</th>
      <th>U_fans</th>
      <th>U_friends</th>
      <th>U_funny</th>
      <th>U_review_count</th>
      <th>U_useful</th>
      <th>stars</th>
      <th>U_years_yelping</th>
      <th>R_state_01</th>
      <th>R_state_AZ</th>
      <th>R_state_BW</th>
      <th>R_state_C</th>
      <th>R_state_EDH</th>
      <th>R_state_ELN</th>
      <th>R_state_FIF</th>
      <th>R_state_HLD</th>
      <th>R_state_IL</th>
      <th>R_state_MLN</th>
      <th>R_state_NC</th>
      <th>R_state_NI</th>
      <th>R_state_NV</th>
      <th>R_state_NY</th>
      <th>R_state_NYK</th>
      <th>R_state_OH</th>
      <th>R_state_ON</th>
      <th>R_state_PA</th>
      <th>R_state_PKN</th>
      <th>R_state_QC</th>
      <th>R_state_RCC</th>
      <th>R_state_SC</th>
      <th>R_state_WA</th>
      <th>R_state_WI</th>
      <th>R_state_WLN</th>
      <th>R_attributes.Alcohol_beer_and_wine</th>
      <th>R_attributes.Alcohol_full_bar</th>
      <th>R_attributes.Alcohol_none</th>
      <th>R_attributes.AgesAllowed_18plus</th>
      <th>R_attributes.AgesAllowed_19plus</th>
      <th>R_attributes.AgesAllowed_21plus</th>
      <th>R_attributes.AgesAllowed_allages</th>
      <th>R_attributes.RestaurantsAttire_casual</th>
      <th>R_attributes.RestaurantsAttire_dressy</th>
      <th>R_attributes.RestaurantsAttire_formal</th>
      <th>R_attributes.RestaurantsPriceRange2_1.0</th>
      <th>R_attributes.RestaurantsPriceRange2_2.0</th>
      <th>R_attributes.RestaurantsPriceRange2_3.0</th>
      <th>R_attributes.RestaurantsPriceRange2_4.0</th>
      <th>R_attributes.Smoking_no</th>
      <th>R_attributes.Smoking_outdoor</th>
      <th>R_attributes.Smoking_yes</th>
      <th>R_attributes.WiFi_free</th>
      <th>R_attributes.WiFi_no</th>
      <th>R_attributes.WiFi_paid</th>
      <th>R_attributes.NoiseLevel_average</th>
      <th>R_attributes.NoiseLevel_loud</th>
      <th>R_attributes.NoiseLevel_quiet</th>
      <th>R_attributes.NoiseLevel_very_loud</th>
      <th>R_attributes.BYOBCorkage_no</th>
      <th>R_attributes.BYOBCorkage_yes_corkage</th>
      <th>R_attributes.BYOBCorkage_yes_free</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>143</td>
      <td>3.0</td>
      <td>3.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>0.0217056</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>110</td>
      <td>3.0</td>
      <td>3.23</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>64</td>
      <td>10</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>268</td>
      <td>3.5</td>
      <td>4.10</td>
      <td>11</td>
      <td>0</td>
      <td>11</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>7</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>True</td>
      <td>10</td>
      <td>186</td>
      <td>0</td>
      <td>70</td>
      <td>4</td>
      <td>3</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>0.49104</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>0.917543</td>
      <td>53</td>
      <td>2.5</td>
      <td>4.64</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>1</td>
      <td>21</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>1145</td>
      <td>4.5</td>
      <td>3.06</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>False</td>
      <td>1</td>
      <td>39</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Creating the train and test sets



```python
msk = np.random.rand(len(data_set)) < 0.5
data_train = flatframe_v4[msk]
data_test = flatframe_v4[~msk]
```


## making the models



```python
Xtrain = data_train.drop(['stars'], axis = 1)
ytrain = data_train['stars']
Xtest = data_test.drop(['stars'], axis = 1)
ytest = data_test['stars']

```




```python
ytest
```





    0        1
    2        3
    3        4
    4        5
    7        3
    8        5
    9        4
    14       5
    16       4
    17       5
    19       5
    20       4
    22       5
    24       3
    25       4
    28       5
    29       1
    32       2
    36       5
    39       2
    41       5
    46       3
    48       1
    51       5
    53       1
    56       5
    60       4
    61       4
    62       1
    65       5
            ..
    99942    5
    99943    3
    99946    2
    99947    4
    99949    4
    99950    5
    99952    5
    99953    4
    99957    1
    99958    5
    99960    4
    99961    2
    99962    5
    99964    3
    99968    5
    99971    4
    99972    1
    99974    5
    99975    4
    99976    4
    99978    4
    99986    3
    99988    3
    99989    2
    99990    5
    99994    1
    99995    3
    99996    3
    99997    1
    99998    5
    Name: stars, Length: 49861, dtype: int64





```python
model = LinearRegression()
model.fit(Xtrain, ytrain)
```





    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



### Linear Model



```python
ypred = model.predict(Xtrain)
ypred_test = model.predict(Xtest)
predround = [int(round(x)) for x in ypred]
print ("The accuracy score of the linear model on the train set is {}"
       .format(metrics.accuracy_score(ytrain, predround)))
predround_test = [int(round(x)) for x in ypred_test]
print ("The accuracy score of the linear model on the test set is {}"
       .format(metrics.accuracy_score(ytest, predround_test)))
```


    The accuracy score of the linear model on the train set is 0.385428508745687
    The accuracy score of the linear model on the test set is 0.380036501474098


### Ridge CV



```python
model_ridge = RidgeCV().fit(Xtrain, ytrain)
```




```python
ridge_ypred = model_ridge.predict(Xtrain)
ridge_ypred_round = [int(round(x)) for x in ridge_ypred]
ridge_ypred_test = model_ridge.predict(Xtest)
ridge_ypred_test_round = [int(round(x)) for x in ridge_ypred_test]
```




```python

print ("The accuracy score of the ridge model on the train set is {}"
       .format(metrics.accuracy_score(ytrain, ridge_ypred_round)))
print ("The accuracy score of the ridge model on the test set is {}"
       .format(metrics.accuracy_score(ytest, ridge_ypred_test_round)))
```


    The accuracy score of the ridge model on the train set is 0.385508286962245
    The accuracy score of the ridge model on the test set is 0.38023705902408694


### Lasso CV



```python
model_lasso = LassoCV().fit(Xtrain, ytrain)
```




```python
lasso_ypred = model_lasso.predict(Xtrain)
lasso_ypred_round = [int(round(x)) for x in lasso_ypred]
lasso_ypred_test = model_lasso.predict(Xtest)
lasso_ypred_test_round = [int(round(x)) for x in lasso_ypred_test]
```




```python
print ("The accuracy score of the lasso model on the train set is {}"
       .format(metrics.accuracy_score(ytrain, lasso_ypred_round)))
print ("The accuracy score of the lasso model on the test set is {}"
       .format(metrics.accuracy_score(ytest, lasso_ypred_test_round)))
```


    The accuracy score of the lasso model on the train set is 0.3619138794152257
    The accuracy score of the lasso model on the test set is 0.3625679388700588


### Baseline Model



```python
base_df = flatframe[['stars', 'business_id', 'user_id']]
```




```python
train_base = base_df[msk]
test_base = base_df[~msk]
```




```python
base_pred = [baseline(x,y) for x,y in zip(train_base['user_id'],train_base['business_id'])]
base_pred_test = [baseline(x,y) for x,y in zip(test_base['user_id'],test_base['business_id'])]
```




```python
print ("The accuracy score of the baseline model on the train set is {}"
       .format(metrics.accuracy_score(ytrain, base_pred)))
print ("The accuracy score of the baseline model on the test set is {}"
       .format(metrics.accuracy_score(ytest, base_pred_test)))
```


    The accuracy score of the baseline model on the train set is 0.37657312670775245
    The accuracy score of the baseline model on the test set is 0.37452116884940134




```python
base_pred_2 = [baseline2(x,y) for x,y in zip(train_base['user_id'],train_base['business_id'])]
base_pred_test_2 = [baseline2(x,y) for x,y in zip(test_base['user_id'],test_base['business_id'])]
print ("The accuracy score of the baseline2 model on the train set is {}"
       .format(metrics.accuracy_score(ytrain, base_pred_2)))
print ("The accuracy score of the baseline2 model on the test set is {}"
       .format(metrics.accuracy_score(ytest, base_pred_test_2)))
```


    The accuracy score of the baseline2 model on the train set is 0.38275593849099504
    The accuracy score of the baseline2 model on the test set is 0.37971560939411564




```python

```

