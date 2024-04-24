from yelpapi import YelpAPI
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk 

## Code will not run without a api_key for security reasons have chosen to ommit my key. 
## simply go to yelpAPi to log in and recieve a personal one. then paste key in the quotes of the apikey variable. 
api_key = ""
yelp_api_instance = YelpAPI(api_key)
search_term = 'pizza'
location_term = 'El Paso, Texas'
analyser = SentimentIntensityAnalyzer()

## Setting limit to 4 to prevent over use of Yelp API 
## searching for pizza in el paso based on ratings. 
search_results = yelp_api_instance.search_query(
    term = search_term, location = location_term, 
    sort_by = 'rating', limit = 15
)

# Setting arrays for simple iteration to gather reviews by id and identify by name in later for loop
id=[]
name=[]
final_list= []

#Populating arrays id and name for uniform iteration of reviews. 
for buisness in search_results['businesses']:
    id.append(buisness['id'])
    name.append(buisness['name'])

#using range len to access corresponding values found in list name and id
for i in range(len(id)):
    #Capturing id in temp variable for easy understanding.
    id_for_reviews = id[i]
    reviews_response = yelp_api_instance.reviews_query(id=id_for_reviews)
    #iterating through reviews to analyse and storing them into a list to create a dataframe
    for review in reviews_response['reviews']:
        sent = analyser.polarity_scores(review['text'])
        store_name = (name[i])
        reviewt = (review['text'])
        #tempvalue 'a' for appending to final list, multiplying each value by 100 allows us to read the values as percentages.
        a = [store_name,reviewt,sent['neg']*100,sent['pos']*100,sent['neu']*100, sent['compound']]
        final_list.append(a)

review_sentiment = pd.DataFrame(final_list, columns = ['Store Name', 'Review Text','Percent Negative','Percent Positive'
                                                       ,'Percent Neutral',' Compound'])
print(review_sentiment)

review_sentiment.to_csv('review_sentiment.csv',index= False)