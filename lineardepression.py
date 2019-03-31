import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import csv


#load test,train and movie data
movies = pd.read_csv('movies.csv')
train_ratings = pd.read_csv('train_ratings.csv')
test_ratings = pd.read_csv('test_ratings.csv')

#convert the movie information to a dataframe indexed by movieId that contains a binary encoding of the genres of each movie
train_movie_genre = pd.concat([movies, movies.genres.str.get_dummies(sep='|')], axis=1)
train_movie_genre = train_movie_genre.drop(['title', 'genres', '(no genres listed)'], axis=1)
train_movie_genre = train_movie_genre.set_index('movieId')

#create dataframe that relates the rating of each movie to the genre encoding previously generated
train_rating_genre = pd.merge(train_ratings, train_movie_genre, on='movieId')
train_rating_genre = train_rating_genre.drop(['movieId', 'userId'], axis=1)

#split the dataframe into independent and dependent variables for training set
train_y = train_rating_genre['rating'].values
train_x = train_rating_genre.drop('rating', axis = 1).values

#do the same with test data
test_data_x = pd.merge(test_ratings, train_movie_genre, on='movieId')
test_x = test_data_x.drop(['Id', 'userId', 'movieId'], axis=1)


#generate the linear regression model
lineardepression = LinearRegression()
lineardepression.fit(train_x,train_y)

#predict the ratings for training and test data
pred_y = lineardepression.predict(test_x.values)
pred_y_train = lineardepression.predict(train_x)

#manipuate the dataframes to generate the right dataframes to analyze the RMSE
pred_y_train = pd.DataFrame(pred_y_train,columns=['Predicted'])
out_train = pd.concat((train_ratings,pred_y_train),axis=1)


pred_y = pd.DataFrame(pred_y,columns=['rating'])
out_test = pd.concat((test_ratings,pred_y),axis=1)

pred_y = pd.concat((pred_y,test_data_x), axis = 1)
pred_y = pred_y[['Id','rating']]
pred_y = pred_y.sort_values(by=['Id'])

#write output to csv files
pred_y.to_csv (r'/Users/amlanbose/Desktop/submission.csv', index = None, header=True)
out_test.to_csv (r'/Users/amlanbose/Desktop/testdata_predictions.csv', index = None, header=True)
out_train.to_csv (r'/Users/amlanbose/Desktop/traindata_predictions.csv', index = None, header=True)





