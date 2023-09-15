# Netflik Film Recommendation System
Em Jager and Sam Whitehurst

### Overview
For many streaming services there is an overwhelming amount of content. Customers may search fruitlessly through thousands of titles only to pick a movie they ultimately don't enjoy. When the initial search begins to feel inconvenient to a customer, Netfilk Film Corp is at risk of losing that client's interest and eventually their business. We designed a predictive algorithm will learn the preferences of each user and build them a profile, which can then be compared to users with similar profiles who have watched and rated some of the same and other similar films. Using the'surprise' package from sklearn with a algorithmic technique called Collaborative Filtering, our system will make recommendations to each user for new, unwatched films based on how highly similar users have rated that film. To maximize convenience, with the customers valuable time in mind, our system is designed to deliever 5 tailor-made recommendations, thus eliminating the need to scroll through endless content to find a film they may enjoy.

### Business Problem
The stakeholders are the new shareholders at the Netflik Film Corp, as they've invested in this new online streaming service. The streaming service provides a personalized experience for every customer by employing the most advanced algorithms and techniques for recommending only the films that are most likely to interest each user. This system builds profiles for each user and then identifies the particular interestes of each user based on the ratings the user provides (on a scale from 0.5 - 5) for each movie they watch. We theorize that similar user profiles will have similar tastes, therefore if our system recognizes that a customer has not seen a film that similar users have rated highly, then our system will recommend that the user watches that film.

We have distilled the search process from thoousands of titles to 5, which allows the user an efficient way to select a movie they are likely to enjoy. Of course, there is the possibility that the user does not like the film they choose from our system's recommendation list, in which case we highly encourage that user to rate that film fairly (and lower) so our system can better adapt to their taste. Because our system relies so heavily on user ratings for input, we have made the rating process as simple as pushing a button as soon as the user finishes their film - that way we our recommendation system can learn from as much data as possible about a user. To address the cold-issue that accompanies a system like this, upon starting a subscription, we ask that all new users rate popular films that tey have seen, to give our system a jump start on making custom recommendations.

In order to measure the success of our model, we iterated through many options, tuning specifc hyperparameters along the way, until we achieved the lowest RMSE (Root Mean Squared Error). We chose this metric because it is easily interpreted as a value in the same units (within the rating scale) and can be measured for all of the model types that we built. For instance a RMSE of 1.0 would imply that a model consistently predicts a user's rating of an unseen film within 1.0 rating point of the actual rating the user would give the film. The 'surprise' package includes many model options: we established a baseilne RMSE score using the NormalPredictor model which we then compared to model iterations generated using more robust models like Singular Value Decomposition (SVD), NMF, and SVDpp model types. Ultimately, we detemrined that the SVD model with optimized paramters (detailed below) resulted in the lowest RMSE, and therefore is the best model to employ to make our predictions. This model was input into our custom recommendation function which generates a tailored list of 5 film recommendations, none of which should have already been viewed by the user.

### Data Understanding
The data for this project comes from the MovieLens dataset provided by the grouplens website. Specifically, we are working with the smaller dataset available on the website, which contains 4 csv files detailing user ratings for a variety of movies. We worked with two of the csv files for our project: the 'ratings.csv' file and the 'movies.csv' file, but we explored each of the four options first to determine which would be needed for designing our recommendation system. Here is a brief description of each of the files available:

The 'ratings.csv' file contains just over 100,000 user rating records from 610 unique users. This is the primary resource we used when biulding our recommendation system. Among the 610 unique users, there is a wide range of inputs, with the highest total number of films rated reaching 2698 and the lowest being 20. For users with more ratings, our system was better able to recognize preferences, leading to recommendations that are more likely to appeal to the user.

### Modeling
We began the modeling phase of our project by establishing a baseline model that we can later use for comparison when we generate RMSE values for each new model iteration. The baseline model we chose is the NormalPrdictor option from the 'surprise' package and we expect it will not perform too well when modeling the data, as it does not take into account some of the more complex algorithmic functions that the later models employ. We chose three other model types to explore after analyzing our baseline results, they are: the SVD model, an NMF model and finally the SVDpp model. For each model type we attempted to optimize the model's hyperparameters via a GridSearch with cross validation. This allowed us to test many hyperparamter options more efficiently than if we had to instantite and build a totally new model every time we wanted to test a specific set of hyperparameters.

*The only downside to this approach to model optimization, is that grid-searching can be a very slow process to run, which can feel like it is slowing project progress, when it is actually still saving time compared to the manual approach for building each and every unique set of model hyperparameter options. GridSearchCV also has built in methods that allow us to see which parameters are optimal, in other words, which set of parameters generate the lowest RMSE for each model type. For certain model types we chose to run additional grid searches, which we have justified below - All done in hopes of building the best model as indicated by the RMSE score.

We will begin the cross validation process by splitting our data using the train_test_split function specific to the 'surprise' package. Because we do not have a 'target' for this type of model, we do not need to specify a 'y' value, rather we just need to set aside some data so the model can be tested when introduced to 'new' data.

### Baseline Model Placeholder
For the baseline model we used NormalPredictor, a simple baseline algorithm that predicts ratings for items by drawing random values from a normal distribution. We then can compare the RMSE of the NormalPredictor algorithm to our tuned, more sophisticated algorithms, and if our models' RMSE and MAE are lower, we know that the models are performing better than random chance.

### SVD: First Complex Model
We chose to then use the Singular Value Decompostion model type to begin our more-complex recommendation system, because an SVD model is categorized as a latent factor model that generates recommendations using specifc user inputs (ratings) tied to specifc items (movies in this case). The SVD model performs matrix factorization without the need for additional setup other than feeding the model a dataset simply consisting of the user identifier (userId), the item identifier (movieId), and the user rating input (rating). Using several iterations of GridSearchCV, we tuned our SVD model and ended up with a final model with an RMSE of 0.85, meaning it can predict movie ratings within 0.85 points. 

### Recommendation System
The next step in this project is to build a function that takes in a the optimal model's predictions and creates a list of all the movies that a user has not seen and then predicts the user's rating for these films. The function then produces the top 5 movies sorted by the highest predicted rating. 

Before the step-by-step function building process, we need to set the 'best_model' variable equal to the model with the lowest RMSE from above.
#### Function Testing: 

- The function appears to cleaning and easily take in a userId as input and then generate a list of the top 5 movies that the model predicts this user will rate the highest.

- Below we've generated a few user recommendations to explore their differences and hyptohesize about why these differneces may have occured.
-
- #### Function Evaluation:

- Each user has a unique set of 5 recommended movies. This optimal model has learned what a user prefers via their ratings and compared their compiled profile to that of similar users in order to predict a rating for an unwatched movie that our system is recommending to them.

- #### Function(s) Evaluations:
- The recommender function works well to generate a list of 5 movies that a user shoul denjoy according to the similar taste of comparable user profiles.
- The profile function acts to help us check if the recommendations would fit for the user. For instance if a user appears to enjoy a certain type of film genre and rates those movies highly often, then we would expect to see that film genre to be in the recommendations list!

- 
### User Examples

### Next Steps
If we were to continue working on this project, we would like to add a content-based model to work in conjunction with our existing model that works by recommending movies to users based on the content of the movies such as genre, runtime, and themes. So rather than relying on other user’s data for comparison, this model would make recommendations based on previously highly rated content that the user watched. Also, we would like to incentive users of Netflik to rate the films they watch, because the more ratings the system has for each user, the better job it can do at predicting the next movies that they will enjoy. Finally we would like to illicit feedback from users regarding their perception of the 5-film recommendation model [ like if the user would prefer 10 options rather then 5 ] so we can fine tune our system to provide the best and easiest movie watching experience for our users. 

