################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")


# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# End of the provided code



# Data overview
head(edx)
str(edx)
summary(edx)

# Unique count of movies, users and genres
n_distinct(edx$movieId)
n_distinct(edx$userId)
n_distinct (edx$genres)


# Ratings count
edx %>% group_by(rating) %>% summarize(count=n()) %>% 
  arrange(desc(rating))


# plot
edx %>% group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_bar(stat = "identity")

# 4. Methods & Analysis - Model Building & Results:

# Create test dataset and train dataset
# set.seed
set.seed(1)
test_ind <- createDataPartition(y = edx$rating, times = 1, p = .1, list=FALSE)
train_ds <- edx[-test_ind,]
test_ds_temp <- edx[test_ind,]

# Make sure userId and movieId in test dataset are also in train dataset
test_ds <- test_ds_temp %>% 
  semi_join(train_ds, by = "movieId") %>%
  semi_join(train_ds, by = "userId")

# Add rows removed from test_ds_temp dataset back into train dataset
rmvd <- anti_join(test_ds_temp, test_ds)
train_ds <- rbind(train_ds, rmvd)


# Model.1: Computing predicted ratings based on the average of population
mu_simple <- mean(train_ds$rating)
mu_simple

mod1_rmse <- RMSE(test_ds$rating, mu_simple)
mod1_rmse



# Model.2: Computing predicted ratings based on movie effects and user effects
mu <- mean(train_ds$rating)

# Add movie effect
movie_effects <- train_ds %>% 
  group_by(movieId) %>% 
  summarize(m_effect = mean(rating - mu))

movie_effects %>% qplot(m_effect, geom ="histogram", bins = 30, data = ., color = I("black"))


# Add user effect
user_effects <- train_ds %>% 
  left_join(movie_effects, by='movieId') %>%
  group_by(userId) %>%
  summarize(u_effect = mean(rating - mu - m_effect))

user_effects %>% qplot(u_effect, geom ="histogram", bins = 30, data = ., color = I("black"))

# Predict ratings
pred_mod2 <- test_ds %>% 
  left_join(movie_effects, by='movieId') %>%
  left_join(user_effects, by='userId') %>%
  mutate(pred = mu + m_effect + u_effect) 

mod2_rmse <- RMSE(test_ds$rating, pred_mod2$pred)
mod2_rmse

head(pred_mod2)
max(pred_mod2$pred)
min(pred_mod2$pred)


# Adjusting the range of possible values (0~5)
pred_mod2$pred[pred_mod2$pred > 5] <- 5
max(pred_mod2$pred)

pred_mod2$pred[pred_mod2$pred < 0] <- 0
min(pred_mod2$pred)

head(pred_mod2)
max(pred_mod2$pred)
min(pred_mod2$pred)

mod2_rmse <- RMSE(test_ds$rating, pred_mod2$pred)
mod2_rmse


# Model.3: Computing predicted ratings based on movie, user & genre effects
# Add genre effect
genre_effects <- train_ds %>% 
  left_join(movie_effects, by='movieId') %>%
  left_join(user_effects, by='userId') %>%
  group_by(genres) %>%
  summarize(g_effect = mean(rating - mu - m_effect - u_effect))

genre_effects %>% qplot(g_effect, geom ="histogram", bins = 30, data = ., color = I("black"))

# Predict ratings
pred_mod3 <- test_ds %>% 
  left_join(movie_effects, by='movieId') %>%
  left_join(user_effects, by='userId') %>%
  left_join(genre_effects, by='genres') %>%
  mutate(pred = mu + m_effect + u_effect + g_effect) 

mod3_rmse <- RMSE(test_ds$rating, pred_mod3$pred)

mod3_rmse

head(pred_mod3)
max(pred_mod3$pred)
min(pred_mod3$pred)



# Adjusting the range of possible values (0~5)
pred_mod3$pred[pred_mod3$pred > 5] <- 5
max(pred_mod3$pred)

pred_mod3$pred[pred_mod3$pred < 0] <- 0
min(pred_mod3$pred)

head(pred_mod3)

mod3_rmse <- RMSE(test_ds$rating, pred_mod3$pred)

mod3_rmse




# Convert timestamp to year (Sample code below)
unixtime = 1459995330
format(as.POSIXct(unixtime, origin="1970-1-1"), format="%Y%m")

# Model.4.1: Computing predicted ratings based on movie, user, genre and year effects
# Add year effect
time_effects <- edx %>%
  left_join(movie_effects, by='movieId') %>%
  left_join(user_effects, by= 'userId') %>%
  left_join(genre_effects, by='genres') %>%
  mutate(timestamp_year = format(as.POSIXct(timestamp, origin="1970-1-1"), format="%Y")) %>%
  group_by(timestamp_year) %>%
  summarize(t_effect = mean(rating - mu - m_effect - u_effect -  g_effect))


# Predict ratings
pred_mod4_1 <- test_ds %>%
  left_join(movie_effects, by='movieId') %>%
  left_join(user_effects, by= 'userId') %>%
  left_join(genre_effects, by='genres') %>%
  mutate(timestamp_year = format(as.POSIXct(timestamp, origin="1970-1-1"), format="%Y")) %>%
  left_join(time_effects, by= 'timestamp_year') %>%
  mutate(pred = mu + m_effect + u_effect + g_effect + t_effect) 


mod4_1_rmse <- RMSE(test_ds$rating, pred_mod4_1$pred)

mod4_1_rmse


pred_mod4_1$pred[pred_mod4_1$pred > 5] <- 5
max(pred_mod4_1$pred)

pred_mod4_1$pred[pred_mod4_1$pred < 0] <- 0
min(pred_mod4_1$pred)


mod4_1_rmse <- RMSE(test_ds$rating, pred_mod4_1$pred)

mod4_1_rmse




# Model.4.2: Computing predicted ratings based on movie, user, genre and month effects
# Add year effect
time_effects <- edx %>%
  left_join(movie_effects, by='movieId') %>%
  left_join(user_effects, by= 'userId') %>%
  left_join(genre_effects, by='genres') %>%
  mutate(timestamp_year = format(as.POSIXct(timestamp, origin="1970-1-1"), format="%Y%m")) %>%
  group_by(timestamp_year) %>%
  summarize(t_effect = mean(rating - mu - m_effect - u_effect -  g_effect))


# Predict ratings
pred_mod4_2 <- test_ds %>%
  left_join(movie_effects, by='movieId') %>%
  left_join(user_effects, by= 'userId') %>%
  left_join(genre_effects, by='genres') %>%
  mutate(timestamp_year = format(as.POSIXct(timestamp, origin="1970-1-1"), format="%Y%m")) %>%
  left_join(time_effects, by= 'timestamp_year') %>%
  mutate(pred = mu + m_effect + u_effect + g_effect + t_effect) 


mod4_2_rmse <- RMSE(test_ds$rating, pred_mod4_2$pred)

mod4_2_rmse



pred_mod4_2$pred[pred_mod4_2$pred > 5] <- 5
max(pred_mod4_2$pred)

pred_mod4_2$pred[pred_mod4_2$pred < 0] <- 0
min(pred_mod4_2$pred)


mod4_2_rmse <- RMSE(test_ds$rating, pred_mod4_2$pred)

mod4_2_rmse








# Model.6.3: Computing predicted ratings based on movie, user, genre and day effects
# Add year effect
time_effects <- edx %>%
  left_join(movie_effects, by='movieId') %>%
  left_join(user_effects, by= 'userId') %>%
  left_join(genre_effects, by='genres') %>%
  mutate(timestamp_year = format(as.POSIXct(timestamp, origin="1970-1-1"), format="%Y%m%d")) %>%
  group_by(timestamp_year) %>%
  summarize(t_effect = mean(rating - mu - m_effect - u_effect -  g_effect))


# Predict ratings
pred_mod4_3 <- test_ds %>%
  left_join(movie_effects, by='movieId') %>%
  left_join(user_effects, by= 'userId') %>%
  left_join(genre_effects, by='genres') %>%
  mutate(timestamp_year = format(as.POSIXct(timestamp, origin="1970-1-1"), format="%Y%m%d")) %>%
  left_join(time_effects, by= 'timestamp_year') %>%
  mutate(pred = mu + m_effect + u_effect + g_effect + t_effect) 


mod4_3_rmse <- RMSE(test_ds$rating, pred_mod4_3$pred)

mod4_3_rmse


pred_mod4_3$pred[pred_mod4_3$pred > 5] <- 5
max(pred_mod4_3$pred)

pred_mod4_3$pred[pred_mod4_3$pred < 0] <- 0
min(pred_mod4_3$pred)

mod4_3_rmse <- RMSE(test_ds$rating, pred_mod4_3$pred)

mod4_3_rmse
# 0.8631282


# Testing the final model on the Validation dataset
predicted_val <- validation %>%
  left_join(movie_effects, by='movieId') %>%
  left_join(user_effects, by='userId') %>%
  left_join(genre_effects, by= 'genres') %>%
  mutate(timestamp_year = format(as.POSIXct(timestamp, origin="1970-1-1"), format="%Y%m%d")) %>%
  left_join(time_effects, by= 'timestamp_year') %>%
  mutate(pred = mu + m_effect + u_effect + g_effect + t_effect) 

# Adjusting the range of possible values (0~5)

predicted_val$pred[predicted_val$pred > 5] <- 5
max(predicted_val$pred)

predicted_val$pred[predicted_val$pred < 0] <- 0
min(predicted_val$pred)

# Computing final RMSE
final_model_rmse <- RMSE(predicted_val$pred, validation$rating)
final_model_rmse
# 0.8647139


# 25 points: RMSE <= 0.8649