# Yelp Recommendation System
This repository consists of the scripts required to train and predict the ratings that a user might provide for any new business on Yelp. The training script `train.py` provides the user with the option of training the Yelp data on three different model options (User based Collaborative Filtering, Alternating Least Squares method using the Scikit-surprise library, and XGBoost) and the performance of these models are evaluated using the prediction script `predict.py` The Usage examples are provided below for reference.

Apache Spark is used for the Big data processing. The code runs in the assumtion that Apache Spark is pre-installed.

## Usage:

Step 1:
Download the required packages:
```sh
$ python3 -m pip install -r requirements.txt
```
Step 2:
To download the training data:
```sh
$ bash download_data.sh
```
Step 3:
To train the models on previously constructed datasets:
```sh
$ spark-submit train.py -tr <Training file path.json> -ts <Test file path.json> -o <output trained model file path.model> -m <Training method [user_based_cf, surprise, xgboost]>
```
Step 4:
To compute the predictions and calculate the RMSE of the predicted models:
```sh
$ spark-submit predict.py -tr <Training file path.json> -ts <Test file path.json> -md <trained model file path.model> -o <output prediction file path> -m <Training method [user_based_cf, surprise, xgboost]>
```