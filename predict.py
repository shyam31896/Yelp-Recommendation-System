import sys
import time
import json
import math
import random
import joblib
import xgboost
import surprise
import argparse
import pandas as pd
from operator import add
from itertools import combinations
from pyspark import SparkContext, SparkConf

parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--train_fp", help="Training data file path", type=str, default="")
parser.add_argument("-ts", "--test_fp", help="Testing data file path", type=str, default="")
parser.add_argument("-md", "--model_fp", help="Trained model file path", type=str, default="")
parser.add_argument("-o", "--output_fp", help="Output file path", type=str, default="")
parser.add_argument("-m", "--method", help="Training method [user_based_cf, surprise, xgboost]", type=str, default="")
args = parser.parse_args()
input_train_file_path, model_file_path, output_file_path, method = args.train_fp, args.model_fp, args.output_fp, args.method
input_test_file_path = args.test_fp if args.test_fp else ""
conf = SparkConf().setMaster('local[*]').setAppName('recsys_predict').set("spark.driver.memory", "4g").set("spark.executor.memory", "4g")
spark = SparkContext(conf=conf).getOrCreate()
spark.setLogLevel("ERROR")
start_time = time.time()

def average(data):
    return sum(list(data)) / len(list(data))

def return_dict(index, method, inv = False):
    train_items = train_inp.map(lambda data: data[index]).distinct().collect()
    if method == "user_based_cf":
        return {k:v for k,v in enumerate(train_items)} if inv else {v:k for k,v in enumerate(train_items)}
    else:
        test_items = test_inp.map(lambda data: data[index]).distinct().collect()
        return {k:v for k,v in enumerate(train_items+test_items)} if inv else {v:k for k,v in enumerate(train_items+test_items)}

def update_dict(dict_list):
    d = {}
    for item in dict_list:
        d[item[0]] = max(item[1], d[item[0]]) if item[0] in d.keys() else item[1]
    return d

def get_cf_model(data):
    return (users[data["u1"]], (users[data["u2"]], data["sim"]))

def get_prediction(data1, data2, tokens, n):
    if data2 not in model_dict:
        return None
    num, den, i = 0, 0, 0
    for item2, similarity in model_dict[data2]:
        if item2 in tokens:
            if data1 in tokens[item2]:
                r = tokens[item2][data1] - user_average_dict[item2]
                num, den, i = num + r * similarity, den + abs(similarity), i + 1
            if i > n:
                break 
    if den != 0:
        rating = (num / den) + user_average_dict[data2]
        return (inv_users[data2], inv_business[data1], rating)
    else:
        return None

def create_json_dict(key):
    f = open(user_avg_fp)
    data = json.load(f)
    f.close()
    return [{str(key):k, "avg_stars":v} for k, v in data.items()]

def test_xgb(xgb_model):
    test_df_xgb, test_feats = pd.read_json(input_test_file_path, lines = True), []
    for _, item in test_df_xgb.iterrows():
        user_useful = user_df[item["user_id"]][0] if str(item["user_id"]) in user_df else 0.0
        user_avgstars = user_df[item["user_id"]][1] if str(item["user_id"]) in user_df else 0.0
        business_latitude = business_df[item["business_id"]][0] if str(item["business_id"]) in business_df else 0.0
        business_longitude = business_df[item["business_id"]][1] if str(item["business_id"]) in business_df else 0.0
        business_avgstars = business_df[item["business_id"]][2] if str(item["business_id"]) in business_df else 0.0
        test_feats.append([user_useful, user_avgstars, business_latitude, business_longitude, business_avgstars])
    predict = xgb_model.predict(test_feats)
    return predict
        
def write_output(triple_list):
    with open(output_file_path, 'w') as f:
        [f.write('{"user_id": "' + str(triple[0]) + '", "business_id": "' + str(triple[1]) + '", "stars": ' + str(triple[2]) + '}\n') for triple in triple_list]
    f.close()

train_inp = spark.textFile(input_train_file_path).map(json.loads).map(lambda data: (data["user_id"], data["business_id"], data["stars"]))
if method == "user_based_cf":
    users = return_dict(0, method)
    business = return_dict(1, method)
    inv_users = return_dict(0, method, True)
    inv_business = return_dict(1, method, True)
    model_dict = spark.textFile(model_file_path).map(json.loads).map(lambda data: get_cf_model(data)).groupByKey().mapValues(set).collectAsMap()
    test_tokens = spark.textFile(input_test_file_path).map(json.loads).map(lambda data: (users.get(str(data["user_id"]), None), business.get(str(data["business_id"]), None)))
    test_tokens = test_tokens.filter(lambda data: data[0] is not None).filter(lambda data: data[1] is not None)
    user_tokens = train_inp.map(lambda data: (users[data[0]], (business[data[1]], data[2]))).groupByKey().mapValues(list)
    user_tokens = user_tokens.map(lambda data: (data[0], update_dict(data[1]))).collectAsMap()
    user_average_dict = train_inp.map(lambda data: (users[data[0]], data[2])).groupByKey().mapValues(average).collectAsMap()
    predict = test_tokens.map(lambda data: get_prediction(data[1], data[0], user_tokens, 11))
    predict = predict.filter(lambda data: data is not None).collect()
    write_output(predict)
elif method == "surprise":
    test_inp = spark.textFile(input_test_file_path).map(json.loads).map(lambda data: (data["user_id"], data["business_id"]))
    users = return_dict(0, method)
    business = return_dict(1, method)
    inv_users = return_dict(0, method, True)
    inv_business = return_dict(1, method, True)
    test_df = test_inp.map(lambda data: (users[data[0]], business[data[1]], 0)).collect()
    model = joblib.load(model_file_path)
    predict = model.test(test_df)
    predict_out = spark.parallelize(predict).map(lambda x: (inv_users[x.uid], inv_business[x.iid], x.est)).collect()
    write_output(predict_out)
elif method == "xgboost":
    user_fp = "data/user.json"
    user_avg_fp = "data/user_avg.json"
    business_fp = "data/business.json"
    business_avg_fp = "data/business_avg.json"
    test_inp = spark.textFile(input_test_file_path).map(json.loads).map(lambda data: (data["user_id"], data["business_id"]))
    train_df = pd.DataFrame(train_inp.map(lambda data: (data[0], data[1], data[2])).collect())
    train_df.columns = ["user_id", "business_id", "rating"]
    test_df = test_inp.map(lambda data: (data[0], data[1])).collect()
    user_df1 = spark.textFile(user_fp).map(json.loads).map(lambda data: (data["user_id"], (data["useful"])))
    user_avg_df = spark.parallelize(create_json_dict("user_id")).map(lambda data: (data["user_id"], (data["avg_stars"])))
    business_df1 = spark.textFile(business_fp).map(json.loads).map(lambda data: (data["business_id"], (data["latitude"], data["longitude"])))
    business_avg_df = spark.parallelize(create_json_dict("business_id")).map(lambda data: (data["business_id"], (data["avg_stars"])))
    user_df = user_df1.leftOuterJoin(user_avg_df)
    business_df = business_df1.leftOuterJoin(business_avg_df)
    user_df = user_df.mapValues(lambda data: (data[0], data[1] if data[1] is not None else 0.0)).collectAsMap()
    business_df = business_df.mapValues(lambda data: (data[0][0], data[0][1], data[1] if data[1] is not None else 0.0)).collectAsMap()
    predict_xgb = test_xgb(joblib.load(model_file_path))
    xgb_out = [(item1[0], item1[1], item2) for item1, item2 in zip(test_df, predict_xgb)]
    write_output(xgb_out)
else:
    print("Wrong method option entered...!!!")
print("Running time = " + str(round(time.time() - start_time, 2)) + "s")
spark.stop()