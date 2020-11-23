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
parser.add_argument("-o", "--model_fp", help="Output model file path", type=str, default="")
parser.add_argument("-m", "--method", help="Training method [user_based_cf, surprise, xgboost]", type=str, default="")
args = parser.parse_args()
input_train_file_path, input_test_file_path, model_file_path, method = args.train_fp, args.test_fp, args.model_fp, args.method
conf = SparkConf().setMaster('local[*]').setAppName('recsys_train').set("spark.driver.memory", "4g").set("spark.executor.memory", "4g")
spark = SparkContext(conf=conf).getOrCreate()
spark.setLogLevel("ERROR")
start_time = time.time()

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

def rm_uncorated(dict1, dict2, num):
    return True if len(set(dict1.keys()).intersection(set(dict2.keys()))) >= num else False

def gen_minhash_rows(hash_coeffs, rows, m):
    return [min([((coeff[0] * row + coeff[1]) % m) for row in rows]) for coeff in hash_coeffs]

def lsh(user_ind, business_ind, b, r):
    return [((i, tuple(business_ind[i*r:(i+1)*r])), user_ind) for i in range(b)]

def jaccard(data, user_dict):        
    if len(set(user_dict[data[0]].keys()).intersection(set(user_dict[data[1]].keys()))) / len(set(user_dict[data[0]].keys()).union(set(user_dict[data[1]].keys()))) >= t:
        return ((data[0], data[1]), 1)
    else:
        return ((data[0], data[1]), 0)

def normalize(ratings):
    avg_rating = sum(ratings) / len(ratings)
    return list(map(lambda data: data - avg_rating, ratings))

def pearson_corr(dict1, dict2):
    corated_users = list(set(dict1.keys()).intersection(set(dict2.keys())))
    rating1, rating2 = [], []
    for key in corated_users:
        rating1.append(dict1[key])
        rating2.append(dict2[key])
    rating1, rating2 = normalize(rating1), normalize(rating2)
    num, den1, den2 = 0, 0, 0
    for i in range(len(rating1)):
        num += rating1[i] * rating2[i]
        den1 += rating1[i] ** 2
        den2 += rating2[i] ** 2
    den = (den1 ** 0.5) * (den2 ** 0.5)
    return num / den if num > 0 else 0

def get_surprise_model(train_df):
    reader = surprise.Reader(rating_scale = (train_df["rating"].min(), train_df["rating"].max()))
    surprise_inp = surprise.Dataset.load_from_df(train_df, reader).build_full_trainset()
    model = surprise.BaselineOnly(bsl_options = {'method': 'als', 'n_epochs': 50, 'reg_u': 12, 'reg_i': 5})
    model.fit(surprise_inp)
    return model

def create_json_dict(key1):
    f = open(user_avg_fp)
    data = json.load(f)
    f.close()
    out = [{str(key1):k, "avg_stars":v} for k, v in data.items()]
    return out

def train_xgb(train_df, user_df, business_df):
    train_feats, train_labels = [], []
    for _, item in train_df.iterrows():
        user_useful, user_avgstars = user_df[item["user_id"]][0], user_df[item["user_id"]][1]
        business_latitude = business_df[item["business_id"]][0]
        business_longitude = business_df[item["business_id"]][1]
        business_avgstars = business_df[item["business_id"]][2]
        train_feats.append([user_useful, user_avgstars, business_latitude, business_longitude, business_avgstars])
        train_labels.append(item["rating"])
    xgb_model = xgboost.XGBRegressor(n_estimators=1000, max_depth=8, nthread=20)
    xgb_model.fit(train_feats, train_labels)
    return xgb_model

def write_output(out_triples, output_file_path):
    keys = ['{"u1": "', '", "u2": "', '", "sim": ']
    with open(output_file_path, 'w') as f:
        [f.write(str(keys[0]) + str(i[0]) + str(keys[1]) + str(i[1]) + str(keys[2]) + str(i[2]) + '}\n') for i in out_triples]
    f.close()

train_inp = spark.textFile(input_train_file_path).map(json.loads).map(lambda data: (data["user_id"], data["business_id"], data["stars"]))
if method == "user_based_cf":
    users = return_dict(0, method)
    business = return_dict(1, method)
    num_hashes, t, b = 50, 0.01, 50
    r = int(num_hashes / b)
    hash_coeffs = [(random.randint(0, 1000), random.randint(0, 1000)) for _ in range(num_hashes)]
    user_tokens = train_inp.map(lambda data: (str(data[0]), (business[data[1]], data[2]))).distinct().groupByKey().mapValues(list)
    user_tokens = user_tokens.filter(lambda data: len(data[1]) >= 3).map(lambda data: (data[0], update_dict(data[1])))
    user_dict = user_tokens.collectAsMap()
    user_hash = user_tokens.map(lambda data: (data[0], gen_minhash_rows(hash_coeffs, data[1], len(users))))
    lsh_out = user_hash.flatMap(lambda data: lsh(data[0], data[1], b, r)).groupByKey().filter(lambda data: len(data[1]) > 1).mapValues(list)
    candidates = lsh_out.flatMap(lambda data: list(combinations(data[1],2)))
    candidates = candidates.map(lambda data: tuple(sorted(data, key = lambda data: data[0])))
    candidates = candidates.groupByKey().flatMap(lambda data: [(data[0], item) for item in set(data[1])])
    candidates = candidates.filter(lambda data: rm_uncorated(user_dict[data[0]], user_dict[data[1]], 3))
    corated_users = candidates.map(lambda data: jaccard(data, user_dict)).filter(lambda data: data[1] > 0).map(lambda data: data[0])
    candidate_users = corated_users.map(lambda data: (data[0], data[1], pearson_corr(user_dict[data[0]], user_dict[data[1]])))
    candidate_users = candidate_users.filter(lambda data: data[2] > 0).collect()
    write_output(candidate_users, model_file_path)
elif method == "surprise":
    test_inp = spark.textFile(input_test_file_path).map(json.loads).map(lambda data: (data["user_id"], data["business_id"]))
    users = return_dict(0, method)
    business = return_dict(1, method)
    inv_users = return_dict(0, method, True)
    inv_business = return_dict(1, method, True)
    train_df = train_inp.map(lambda data: (users[data[0]], business[data[1]], data[2])).collect()
    test_df = test_inp.map(lambda data: (users[data[0]], business[data[1]], 0)).collect()
    train_df = pd.DataFrame(train_df)
    train_df.columns = ["user_id", "business_id", "rating"]
    model = get_surprise_model(train_df)
    joblib.dump(model, model_file_path)
elif method == "xgboost":
    user_fp = "data/user.json"
    user_avg_fp = "data/user_avg.json"
    business_fp = "data/business.json"
    business_avg_fp = "data/business_avg.json"
    test_inp = spark.textFile(input_test_file_path).map(json.loads).map(lambda data: (data["user_id"], data["business_id"]))
    users, business = return_dict(0, method), return_dict(1, method)
    inv_users, inv_business = return_dict(0, method, True), return_dict(1, method, True)
    train_df = pd.DataFrame(train_inp.map(lambda data: (data[0], data[1], data[2])).collect())
    train_df.columns = ["user_id", "business_id", "rating"]
    test_df = test_inp.map(lambda data: (data[0], data[1])).collect()
    user_df1 = spark.textFile(user_fp).map(json.loads).map(lambda data: (data["user_id"], (data["useful"])))
    user_avg_df = spark.parallelize(create_json_dict("user_id")).map(lambda data: (data["user_id"], (data["avg_stars"])))
    business_df1 = spark.textFile(business_fp).map(json.loads).map(lambda data: (data["business_id"], (data["latitude"], data["longitude"])))
    business_avg_df = spark.parallelize(create_json_dict("business_id")).map(lambda data: (data["business_id"], (data["avg_stars"])))
    user_df, business_df = user_df1.leftOuterJoin(user_avg_df), business_df1.leftOuterJoin(business_avg_df)
    user_df = user_df.mapValues(lambda data: (data[0], data[1] if data[1] is not None else 0.0)).collectAsMap()
    business_df = business_df.mapValues(lambda data: (data[0][0], data[0][1], data[1] if data[1] is not None else 0.0)).collectAsMap()
    model = train_xgb(train_df, user_df, business_df)
    joblib.dump(model, model_file_path)
else:
    print("Wrong option entered...!!!")
print("Running time = " + str(round(time.time() - start_time, 2)) + "s")
spark.stop()