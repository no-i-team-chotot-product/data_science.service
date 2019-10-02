# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 04:01:18 2019

@author: ASUS
"""

#import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse 
class CF(object):
    
    def __init__(self, Y_data, k, dist_func = cosine_similarity, uuCF = 1):
        self.uuCF = uuCF 
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]
        self.k = k
        self.dist_func = dist_func
        self.Ybar_data = None
       
        self.n_users = len(np.unique(Y_data[:,0]))
        self.n_items = len(np.unique(Y_data[:,1]))
    
    def add(self, new_data):
        self.Y_data = np.concatenate((self.Y_data, new_data), axis = 0)
    
    def normalize_Y(self):
        users = self.Y_data[:, 0] 
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((len(np.unique(Y_data[:,0])),))
        for n in range(self.n_users):
            ids = np.where(users == n)[0].astype(np.int32)
            item_ids = self.Y_data[ids, 1] 
            ratings = self.Y_data[ids, 2]
            # take mean
            m = np.mean(ratings) 
            #print(m)
            if np.isnan(m):
                m = 0 
            self.mu[n] = m
            # normalize
            self.Ybar_data[ids, 2] = ratings - self.mu[n]
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
            (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()

    def similarity(self):
        eps = 1e-6
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)
        
    def refresh(self):
        self.normalize_Y()
        self.similarity() 
        
    def fit(self):
        self.refresh()
        
    def __pred(self, u, i, normalized = 1):
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        sim = self.S[u, users_rated_i]
        a = np.argsort(sim)[-self.k:] 
        nearest_s = sim[a]
        r = self.Ybar[i, users_rated_i[a]]

        if normalized:
            return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8)
        return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8) + self.mu[u]
    
    def pred(self, u, i, normalized = 1):
        if self.uuCF: return self.__pred(u, i, normalized)
        return self.__pred(i, u, normalized)
            
    def recommend(self, u):
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()              
        recommended_items = []
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                rating = self.__pred(u, i)
                if rating > 2: 
                    recommended_items.append(i)
        
        return recommended_items 
    
    def recommend2(self, u):
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()              
        recommended_items = []
    
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                rating = self.__pred(u, i)
                if rating > 2: 
                    recommended_items.append(i)
                    
        return recommended_items 

    def print_recommendation(self):
        print( 'Recommendation: ')
        for u in range(self.n_users):
            recommended_items = self.recommend(u)
            if self.uuCF:
                print( '    Recommend item(s):', recommended_items, 'for user', u)
            else: 
                print( '    Recommend item', u, 'for user(s) : ', recommended_items)
 
from flask import Flask, request, jsonify
               
import os
import pickle
import re
import json
app = Flask(__name__)

from flask import Flask, request, jsonify
rs = None
try:
    with open('D:/bao/Coder-School/project/git/data_science.service/ChotoRecommenderSystemModel.pkl', 'rb') as model:
        rs = pickle.load(model)
except IOError:
    print("File not found!!")

@app.route("/recommend/user/<id>", methods=["GET"])
def recommendforuser(id):
    list_item = rs.recommend(int(id))
    return jsonify({'item': list_item})

app.run(debug=True)