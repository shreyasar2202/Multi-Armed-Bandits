

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import geopy.distance
import ast
import nltk
import math
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from scipy.special import expit
import random


ad = pd.read_csv('ads.csv',encoding='iso-8859-1')[:50]
srch = pd.read_csv('search.csv',encoding='iso-8859-1')[:10000]

class Search:
  def __init__(self,search):
    self.n = len(search)
    self.loc = search[['Latitude', 'Longitude']]
    self.text = search[['SearchParams', 'SearchQuery']]
    self.corp = [ [i for i in nltk.word_tokenize(sent.lower()) if i not in Operations.stop()] for sent in self.text.apply(Operations.vectorize,axis=1).tolist()]
    self.budget = search.Budget.tolist()
    self.t = 0
    
  def getNextRow(self):
    self.t+=1
    return (self.loc.iloc[self.t-1], self.corp[self.t-1], self.budget[self.t-1])
  
  def getRow(self, index):
    return (self.loc[index], self.corp[index], self.budget[index])

class Ads:
  def __init__(self,ads):
    self.n = len(ads)
    self.loc = ads[['Latitude', 'Longitude']]
    self.text = ads[['Params', 'Title']]
    self.corp = [ [i for i in nltk.word_tokenize(sent.lower()) if i not in Operations.stop()] for sent in self.text.apply(Operations.vectorize,axis=1).tolist()]
    self.price = ads.Price.tolist()
  
  def getRow(self, index):
    return (self.loc.iloc[index], self.corp[index], self.price[index])

class Context:
  def __init__(self):
    arrs = np.load('arrs.npy')
    words = np.load('words.npy')
    self.vec_dict = dict(zip(words, arrs))      
    global ad 
    self.ads = Ads(ad)
    self.ad_centroids = [self.computeCentroid(self.ads.getRow(i)[1]) for i in range(self.ads.n)]
    self.max_geo_distance=Operations.computeDistance(50,30,70,140)
    global srch
    self.max_srch_budget=max(srch.Budget)
    #global srch
    self.min_srch_budget=min(srch.Budget)
    self.max_ad_price=max(self.ads.price)
    self.min_ad_price=min(self.ads.price)
    self.weights = np.array([5, 2, 1])
    self.w_sum = sum(self.weights)
    
  def computeCentroid(self,query):
    return np.mean(np.array([self.vec_dict[word] for word in query]), axis = 0)
  
  def getContext(self,ad_id,search_tuple):
    
    centroidDistance = 1 - self.getCentroidDistance(ad_id,search_tuple[1])
    if centroidDistance < 0:
      centroidDistance = 0
    return np.array([(self.weights[0]/self.w_sum)*centroidDistance, (self.weights[1]/self.w_sum)*(1-self.getLocationDistance(ad_id,search_tuple[0])), (self.weights[2]/self.w_sum)*(1-self.getBudgetDistance(ad_id,search_tuple[2]))])
   
    
  def getCentroidDistance(self, ad_id,query):
    query_centroid=self.computeCentroid(query)
    return np.linalg.norm(query_centroid-self.ad_centroids[ad_id]) #(np.sqrt(len(query_centroid)*4)
  
  def getLocationDistance(self, ad_id,search_location):
    ad_location=self.ads.getRow(ad_id)[0]
    distance=Operations.computeDistance(ad_location[0],ad_location[1],search_location[0],search_location[1])
    return distance/self.max_geo_distance
  
  def getBudgetDistance(self, ad_id,search_budget):
    return np.linalg.norm(search_budget-self.ads.getRow(ad_id)[2])/(max(self.max_srch_budget-self.min_ad_price, self.max_ad_price-self.min_srch_budget))

class Bandit(object):

    def generate_reward(self, i):
        raise NotImplementedError


class BernoulliBandit(Bandit):

    def __init__(self, n=10):
        global ad
        self.ads=ad
        self.n = n
        self.d = 3
        #self.theta = np.random.rand(self.d, self.n)
        self.bias, self.theta=self.generate_thetas()
        self.x = np.random.rand(self.d, self.n)
        print ('\n\noptimal theta :\n', self.theta)
        #print ('\n\nproduct : \n', (np.matmul(np.transpose(self.x), self.theta)*np.eye(self.n)).sum(axis = 0))
        self.avg_reward = [0 for _ in range(self.n)]
        self.counts = [0 for _ in range(self.n)]
        self.m = int(0.1*len(self.ads))
        

    def generate_reward(self, i_arr, x, sleep):

        proba_vector=expit((np.matmul(np.transpose(x), self.theta)*np.eye(self.n)).sum(axis = 0)+self.bias)
        proba_vector = proba_vector * sleep
        best_probas = np.sort(proba_vector)[::-1][:self.m]
        rewards = np.array([1 if np.random.random() < proba_vector[i] else 0 for i in i_arr])
        chosen_probas = np.array([proba_vector[i] for i in i_arr])
        
        return rewards, sum(best_probas) - sum(chosen_probas)
                
    def generate_thetas(self):     
      ctrs=[x*50 for x in self.ads.HistCTR.tolist()]

      context = Context()
      thetas = np.zeros((self.d,self.n))
      w = [1,1,1]
      w_sum = sum(w)
      bias = np.zeros(self.n)
      for i in range(self.n):
        res = math.log(ctrs[i]/(1-ctrs[i]))
        thetas[:, i] = [weight/w_sum for weight in w] 
        bias[i] = res - np.dot(context.weights,thetas[:,i])/context.w_sum
      return bias,thetas

class Operations:
  
  @staticmethod
  def vectorize(row):
    cell=row[0]
    vector=[]
    if cell is not None:
        cell = ast.literal_eval(cell)
        assert(type(cell)==dict)
    
        vector=list(cell.values())
    if row[1] is not None:
        vector.append(row[1])
        
    return " ".join(vector).lower()

  
  @staticmethod
  def computeDistance(lat1, long1, lat2, long2):
    coords_1 = (lat1, long1)
    coords_2 = (lat2, long2)

    return geopy.distance.distance(coords_1,coords_2).km
  
  @staticmethod
  def stop():
    return stopwords.words('english') + list(string.punctuation) + ["''", "'s"]

class LogisticUCB(object):
    def __init__(self, bandit, dimensions=3, delta=1):
        """
        bandit (Bandit): the target bandit to solve.
        """
        assert isinstance(bandit, BernoulliBandit)
        np.random.seed(int(time.time()))

        self.bandit = bandit

        #self.chosen_arms = []  # A list of machine ids, 0 to bandit.n-1.
        self.regret = 0.  # Cumulative regret.
        self.regrets = [0.]  # History of cumulative regret.
        self.k = self.bandit.n
        self.d = dimensions
        self.t = 0
        
        self.theta = np.zeros((self.d,self.k))    
        self.p = np.zeros(self.k)
        self.r = [np.random.random()/10 for _ in range(self.k)]
        self.alpha = 1.1 #+ np.sqrt(np.log(2/delta)/2)
        self.eta = 10000

        self.x = np.zeros((self.d, self.k))
        self.q = [0 for _ in range(self.k)]
        self.d_csmab = [0 for _ in range(self.k)]
        self.logregs = [LogisticRegression(C=1e7, solver='lbfgs') for _ in range(self.k)]
        self.q_tanh = [0 for _ in range(self.k)]
        
        #print ('\n\nx : \n', self.x)
        #print ('search')
        
        
        self.counts = [2 for _ in range(self.k)]
        global srch
        self.search = Search(srch)
        self.context = Context()
        self.A = np.array([np.identity(self.d) for _ in range(self.k)])
        
        
    def update_regret(self, diff):
        self.regret += diff/len(self.bandit.ads)
        self.regrets.append(self.regret)

    @property
    def estimated_probas(self):
        raise NotImplementedError

    def run_one_step(self):
        """Return the machine index to take action on."""
        self.t += 1
        
        #self.x = np.random.rand(self.d,self.k)
        query = self.search.getNextRow()

        while True:
          self.sleep = np.array([random.choice([0, 1, 1, 1]) for _ in range(self.bandit.n)])	
          if(sum(self.sleep)>0):
            break
            
        for i in range(self.bandit.n):
          self.q[i] = max(0, (self.q[i] + self.r[i]*sum(self.x[:,i]) - self.d_csmab[i]))
        
        self.q_tanh = np.tanh(self.q)
        self.x = np.transpose(np.array([self.context.getContext(i, query) for i in range(self.k)]))

        logreg_score=[0 for _ in range(self.k)]
        for i in range(self.k):
          inverse=np.linalg.inv(self.A[i])
          if self.counts[i]!=2:
            self.p[i] = self.eta*expit(np.dot(self.x[:, i], np.ravel(self.logregs[i].coef_))+self.logregs[i].intercept_)
            logreg_score[i]=self.p[i]

            self.p[i] += self.eta*self.alpha*(np.sqrt(np.matmul(np.matmul(np.transpose(self.x[:,i]),inverse),self.x[:,i]))) + self.q_tanh[i]*sum(self.x[:,i])
          else:
            #self.p[i] = self.alpha*(np.sqrt(np.matmul(np.matmul(np.transpose(self.x[:,i]),np.eye(self.d)),self.x[:,i])))
            self.p[i] = self.eta*self.alpha*(np.sqrt(np.matmul(np.matmul(np.transpose(self.x[:,i]),inverse),self.x[:,i]))) + self.q_tanh[i]*sum(self.x[:,i])
        
        self.p = np.multiply(self.p, self.sleep)      
        
        chosen_array = np.array(self.p.argsort()[-min(self.bandit.m, sum(self.sleep)):][::-1])
        #print("Chosen ", chosen_array)
        for chosen in chosen_array:
           self.A[chosen] = self.A[chosen] + np.matmul(np.reshape(self.x[:,chosen],(self.d,1)), np.reshape(self.x[:,chosen],(1,self.d)))
        self.d_csmab = [1 if _ in chosen_array else 0 for _ in range(self.k)]
        
        rewards, diff = self.bandit.generate_reward(chosen_array, self.x, self.sleep)
        self.train_update(chosen_array, rewards)  
        return diff
      
    def train_update(self, chosen_array, rewards):
      for chosen in chosen_array:
        self.counts[chosen] += 1
        count = self.counts[chosen]
        self.trains[chosen][:,count-1] = self.x[:,chosen]
        self.tests[chosen][count-1] = rewards[np.argwhere(chosen_array == chosen)[0]]
        self.X_train = self.trains[chosen][:, :count]
        self.y_train = self.tests[chosen][:count]
        self.logregs[chosen] = LogisticRegression(C=1e7, solver='lbfgs')
        self.logregs[chosen].fit(np.transpose(self.X_train), self.y_train)
      
        
    def run(self, num_steps):
        assert self.bandit is not None
        self.time_steps = num_steps
        self.trains = np.array([np.zeros((self.d, num_steps+1)) for _ in range(self.k)])
        self.tests = np.array([np.zeros((num_steps+1)) for _ in range(self.k)])
        for k in range(self.k):
          self.trains[k,:,0] = self.context.weights/self.context.w_sum
          self.tests[k,0]=1
        if num_steps>self.search.n:
          num_steps = self.search.n
        for _ in range(num_steps):
            #if _%1000==0:
            print('\n\n\nstep : ', _+1 )
            diff = self.run_one_step()

            self.update_regret(diff)

def plot_results(regrets):

    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(regrets)), regrets)

    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Cumulative Regret')
    #ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax1.legend(loc = 'best')
    ax1.grid('k', ls='--', alpha=0.3)

    plt.show()

k = BernoulliBandit(n = len(ad))

ucb = LogisticUCB(k)

ucb.run(100)

plot_results(ucb.regrets)
