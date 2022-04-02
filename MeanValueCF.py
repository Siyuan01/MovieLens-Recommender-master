# -*- coding = utf-8 -*-
"""
User-based Collaborative filtering.


"""
import collections
from operator import itemgetter

import math

from collections import defaultdict

import similarity
import utils
from utils import LogTime


class MeanValueCF:
    """
    MeanValue Collaborative filtering.
    Top-N recommendation.
    """

    def __init__(self, k_sim_user=20, n_rec_movie=10, dataset_name='ml-100k',save_model=True):
        """
        Init UserBasedCF with n_sim_user and n_rec_movie.
        :return: None
        """
        print("MeanValueCF start...\n")
        self.k_sim_user = k_sim_user
        self.n_rec_movie = n_rec_movie
        self.trainset = None
        self.filledset=None
        self.save_model = save_model
        self.usernum,self.itemnum=(943,1682) if dataset_name == 'ml-100k' else (6040,3952)
        self.records=[]#存放用户评分数据，令records[i] = [u,i,rui,pui]，其中rui是用户u对物品i的实际评分，pui是算法预测出来的用户u对物品i的评分
        self.user_sim_mat=None
    def fillMissingValue(self,trainset):
        '''
        Fill the trainset with mean value.
        :param trainset: train dataset(dict)
        :return: filledset(dict)
        '''
        filledset = defaultdict(dict)
        for user,movies in trainset.items():
            for movie,rating in movies.items():
                filledset[user][movie]=trainset[user][movie]
        # print(filledset)
        # print((list(trainset.keys())))
        user_mean={}
        for user,movies in trainset.items():
            user_mean[user]=sum(trainset[user].values())/(1.0*len(trainset[user]))
            # print(user,sum(trainset[user].values()),len(trainset[user]))
        # print(user_mean)
        for u in range(1,self.usernum+1):
            for i in range(1,self.itemnum+1):
                filledset[str(u)][str(i)]=user_mean[str(u)]
        return filledset


    def fit(self, trainset):
        """
        Fit the trainset by fill the missing value.
        :param trainset: train dataset
        :return: None
        """
        model_manager = utils.ModelManager()
        try:
            self.movie_popular = model_manager.load_model('movie_popular')
            self.movie_count = model_manager.load_model('movie_count')
            self.trainset = model_manager.load_model('trainset')
            self.filledset = self.fillMissingValue(self.trainset)
            print('User origin similarity model has saved before.\nLoad model success...\n')
        except OSError:
            print('No model saved before.\nTrain a new model...')
            self.user_sim_mat, self.movie_popular, self.movie_count = \
                similarity.calculate_user_similarity(trainset=trainset)
            self.filledset=self.fillMissingValue(trainset)
            self.trainset = trainset
            print('Train a new model success.')
            if self.save_model:
                model_manager.save_model(self.movie_popular, 'movie_popular')
                model_manager.save_model(self.movie_count, 'movie_count')
            print('The new model has saved success.\n')

    def recommend(self, user):
        """
        Find K similar users and recommend N movies for the user.
        :param user: The user we recommend movies to.
        :return: the N best score movies
        """
        if not self.n_rec_movie or \
                not self.trainset or not self.movie_popular or not self.movie_count:
            raise NotImplementedError('UserCF has not init or fit method has not called yet.')
        N = self.n_rec_movie
        if user not in self.trainset:
            print('The user (%s) not in trainset.' % user)
            return
        # print('Recommend movies to user start...')
        predict_score=defaultdict(int)
        watched_movies = self.trainset[user]
        for movie, rating in self.filledset[user].items():
                if movie in watched_movies:
                    continue
                # predict the user's "interest" for each movie
                predict_score[movie] = rating
                self.records.append([user,movie,rating])#除了看过的之外的电影预测评分
        # print('Recommend movies to user success.')
        # return the N best score movies
        # print([score for  _, score in sorted(predict_score.items(), key=itemgetter(1), reverse=True)[0:N]])
        return [movie for movie, _ in sorted(predict_score.items(), key=itemgetter(1), reverse=True)[0:N]]

    def test(self, testset):
        """
        Test the recommendation system by recommending scores to all users in testset.
        :param testset: test dataset
        :return:
        """
        if not self.n_rec_movie or not self.trainset or not self.movie_popular or not self.movie_count:
            raise ValueError('MVCF has not init or fit method has not called yet.')
        self.testset = testset
        print('Test recommendation system start...')
        # print(testset)
        N = self.n_rec_movie
        #  varables for precision and recall
        hit = 0
        rec_count = 0
        test_count = 0
        # varables for coverage
        all_rec_movies = set()
        # varables for popularity
        popular_sum = 0

        # record the calculate time has spent.
        test_time = LogTime(print_step=1000)

        RMSE = 0
        MAE = 0
        sum_R=0
        sum_M=0
        for user,movie,pui in self.records:
            #records：self.records.append([user,movie,rating])#除了看过的之外的电影预测评分
            rui=self.testset[user][movie]
            sum_R+=((rui-pui)*(rui-pui))
            sum_M+=math.fabs(rui-pui)
            test_time.count_time()
        RMSE=sum_R/float(len(self.records))
        MAE=sum_M/float(len(self.records))

        for i, user in enumerate(self.trainset):
            test_movies = self.testset.get(user, {})
            rec_movies = self.recommend(user)  # type:list
            for movie in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
                popular_sum += math.log(1 + self.movie_popular[movie])
                # log steps and times.
            rec_count += N
            test_count += len(test_movies)
            # print time per 500 times.
            test_time.count_time()
        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        popularity = popular_sum / (1.0 * rec_count)

        print('Test recommendation system success.')
        test_time.finish()

        print('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f\n' %
              (precision, recall, coverage, popularity))

        print('RMSE=%.4f\tMAE=%.4f\n' %(RMSE, MAE))

    def predict(self, testset):
        """
        Recommend movies to all users in testset.
        :param testset: test dataset
        :return: `dict` : recommend list for each user.
        """
        movies_recommend = defaultdict(list)
        print('Predict scores start...')
        # record the calculate time has spent.
        predict_time = LogTime(print_step=500)
        for i, user in enumerate(testset):
            rec_movies = self.recommend(user)  # type:list
            movies_recommend[user].append(rec_movies)
            # log steps and times.
            predict_time.count_time()
        print('Predict scores success.')
        predict_time.finish()
        return movies_recommend
