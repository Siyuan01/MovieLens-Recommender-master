#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description : Item-based Collaborative filtering.
"""
import collections
from operator import itemgetter

import math

from collections import defaultdict

import similarity
import utils
from utils import LogTime


class ItemBasedCF:
    """
    Item-based Collaborative filtering.
    Top-N recommendation.
    """
    def __init__(self, k_sim_movie=20, n_rec_movie=10, use_iuf_similarity=False, save_model=True):
        """
        Init UserBasedCF with n_sim_user and n_rec_movie.
        :return: None
        """
        print("ItemBasedCF start...\n")
        self.k_sim_movie = k_sim_movie
        self.n_rec_movie = n_rec_movie
        self.trainset = None
        self.save_model = save_model
        self.use_iuf_similarity = use_iuf_similarity
        self.filledset = defaultdict(dict)
        self.result = {}
        self.user_mean = None

    def fit(self, trainset):
        """
        Fit the trainset by calculate movie similarity matrix.
        :param trainset: train dataset
        :return: None
        """
        model_manager = utils.ModelManager()
        try:
            self.movie_sim_mat = model_manager.load_model(
                'movie_sim_mat-iif' if self.use_iuf_similarity else 'movie_sim_mat')
            self.movie_popular = model_manager.load_model('movie_popular')
            self.movie_count = model_manager.load_model('movie_count')
            self.trainset = model_manager.load_model('trainset')
            print('Movie similarity model has saved before.\nLoad model success...\n')
        except OSError:
            print('No model saved before.\nTrain a new model...')
            # self.movie_sim_mat, self.movie_popular, self.movie_count = \
            #     similarity.calculate_item_cosine_similarity(trainset=trainset,
            #                                          use_iuf_similarity=self.use_iuf_similarity)
            self.movie_sim_mat, self.movie_popular, self.movie_count = \
                similarity.calculate_item_cosine_similarity(trainset=trainset,)
            self.trainset = trainset
            print('Train a new model success.')
            if self.save_model:
                model_manager.save_model(self.movie_sim_mat,
                                         'movie_sim_mat-iif' if self.use_iuf_similarity else 'movie_sim_mat')
                model_manager.save_model(self.movie_popular, 'movie_popular')
                model_manager.save_model(self.movie_count, 'movie_count')
                #model_manager.save_model(self.trainset, 'trainset')
                print('The new model has saved success.\n')
    def get_usermean(self,trainset=None):
        '''
        Get the mean value of each user
        :param self.trainset: The rating matrix.
        :return: the mean value of each user
        '''
        if self.user_mean:
            return
        if not trainset:
            trainset=self.trainset
        self.user_mean={}
        for user,movies in trainset.items():
            self.user_mean[user]=sum(movies.values())/len(movies)
    def recommend(self, user):
        """
        Find K similar movies and recommend N movies for the user.
        :param user: The user we recommend movies to.
        :return: the N best score movies
        """
        if not self.movie_sim_mat or not self.n_rec_movie or \
                not self.trainset or not self.movie_popular or not self.movie_count:
            raise NotImplementedError('ItemCF has not init or fit method has not called yet.')
        if not self.user_mean:
            self.get_usermean()
        K = self.k_sim_movie
        N = self.n_rec_movie
        predict_score = collections.defaultdict(int)
        sim_sum=collections.defaultdict(float)
        if user not in self.trainset:
            print('The user (%s) not in trainset.' % user)
            return
        # print('Recommend movies to user start...')
        watched_movies = self.trainset[user]
        for movie, rating in watched_movies.items():
            for related_movie, similarity_factor in sorted(self.movie_sim_mat[movie].items(),
                                                           key=itemgetter(1), reverse=True)[0:K]:
                if related_movie in watched_movies:
                    continue
                # predict the user's "interest" for each movie
                # the predict_score is sum(similarity_factor * rating)
                predict_score[related_movie] += similarity_factor * rating
                sim_sum[related_movie]+= similarity_factor
        # print('Recommend movies to user success.')
        for movie in predict_score.keys():
            predict_score[movie]=predict_score[movie]/sim_sum[movie]
            self.filledset[user][movie]=predict_score[movie]
        return [movie for movie, _ in sorted(predict_score.items(), key=itemgetter(1), reverse=True)[0:N]]

    def test(self, testset):
        """
        Test the recommendation system by recommending scores to all users in testset.
        :param testset: test dataset
        :return:
        """
        if not self.n_rec_movie or not self.trainset or not self.movie_popular or not self.movie_count:
            raise ValueError('ItemCF has not init or fit method has not called yet.')
        if not self.user_mean:
            self.get_usermean(self.filledset)
        self.testset = testset
        print('Test recommendation system start...')
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

        sum_R = 0
        sum_M = 0
        testsize = 0
        for user, movies in self.testset.items():
            for movie, rating in movies.items():
                if movie in self.filledset[user]:
                    pui = self.filledset[user][movie]
                else:
                    pui=self.user_mean[user]# 有一部分电影没有被评价过，用平均值填充（总是会使误差变大）
                sum_R += ((rating - pui) ** 2)
                sum_M += math.fabs(rating - pui)
                testsize += 1
            test_time.count_time()

        RMSE = math.sqrt(sum_R / float(testsize))
        MAE = sum_M / float(testsize)

        print('Test recommendation system success.')
        test_time.finish()

        print('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f\n' %
              (precision, recall, coverage, popularity))
        print('RMSE=%.4f\tMAE=%.4f\n' % (RMSE, MAE))
        print(testsize)
        self.result['RMSE'] = RMSE
        self.result['MAE'] = MAE
        self.result['precision'] = precision
        self.result['recall'] = recall
        self.result['coverage'] = coverage
        self.result['popularity'] = popularity

        # num = sum([len(movies) for user, movies in self.filledset.items()])
        # print("now the number of the records is %d, the data sparsity level is %.2f\n" % (
        #     num, 1 - (num / (1.0 * 943 * 1682))))

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
