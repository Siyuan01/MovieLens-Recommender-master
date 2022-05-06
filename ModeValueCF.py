# -*- coding = utf-8 -*-
"""
ModeValue Collaborative filtering.
fill the array with user's mode rating.
"""
import collections
from operator import itemgetter

import math

from collections import defaultdict

import similarity
import utils
from utils import LogTime


class ModeValueCF:
    """
    ModeValue Collaborative filtering.
    Top-N recommendation.
    """

    def __init__(self, k_sim_user=20, n_rec_movie=10, dataset_name='ml-100k', save_model=True):
        """
        Init with n_rec_movie.
        :return: None
        """
        print("MeanValueCF start...\n")
        self.K = k_sim_user
        self.n_rec_movie = n_rec_movie
        self.trainset = None
        self.filledset = None
        self.save_model = save_model
        self.usernum, self.itemnum = (943, 1682) if dataset_name == 'ml-100k' else (6040, 3952)
        self.user_sim_mat = None
        self.result = {}

    def fillMissingValue(self, trainset):
        '''
        Fill the trainset with Mode value.
        :param trainset: train dataset(dict)
        :return: filledset(dict)
        '''
        print("FillMissingValue start...\n")
        filledset = defaultdict(dict)
        for user, movies in trainset.items():
            for movie, rating in movies.items():
                filledset[user][movie] = trainset[user][movie]

        for user, movies in filledset.items():
            user_similarity = sorted(self.user_sim_mat[user].items(), key=itemgetter(1), reverse=True)
            # the len of user_similarity is between 0 -- self.usernum-1
            if len(user_similarity) > self.K:
                user_similarity = user_similarity[:self.K]
            # Find items that have not been graded
            for i in range(1, self.itemnum + 1):
                if str(i) not in movies:
                    score_mode = defaultdict(int)
                    for sim_user, _ in user_similarity:
                        if str(i) in self.trainset[sim_user]:
                            score = self.trainset[sim_user][str(i)]
                            score_mode[score] += 1
                    maxcount, res = 0, 0
                    for score, count in score_mode.items():
                        if count > maxcount:
                            maxcount, res = count, score
                    if maxcount != 0:
                        filledset[user][str(i)] = res
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
            self.trainset = trainset
            self.filledset = self.fillMissingValue(trainset)
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
        predict_score = defaultdict(int)
        watched_movies = self.trainset[user]
        for movie, rating in self.filledset[user].items():
            if movie in watched_movies:
                continue
            # predict the user's "interest" for each movie
            predict_score[movie] = rating
        # print('Recommend movies to user success.')
        # return the N best score movies
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
                    pui = sum(self.filledset[user].values()) / (1.0 * len(self.filledset[user]))
                sum_R += ((rating - pui) * (rating - pui))
                sum_M += math.fabs(rating - pui)
                testsize += 1
                test_time.count_time()
        RMSE = sum_R / float(testsize)
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
