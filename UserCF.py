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


class UserBasedCF:
    """
    User-based Collaborative filtering.
    """
    def __init__(self, k_sim_user=20, n_rec_movie=10, use_iif_similarity=False, save_model=True):
        """
        Init UserBasedCF with n_sim_user and n_rec_movie.
        :return: None
        """
        print("UserBasedCF start...\n")
        self.k_sim_user = k_sim_user
        self.n_rec_movie = n_rec_movie
        self.trainset = None
        self.filledset=defaultdict(dict)
        self.save_model = save_model
        self.use_iif_similarity = use_iif_similarity
        self.user_mean=None
        self.result = {}

    def fit(self, trainset):
        """
        Fit the trainset by calculate user similarity matrix.
        :param trainset: train dataset
        :return: None
        """
        model_manager = utils.ModelManager()
        try:
            self.user_sim_mat = model_manager.load_model(
                'user_sim_mat-iif' if self.use_iif_similarity else 'user_sim_mat')
            self.movie_popular = model_manager.load_model('movie_popular')
            self.movie_count = model_manager.load_model('movie_count')
            self.trainset = model_manager.load_model('trainset')
            print('User origin similarity model has saved before.\nLoad model success...\n')
        except OSError:
            print('No model saved before.\nTrain a new model...')
            # self.user_sim_mat, self.movie_popular, self.movie_count = \
            #     similarity.calculate_user_similarity(trainset=trainset,
            #                                          use_iif_similarity=self.use_iif_similarity)
            self.user_sim_mat, self.movie_popular, self.movie_count = \
                similarity.calculate_user_cosine_similarity(trainset=trainset)
            self.trainset = trainset
            print('Train a new model success.')
            if self.save_model:
                model_manager.save_model(self.user_sim_mat,
                                         'user_sim_mat-iif' if self.use_iif_similarity else 'user_sim_mat')
                model_manager.save_model(self.movie_popular, 'movie_popular')
                model_manager.save_model(self.movie_count, 'movie_count')
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
        Find K similar users and recommend N movies for the user.
        :param user: The user we recommend movies to.
        :return: the N best score movies
        """
        if not self.user_sim_mat or not self.n_rec_movie or \
                not self.trainset or not self.movie_popular or not self.movie_count:
            raise NotImplementedError('UserCF has not init or fit method has not called yet.')
        K = self.k_sim_user
        N = self.n_rec_movie
        predict_score = collections.defaultdict(int)
        movie_simsum=collections.defaultdict(float)
        if user not in self.trainset:
            print('The user (%s) not in trainset.' % user)
            return
        # print('Recommend movies to user start...')
        watched_movies = self.trainset[user]

        similar_user_fac=sorted(self.user_sim_mat[user].items(),key=itemgetter(1), reverse=True)[0:K]
        # sum_sim = 0
        for similar_user, similarity_factor in similar_user_fac:
            for movie, rating in self.trainset[similar_user].items():
                if movie in watched_movies:
                    continue
                # predict the user's "interest" for each movie
                # the predict_score is sum(similarity_factor * rating)
                predict_score[movie] += (similarity_factor * (rating))#-self.user_mean[similar_user]))
                movie_simsum[movie]+=similarity_factor

        for movie in predict_score.keys():
            predict_score[movie]= (predict_score[movie]/movie_simsum[movie])#+self.user_mean[user]
            self.filledset[user][movie]=predict_score[movie]
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
            raise ValueError('UserCF has not init or fit method has not called yet.')
        self.testset = testset
        print('Test recommendation system start...')
        if not self.user_mean:
            self.get_usermean(self.filledset)
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
                    pui=self.user_mean[user]
                sum_R += ((rating - pui) **2)
                sum_M += math.fabs(rating - pui)
                # print(pui,rating)
                testsize += 1
            test_time.count_time()

        RMSE = math.sqrt(sum_R / float(testsize))
        MAE = sum_M / float(testsize)

        print('Test recommendation system success.')
        test_time.finish()

        print('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f\n' %
              (precision, recall, coverage, popularity))
        print('RMSE=%.4f\tMAE=%.4f\n' % (RMSE, MAE))
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
