# -*- coding = utf-8 -*-
"""
Main function to build recommendation systems.

"""
import utils
from ItemCF import ItemBasedCF
from LFM import LFM
from UserCF import UserBasedCF
from dataset import DataSet
from most_popular import MostPopular
from random_pred import RandomPredict
from utils import LogTime
from MeanValueCF import MeanValueCF

def run_model(model_name, dataset_name, test_size=0.3, clean=False):
    print('*' * 70)
    print('\tThis is %s model trained on %s with test_size = %.2f' % (model_name, dataset_name, test_size))
    print('*' * 70 + '\n')
    model_manager = utils.ModelManager(dataset_name, test_size)
    try:
        trainset = model_manager.load_model('trainset')
        testset = model_manager.load_model('testset')
    except OSError:
        ratings = DataSet.load_dataset(name=dataset_name)
        trainset, testset = DataSet.train_test_split(ratings, test_size=test_size)
        model_manager.save_model(trainset, 'trainset')
        model_manager.save_model(testset, 'testset')
    '''Do you want to clean workspace and retrain model again?'''
    '''if you want to change test_size or retrain model, please set clean_workspace True'''
    model_manager.clean_workspace(clean)
    if model_name == 'UserCF':
        model = UserBasedCF(n_rec_movie=20)
    elif model_name == 'ItemCF':
        model = ItemBasedCF()
    elif model_name == 'MeanValueCF':
        model = MeanValueCF(dataset_name=dataset_name)
    elif model_name == 'Random':
        model = RandomPredict()
    elif model_name == 'MostPopular':
        model = MostPopular()
    elif model_name == 'UserCF-IIF':
        model = UserBasedCF(use_iif_similarity=True)
    elif model_name == 'ItemCF-IUF':
        model = ItemBasedCF(use_iuf_similarity=True)
    elif model_name == 'LFM':
        # K, epochs, alpha, lamb, n_rec_movie
        model = LFM(10, 20, 0.1, 0.01, 10)
    else:
        raise ValueError('No model named ' + model_name)
    model.fit(trainset)

    # adjust the number of test user
    # user_list=[1, 100, 233, 666, 888]
    user_list=list(range(1,943+1))  if dataset_name == 'ml-100k' else list(range(1,6040+1))

    recommend_test(model, user_list)
    model.test(testset)
    print(model.user_sim_mat)

def recommend_test(model, user_list):
    for user in user_list:
        recommend = model.recommend(str(user))
        print("recommend for userid = %s:" % user)
        print(recommend)
        print()


if __name__ == '__main__':
    main_time = LogTime(words="Main Function")
    dataset_name = 'ml-100k'
    # dataset_name = 'ml-1m'
    model_type = 'UserCF'
    # model_type = 'UserCF-IIF'
    # model_type = 'ItemCF'
    # model_type = 'Random'
    # model_type = 'MostPopular'
    # model_type = 'ItemCF-IUF'
    # model_type = 'LFM'
    model_type= 'MeanValueCF'
    test_size = 0.1
    run_model(model_type, dataset_name, test_size, False)
    main_time.finish()
