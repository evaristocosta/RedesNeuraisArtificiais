# https://www.kaggle.com/alvations/xor-with-mlp
import random
random.seed(0)


def generate_zero():
    return random.uniform(0, 49) / 100


def generate_one():
    return random.uniform(50, 100) / 100


def generate_and(num_data_points):
    Xs, Ys = [], []
    for _ in range(num_data_points):
        # and(0, 0) -> 0
        Xs.append([generate_zero(), generate_zero()])
        Ys.append([0])
        # and(1, 0) -> 0
        Xs.append([generate_one(), generate_zero()])
        Ys.append([0])
        # and(0, 1) -> 0
        Xs.append([generate_zero(), generate_one()])
        Ys.append([0])
        # and(1, 1) -> 1
        Xs.append([generate_one(), generate_one()])
        Ys.append([1])
    return Xs, Ys


def generate_or(num_data_points):
    Xs, Ys = [], []
    for _ in range(num_data_points):
        # or(0, 0) -> 0
        Xs.append([generate_zero(), generate_zero()])
        Ys.append([0])
        # or(1, 0) -> 1
        Xs.append([generate_one(), generate_zero()])
        Ys.append([1])
        # or(0, 1) -> 1
        Xs.append([generate_zero(), generate_one()])
        Ys.append([1])
        # or(1, 1) -> 1
        Xs.append([generate_one(), generate_one()])
        Ys.append([1])
    return Xs, Ys


def generate_xor(num_data_points):
    Xs, Ys = [], []
    for _ in range(num_data_points):
        # xor(0, 0) -> 0
        Xs.append([generate_zero(), generate_zero()])
        Ys.append([0])
        # xor(1, 0) -> 1
        Xs.append([generate_one(), generate_zero()])
        Ys.append([1])
        # xor(0, 1) -> 1
        Xs.append([generate_zero(), generate_one()])
        Ys.append([1])
        # xor(1, 1) -> 0
        Xs.append([generate_one(), generate_one()])
        Ys.append([0])
    return Xs, Ys
