import os
from array import array

import numpy
import csv

# read data from file
def readFile(fileName):
    with open(fileName) as f:
        lines = csv.reader(f, delimiter=',')
        matrix = []
        for line in lines:
            if line[-1] == '':
                array = line[:-1]
            else:
                array = line
            matrix.append(array)
        matrix = numpy.asarray(matrix, dtype=numpy.float32)
    # instances=matrix[:,1:]
    # values=numpy.ones((instances.shape[0],1))-matrix[:,0:1]
    # one_hot_labels=numpy.c_[matrix[:,0:1],values]
    return matrix

class DataSet(object):
    def __init__(self,instances,labels, groups=[]):
        self._instances=instances
        self._labels=labels
        self._num_instances=instances.shape[0]
        self._epochs_completed=0
        self._index_in_epoch=0
        self._groups=groups
    @property
    def instances(self):
        return self._instances
    @property
    def labels(self):
        return self._labels
    @property
    def groups(self):
        return self._groups
    @property
    def num_instances(self):
        return self._num_instances
    @property
    def epochs_completed(self):
        return self._epochs_completed
    def pos_instance_ratio(self):
        pos_instances=0.0
        for label in self._labels[:,0]:
                if label==1:
                        pos_instances+=1
        return pos_instances/self._labels.shape[0]

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_instances:
            self._epochs_completed += 1
            shuffled = numpy.arange(self._num_instances)
            numpy.random.shuffle(shuffled)
            self._instances=self._instances[shuffled]
            self._labels=self._labels[shuffled]
            if self._groups != []:
                self._groups=self._groups[shuffled]
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self._instances[start:end], self._labels[start:end]

def create_train_sets(train_instances, train_labels):
    class DataSets(object):
        pass
    data_sets = DataSets()

    data_sets.train = DataSet(train_instances, train_labels)
    return data_sets
def read_data_sets(train_file, train_label_file, test_file, test_label_file):
    class DataSets(object):
        pass
    data_sets = DataSets()
    train_instances = readFile(train_file)
    train_labels = readFile(train_label_file)
    test_instances = readFile(test_file)
    test_labels = readFile(test_label_file)

    if False:
        shuffled = numpy.arange(test_instances.shape[1])
        numpy.random.shuffle(shuffled)
        test_instances = test_instances[:, shuffled]
        train_instances = train_instances[:, shuffled]

    data_sets.train = DataSet(train_instances, train_labels)
    data_sets.test = DataSet(test_instances, test_labels)
    return data_sets

# min-max normalization
def normalize(train_instances,test_instances):
        #train_size=train_instances.shape[0]
        train_min=train_instances.min(0)
        train_max=train_instances.max(0)
        train_instances=(train_instances-train_min)/(train_max-train_min)
        test_instances=(test_instances-train_min)/(train_max-train_min)
        return train_instances,test_instances
        #comb=numpy.r_[train_instances,test_instances]
        #comb=(comb-comb.min(0))/(comb.max(0)-comb.min(0))
        #return comb[0:train_size,:],comb[train_size:,:]

# test normalize() method
def testNormalize():
        a=numpy.arange(15).reshape(-1,5)
        b=numpy.arange(15,25).reshape(-1,5)
        a=a.astype(numpy.float32)
        b=b.astype(numpy.float32)
        a,b=normalize(a,b)
        print(a)
        print(b)

# test read_data_sets() method
def testReadDataSets():
        dir='./Results/Time/1/'
        datasets=read_data_sets(dir+'Train.csv',dir+'TrainLabel.csv',dir+'Test.csv',dir+'TestLabel.csv')
        print(datasets.train.pos_instance_ratio())
        print(datasets.test.pos_instance_ratio())

