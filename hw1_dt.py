import numpy as np
import utils as Util

class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        assert (len(features) == len(labels))
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()
        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred

class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.pruneTested = False
        self.parent = None
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
        # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self):

        if(len(self.features[0]) == 0):
            self.splittable = False
            return

        num_attrs = len(self.features[0])

        #find splitting attr

        info_gain_by_attributes = []
        distinct_attribute_vals = []

        for i in range(num_attrs):
            info_gain_by_attributes.append(None)
            distinct_attribute_vals.append(set())

        for feature in self.features:
            for attr_index in range(len(feature)):
                distinct_attribute_vals[attr_index].add(feature[attr_index])

        #distinct_attribute_vals.sort()

        for attr_index in range(0, num_attrs):
            unique_labels = list(np.unique(self.labels))
            num_branches = len(distinct_attribute_vals[attr_index])
            branches = np.zeros((num_branches,self.num_cls))
            enum_distinct_attribute_vals = enumerate(distinct_attribute_vals[attr_index])

            for distinct_attribute_val_index,distinct_attribute_val in list(enum_distinct_attribute_vals):
                for feature_index in range(len(self.features)):
                    if(self.features[feature_index][attr_index] == distinct_attribute_val):
                        branches[distinct_attribute_val_index][unique_labels.index(self.labels[feature_index])] += 1

            parent_dist = np.zeros(self.num_cls)
            for cls in self.labels:
                parent_dist[unique_labels.index(cls)] += 1

            info_gain_by_attributes[attr_index] = Util.Information_Gain(Util.entropy(parent_dist),branches)
        max_information_gains_indices = [index for index, val in enumerate(info_gain_by_attributes) if val == max(info_gain_by_attributes)]

        flag = True
        for index in max_information_gains_indices:
            if(info_gain_by_attributes[index] == 0):
                continue
            else:
                flag = False
                break

        if(flag is True):
            self.splittable = False
            return

        if len(max_information_gains_indices) == 1:
            self.dim_split = max_information_gains_indices[0]
        else:
            distinct_attr_vals = []
            for index in max_information_gains_indices:
                distinct_attr_vals.append(len(distinct_attribute_vals[index]))

            self.dim_split = max_information_gains_indices[distinct_attr_vals.index(max(distinct_attr_vals))]

        self.feature_uniq_split = list(distinct_attribute_vals[self.dim_split])

        #Recursively create tree

        self.feature_uniq_split.sort()

        for distinct_attribute_val in self.feature_uniq_split:
            new_features = []
            new_labels = []

            for feature_index, feature in enumerate(self.features):
                if feature[self.dim_split] == distinct_attribute_val:
                    new_features.append(feature[:self.dim_split] + feature[self.dim_split+1:])
                    new_labels.append(self.labels[feature_index])

            if(len(new_features) > 0 and (len(new_features) == len(new_labels))):
                new_child = TreeNode(new_features,new_labels,np.unique(new_labels).size)
                new_child.parent = self
                self.children.append(new_child)

        for child in self.children:
            if(child.splittable):
                child.split()
        return

    # TODO: predict the branch or the class
    def predict(self, feature_org):
        feature = feature_org[:]
        # feature: List[any]
        # return: int

        if(len(feature) == 0):
            return self.cls_max

        if(len(self.children) == 0 ):
            return self.cls_max
        else:
            child_index = -1
            for index, distinct_attr_val in enumerate(self.feature_uniq_split):
                if(feature[self.dim_split] == distinct_attr_val):
                    child_index = index
                    break

            feature.pop(self.dim_split)
            return self.children[child_index].predict(feature)