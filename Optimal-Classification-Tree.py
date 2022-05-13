
# coding: utf-8

# In[106]:


from anytree import Node, RenderTree, PreOrderIter
import random
import pandas as pd
import numpy as np


# creates a data frame from a list of lists
def to_data_frame(data, feature_names):
    return pd.DataFrame(data, columns = feature_names)

#split data into decision sets
def split_data(dataFrame, feature, operator, value):
    try:
        leftDataFrame = dataFrame.query('{feature} {operator} {value}'.format(feature = feature, operator = operator, value = str(value))).reset_index(drop=True)
        rightDataFrame = dataFrame.query('~({feature} {operator} {value})'.format(feature = feature, operator = operator, value = str(value))).reset_index(drop=True)
        return leftDataFrame, rightDataFrame
    except Exception as e:
        print(e,get_features(dataFrame), feature, operator, value)
        return pd.DataFrame(), pd.DataFrame()

# gets all features of the dataset, removing class
def get_features(dataFrame):
    features = []
    for column in dataFrame.columns:
        if column != "Class":
            features.append(column)
    return features

# Gets classes that are available in a data frame
def get_classes(dataFrame):
    return dataFrame["Class"].unique()


# In[96]:


'''

Error and Loss calculations

'''


# Calculates the error of a dataframe given a target class
def calculate_combined_error(dataFrame, targetClass):
    return len(dataFrame.query("Class != @targetClass").index)

# Calculates the error of a dataframe using the most popular class
def popular_error(dataFrame):
    return calculate_combined_error(dataFrame, dataFrame["Class"].value_counts().first_valid_index())

# Calculates the error value of how well the tree fits the data
# Input: Tree, Dataframe
# Recursive function
def error_value(tree, dataFrame):
    # misclassification error
    error = 0
    if not tree.is_leaf:
        
        # split data by decision
        feature, operator, value = tree.name.split()
        leftData,rightData  = split_data(dataFrame, feature, operator, value)
        # recure through left and right nodes
        if((not leftData.empty) and not tree.children[0].right):
            error += error_value(tree.children[0],leftData)
        if((not rightData.empty) and len(tree.children)>1):
            error += error_value(tree.children[1],rightData)
        elif((not rightData.empty) and tree.children[0].right):
            error += error_value(tree.children[0],rightData)
    else:
        error = len(dataFrame.query("Class != '{treeClass}'".format(treeClass = tree.Class)).index)
        
    return error



# Calculates the penalizing value for the complexity of the tree
# Input: Tree
# Recursive function
def complexity(tree):
    length = 0
    if not tree.is_leaf:
        # increase complexity by number of splits in the tree
        length += 1
        for child in tree.children:
            length += complexity(child)
    return length


# calculates the loss function for the tree and the respective data
def loss_function(tree, dataFrame, alpha = 0.5,baseLineError = 1):
    if baseLineError <= 0:
        baseLineInverse = 0
    else:
        baseLineInverse = (1/baseLineError)
    return (1/baseLineError) * error_value(tree, dataFrame) + alpha * complexity(tree)

# calculates the loss function for the tree and the respective data
def dict_loss_function(tree, error_dict, alpha = 0.5,baseLineError = 1):
    if baseLineError <= 0:
        baseLineInverse = 0
    else:
        baseLineInverse = (1/baseLineError)
    return baseLineInverse * get_dict_error(error_dict) + alpha * complexity(tree)


# In[97]:


def check_features(dataFrame, features, printErrors = False):
    if(printErrors):
        print(*(("Error", feature, " not in list") for feature in features if feature not in get_features(dataFrame)), sep='\n')
    if features:
        return [feature for feature in features if feature in get_features(dataFrame)]
    else:
        return []

#def check_splits(dataframe, n_min):
    #if len(dataFrame.index) > n_min * 2
    # and either 
    #     2 n_min unique values
    #     n_min <= 2 and 2 unique sets of duplicates
    #     n_min > 2 and bin packing 
    #elif unique values are 
    #

# Create a leaf
# Uses most populous class in available data frame
def create_leaf(dataFrame,parent=None,right=False):
    
    if not dataFrame.empty:
        
        # Get the most populous class
        Class = dataFrame['Class'].value_counts().first_valid_index()

        # Get accuracy of class
        accuracy = (dataFrame['Class'].value_counts().iloc[0]/len(dataFrame.index)).round(3)
        error = calculate_combined_error(dataFrame, Class)
        size = len(dataFrame.index)
        
        # Create Node using parent node
        leaf = Node("class = {Class}, accuracy = {accuracy}, size = {size} error = {error}".format(Class=Class,accuracy=accuracy,size = size, error = error),parent=parent, Class = Class, accuracy = accuracy, error = error,size = size, right = right, checked = False)

        return leaf
    else:
        return None
    
# recursive function for creating random trees
def random_tree(dataFrame, features = None, parent = None, remainingDepth = 0, right = False, operators = ["<"], n_min = 1):
    if(n_min < 1):
        n_min = 1
    # create node with splits
    if(remainingDepth > 0 and len(dataFrame.index) >= n_min * 2):
        features = check_features(dataFrame,features)
        if not features or len(features) == 0:
            features = get_features(dataFrame)
        
        # Add check to ensure that a split is even possible, points may share same values
        leftDataFrame, rightDataFrame = pd.DataFrame(), pd.DataFrame()
        # Restrict possible dataframes by minimum data points
        i = 0
        loopLimit = 5
        while (len(leftDataFrame.index) < n_min or len(rightDataFrame.index) < n_min) and i < loopLimit:
            # split decision
            ############################################
            # Pick random feature and operator
            feature, operator = random.choice(features), random.choice(operators)
            # Choose random value in available feature
            value = random.choice(dataFrame[feature])
            ############################################
            

            # Split dataset on decision
            leftDataFrame, rightDataFrame = split_data(dataFrame, feature, operator, value)
            if (len(leftDataFrame.index) < n_min or len(rightDataFrame.index) < n_min):
                i += 1
            
        if i < loopLimit:
            # Create decision node
            tree = Node('{feature} {operator} {value}'.format(feature = feature,operator = operator,value = value),parent = parent, right = right, checked = False)

            # create child nodes of decision node
            random_tree(leftDataFrame,features,tree,remainingDepth=remainingDepth-1,right=False, operators=operators, n_min = n_min)
            random_tree(rightDataFrame,features,tree,remainingDepth=remainingDepth-1,right=True, operators=operators, n_min = n_min)
            
        else:
            tree = create_leaf(dataFrame, parent, right)
        # Create a class node when at max length or no possible splits remaining
    else:
        # create leaf
        tree = create_leaf(dataFrame, parent, right)
    return tree

# prints the tree in ascii
def print_tree(tree):
    for pre, fill, node in RenderTree(tree):
        print("%s%s, %s, %s" % (pre, node.name,"Left" if not node.right else "Right", node.checked))


# In[102]:


############################################################

# Trims the dataset to appropriate data for the node
# Recurses from bottom to top
def data_values(node, dataFrame,right = None):
    
    # trim data starting from parent value
    if node.parent:
        dataFrame = data_values(node.parent,dataFrame,node.right)
        
    # if data is being split at node and the node has a split
    # Split data according to next node direction
    if right != None and node.children:
        
        feature, operator, value = node.name.split()
        return split_data(dataFrame, feature, operator, value)[int(right)]
    # If leaf or not splitting data at node
    # Return data at that node
    else:
        return dataFrame

# Verifies if all leaves in a tree have the minimum amount of data points
def verify_minimum_leaves(tree, n_min = 1):
    truth = True
    if tree.is_leaf:
        if tree.size < n_min:
            truth = False
    else:
        for child in tree.children:
            truth = truth == verify_minimum_leaves(child, n_min)
    
    return truth

############################################################

# Gets list of leaves of tree
def get_leaves(tree):
    leaves = []
    if not tree.is_leaf:
        for children in tree.children:
            leaves += get_leaves(children)
    else:
        leaves.append(tree)
    return leaves

# Creates dictionary of hashed leaves with values of 0
# Used for abstraction of leaves
def leaves_to_dict(tree):
    leaf_dict = {}
    leaves = get_leaves(tree)
    for leaf in leaves:
        leaf_dict[hash(leaf)] = 0
    return leaf_dict

#  Updates hashed leaf dictionary with data
def put_leaf_dict_data(tree, dataFrame, leafDict):
    # Update children of tree
    if not tree.is_leaf:
        
        # Get new data split for child nodes
        feature, operator, value = tree.name.split()
        leftData,rightData  = split_data(dataFrame, feature, operator, value)
        # Recure through all children
        if((not leftData.empty) and not tree.children[0].right):
            put_leaf_dict_data(tree.children[0],leftData,leafDict)
        if((not rightData.empty) and len(tree.children)>1):
            put_leaf_dict_data(tree.children[1],rightData,leafDict)
        elif((not rightData.empty) and tree.children[0].right):
            put_leaf_dict_data(tree.children[0],rightData,leafDict)
    else:
        leafDict[hash(tree)] = dataFrame['Class'].value_counts()

# Removes data values from hashed leaf dictionary
def minus_leaf_dict_data(tree, dataFrame, leafDict):
    # Update children of tree
    if not tree.is_leaf:
        
        # Get new data split for child nodes
        feature, operator, value = tree.name.split()
        leftData,rightData  = split_data(dataFrame, feature, operator, value)
        # Recure through all children
        if((not leftData.empty) and not tree.children[0].right):
            minus_leaf_dict_data(tree.children[0],leftData,leafDict)
        if((not rightData.empty) and len(tree.children)>1):
            minus_leaf_dict_data(tree.children[1],rightData,leafDict)
        elif((not rightData.empty) and tree.children[0].right):
            minus_leaf_dict_data(tree.children[0],rightData,leafDict)
    else:
        leafDict[hash(tree)] = leafDict[hash(tree)].subtract(dataFrame['Class'].value_counts(),fill_value = 0).sort_values(ascending = False)

# Adds data values to hashed leaf dictionary        
def plus_leaf_dict_data(tree, dataFrame, leafDict):
    # Update children of tree
    if not tree.is_leaf:
        
        # Get new data split for child nodes
        feature, operator, value = tree.name.split()
        leftData,rightData  = split_data(dataFrame, feature, operator, value)
        # Recure through all children
        if((not leftData.empty) and not tree.children[0].right):
            plus_leaf_dict_data(tree.children[0],leftData,leafDict)
        if((not rightData.empty) and len(tree.children)>1):
            plus_leaf_dict_data(tree.children[1],rightData,leafDict)
        elif((not rightData.empty) and tree.children[0].right):
            plus_leaf_dict_data(tree.children[0],rightData,leafDict)
    else:
        leafDict[hash(tree)] = dataFrame['Class'].value_counts().add(leafDict[hash(tree)],fill_value = 0).sort_values(ascending = False)

# Gets error of a sorted series (using most popular class)
def series_error(series):
    return series.sum() -  series.iloc[0] 

# Outputs error of series
def check_dict_error(item):
    if type(item) == pd.Series:
        return series_error(item)
    else:
        return np.inf

# Gets error of a hashed leaf dictionary
def get_dict_error(leafDict):
    error = 0
    for key in leafDict.keys():
        error += check_dict_error(leafDict[key])
    return error

# Outputs error of series
def get_dict_size(item):
    if type(item) == pd.Series:
        return item.sum()
    else:
        return 0

def min_leaf(leafDict,n_min):
    satisfies = True
    for leafSize in leafDict.values():
        if(get_dict_size(leafSize) < n_min):
            satisfies = False
    return satisfies
    

# Debugging function for error of a hashed leaf dictionary
def get_each_dict_error(leafDict):
    for key in leafDict.keys():
        print(leafDict[key])
        print(check_dict_error(leafDict[key]))

# Exhaustively finds the local optimal data split of a node while keeping it's children
def optimal_node_data_split_dict(node, dataFrame, alpha = 0.5, baseLineError = 1, n_min = 1): 
    # Gets children nodes of node
    childNodes = node.children
    
    # Sets best error as error of current node
    bestError = [error_value(node, dataFrame)*(1/baseLineError)]
    
    # Only runs on nodes 
    if(len(childNodes) != 1):
        # Creates empty child nodes for lead nodes
        if(len(childNodes) == 0):
            leftNode = Node("", right = False, checked = False)
            rightNode = Node("", right = True, checked = False)
            newSplitCost = 1
            
        else:
            leftNode = childNodes[0]
            rightNode = childNodes[1]
            newSplitCost = 0
        # Adjust best error for complexity cost
        bestError[0] -= alpha * newSplitCost
        

        
        # Loop through all features
        for feature in get_features(dataFrame):
            
            # Remove duplicate data points for splits in feature
            values = dataFrame[feature].drop_duplicates().sort_values().reset_index(drop=True)
            
            # For every feature, create/reset left and right dictionary
            left_leaf_dict = leaves_to_dict(leftNode)
            right_leaf_dict = leaves_to_dict(rightNode)
            
            # Place all data points in right leaf dictionary
            put_leaf_dict_data(rightNode, dataFrame, right_leaf_dict)
            
            # Loop through splits in feature
            for i, point in enumerate(values[:-1]):
                
                # Get all points in between current split and previous split
                samePoints = dataFrame.query('{feature} == {value}'.format(feature = feature,value = point))
                
                
                # Remove points from right leaf dictionary
                minus_leaf_dict_data(rightNode, samePoints, right_leaf_dict)

                # Add points from left leaf dictionary
                plus_leaf_dict_data(leftNode, samePoints, left_leaf_dict)
                
                
                # Check if new split satisfies n_min
                if(min_leaf(left_leaf_dict,n_min) and min_leaf(right_leaf_dict,n_min)):
                    
                    #Calculate error of split
                    new_error = (get_dict_error(left_leaf_dict) + get_dict_error(right_leaf_dict))*(1/baseLineError)
                    
                    # Set new best split
                    if(new_error < bestError[0]):
                        bestError = [new_error,feature,"<",(1/2 * (values.iloc[i] + values.iloc[i+1]))]
                        
                        # For debugging purposes
                        #print(bestError)
        # Replace node if best new split is better than current
        if(len(bestError) > 1):
            # Create split node
            bestErrorNode = Node('{feature} {operator} {value}'.format(feature = bestError[1],operator = bestError[2],value = bestError[3]), children = childNodes, checked = True)
            
            if(childNodes):
                # Update children of replacement node at new decision point
                update_tree(bestErrorNode,dataFrame)
                bestErrorNode.checked = True
            else:
                # Create children at new decision point for nodes which were previously leaves
                leftDataFrame,rightDataFrame = split_data(dataFrame,bestError[1],"<",bestError[3])
                create_leaf(leftDataFrame,parent=bestErrorNode,right=False)
                create_leaf(rightDataFrame,parent=bestErrorNode,right=True)
                
            # Replace old node with new node
            replace_node(bestErrorNode, node)

        else:
            node.checked = True
            bestErrorNode = node
        
        return bestErrorNode
        # If checking to update node here, add one to complexity when splitting on leaf
    else:
        # Runs delete node when only one child node exists
        return delete_split(node, dataFrame, alpha, baseLineError)
    
#######

def replace_node(newNode,oldNode):
    # Add new node to tree
    newNode.parent = oldNode.parent
                
    # Get the position of the node relative to the parent
    newNode.right = oldNode.right
            
    # Remove original node from tree
    oldNode.parent = None
                
    # Reset left positioned nodes
    place_left_node(newNode)
                
    #Uncheck parents for improvement
    uncheck_parents(newNode)

def place_left_node(node):
    if (not node.right and node.parent):
        if(len(node.parent.children)>1):
            node.parent.children = [node.parent.children[1],node.parent.children[0]]
    
# Create new split at given node
def optimal_split(node, dataFrame, alpha = 0.5, baseLineError = 1, n_min = 1):
    
    
    # Create new optimal split if it does not violate minimum number of data points
    if(len(dataFrame.index) >= n_min * 2):
        # Create parallel split
        newNode = optimal_node_data_split_dict(node, dataFrame, alpha = alpha, baseLineError = baseLineError, n_min = n_min)
        # change node to decision
        return newNode
    else:
        return node

# Updates tree with new data values
def update_tree(tree, dataFrame):
    # Reset checked flag
    tree.checked = False
    
    # Update children of tree
    if not tree.is_leaf:
        
        # Get new data split for child nodes
        feature, operator, value = tree.name.split()
        leftData,rightData  = split_data(dataFrame, feature, operator, value)
        
        # Recure through all children
        if((not leftData.empty) and not tree.children[0].right):
            update_tree(tree.children[0],leftData)
        if((not rightData.empty) and len(tree.children)>1):
            update_tree(tree.children[1],rightData)
        elif((not rightData.empty) and tree.children[0].right):
            update_tree(tree.children[0],rightData)
    else:
        # debugging
        if(len(dataFrame.index) == 0):
            print(" No items at tree: ", tree)
        
        
        # Update leaf for new data points
        Class = dataFrame['Class'].value_counts().first_valid_index()
        tree.error = calculate_combined_error(dataFrame, Class)
        tree.size = len(dataFrame.index)
        tree.accuracy =  (dataFrame['Class'].value_counts().iloc[0]/len(dataFrame.index)).round(3)
        tree.name = "class = {Class}, accuracy = {accuracy}, size = {size} error = {error}".format(Class=Class,accuracy=tree.accuracy,size = tree.size, error = tree.error)
        tree.Class =  Class

    
# Improve tree by replacing current split with one of it's children
def delete_split(node, dataFrame, alpha = 0.5, baseLineError = 1):
    # Set default node to original node
    newNode = node
    
    # Check if node has children
    if not node.is_leaf:
        
        # If multiple children
        if len(node.children) > 1:
            
            # Try replacing with left child
            leftNode = node.children[0]
            
            # Try replacing with right child
            rightNode = node.children[1]
            
            
            # Create leaf dictionaries for left and right child nodes
            left_leaf_dict = leaves_to_dict(leftNode)
            right_leaf_dict = leaves_to_dict(rightNode)
            
            # Place data in left and right leaf dictionaries
            put_leaf_dict_data(leftNode, dataFrame, left_leaf_dict)
            put_leaf_dict_data(rightNode, dataFrame, right_leaf_dict)
            
            # Get loss of  left and right leaf dictionaries
            rightLoss = dict_loss_function(rightNode, right_leaf_dict, alpha, baseLineError)
            leftLoss = dict_loss_function(leftNode, left_leaf_dict, alpha, baseLineError)
            
            # Get loss of original node
            currentLoss = loss_function(node, dataFrame, alpha, baseLineError)
            
            improved = False
            
            # Replace old node with left or right node if they have lower loss
            if leftLoss < rightLoss:
                if leftLoss < currentLoss:
                    newNode = leftNode
                    improved = True
            else:
                if rightLoss < currentLoss:
                    newNode = rightNode
                    improved = True
            
            
            # Place replacement node in the original tree
            if improved:
                
                # Replace node with new node
                replace_node(newNode,node)
                
                # Update nodes
                update_tree(newNode, dataFrame)
            else:
                # Check non-improved 
                node.checked = True
                
        # Single child
        else:
            newNode = node.children[0]
            
    return newNode

# Creates a new split at a leaf
# Buffer function to reduce potential unnecessary calculations
def create_split(node, dataFrame, alpha = 0.5, baseLineError = 1, n_min = 1):
    
    newNode = node
    
    if node.is_leaf:
        
        # Only creates leaf if accuracy cannot be improved or improved node would violate minimum data count at leaves
        if node.error > 0 and len(dataFrame.index) >= n_min * 2:
            
            newNode = optimal_split(node, dataFrame, alpha, baseLineError, n_min)
        else:
            # Checks the leaf if it cannot be improved
            node.checked = True
    
    # Incase non-leaves are run through create split
    else:
        newNode = optimal_split(node, dataFrame, alpha, baseLineError, n_min)
        
    return newNode
            
# Returns random unchecked node from tree
def get_random_node(tree):
    availableNodes = [node for node in PreOrderIter(tree) if not node.checked]
    if availableNodes:
        return random.choice(availableNodes)
    else:
        return None

# Gets root of node
# If no root, returns node
def get_root(node):
    return node.root

# Calculates baseline error of a dataset
def calculate_baseline_error(dataFrame):
    return len(dataFrame.index) - dataFrame["Class"].value_counts().iloc[0]

# Unchecks all parents of a node recursively
def uncheck_parents(node):
    if node.parent:
        node.parent.checked = False
        uncheck_parents(node.parent)

# Main loop for creating Optimal Classification Trees
def random_node_modification(tree, dataFrame, n_min = 1, D_max = 5, alpha = 0.5):
    
    # Calculate baseline error of data
    baseLineError = calculate_baseline_error(dataFrame)
    
    # Loops through while unchecked nodes are in the tree
    while get_random_node(tree):
        
        # Pick random unchecked node for modification
        node = tree#get_random_node(tree)
            
        # Get data frame at specific node
        dataFrameNode = data_values(node,dataFrame)
        # Runs update (parallel) split and delete split on decision nodes
        if not node.is_leaf:
            
            # Run update split
            node = optimal_split(node, dataFrameNode, alpha, baseLineError, n_min)
            
            # Run delete split
            node = delete_split(node, dataFrameNode, alpha, baseLineError)
        else:
            
            # Only creates new split if leaf is not at max depth
            if node.depth < D_max:
                node = create_split(node, dataFrameNode, alpha, baseLineError, n_min)
            else:
                node.checked = True
                
        # Reset the root if the node chosen for modification is the root
        if (node == get_root(node)):
            tree = node
        
    return tree


# In[105]:


# Finds best local optimum tree with given hyperparameters and tree count
def best_local_optimum_tree(dataFrame, n = 1, n_min = 1, D_max = 5, alpha = 0.5):
    
    # Initialse baseline
    baseLineError = calculate_baseline_error(dataFrame)
    
    # Intiliase error for baseline leaf
    optimalTree = create_leaf(dataFrame)
    bestLoss = loss_function(optimalTree, dataFrame, alpha, baseLineError)
    # Loop through number of trees
    for i in range(0,n):
        
        # Create random tree
        startingTree = random_tree(dataFrame, get_features(dataFrame), remainingDepth = D_max, n_min = n_min)
        
        # Run optimal tree function to find local optimum tree
        localOptimalTree = random_node_modification(startingTree,dataFrame,alpha = alpha, D_max = D_max, n_min = n_min)
        
        # Get loss of local optimum tree
        localLoss = loss_function(localOptimalTree, dataFrame, alpha, baseLineError)
        
        # Replace best tree with new local optimum tree if better
        if(localLoss < bestLoss):
            
            optimalTree = localOptimalTree
            
            # Update best loss
            bestLoss = localLoss
    
    return optimalTree

