

import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import itertools
from collections import Counter
import numpy as np
from scipy import stats
import statistics
from networkx.algorithms.community.centrality import girvan_newman


### function to plot the degree distribution of the graph
def plot_deg_dist(G):
    degrees = nx.degree(G)
#    print(degrees)
    all_deg = [v for k, v in degrees]
    unique_deg = list(set(all_deg))
    unique_deg.sort()
#    print(unique_deg)
    count_of_deg = [all_deg.count(x) for x in unique_deg]
    plt.plot(unique_deg, count_of_deg, '-ok')
    plt.ylabel('Number of Nodes')
    plt.xlabel('Degrees')
    plt.title('Degree Distribution')
    plt.show()
    
#    plt.loglog(unique_deg, count_of_deg,'-ok')
#    plt.ylabel('Number of Nodes')
#    plt.xlabel('Degrees')
#    plt.title('loglog Degree Distribution')
#    plt.show()


### function for applying the girvan newmann algorithm
def girvan(G):
    communities = girvan_newman(G)
    node_groups = []
    
    '''
    ### if we want only k clusters
    k = 1
    limited = itertools.takewhile(lambda c: len(c) <= k, communities)
    for com in limited:
        node_groups.append(list(sorted(c) for c in com))
    print(node_groups)
    '''
    
    ### for all clusters
    for com in next(communities):
      node_groups.append(list(com))
    
#    print(node_groups)
#    print(G.nodes())
    index = {}
    i=0
    for node in G:
        index[node] = i
        i+=1
    
    # color the clusters
    some_colors = ['red','blue','green','yellow','purple','cyan','magenta','brown','aqua','violet','orchid','pink','orange','royalblue']
    color_map = ['red']*G.order()
    for com in node_groups:
        col = random.choice(some_colors)
        for node in com:
            color_map[index[node]] = col
#    print(color_map)
#    nx.draw(G, node_color=color_map, with_labels=True)
#    plt.show()
    
    return node_groups



'''
#### not complete yet
### function to iterate recursively on the friends in a cluster
def decideRecursively(node, visited, response, index, SG):
    # mark as visited so that we don't call recursion on it again
    visited[node] = 1
    count_of_accepted_neighbors = 0
    neighbors = SG.neighbors(node)
    for ngh in neighbors:
        # if neighbor has been visited see it's response
        if visited[ngh]==1:
            # check if the neighbor has responded
            if response[ngh]==1:
                count_of_accepted_neighbors += 1
        else:
            decideRecursively(ngh, group_deg_pair, visited, SG)
    # probabilty to accept is 70% here
    if len(neighbors)>0 and count_of_accepted_neighbors/len(neighbors) > 0.70:
        response[index[node]] = 1
    else:
        response[index[node]] = 0
'''


def selectTargets(G, all_nodes):
    
    
    ### Method 1: probabilty of recieving proportional to popularity by degree
    # set the probability of getting new Request based on degree of that node
#    probOfGettingNewReq = [0.5]*len(all_nodes)
#    total_sum_deg = sum(all_deg)
#    for node in all_nodes:
#        probOfGettingNewReq[node] = G.degree(node)/total_sum_deg
    #print(probOfGettingNewReq)
    
    
    ### Method 2: probabilty of recieving inversely proportional to popularity by degree
    # set the probability of getting new Request based on (inverse)degree of that node
#    probOfGettingNewReqInv = [0.5]*len(all_nodes)
#    all_inv_deg = [1/dg for dg in all_deg]
#    total_sum_inv_deg = sum(all_inv_deg)
#    for node in all_nodes:
#        probOfGettingNewReqInv[node] = 1/(G.degree(node)*total_sum_inv_deg)
    #print(probOfGettingNewReqInv)
    
    
    ### Method 3: probabilty of recieving inversely proportional to popularity by degree
    # set the probability of getting new Request based on (Max Degree-degree) of that node
    probOfGettingNewReqSub = [0.5]*len(all_nodes)
    mx_dg = max(all_deg)
    all_sub_deg = [mx_dg-dg for dg in all_deg]
    total_sum_sub_deg = sum(all_sub_deg)
    for node in all_nodes:
        probOfGettingNewReqSub[node] = (mx_dg-G.degree(node))/total_sum_sub_deg
    #print(probOfGettingNewReqSub)
    
    
    # choose a method for target selection
    reqGettingProb = probOfGettingNewReqSub.copy()
    
    
    # select targets by random sampling of above "average" nodes
    above_avg_nodes = []
    std_dev = statistics.stdev(reqGettingProb)
    meanProb = statistics.mean(reqGettingProb)
#    print("sd = ",std_dev," mean = ",meanProb)
    for nd in all_nodes:
        # filter out nodes with high probability (but not very high)
        if reqGettingProb[nd] > meanProb-std_dev/4 and reqGettingProb[nd] < meanProb+std_dev:
            above_avg_nodes.append(nd)
    k = random.randrange(20,25)   # number of seed targets (randomly between 20-25)
    
    
    seed_targets = random.sample(above_avg_nodes,k)
#    print("seed_targets = ",seed_targets)
    
    
    
    # Visualisation of the seed targets
    
#    SeedGraph = G.subgraph(seed_targets).copy()
#    print(nx.info(SeedGraph))
#    tempo_gir = girvan(SeedGraph)
    

    
    targets = seed_targets.copy()
    # after choosing seed targets we will randomly select few friends of each
    for sd_trgt in seed_targets:
        neighbors = list(G.neighbors(sd_trgt))
        num = random.randrange(0,min(3,len(neighbors))+1)
        # select some neighbors randomly (or target them also based on probabilty of attack)
        random_neigh = random.sample(neighbors,num)
        for neigh in random_neigh:
            if neigh not in targets:
                targets.append(neigh)
    
    print("total targets = ",len(targets))
    
    # extract the subgraph of targets
    SG = G.subgraph(targets).copy()
    print(nx.info(SG))
    
    # plot the subgraph of targets
#    sp = nx.spring_layout(SG)
#    plt.axis('off')
#    nx.draw_networkx(SG,pos=sp,with_labels=False,node_size=30, node_color='blue')
#    plt.show()
    
#    nx.draw_spectral(SG, with_lables='True')
#    plt.show()
    
    
    
    ## splitting the targets into communities (using girvan newmann)
    
    node_groups = girvan(SG)
#    print(node_groups)
    
    # return the targets
    return targets





### auxiliary function to generate weighted choices from a list
def weighted_choices(w, mu, sd, k=1):
    # where mu is the index of w you want to be the most likely,
    # and sd is how tight you want that selection to be
    # k = number of elements to be selected
    weights = stats.norm(mu, sd).pdf(range(len(w)))
    return random.choices(w, weights=weights, k=k)




### this will classify nodes acc. to sybilEdge algorithm
def classifyNodeSybilEdge(G, all_nodes, nodeLabels, being_targeted_rate, accept_rate_from_real, accept_rate_from_fake, all_targets, fake_percentage):
    # G is the graph of social network
    # all_nodes is a list of all nodes in a sorted order
    # nodeLabels has two fields for each node, true_label and predicted
    # being_targeted_rate is a list of rate of selection by real vs fake
    # accept_rate_from_real and accept_rate_from_fake are list of response to real and fake nodes
    # targets is the list of targets selected
    
#    y1 = [being_targeted_rate[trgt]/(being_targeted_rate[trgt]+1.0) for trgt in all_nodes]
#    y2 = [1.0/(being_targeted_rate[trgt]+1.0) for trgt in all_nodes]
#    plt.hist(y1,bins=200,label='real/(real+fake) ratio',histtype=u'step')
#    plt.hist(y2,bins=200,label='fake/(real+fake) ratio',histtype=u'step')
#    plt.ylabel('Number of Nodes')
#    plt.title('Targets Recieve probabilty by real nodes vs fake nodes for all nodes')
#    plt.legend()
#    plt.show()
    
    
    all_percent_predictions = []
    
    for node in all_nodes:
        if nodeLabels[node]["predicted"]=='unknown':
            ### use SybilEdge formula to compute fake probability
            prior_fake_prob = fake_percentage/100 #(5-10% according to standards)
            
            fake_attacker = prior_fake_prob*1e30
            real_attacker = (1-prior_fake_prob)*1e30
            
            ### because the being_targeted_rate tells the ratio of selection by real/fake attacker numbers
            ### so probability of being selected by real attacker is (being_targeted_rate[trgt]/(being_targeted_rate[trgt]+1))
            ### so probability of being selected by real attacker is (1/(being_targeted_rate[trgt]+1))
            
            
            #### Method: when being_targeted_rate is sent
            # here the prob of getting a request is
            # dependent on the target's statistics of getting real/fake req.
            '''
            ## for targets who accepted the request
            targets = all_targets[node]['accept']
            # calculate the attacker being fake
            for trgt in targets:
                fake_attacker = fake_attacker*accept_rate_from_fake[trgt]*(1.0/(being_targeted_rate[trgt]+1.0))
                
            # calculate the attacker being real
            for trgt in targets:
                real_attacker = real_attacker*accept_rate_from_real[trgt]*(being_targeted_rate[trgt]/(being_targeted_rate[trgt]+1.0))
            
            
            ## for targets who rejected the request
            targets = all_targets[node]['reject']
            # calculate the attacker being fake
            for trgt in targets:
                fake_attacker = fake_attacker*(1.0-accept_rate_from_fake[trgt])*(1.0/(being_targeted_rate[trgt]+1.0))
                
            # calculate the attacker being real
            for trgt in targets:
                real_attacker = real_attacker*(1.0-accept_rate_from_real[trgt])*(being_targeted_rate[trgt]/(being_targeted_rate[trgt]+1.0))
            '''
            
            
            #### Method when real_fake_ratio is sent
            # here the prob of getting a req from atatcker
            # depends on the target nature(S/B)

            ## for targets who accepted the request
            targets = all_targets[node]['accept']
            # calculate the attacker being fake
            for trgt in targets:
                if nodeLabels[trgt]["true_label"]=='real':
                    fake_attacker = fake_attacker*accept_rate_from_fake[trgt]*(being_targeted_rate[node]/(being_targeted_rate[node]+1.0))
                else:
                    fake_attacker = fake_attacker*accept_rate_from_fake[trgt]*(1.0/(being_targeted_rate[node]+1.0))
                
            # calculate the attacker being real
            for trgt in targets:
                if nodeLabels[trgt]["true_label"]=='real':
                    real_attacker = real_attacker*accept_rate_from_real[trgt]*(being_targeted_rate[node]/(being_targeted_rate[node]+1.0))
                else:
                    real_attacker = real_attacker*accept_rate_from_real[trgt]*(1.0/(being_targeted_rate[node]+1.0))
            
            
            ## for targets who rejected the request
            targets = all_targets[node]['reject']
            # calculate the attacker being fake
            for trgt in targets:
                if nodeLabels[trgt]["true_label"]=='real':
                    fake_attacker = fake_attacker*(1.0-accept_rate_from_fake[trgt])*(being_targeted_rate[node]/(being_targeted_rate[node]+1.0))
                else:
                    fake_attacker = fake_attacker*(1.0-accept_rate_from_fake[trgt])*(1.0/(being_targeted_rate[node]+1.0))
                
            # calculate the attacker being real
            for trgt in targets:
                if nodeLabels[trgt]["true_label"]=='real':
                    real_attacker = real_attacker*(1.0-accept_rate_from_real[trgt])*(being_targeted_rate[node]/(being_targeted_rate[node]+1.0))
                else:
                    real_attacker = real_attacker*(1.0-accept_rate_from_real[trgt])*(1.0/(being_targeted_rate[node]+1.0))
            
            # final probability of being fake
            try:
                probabilty_being_fake = fake_attacker/(fake_attacker+real_attacker)
            except ZeroDivisionError:
               probabilty_being_fake = 0.5
#            print("probabilty_being_fake = ",probabilty_being_fake," ",nodeLabels[node]["true_label"])
            # all_percent_predictions.append(tuple([100*probabilty_being_fake,nodeLabels[node]["true_label"]]))
            all_percent_predictions.append(100*probabilty_being_fake)
    
#    print(all_percent_predictions)
#    plt.hist(all_percent_predictions,bins=200)
#    plt.title('Percentage Prediction of unknown nodes (SybilEdge)')
#    plt.xlabel('probability*100')
#    plt.ylabel('Number of Nodes')
#    plt.show()
    
    
    
    
    '''
    ##### Modified Algo using graph based property
    all_percent_predictions_modified = []
    
    for node in all_nodes:
        if nodeLabels[node]["predicted"]=='unknown':            
            ### use SybilEdge formula to compute fake probability
            prior_fake_prob = fake_percentage/100 #(5-10% according to standards)
            
            fake_attacker = prior_fake_prob*1e30
            real_attacker = (1-prior_fake_prob)*1e30
    
            #### Method when real_fake_ratio is sent
            # here the prob of getting a req from atatcker
            # depends on the target nature(S/B)

            ## for targets who accepted the request
            targets = all_targets[node]['accept']
            # calculate the attacker being fake
            for trgt in targets:
                if nodeLabels[trgt]["true_label"]=='real':
                    fake_attacker = fake_attacker*accept_rate_from_fake[trgt]*(being_targeted_rate[node]/(being_targeted_rate[node]+1.0))
                else:
                    fake_attacker = fake_attacker*accept_rate_from_fake[trgt]*(1.0/(being_targeted_rate[node]+1.0))
                
            # calculate the attacker being real
            for trgt in targets:
                if nodeLabels[trgt]["true_label"]=='real':
                    real_attacker = real_attacker*accept_rate_from_real[trgt]*(being_targeted_rate[node]/(being_targeted_rate[node]+1.0))
                else:
                    real_attacker = real_attacker*accept_rate_from_real[trgt]*(1.0/(being_targeted_rate[node]+1.0))
            
            
            ## for targets who rejected the request
            targets = all_targets[node]['reject']
            # calculate the attacker being fake
            for trgt in targets:
                if nodeLabels[trgt]["true_label"]=='real':
                    fake_attacker = fake_attacker*(1.0-accept_rate_from_fake[trgt])*(being_targeted_rate[node]/(being_targeted_rate[node]+1.0))
                else:
                    fake_attacker = fake_attacker*(1.0-accept_rate_from_fake[trgt])*(1.0/(being_targeted_rate[node]+1.0))
                
            # calculate the attacker being real
            for trgt in targets:
                if nodeLabels[trgt]["true_label"]=='real':
                    real_attacker = real_attacker*(1.0-accept_rate_from_real[trgt])*(being_targeted_rate[node]/(being_targeted_rate[node]+1.0))
                else:
                    real_attacker = real_attacker*(1.0-accept_rate_from_real[trgt])*(1.0/(being_targeted_rate[node]+1.0))
            
            # final probability of being fake
            try:
                probabilty_being_fake = fake_attacker/(fake_attacker+real_attacker)
            except ZeroDivisionError:
               probabilty_being_fake = 0.5
#            print("probabilty_being_fake = ",probabilty_being_fake," ",nodeLabels[node]["true_label"])
            all_percent_predictions_modified.append(tuple([100*probabilty_being_fake,nodeLabels[node]["true_label"]]))
    
#    print(all_percent_predictions_modified)
#    plt.hist(all_percent_predictions_modified,bins=200)
#    plt.title('Percentage Prediction of unknown nodes (SybilEdge)')
#    plt.xlabel('probability*100')
#    plt.ylabel('Number of Nodes')
#    plt.show()
    '''
    




def SybilEdge(G, all_nodes, nodeLabels, fake_percentage):
    ### first generate the prior data needed for the nodes
    # accept/reject rates for nodes
    # being targeted rates for nodes
    
    # for each node, we suppose she sent between 110-310% requests
    # keeping the weight around 150% and varying over different standard deviations
    
    
    excess_lower_bound = 110
    excess_upper_bound = 410
    percent_excess_requests = np.linspace(excess_lower_bound,excess_upper_bound,excess_upper_bound-excess_lower_bound) #create percentages at gap of 1%
    
    # mean values for real and fake nodes
    real_node_mean = random.uniform(135,145)-100
    fake_node_mean = random.uniform(250,270)-100
    
    # standard deviation for real and fake nodes
    # keep tight for real and loose for fake
    sd_real = random.randrange(12,15)
    sd_fake = random.randrange(23,25)
    
    real_req = []
    fake_req = []
    
    
    # set the number of (%)requests based on the labels of the nodes
    total_requests = []
    for node in all_nodes:
        # it is a real node
        if(nodeLabels[node]["true_label"]=='real'):
            total_requests.append(weighted_choices(percent_excess_requests, real_node_mean, sd_real)[0])
            total_requests[-1] = round(total_requests[-1])
            if total_requests[-1] > excess_upper_bound:
                total_requests[-1] = excess_upper_bound
            if total_requests[-1] < excess_lower_bound:
                total_requests[-1] = excess_lower_bound
            real_req.append(total_requests[-1])
        # it is fake node
        else:
            total_requests.append(weighted_choices(percent_excess_requests, fake_node_mean, sd_fake)[0])
            if total_requests[-1] > excess_upper_bound:
                total_requests[-1] = excess_upper_bound
            if total_requests[-1] < excess_lower_bound:
                total_requests[-1] = excess_lower_bound
            fake_req.append(total_requests[-1])
    
    ## plot and see the extent of requests sent out by real and fake nodes
#    real_req.sort()
#    fake_req.sort()
#    plt.plot(np.linspace(0,len(fake_req),len(fake_req)),fake_req,label="fake nodes",color="r")
#    plt.plot(np.linspace(0,len(real_req),len(real_req)),real_req,label="real nodes",color="blue")
#    plt.xlabel('nodes')
#    plt.ylabel('total_requests_excess_percent')
#    plt.legend()
#    plt.show()
    
    
    ## Now we will assign the targets who rejected requests for each node
    ## The friend nodes already accepted, so we will randomly select nodes
    
    # stores the count of real and fake requests received by each node (2D list)
    being_targeted_real_vs_fake = [(0,0,0,0)]*len(all_nodes)
    # stores the count of real and fake requests accepted/rejected by each node (2D list)
    accept_response_real_vs_fake = [(0,0,0,0)]*len(all_nodes)
    reject_response_real_vs_fake = [(0,0,0,0)]*len(all_nodes)
    ### -> the 0 index denotes the real->real part
    ### -> the 1 index denotes the fake->real part
    ### -> the 2 index denotes the real->fake part
    ### -> the 3 index denotes the fake->fake part
    
    # separate the real and fake nodes
    real_nodes = []
    fake_nodes = []
    for node in all_nodes:
        if nodeLabels[node]["true_label"]=='real':
            real_nodes.append(node)
        else:
            fake_nodes.append(node)
    
    # make a set of all nodes for faster implementation
    set_of_real_nodes = set(real_nodes)
    set_of_fake_nodes = set(fake_nodes)
    
    
    ## stores number of real requests sent
    real_sent = [0]*len(all_nodes)
    ## stores number of fake requests sent
    fake_sent = [0]*len(all_nodes)
    
    
    ## list of all friend requests made by a node(accepted + rejected)
    all_requests = []
    
    
    for node in all_nodes:
        neighbors = set(G.neighbors(node))
        excess_req = math.ceil((total_requests[node]/100-1)*len(neighbors))
        ## set up a dict for accept and reject requests made 
        friend_requests_made = dict()
        accept_friend_requests_made = neighbors.copy()
        reject_friend_requests_made = set()
        friend_requests_made['accept'] = accept_friend_requests_made
        # add the node itself to neighbors set to avoid choosing it as target
        neighbors.add(node)
        # we select targets only from nodes who are not friend
        choose_fake_from = set_of_fake_nodes.copy()
        choose_fake_from.difference(neighbors)
        choose_real_from = set_of_real_nodes.copy()
        choose_real_from.difference(neighbors)
        
        # if node is real
        # for real nodes -> we select 85% of excess requests for reals and 15% for fakes
        if nodeLabels[node]["true_label"]=='real':
            ## fake part (15%)
            fake_excess = math.ceil(0.15*excess_req)
            # update fake sent
            fake_sent[node] = fake_sent[node]+fake_excess
            # now randomly choose fake targets from nodes who are not friends
            fake_targets = random.sample(choose_fake_from, min(fake_excess,len(choose_fake_from)))
            ## add the targets in all_requests list
            reject_friend_requests_made = reject_friend_requests_made.union(set(fake_targets))
            # now update counts in the being_targeted and response lists
            for trgt in fake_targets:
                # increse count of a fake node being targeted by a real node
                tup = list(being_targeted_real_vs_fake[trgt])
                tup[2] += 1
                being_targeted_real_vs_fake[trgt] = tuple(tup)
                # increase count of a fake node rejecting a real node
                tup1 = list(reject_response_real_vs_fake[trgt])
                tup1[2] += 1
                reject_response_real_vs_fake[trgt] = tuple(tup1)
            
            ## real part (85%)
            real_excess = excess_req-fake_excess
            # update real sent
            real_sent[node] = real_sent[node]+real_excess
            # now randomly choose real targets from nodes who are not friends
            real_targets = random.sample(choose_real_from, min(real_excess,len(choose_real_from)))
            ## add the targets in all_requests list
            reject_friend_requests_made = reject_friend_requests_made.union(set(real_targets))
            # now update counts in the being_targeted and response lists
            for trgt in real_targets:
                # increse count of a real node being targeted by a real node
                tup2 = list(being_targeted_real_vs_fake[trgt])
                tup2[0] += 1
                being_targeted_real_vs_fake[trgt] = tuple(tup2)
                # increase count of a real node rejecting a real node
                tup3 = list(reject_response_real_vs_fake[trgt])
                tup3[0] += 1
                reject_response_real_vs_fake[trgt] = tuple(tup3)
        
        # if node is fake
        # for fake nodes -> we select 30% of excess requests from reals and 70% from fakes
        else:
            ## fake part (70%)
            fake_excess = math.ceil(0.7*excess_req)
            # update fake sent
            fake_sent[node] = fake_sent[node]+fake_excess
            # now randomly choose fake targets from nodes who are not friends
            fake_targets = random.sample(choose_fake_from, min(fake_excess,len(choose_fake_from)))
            ## add the targets in all_requests list
            reject_friend_requests_made = reject_friend_requests_made.union(set(fake_targets))
            # now update counts in the being_targeted and response lists
            for trgt in fake_targets:
                # increse count of a fake node being targeted by a fake node
                tup4  = list(being_targeted_real_vs_fake[trgt])
                tup4[3] += 1
                being_targeted_real_vs_fake[trgt] = tuple(tup4)
                # increase count of a fake node rejecting a fake node
                tup5 = list(reject_response_real_vs_fake[trgt])
                tup5[3] += 1
                reject_response_real_vs_fake[trgt] = tuple(tup5)
            
            ## real part (30%)
            real_excess = excess_req-fake_excess
            # update real sent
            real_sent[node] = real_sent[node]+real_excess
            # now randomly choose real targets from nodes who are not friends
            real_targets = random.sample(choose_real_from, min(real_excess,len(choose_real_from)))
            ## add the targets in all_requests list
            reject_friend_requests_made = reject_friend_requests_made.union(set(real_targets))
            # now update counts in the being_targeted and response lists
            for trgt in real_targets:
                # increse count of a real node being targeted by a fake node
                tup6  = list(being_targeted_real_vs_fake[trgt])
                tup6[1] += 1
                being_targeted_real_vs_fake[trgt] = tuple(tup6)
                # increase count of a real node rejecting a fake node
                tup7 = list(reject_response_real_vs_fake[trgt])
                tup7[1] += 1
                reject_response_real_vs_fake[trgt] = tuple(tup7)
        
        friend_requests_made['reject'] = reject_friend_requests_made
        all_requests.append(friend_requests_made)
    
    
    
    # now iterate through all edges and udpate the targeted rate and accept rates
    for edge in G.edges():
        # randomly choose one node to be a target and other to be targeter
        trgt = random.choice(edge)
        if trgt==edge[0]:
            sender = edge[1]
        else:
            sender = edge[0]
        if nodeLabels[trgt]["true_label"]=='real':
            # update real sent
            real_sent[sender] = real_sent[sender]+1
            
            if nodeLabels[sender]["true_label"]=='real':
                tup  = list(being_targeted_real_vs_fake[trgt])
                tup[0] += 1
                being_targeted_real_vs_fake[trgt] = tuple(tup)
                tup1 = list(accept_response_real_vs_fake[trgt])
                tup1[0] += 1
                accept_response_real_vs_fake[trgt] = tuple(tup1)
            else:
                tup2  = list(being_targeted_real_vs_fake[trgt])
                tup2[1] += 1
                being_targeted_real_vs_fake[trgt] = tuple(tup2)
                tup3 = list(accept_response_real_vs_fake[trgt])
                tup3[1] += 1
                accept_response_real_vs_fake[trgt] = tuple(tup3)
        else:
            # update fake sent
            fake_sent[sender] = fake_sent[sender]+1
            
            if nodeLabels[sender]["true_label"]=='real':
                tup4  = list(being_targeted_real_vs_fake[trgt])
                tup4[2] += 1
                being_targeted_real_vs_fake[trgt] = tuple(tup4)
                tup5 = list(accept_response_real_vs_fake[trgt])
                tup5[2] += 1
                accept_response_real_vs_fake[trgt] = tuple(tup5)
            else:
                tup6  = list(being_targeted_real_vs_fake[trgt])
                tup6[3] += 1
                being_targeted_real_vs_fake[trgt] = tuple(tup6)
                tup7 = list(accept_response_real_vs_fake[trgt])
                tup7[3] += 1
                accept_response_real_vs_fake[trgt] = tuple(tup7)
    
    
    
    ## this ratio will tell considering all sent friend requests
    ## what is ratio of real/fake sent
    real_fake_ratio_of_req_sent = [0.5]*len(all_nodes)
    for nd in all_nodes:
        if real_sent[nd]==0:
            real_fake_ratio_of_req_sent[nd] = (1+real_sent[nd])/(1+fake_sent[nd])
        else:
            real_fake_ratio_of_req_sent[nd] = real_sent[nd]/fake_sent[nd]
#    plt.hist(real_fake_ratio_of_req_sent,bins=200,histtype=u'step')
#    plt.title('Real/Fake ratio of friend requests sent for all nodes')
#    plt.ylabel('Number of Nodes')
#    plt.show()
    
    
    
    ### plot and visualise the reject rates for real and fake nodes
    
    ### reject rate = #rejects/#total requests
#    plot_reject_rate_real = [(total_requests[node]-100)/total_requests[node] for node in real_nodes if total_requests[node]>0]
#    plot_reject_rate_fake = [(total_requests[node]-100)/total_requests[node] for node in fake_nodes if total_requests[node]>0]
#    plt.hist(plot_reject_rate_real,bins=200,label='real',histtype=u'step')
#    plt.hist(plot_reject_rate_fake,bins=200,label='fake',histtype=u'step')
#    plt.ylabel('Number of Nodes')
#    plt.title('Reject Rates')
#    plt.legend()
#    plt.show()
    
    
    
    ### plot and visualise the targets' response rates to real and fake nodes
    
    # accept_rate from real to real is #accept from reals to reals/(#accept from reals to reals + #rejects from reals to reals)
#    plot_accept_rate_from_real_to_real = [accept_response_real_vs_fake[node][0]/(accept_response_real_vs_fake[node][0]+reject_response_real_vs_fake[node][0]) for node in real_nodes if accept_response_real_vs_fake[node][0]+reject_response_real_vs_fake[node][0]>0]
#    plot_accept_rate_from_fake_to_real = [accept_response_real_vs_fake[node][1]/(accept_response_real_vs_fake[node][1]+reject_response_real_vs_fake[node][1]) for node in real_nodes if accept_response_real_vs_fake[node][1]+reject_response_real_vs_fake[node][1]>0]
#    plot_accept_rate_from_real_to_fake = [accept_response_real_vs_fake[node][2]/(accept_response_real_vs_fake[node][2]+reject_response_real_vs_fake[node][2]) for node in fake_nodes if accept_response_real_vs_fake[node][2]+reject_response_real_vs_fake[node][2]>0]
#    plot_accept_rate_from_fake_to_fake = [accept_response_real_vs_fake[node][3]/(accept_response_real_vs_fake[node][3]+reject_response_real_vs_fake[node][3]) for node in fake_nodes if accept_response_real_vs_fake[node][3]+reject_response_real_vs_fake[node][3]>0]
#    plt.hist(plot_accept_rate_from_real_to_real,bins=200,label='real to real',histtype=u'step')
#    plt.hist(plot_accept_rate_from_fake_to_real,bins=200,label='fake to real',histtype=u'step')
#    plt.hist(plot_accept_rate_from_real_to_fake,bins=200,label='real to fake',histtype=u'step')
#    plt.hist(plot_accept_rate_from_fake_to_fake,bins=200,label='fake to fake',histtype=u'step')
#    plt.ylabel('Number of nodes')
#    plt.xlabel('(accept rate for reals)/(accept rate for fakes)')
#    plt.title('Targets\' accept rates for requests(finer version)')
#    plt.legend()
#    plt.show()
    
#    accept_rate_ratio_for_real = [(accept_response_real_vs_fake[node][0]*(accept_response_real_vs_fake[node][1]+reject_response_real_vs_fake[node][1]))/(accept_response_real_vs_fake[node][1]*(accept_response_real_vs_fake[node][0]+reject_response_real_vs_fake[node][0])) for node in real_nodes if accept_response_real_vs_fake[node][1]>0 and accept_response_real_vs_fake[node][0]+reject_response_real_vs_fake[node][0]>0]
#    accept_rate_ratio_for_fake = [(accept_response_real_vs_fake[node][2]*(accept_response_real_vs_fake[node][3]+reject_response_real_vs_fake[node][3]))/(accept_response_real_vs_fake[node][3]*(accept_response_real_vs_fake[node][2]+reject_response_real_vs_fake[node][2])) for node in fake_nodes if accept_response_real_vs_fake[node][3]>0 and accept_response_real_vs_fake[node][2]+reject_response_real_vs_fake[node][2]>0]
#    plt.hist(accept_rate_ratio_for_real,bins=200,label='real',histtype=u'step')
#    plt.hist(accept_rate_ratio_for_fake,bins=200,label='fake',histtype=u'step')
#    plt.ylabel('Number of nodes')
#    plt.xlabel('(accept rate for reals)/(accept rate for fakes)')
#    plt.title('Targets\' accept rate ratio for requests from real senders vs. fakes')
#    plt.legend()
#    plt.show()
    
    
    ### keep the accept_rate > 0
    accept_rate_from_real = []
    accept_rate_from_fake = []
    for node in all_nodes:
        if nodeLabels[node]["true_label"]=='real':
            # set real accept rate
            if accept_response_real_vs_fake[node][0]+reject_response_real_vs_fake[node][0]>0:
                prob_acpt = (1+accept_response_real_vs_fake[node][0])/(1+accept_response_real_vs_fake[node][0]+reject_response_real_vs_fake[node][0])
                if prob_acpt>0.99:
                    prob_acpt = random.uniform(0.96,0.98)
                accept_rate_from_real.append(prob_acpt)
            else:
                # no requests recieved by this real node, set accept rate higher than average
                acptreal = random.uniform(0.5,0.7)
                accept_rate_from_real.append(acptreal)
            # set fake accept rate
            if accept_response_real_vs_fake[node][1]+reject_response_real_vs_fake[node][1]>0:
                prob_acpt = (1+accept_response_real_vs_fake[node][1])/(1+accept_response_real_vs_fake[node][1]+reject_response_real_vs_fake[node][1])
                if prob_acpt>0.99:
                    prob_acpt = random.uniform(0.96,0.98)
                accept_rate_from_fake.append(prob_acpt)
            else:
                # no requests recieved by this real node, set accept rate lower than average
                acptfake = random.uniform(0.2,0.4)
                accept_rate_from_fake.append(acptfake)
        else:
            # set real accept rate
            if accept_response_real_vs_fake[node][2]+reject_response_real_vs_fake[node][2]>0:
                prob_acpt = (1+accept_response_real_vs_fake[node][2])/(1+accept_response_real_vs_fake[node][2]+reject_response_real_vs_fake[node][2])
                if prob_acpt>0.99:
                    prob_acpt = random.uniform(0.96,0.98)
                accept_rate_from_real.append(prob_acpt)
            else:
                # no requests recieved by this fake node, set accept rate very high
                acptreal = random.uniform(0.8,0.9)
                accept_rate_from_real.append(acptreal)
            # set fake accept rate
            if accept_response_real_vs_fake[node][3]+reject_response_real_vs_fake[node][3]>0:
                prob_acpt = (1+accept_response_real_vs_fake[node][3])/(1+accept_response_real_vs_fake[node][3]+reject_response_real_vs_fake[node][3])
                if prob_acpt>0.99:
                    prob_acpt = random.uniform(0.96,0.98)
                accept_rate_from_fake.append(prob_acpt)
            else:
                # no requests recieved by this fake node, set accept rate lower than average
                acptfake = random.uniform(0.4,0.5)
                accept_rate_from_fake.append(acptfake)
    
#    plt.hist(accept_rate_from_real,bins=200,label='from real',histtype=u'step')
#    plt.hist(accept_rate_from_fake,bins=200,label='from fake',histtype=u'step')
#    plt.title('Accept rates (for all nodes)')
#    plt.ylabel('Number of nodes')
#    plt.legend()
#    plt.show()
    
    
    
    ### plot and visualise the being_targeted_rates by real and fake nodes
    
#    plot_being_targeted_count_real_to_real = [being_targeted_real_vs_fake[node][0] for node in real_nodes]
#    plot_being_targeted_count_fake_to_real = [being_targeted_real_vs_fake[node][1] for node in real_nodes]
#    plot_being_targeted_count_real_to_fake = [being_targeted_real_vs_fake[node][2] for node in fake_nodes]
#    plot_being_targeted_count_fake_to_fake = [being_targeted_real_vs_fake[node][3] for node in fake_nodes]
#    plt.hist(plot_being_targeted_count_real_to_real,bins=200,label='real to real',histtype=u'step')
#    plt.hist(plot_being_targeted_count_fake_to_real,bins=200,label='fake to real',histtype=u'step')
#    plt.hist(plot_being_targeted_count_real_to_fake,bins=200,label='real to fake',histtype=u'step')
#    plt.hist(plot_being_targeted_count_fake_to_fake,bins=200,label='fake to fake',histtype=u'step')
#    plt.title('Targets\' receive counts from real senders vs. fakes(finer version)')
#    plt.legend()
#    plt.show()
    
#    plot_being_targeted_count_real = [being_targeted_real_vs_fake[node][0]+being_targeted_real_vs_fake[node][1] for node in real_nodes]
#    plot_being_targeted_count_fake = [being_targeted_real_vs_fake[node][2]+being_targeted_real_vs_fake[node][3] for node in fake_nodes]
#    plt.hist(plot_being_targeted_count_real,bins=200,label='real',histtype=u'step')
#    plt.hist(plot_being_targeted_count_fake,bins=200,label='fake',histtype=u'step')
#    plt.title('Targets\' receive counts from real senders vs. fakes')
#    plt.ylabel('Number of nodes')
#    plt.legend()
#    plt.show()
    
    
    being_targeted_rate = []
    for node in all_nodes:
        if nodeLabels[node]["true_label"]=='real':
            if being_targeted_real_vs_fake[node][1]>0:
                being_targeted_rate.append(being_targeted_real_vs_fake[node][0]/being_targeted_real_vs_fake[node][1])
            else:
                being_targeted_rate.append(random.uniform(1,3))
        else:
            if being_targeted_real_vs_fake[node][3]>0:
                being_targeted_rate.append(being_targeted_real_vs_fake[node][2]/being_targeted_real_vs_fake[node][3])
            else:
                being_targeted_rate.append(random.uniform(1,3))
#    plt.hist(being_targeted_rate,bins=200,histtype=u'step')
#    plt.title('Targets\' receive rates from real senders vs. fakes')
#    plt.ylabel('Number of nodes')
#    plt.xlabel('(fraction of reals\' requests to target)/(fraction of fakes\' requests to target)')
#    plt.show()
    
    
    
    
    ### now we have setup the prior information
    ### now we will stimulate the newNodeSybilEdge
    
    # choose targets
#    targets = selectTargets(G, all_nodes)
    
    
    # call the score assigning function
    # with real_fake_ratio
    classifyNodeSybilEdge(G, all_nodes, nodeLabels, real_fake_ratio_of_req_sent, accept_rate_from_real, accept_rate_from_fake, all_requests, fake_percentage)
    # with being_targeted_rate
#    classifyNodeSybilEdge(G, all_nodes, nodeLabels, being_targeted_rate, accept_rate_from_real, accept_rate_from_fake, all_requests, fake_percentage)





def classifyNodeGraphbasedAlgo(G, all_nodes, nodeLabels, being_targeted_rate, accept_rate_from_real, accept_rate_from_fake, all_requests, fake_percentage):
    
    
    
    
    return





### function for adding the new node to the graph
def newNodeAddition(G, all_nodes, nodeLabels, reqAcceptProb, reqGettingProb):
    #plt.plot(all_nodes,reqGettingProb)
    #plt.show()
    
    # select targets by random sampling of above "average" nodes
    above_avg_nodes = []
    std_dev = statistics.stdev(reqGettingProb)
    meanProb = statistics.mean(reqGettingProb)
    print("sd = ",std_dev," mean = ",meanProb)
    for nd in all_nodes:
        # filter out nodes with high probability (but not very high)
        if reqGettingProb[nd] > meanProb-std_dev/4 and reqGettingProb[nd] < meanProb+std_dev:
            above_avg_nodes.append(nd)
    k = random.randrange(20,25)   # number of seed targets (randomly between 20-25)
    
    
    seed_targets = random.sample(above_avg_nodes,k)
    print("seed_targets = ",seed_targets)
    
    
    
    # Visualisation of the seed targets
    
#    SeedGraph = G.subgraph(seed_targets).copy()
#    print(nx.info(SeedGraph))
#    tempo_gir = girvan(SeedGraph)
    

    
    targets = seed_targets.copy()
    # after choosing seed targets we will randomly select few friends of each
    for sd_trgt in seed_targets:
        neighbors = list(G.neighbors(sd_trgt))
        num = random.randrange(0,min(3,len(neighbors))+1)
        # select some neighbors randomly (or target them also based on probabilty of attack)
        random_neigh = random.sample(neighbors,num)
        for neigh in random_neigh:
            if neigh not in targets:
                targets.append(neigh)
    
    print("total targets = ",len(targets))
    
    # extract the subgraph of targets
    SG = G.subgraph(targets).copy()
    print(nx.info(SG))
    
    # plot the subgraph of targets
#    sp = nx.spring_layout(SG)
#    plt.axis('off')
#    nx.draw_networkx(SG,pos=sp,with_labels=False,node_size=30, node_color='blue')
#    plt.show()
    
#    nx.draw_spectral(SG, with_lables='True')
#    plt.show()
    
    
    
    ## splitting the targets into communities (using girvan newmann)
    
    node_groups = girvan(SG)
#    print(node_groups)
    
    
    
    
    ##### Method 1
    
    # now we decide for each node whether to accept or reject
    # for less degree nodes, we do a random thing (we can use prior if we have the data)
    # for higher degree nodes, we decide on the basis of fraction of neighbors accept(or reject) ratio
    # OR
    # we can recursively start from the highest degree node and call upon all its neighbors
    # then decide based on their decision
    
    
    # make a node-deg pair for each target node
    target_deg_pair = []
    for node in targets:
        # we are using the degree from subgraph of targets and not the main graph
        target_deg_pair.append([node, SG.degree(node)])
    
    # sort all the nodes based on their degrees in sub graph
    target_deg_pair.sort(key=lambda x:x[1])
#    print(target_deg_pair)
    
    
    # once sorted with respect to their degrees (alternative is to sort wrt time of acceptance)
    min_degree = target_deg_pair[0][1]
    
    
    # response list stores all of the targets' responses
    response = [-1]*len(target_deg_pair)
    
    # indexing the targets for easier implementation
    index_targets = {}
    for i in range(len(targets)):
        index_targets[targets[i]] = i
    
    '''
    # setting up the targets' responses
    for i in range(len(target_deg_pair)):
        # if smallest degree nodes, decide randomly (or we can use prior bias) (or 50% - something proportional to degree-min_degree)
        if(target_deg_pair[i][1]==min_degree):
            x = random.random()
            if x > reqAcceptProb[target_deg_pair[i][0]]:
                response[i] = 1
            else:
                response[i] = 0
        # else we decide by seeing the behaviour of neighbors
        else:
            # get the list of neighbors who were targeted
            neighbors = list(SG.neighbors(target_deg_pair[i][0]))
            # fraction of neighbors who accepted the request
            total_ngh_responses = len(neighbors)
            fraction = 0.0
            for ngh in neighbors:
                # if the neighbor has accepted
                if response[index_targets[ngh]] == 1:
                    fraction+=1.0
                # if the neighbor has not responded yet
                elif response[index_targets[ngh]] == -1:
                    total_ngh_responses-=1.0
            if total_ngh_responses>0:
                fraction = fraction/total_ngh_responses
            print(total_ngh_responses," ",len(neighbors))
            # accept if (70% + something proportional to degree-min_degree)
            if fraction >= 0.70:
                response[i] = 1
            else:
                response[i] = 0
#    print(response)
    '''
    
    # iterating over communities
    for comm in node_groups:
        print("comm = ",comm)
        deg_com = [SG.degree(nd) for nd in comm]
        min_degree = min(deg_com)
        print(deg_com," ",min_degree)
        for node in comm:
            # if smallest degree node, decide randomly (or we can use prior bias) (or 50% - something proportional to degree-min_degree)
            if(target_deg_pair[index_targets[node]][1]==min_degree):
                x = random.random()
                if x > reqAcceptProb[target_deg_pair[index_targets[node]][0]]:
                    response[i] = 1
                else:
                    response[i] = 0
            # else we decide by seeing the behaviour of neighbors
            else:
                # get the list of neighbors who were targeted
                neighbors = list(SG.neighbors(target_deg_pair[index_targets[node]][0]))
                # fraction of neighbors who accepted the request
                total_ngh_responses = len(neighbors)
                fraction = 0.0
                for ngh in neighbors:
                    # if the neighbor has accepted
                    if response[index_targets[ngh]] == 1:
                        fraction+=1.0
                    # if the neighbor has not responded yet
                    elif response[index_targets[ngh]] == -1:
                        total_ngh_responses-=1.0
                if total_ngh_responses>0:
                    fraction = fraction/total_ngh_responses
                print(total_ngh_responses," ",len(neighbors))
                # accept if (70% + something proportional to degree-min_degree)
                if fraction >= 0.70:
                    response[i] = 1
                else:
                    response[i] = 0
#    print(response)
    
    
    
    
    # set the values of accept_count and reject_count
    accept_count = 0
    reject_count = 0
    for resp in response:
        if resp==1:
            accept_count+=1
        else:
            reject_count+=1
    print("accept rate = ",100*accept_count/len(targets)," reject rate = ",100*reject_count/len(targets))

    
    
    '''
    ##### Method 2
    
    # make a node-deg pair for each community
    node_group_deg_pair = []
    for group in node_groups:
        grp_pair = []
        for node in group:
            grp_pair.append([node, SG.degree(node)])
        node_group_deg_pair.append(grp_pair)        
    
    # sort all the groups based on their degrees in main graph (in reverse order)
    for group in node_group_deg_pair:
        group.sort(key=lambda x:x[1], reverse=True)
    print(node_group_deg_pair)
    
    
    # response list stores all of the targets' responses
    response = [-1]*len(target_deg_pair)
    
    # indexing the targets for easier implementation
    index_targets = {}
    for i in range(len(targets)):
        index_targets[targets[i]] = i

    
    # call the recursive function from highest degree node
#    for grp in node_group_deg_pair:
        # visited list to mark nodes already called in recursion
#        visited = [0]*len(grp)
#        decideRecursively(grp[0][0], visited, index_targets, SG)
    
#    print(response)
    '''
    
    
    # add the newNode to the network after all the processing
    n = G.order()
    G.add_node(n)
    for trgt in targets:
        # this target has accepted the request
        if response[index_targets[trgt]]==1:
            G.add_edge(trgt,n)
            
    # set the label of the new node (sybil/benign)
    nodeLabels[n] = {"true_label":'unknown',"predicted":'unknown'}
    
    # how to tag it as real or fake??
    # using sybilEdge formula ?








# read the data from the file
G = nx.read_edgelist('facebook_combined.txt',create_using=nx.Graph(),nodetype=int)

print(nx.info(G))

# spring is better than spectral
#sp = nx.spring_layout(G)
#plt.axis('off')
#nx.draw_networkx(G,pos=sp,with_labels=False,node_size=10, node_color='r')
#plt.show()

#nx.draw_spectral(G, with_lables='True')
#plt.show()



#girvan(G)



#plot_deg_dist(G)

### density plots
#print("density = ",nx.density(G))
#clustering = nx.clustering(G)
#print(clustering)
#print("average clustering = ",nx.average_clustering(G))
#print("diameter = ",nx.diameter(G))



all_nodes = list(G.nodes)
all_nodes.sort()

all_deg = [G.degree(node) for node in all_nodes]



probOfGettingNewReq = [0.5]*len(all_nodes)
# can we set some prior?? for accepting new nodes and then update it with new additions
probOfAcceptingNewReq = [0.5]*len(all_nodes)


### Method 1: probabilty of recieving proportional to popularity by degree
# set the probability of getting new Request based on degree of that node
total_sum_deg = sum(all_deg)
for node in all_nodes:
    probOfGettingNewReq[node] = G.degree(node)/total_sum_deg
#print(probOfGettingNewReq)


### Method 2: probabilty of recieving inversely proportional to popularity by degree
# set the probability of getting new Request based on (inverse)degree of that node
probOfGettingNewReqInv = [0.5]*len(all_nodes)
all_inv_deg = [1/dg for dg in all_deg]
total_sum_inv_deg = sum(all_inv_deg)
for node in all_nodes:
    probOfGettingNewReqInv[node] = 1/(G.degree(node)*total_sum_inv_deg)
#print(probOfGettingNewReqInv)


### Method 3: probabilty of recieving inversely proportional to popularity by degree
# set the probability of getting new Request based on (Max Degree-degree) of that node
probOfGettingNewReqSub = [0.5]*len(all_nodes)
mx_dg = max(all_deg)
all_sub_deg = [mx_dg-dg for dg in all_deg]
total_sum_sub_deg = sum(all_sub_deg)
for node in all_nodes:
    probOfGettingNewReqSub[node] = (mx_dg-G.degree(node))/total_sum_sub_deg
#print(probOfGettingNewReqSub)


'''
### Method 4: Based on cluster coefficients (low cluster coeff => more vulnerable)
# set the probability of getting new Request based on cluster coeff of that node
probOfGettingNewReqCluster = [0.5]*len(all_nodes)
for node in all_nodes:
    probOfGettingNewReqCluster[node] = 
#print(probOfGettingNewReqCluster)
'''






### make the graph ready for fake and real nodes

n = G.order()

fake_percentage = 10

# assign labels to all the nodes
# true_label denotes the actual nature of node
# predicted shows the result of algorithm

# we make a list of nodes with the degree and sort it
node_deg = []
for idx in range(len(all_nodes)):
    node_deg.append((G.degree(all_nodes[idx]),all_nodes[idx],idx))
node_deg.sort()

# this makes sure that very high degree nodes are not fake
nodeLabels = {}
for ndTup in node_deg:
    p = random.random()
    if(p < fake_percentage/100):
        nodeLabels[ndTup[2]] = {"true_label":'fake',"predicted":'fake'}
    else:
        nodeLabels[ndTup[2]] = {"true_label":'real',"predicted":'real'}


# we are assuming only percent_unknown nodes
# are left to classify using the algorithm
# we are aslo assuming that rest of the nodes have
# classified correctly. [This would help us to see
# how good our algorithm is]
percent_unknown = 20
nodes_copy = all_nodes.copy()
random.shuffle(nodes_copy)

unknown_nodes = nodes_copy[:int(percent_unknown*n/100)]

# mark the unknown nodes' predicted label as unknown
for node in unknown_nodes:
    nodeLabels[node]["predicted"] = 'unknown'


#nx.draw_spectral(G, with_lables=False, node_size=10, node_color='cyan')

#nx.draw_spectral(G, lables=nodeLabels, with_lables=True, node_size=10, node_color='cyan')
#nx.draw_networkx_labels(G, nx.spectral_layout(G), labels=nodeLabels, font_size=5, font_color='k')
#plt.show()







### start adding new Nodes
#newNodeAddition(G, all_nodes, nodeLabels, probOfAcceptingNewReq, probOfGettingNewReqSub)


# run sybilEdge
SybilEdge(G, all_nodes, nodeLabels, fake_percentage)






### cluster coefficients
##print("nodes = ",all_nodes)
#count = 0
#avg = 0
#cluster_coeff = []
#for person_i in all_nodes:
#    if(G.degree(person_i)>50 and G.degree(person_i)<200):
##        print(person_i)
#        val_cluster = nx.clustering(G,person_i)
#        avg += val_cluster
#        cluster_coeff.append(val_cluster)
#        count+=1


#unique_cluster_coeff = list(set(cluster_coeff))
#unique_cluster_coeff.sort()
#plot_cluster = []
#for val in unique_cluster_coeff:
#    cnt = 0
#    for cntr in unique_cluster_coeff:
#       if(abs(cntr-val)<(0.0001)):
#           cnt+=1
#    plot_cluster.append(cnt)
#plt.plot(unique_cluster_coeff,plot_cluster,color='green',marker='o',linewidth=25)
#plt.xlabel('cluster_coeff')
#plt.ylabel('number of nodes')

#avg/=count
#print("average clustering coeff = ",avg)
