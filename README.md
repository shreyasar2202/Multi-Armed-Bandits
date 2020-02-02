# Contextual-Bandits

A multi-armed bandit is a popular problem in probability theory and reinforcement learning,
in which a given slot machine (resource pool) has n arms (bandit/resource), each rigged
with their own probability of success. This is a classic sequential resource allocation problem
where the objective is to pull the arms (resources) in a sequence so as to obtain the
maximum cumulative reward in the face of uncertainty. Traditional A/B testing methods
are offline resource allocation methods that dedicate a period of time purely to exploration
where traffic is equally allocated to the two possible versions due to which a lot of time
and revenue is wasted on the losing variant.On the other hand, MABs successfully balance
exploration and exploitation as more knowledge is gained. In the context of internet advertisements,
the goal is to learn the click-through rates of several competing advertisements
in order to converge to the most appropriate set of advertisements for the user in question.

The proposed system implements the CSlogUCB-F algorithm, that we have come up with,
which makes use of side information available in each round - user context and advertisement
context - to improve the relevance of the chosen ads to the user making the search
query, while ensuring fairness to all advertisers. Initially, the data is preprocessed to form
normalised context vectors, which are fed to a module that implements the proposed algorithm
and selects the best advertisement for the current query based on the input context.
Following this, the various parameters of the algorithm including estimated rewards and
fairness debts are updated based on user feedback.

The introduction of context into CSMAB-F algorithm, resulting in the proposed algorithm
being developed, helped reduce the regret incurred through the entire run across time steps
empirically. An improved fairness notion was proposed that models the real world scenarios
more accurately, and was found to reduce the regret based on the experiments conducted.
Logistic regression was used to learn the coefficient vectors for the context, which was also
a factor in the reduced regret.
