import numpy as np
from scipy.special import erf
from tqdm import tqdm
from time import time

from .signals import split_discont_ids, duration_split


def prob_gaussian_dist(dI, sigma):
    return 1.0 - erf(dI/np.sqrt(2.0*np.square(sigma)))


def boundary_dist(y, y0, y1):
    y_min = min(y0,y1)
    y_max = max(y0,y1)
    m_sup = (y > y_max).astype(float)
    m_inf = (y < y_min).astype(float)
    return (y - y_max) * m_sup + (y_min - y) * m_inf


def near_neighbours(r, x, xlim=(-np.inf, np.inf)):
    N = r.shape[0]
    M = x.shape[0]

    m_pairs = []

    ref_start = 0
    for k in range(N):
        if k == 0:
            lims = [xlim[0], r[k+1]]
        elif k == N-1:
            lims = [r[k-1], xlim[1]]
        else:
            lims = [r[k-1], r[k+1]]

        for i in range(ref_start, M):
            if (x[i] > lims[0]) & (x[i] < lims[1]):
                m_pairs.append([k,i])
            elif (x[i] <= lims[0]):
                ref_start += 1
            elif (x[i] >= lims[1]):
                break

    return np.array(m_pairs)


def links_array_to_dict(links_array):
    links_dict = {}
    for link in links_array:
        if link[0] in links_dict:
            links_dict[link[0]] += [link[1]]
        else:
            links_dict[link[0]] = [link[1]]

    return links_dict


def links_dict_to_array(links_dict):
    links_l = []
    for k in links_dict:
        for i in links_dict[k]:
            links_l.append([k,i])

    return np.array(links_l)


def revise_links(ids_links, prob_links):
    # convert links array to dictionnary for easier access
    links_dict = links_array_to_dict(ids_links)

    # keep track of which links are unbreakable
    ilinks_ubrk = []
    i_ubrk = []

    # from strongest to weakest link
    for k in np.argsort(prob_links)[::-1]:
        # get strongest link
        ilink_ubrk = ids_links[k]
        i, i_ref = ilink_ubrk

        # add it to the list of unbreakable links
        if i not in i_ubrk:
            i_ubrk.append(i)
            ilinks_ubrk.append([i, i_ref])

        if i in links_dict:
            # break links from previous point in conflict with current unbreakable link
            if i-1 in links_dict:
                ilinks_prev = links_dict[i-1]
                ilinks_keep = [k for k in ilinks_prev if k <= i_ref or [i-1, k] in ilinks_ubrk]
                if len(ilinks_keep) > 0:
                    links_dict[i-1] = ilinks_keep
                else:
                    links_dict.pop(i-1)

            # break links from next point in conflict with current unbreakable link
            if i+1 in links_dict:
                ilinks_next = links_dict[i+1]
                ilinks_keep = [k for k in ilinks_next if k >= i_ref or [i+1, k] in ilinks_ubrk]
                if len(ilinks_keep) > 0:
                    links_dict[i+1] = ilinks_keep
                else:
                    links_dict.pop(i+1)

    return links_dict_to_array(links_dict)


def links_max_prob(ids_links, prob_links, N):
    ids_best_links = []
    prob_best_links = []
    for i in range(N):
        ids_links_i = np.where(ids_links[:,0] == i)[0]

        if len(ids_links_i) > 0:
            sub_id_best_link = np.argmax(prob_links[ids_links_i])

            ids_best_links.append(ids_links[ids_links_i[sub_id_best_link]])
            prob_best_links.append(prob_links[ids_links_i[sub_id_best_link]])

        else:
            ids_best_links.append([-1,-1])
            prob_best_links.append(0.0)

    return np.array(ids_best_links), np.array(prob_best_links)


#def segment_within_steps(t, t_steps):
#    ids_segs = []
#    for t0, t1 in zip(t_steps[:-1], t_steps[1:]):
#        ids_segs.append(np.where((t0 <= t) & (t <= t1))[0])
#
#    return ids_segs


def segment_within_steps(t, t_steps):
    N = len(t)
    M = len(t_steps)

    ids_step = []
    ids_in = []
    i = 0
    for j in range(N):
        while t[j] > t_steps[i+1]:
            i += 1
            if i+1 == M:
                return np.array(ids_step), ids_in

        if t_steps[i] < t[j] < t_steps[i+1]:
            if i in ids_step:
                ids_in[-1].append(j)
            else:
                ids_step.append(i)
                ids_in.append([j])

    return np.array(ids_step), ids_in


def extract_best_sub_patterns(p, x, x_nodes, L, M=1, merge=True):
    # empty lists for possible sub-patterns
    sub_p_l, sub_x_l, sub_x_nodes_l = [], [], []

    # count number of elements
    n = len(p)

    # chack that length asked is not too big for input
    if n > L:
        # extract probabilities for all sub-patterns of size L
        p_sub_segs = np.stack([p[i:n-L+i+1] for i in range(L)])

        # get the indices for top M best sub-patterns
        ids_bss = np.argsort(np.prod(p_sub_segs,0))[::-1]
        ids_sub_segs = np.stack([np.arange(len(p))[i:n-L+i+1] for i in range(L)])

        # if merging segments
        if merge:
            ids_ts = split_discont_ids(np.unique(np.sort(ids_sub_segs.T[ids_bss[:M]].ravel())))

        # else
        else:
            ids_ts = ids_sub_segs.T[ids_bss[:M]]

        # for each top M best sub-patterns
        for ids in ids_ts:
            # extract corresponding sub-pattern nodes and time ranges
            sub_x_nodes = x_nodes[ids[0]:ids[-1]+2,:]

            t0 = sub_x_nodes[0,0]
            t1 = sub_x_nodes[-1,0]

            # extract corresponding data in the range of the selected sub-pattern
            ids_step, ids_in = segment_within_steps(x[:,0], [t0,t1])
            sub_x = x[ids_in[0]]

            # store results
            sub_p_l.append(p[ids[0]:ids[-1]+1])
            sub_x_l.append(sub_x)
            sub_x_nodes_l.append(sub_x_nodes)

    return sub_p_l, sub_x_l, sub_x_nodes_l


class Pattern:
    def __init__(self, i, x, x_nodes, sigma):
        # initailize pattern data
        self.i = i
        self.x = x
        self.x_nodes = x_nodes
        self.sigma = sigma
        self.ids_links = []
        self.ids_best_links = []
        self.prob_links = []
        self.prob_best_links = []
        self.prob_infills = []

        # compute probability of jump to be outside the noise
        self.prob_elements = 1.0 - prob_gaussian_dist(np.abs(np.diff(self.x_nodes[:,1])), self.sigma)

        # profiler
        self.profiler = {'calls':0, 'dt0':0.0, 'dt1':0.0, 'dt2':0.0, 'dt3':0.0, 'dt4':0.0, 'dt5':0.0}

    def link(self, pattern):
        # profiler update
        self.profiler['calls'] += 1

        # find pattern links
        t0 = time()
        lim = (np.min(self.x_nodes[:,0]), np.max(self.x_nodes[:,0]))
        ids_links = near_neighbours(self.x_nodes[:,0], pattern.x_nodes[:,0], xlim=lim)
        self.profiler['dt0'] += (time()-t0)

        # check that links exist
        if len(ids_links) > 0:
            # compute match probability
            t0 = time()
            dx_nodes = np.abs(self.x_nodes[ids_links[:,0],1] - pattern.x_nodes[ids_links[:,1],1])
            prob_links = prob_gaussian_dist(dx_nodes, self.sigma)
            self.profiler['dt1'] += (time()-t0)

            # break unprobable links
            t0 = time()
            ids_links = revise_links(ids_links, prob_links)
            self.profiler['dt2'] += (time()-t0)

            # recompute match probability
            # TODO: include in revise links
            t0 = time()
            dx_nodes = np.abs(self.x_nodes[ids_links[:,0],1] - pattern.x_nodes[ids_links[:,1],1])
            prob_links = prob_gaussian_dist(dx_nodes, self.sigma)
            self.profiler['dt3'] += (time()-t0)

            # extract best links
            t0 = time()
            ids_best_links, prob_best_links = links_max_prob(ids_links, prob_links, self.x_nodes.shape[0])
            self.profiler['dt4'] += (time()-t0)

            # compute infill probability
            t0 = time()
            ids_step, ids_in = segment_within_steps(pattern.x_nodes[:,0], self.x_nodes[:,0])
            prob_infills = np.ones(len(self.prob_elements), dtype=float)
            for i_step, ids_in in zip(ids_step, ids_in):
                infill_dist = boundary_dist(pattern.x_nodes[ids_in,1], self.x_nodes[i_step,1], self.x_nodes[i_step+1,1])
                probs_seg = prob_gaussian_dist(infill_dist, self.sigma)
                prob_infills[i_step] = np.power(np.prod(probs_seg), 1.0 / float(len(ids_in)))

            self.profiler['dt5'] += (time()-t0)

            # store results
            self.ids_links.append(ids_links)
            self.ids_best_links.append(ids_best_links)
            self.prob_links.append(prob_links)
            self.prob_best_links.append(prob_best_links)
            self.prob_infills.append(prob_infills)
        else:
            self.prob_best_links.append(np.zeros(self.prob_elements.shape[0]+1))
            self.prob_infills.append(np.zeros(self.prob_elements.shape[0]))

    def get_prob(self):
        prob_pattern = []
        for p_best_links, p_infill in zip(self.prob_best_links, self.prob_infills):
            # compute probability match extremities
            p_elems_ext = np.sqrt(p_best_links[:-1] * p_best_links[1:])

            # apply geometric mean of the total probabilities
            prob_pattern.append(np.power(p_elems_ext * self.prob_elements * p_infill, 1.0/3.0))

        return np.array(prob_pattern)

    def clear_links(self):
        self.ids_links = []
        self.ids_best_links = []
        self.prob_links = []
        self.prob_best_links = []
        self.prob_infills = []


class PatternsAnalysisTool:
    def __init__(self, patterns_x, patterns_x_nodes, sigma, tol):
        # store parameters
        self.sigma = sigma
        self.tol = tol

        # store patterns
        self.patterns = []
        for i in range(len(patterns_x)):
            self.patterns.append(Pattern(i, patterns_x[i], patterns_x_nodes[i], sigma))

        # time clustering
        clust_labels = duration_split(np.array([v[-1,0] for v in patterns_x]), tol=tol)
        self.ids_clusters = [np.where(clust_labels == k)[0] for k in range(np.max(clust_labels))]

        # debug print
        A_covered = float(np.sum([len(ids_clust)*len(ids_clust) for ids_clust in self.ids_clusters]))
        A_tot = float(np.square(len(self.patterns)))
        print("Clustering coverage: {:.2f}%".format(100.0*A_covered/A_tot))
        print("Sigma: {:.2f}".format(self.sigma))

    def link_self(self):
        N = len(self.patterns)
        for k in tqdm(range(N)):
            self.patterns[k].link(self.patterns[k])

    def link_all_in_clusters(self):
        M = len(self.ids_clusters)
        for k in tqdm(range(M)):
            for i in self.ids_clusters[-k-1]:
                for j in self.ids_clusters[-k-1]:
                    self.patterns[i].link(self.patterns[j])

    def link_all_in_sub(self):
        N = len(self.sub_patterns)
        for i in tqdm(range(N)):
            for j in range(N):
                self.sub_patterns[i].link(self.sub_patterns[j])

    def scan_for_best_patterns(self, L, M=1):
        N = len(self.patterns)

        self.sub_patterns = []
        for i in range(N):
            # get i-th pattern
            pat = self.patterns[i]

            # get probability and check if not empty
            p = pat.get_prob()

            if len(p)>0:
                # find best patterns (based on mean)
                sub_p_l, sub_x_l, sub_x_nodes_l = extract_best_sub_patterns(np.mean(p,0), pat.x, pat.x_nodes, L, M)

                for sub_x, sub_x_nodes in zip(sub_x_l, sub_x_nodes_l):
                    self.sub_patterns.append(Pattern(pat.i, sub_x, sub_x_nodes, self.sigma))

    def get_reduced_prob(self, fct=np.mean):
        N = len(self.patterns)
        M = len(self.ids_clusters)
        P = np.zeros((N,N))

        for k in range(M):
            for i in self.ids_clusters[k]:
                P[i, self.ids_clusters[k]] = fct(self.patterns[i].get_prob(), 1)

        return P

    def get_sub_reduced_prob(self, fct=np.mean):
        N = len(self.sub_patterns)
        P = np.zeros((N,N))

        for i in range(N):
            P[i, :] = fct(self.sub_patterns[i].get_prob(), 1)

        return P
