import numpy as np

class KDTree(object):

    def __init__(self, pts, depth=0):

        assert(pts.size > 0)

        n, k = pts.shape
        axis = depth % k

        # Sort point list and choose median as pivot element
        (sorted_indices, pts_list) = zip(*sorted(enumerate(pts), key=lambda point: point[1][axis]))
        median = n // 2 # choose median
        sorted_indices = np.array(sorted_indices)
        pts = pts[sorted_indices, :]

        self.location = pts[median,:]
        self.median_idx = median
        self.sorted_indices = sorted_indices
        self.npts = n
        self.bottom_corner = np.min(pts, axis=0)
        self.top_corner = np.max(pts, axis=0)
        self.width = self.top_corner - self.bottom_corner

        # construct subtrees
        self.left_child = KDTree(pts=pts[:median, :], depth=depth + 1) if median > 0 else None
        self.right_child = KDTree(pts=pts[median+1:, :], depth=depth + 1) if median < n-1 else None

    def set_v(self, v):
        v = v[self.sorted_indices]
        self.vi = v[self.median_idx]
        self.unweighted_sum = np.sum(v) - self.vi

        if self.left_child:
            self.left_child.set_v(v[:self.median_idx])
        if self.right_child:
            self.right_child.set_v(v[self.median_idx+1:])

    def get_v(self):
        v = np.zeros((self.npts,))
        if self.left_child:
            v[:self.median_idx] = self.left_child.get_v()
        if self.right_child:
            v[self.median_idx+1:] = self.right_child.get_v()
        v[self.median_idx] = self.vi
        v = v[np.argsort(self.sorted_indices)]
        return v

    # compute the weighted sum
    # sum_i w(query_pt, x_i) * v_i
    # where i ranges over all points in the tree
    # and w is assumed to be isotropic and monotonically decreasing in ||query_pt-x||
    def weighted_sum(self, w, query_pt, weight_sofar=0, eps=0.001):
        self_sum = self.vi * w(query_pt, self.location)
        weighted_sum = self_sum
        weight_sofar += self_sum

        if self.npts > 1:

            # find bounds on the weight of points in this box, to
            # decide whether to cut off computation
            k = len(query_pt)

            d1 = query_pt - self.bottom_corner
            d2 = self.top_corner - query_pt
            query_in_bounds= (d1 > 0).all() and (d2 > 0 ).all()

            if not query_in_bounds:

                # the max bound comes from projecting the query onto the bounding box
                maxbound_pt = np.copy(query_pt)
                query_below_bottom = (d1 < 0)
                maxbound_pt[query_below_bottom] = self.bottom_corner[query_below_bottom]
                query_above_top = (d2 < 0)
                maxbound_pt[query_above_top] = self.top_corner[query_above_top]
                max_weight = w(query_pt, maxbound_pt)

                # the min bound comes from the corner furthest away from the query
                minbound_pt = np.zeros((k,))
                query_closer_to_bottom = d1 < self.width/2
                minbound_pt[query_closer_to_bottom] = self.top_corner[query_closer_to_bottom]
                minbound_pt[~query_closer_to_bottom] = self.bottom_corner[~query_closer_to_bottom]
                min_weight = w(query_pt, minbound_pt)

                if not (max_weight >= min_weight):
                    print "wtf? weight bounds are wrong"
                    import pdb; pdb.set_trace()

                # unlike the paper, we're allowing nodes to live in the tree (not just at leaves).
                # since we've already accounted for the point at this node, we use (npts-1).
                cutoff_threshold = 2 * eps * (weight_sofar + (self.npts-1) * min_weight)

            if not query_in_bounds and (max_weight - min_weight) < cutoff_threshold:
                # TODO: it's not clear that averaging the min and max
                # weight is reasonable to get the average weight,
                # especially if the weights fall off e.g. exponentially
                # with distance

                weighted_sum += .5 * (max_weight + min_weight) * self.unweighted_sum
                weight_sofar += min_weight * (self.npts-1)
            else:
                closest_child = self.left_child
                other_child = self.right_child
                self.closest="left"
                if self.right_child and np.linalg.norm(query_pt-self.right_child.location, 2) < np.linalg.norm(query_pt-self.left_child.location, 2):
                    closest_child = self.right_child
                    other_child = self.left_child
                    self.closest="right"
                (closest_sum, closest_bound) = closest_child.weighted_sum(w=w, query_pt=query_pt, weight_sofar=weight_sofar, eps=eps)
                self.closest_sum = closest_sum
                weighted_sum += closest_sum
                weight_sofar = closest_bound
                if other_child:
                    other_sum, other_bound = other_child.weighted_sum(w=w, query_pt=query_pt, weight_sofar=weight_sofar, eps=eps)
                    self.other_sum = other_sum
                    weighted_sum += other_sum
                    weight_sofar = other_bound

        return (weighted_sum, weight_sofar)
            

        
def main():
    X = np.random.normal(size=(1000, 2))
    tree = KDTree(pts=X)

    v = np.arange(1000)
    tree.set_v(v)
    newv = tree.get_v()
    if (v == newv).all():
        print "setting and getting successful!"

    w = lambda x1, x2 : np.exp(-.5 * np.linalg.norm(x1-x2, 2)**2 / .0001)
    query_pt = np.array((.1, -.3))
    k = [w(query_pt, x) for x in X]
    kv = np.dot(k, v)

    kv_tree, weight_sofar = tree.weighted_sum(w=w, query_pt=query_pt, eps=1e-4)
    if np.abs(kv_tree - kv) < 0.01 * kv:
        print "vector multiplication successful!"
    else:
        print "vector multiplication unsuccessful:"
    print kv
    print kv_tree
       

if __name__ == "__main__":
    main()
