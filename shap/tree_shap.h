/**
 * Fast recursive computation of SHAP values in trees.
 * See https://arxiv.org/abs/1802.03888 for details.
 *
 * Scott Lundberg, 2018 (independent algorithm courtesy of Hugh Chen 2018)
 */

#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdio.h> 
#include <cmath>
using namespace std;

typedef double tfloat;

namespace FEATURE_DEPENDENCE {
    const unsigned independent = 0;
    const unsigned tree_path_dependent = 1;
    const unsigned global_path_dependent = 2;
}

namespace MODEL_OUTPUT {
    const unsigned margin = 0;
    const unsigned probability = 1;
    const unsigned log_loss = 2;
}

namespace OBJECTIVE { 
    const unsigned squared_error = 0;
    const unsigned logistic = 1;
}

namespace TRANSFORM { 
    const unsigned identity = 0;
    const unsigned logistic = 1;
}

struct TreeEnsemble {
    int *children_left;
    int *children_right;
    int *children_default;
    int *features;
    tfloat *thresholds;
    tfloat *values;
    tfloat *node_sample_weights;
    bool less_than_or_equal;
    unsigned max_depth;
    unsigned tree_limit;
    tfloat base_offset;
    unsigned max_nodes;
    unsigned num_outputs;

    TreeEnsemble() {}
    TreeEnsemble(int *children_left, int *children_right, int *children_default, int *features,
                 tfloat *thresholds, tfloat *values, tfloat *node_sample_weights,
                 bool less_than_or_equal, unsigned max_depth, unsigned tree_limit, tfloat base_offset,
                 unsigned max_nodes, unsigned num_outputs) :
        children_left(children_left), children_right(children_right),
        children_default(children_default), features(features), thresholds(thresholds),
        values(values), node_sample_weights(node_sample_weights),
        less_than_or_equal(less_than_or_equal), max_depth(max_depth), tree_limit(tree_limit),
        base_offset(base_offset), max_nodes(max_nodes), num_outputs(num_outputs) {}

    void get_tree(TreeEnsemble &tree, const unsigned i) const {
        const unsigned d = i * max_nodes;

        tree.children_left = children_left + d;
        tree.children_right = children_right + d;
        tree.children_default = children_default + d;
        tree.features = features + d;
        tree.thresholds = thresholds + d;
        tree.values = values + d * num_outputs;
        tree.node_sample_weights = node_sample_weights + d;
        tree.less_than_or_equal = less_than_or_equal;
        tree.max_depth = max_depth;
        tree.tree_limit = 1;
        tree.base_offset = base_offset;
        tree.max_nodes = max_nodes;
        tree.num_outputs = num_outputs;
    }

    void allocate(unsigned tree_limit_in, unsigned max_nodes_in, unsigned num_outputs_in) {
        tree_limit = tree_limit_in;
        max_nodes = max_nodes_in;
        num_outputs = num_outputs_in;
        children_left = new int[tree_limit * max_nodes];
        children_right = new int[tree_limit * max_nodes];
        children_default = new int[tree_limit * max_nodes];
        features = new int[tree_limit * max_nodes];
        thresholds = new tfloat[tree_limit * max_nodes];
        values = new tfloat[tree_limit * max_nodes * num_outputs];
        node_sample_weights = new tfloat[tree_limit * max_nodes];
    }

    void free() {
        delete[] children_left;
        delete[] children_right;
        delete[] children_default;
        delete[] features;
        delete[] thresholds;
        delete[] values;
        delete[] node_sample_weights;
    }
};

struct ExplanationDataset {
    tfloat *X;
    bool *X_missing;
    tfloat *y;
    tfloat *R;
    bool *R_missing;
    unsigned num_X;
    unsigned M;
    unsigned num_R;

    ExplanationDataset() {}
    ExplanationDataset(tfloat *X, bool *X_missing, tfloat *y, tfloat *R, bool *R_missing, unsigned num_X,
                       unsigned M, unsigned num_R) : 
        X(X), X_missing(X_missing), y(y), R(R), R_missing(R_missing), num_X(num_X), M(M), num_R(num_R) {}

    void get_x_instance(ExplanationDataset &instance, const unsigned i) const {
        instance.M = M;
        instance.X = X + i * M;
        instance.X_missing = X_missing + i * M;
        instance.num_X = 1;
    }
};

// struct TreeNode<P> {
//   int left_child;
//   int right_child;
//   int default_child;
//   int feature;
//   tfloat threshold;
//   tfloat values[P];
//   tfloat sample_weight;
//   TreeNode() {}
//   TreeNode(int lc, int rc, int dc, int f, tfloat t, tfloat v, tfloat w) :
//       left_child(lc), right_child(rc), default_child(dc), feature(f),
//       threshold(t), sample_weight(w) {
//     for (unsigned i = 0; i < P; ++i) {
//       values[i] = v[i];
//     }
//   }
// };

// void convert_from_parallel_arrays<P>(const unsigned num_outputs, const int *children_left,
//                                   const int *children_right,
//                                   const int *children_default, const int *features,
//                                   const tfloat *thresholds, const tfloat *values,
//                                   const tfloat *node_sample_weights, TreeNode<P> *new_tree,
//                                   unsigned i = 0, unsigned pos = 0) {
//   new_tree[pos] = TreeNode<P>(
//     children_left[i], children_right[i], children_default[i], features[i],
//     thresholds[i], values[i * P], node_sample_weights[i]
//   )
//   ++pos;

//   if (children_left[i] > )
// }

// data we keep about our decision path
// note that pweight is included for convenience and is not tied with the other attributes
// the pweight of the i'th path element is the permuation weight of paths with i-1 ones in them
struct PathElement {
  int feature_index;
  tfloat zero_fraction;
  tfloat one_fraction;
  tfloat pweight;
  PathElement() {}
  PathElement(int i, tfloat z, tfloat o, tfloat w) :
    feature_index(i), zero_fraction(z), one_fraction(o), pweight(w) {}
};


// extend our decision path with a fraction of one and zero extensions
inline void extend_path(PathElement *unique_path, unsigned unique_depth,
                        tfloat zero_fraction, tfloat one_fraction, int feature_index) {
  unique_path[unique_depth].feature_index = feature_index;
  unique_path[unique_depth].zero_fraction = zero_fraction;
  unique_path[unique_depth].one_fraction = one_fraction;
  unique_path[unique_depth].pweight = (unique_depth == 0 ? 1.0f : 0.0f);
  for (int i = unique_depth - 1; i >= 0; i--) {
    unique_path[i + 1].pweight += one_fraction * unique_path[i].pweight * (i + 1)
                                  / static_cast<tfloat>(unique_depth + 1);
    unique_path[i].pweight = zero_fraction * unique_path[i].pweight * (unique_depth - i)
                             / static_cast<tfloat>(unique_depth + 1);
  }
}

// undo a previous extension of the decision path
inline void unwind_path(PathElement *unique_path, unsigned unique_depth, unsigned path_index) {
  const tfloat one_fraction = unique_path[path_index].one_fraction;
  const tfloat zero_fraction = unique_path[path_index].zero_fraction;
  tfloat next_one_portion = unique_path[unique_depth].pweight;

  for (int i = unique_depth - 1; i >= 0; --i) {
    if (one_fraction != 0) {
      const tfloat tmp = unique_path[i].pweight;
      unique_path[i].pweight = next_one_portion * (unique_depth + 1)
                               / static_cast<tfloat>((i + 1) * one_fraction);
      next_one_portion = tmp - unique_path[i].pweight * zero_fraction * (unique_depth - i)
                               / static_cast<tfloat>(unique_depth + 1);
    } else {
      unique_path[i].pweight = (unique_path[i].pweight * (unique_depth + 1))
                               / static_cast<tfloat>(zero_fraction * (unique_depth - i));
    }
  }

  for (unsigned i = path_index; i < unique_depth; ++i) {
    unique_path[i].feature_index = unique_path[i+1].feature_index;
    unique_path[i].zero_fraction = unique_path[i+1].zero_fraction;
    unique_path[i].one_fraction = unique_path[i+1].one_fraction;
  }
}

// determine what the total permuation weight would be if
// we unwound a previous extension in the decision path
inline tfloat unwound_path_sum(const PathElement *unique_path, unsigned unique_depth,
                                  unsigned path_index) {
  const tfloat one_fraction = unique_path[path_index].one_fraction;
  const tfloat zero_fraction = unique_path[path_index].zero_fraction;
  tfloat next_one_portion = unique_path[unique_depth].pweight;
  tfloat total = 0;
  
  if (one_fraction != 0) {
    for (int i = unique_depth - 1; i >= 0; --i) {
      const tfloat tmp = next_one_portion / static_cast<tfloat>((i + 1) * one_fraction);
      total += tmp;
      next_one_portion = unique_path[i].pweight - tmp * zero_fraction * (unique_depth - i);
    }
  } else {
    for (int i = unique_depth - 1; i >= 0; --i) {
      total += unique_path[i].pweight / (zero_fraction * (unique_depth - i));
    }
  }
  return total * (unique_depth + 1);
}

// recursive computation of SHAP values for a decision tree
inline void tree_shap_recursive(const unsigned num_outputs, const int *children_left,
                                const int *children_right,
                                const int *children_default, const int *features,
                                const tfloat *thresholds, const tfloat *values,
                                const tfloat *node_sample_weight,
                                const tfloat *x, const bool *x_missing, tfloat *phi,
                                unsigned node_index, unsigned unique_depth,
                                PathElement *parent_unique_path, tfloat parent_zero_fraction,
                                tfloat parent_one_fraction, int parent_feature_index,
                                int condition, unsigned condition_feature,
                                tfloat condition_fraction, bool less_than_or_equal) {

  // stop if we have no weight coming down to us
  if (condition_fraction == 0) return;

  // extend the unique path
  PathElement *unique_path = parent_unique_path + unique_depth + 1;
  std::copy(parent_unique_path, parent_unique_path + unique_depth + 1, unique_path);

  if (condition == 0 || condition_feature != static_cast<unsigned>(parent_feature_index)) {
    extend_path(unique_path, unique_depth, parent_zero_fraction,
                parent_one_fraction, parent_feature_index);
  }
  const unsigned split_index = features[node_index];

//   std::cout << "node_index " << node_index << " " << split_index;
//   std::cout << " " << x;
//   std::cout << " " << x[split_index];
  
//   if (thresholds[node_index] > 0) {
//       std::cout << " ";
//   }
//   if (x[node_index] > 0) {
//       std::cout << " ";
//   }
//   if (children_right[node_index] < 0) {
//       std::cout << " ";
//   }
//   if (x_missing[split_index] < 0) {
//       std::cout << " ";
//   }
//   std::cout << " " << x[split_index] << "\n";

  // leaf node
  if (children_right[node_index] < 0) {
    for (unsigned i = 1; i <= unique_depth; ++i) {
      const tfloat w = unwound_path_sum(unique_path, unique_depth, i);
      const PathElement &el = unique_path[i];
      const unsigned phi_offset = el.feature_index * num_outputs;
      const unsigned values_offset = node_index * num_outputs;
      const tfloat scale = w * (el.one_fraction - el.zero_fraction) * condition_fraction;
      for (unsigned j = 0; j < num_outputs; ++j) {
        phi[phi_offset + j] += scale * values[values_offset + j];
      }
    }

  // internal node
  } else {
    // find which branch is "hot" (meaning x would follow it)
    unsigned hot_index = 0;
    if (x_missing[split_index]) {
      hot_index = children_default[node_index];
    } else if ((less_than_or_equal && x[split_index] <= thresholds[node_index]) ||
               (!less_than_or_equal && x[split_index] < thresholds[node_index])) {
      hot_index = children_left[node_index];
    } else {
      hot_index = children_right[node_index];
    }
    const unsigned cold_index = (static_cast<int>(hot_index) == children_left[node_index] ?
                                 children_right[node_index] : children_left[node_index]);
    const tfloat w = node_sample_weight[node_index];
    const tfloat hot_zero_fraction = node_sample_weight[hot_index] / w;
    const tfloat cold_zero_fraction = node_sample_weight[cold_index] / w;
    tfloat incoming_zero_fraction = 1;
    tfloat incoming_one_fraction = 1;

    // see if we have already split on this feature,
    // if so we undo that split so we can redo it for this node
    unsigned path_index = 0;
    for (; path_index <= unique_depth; ++path_index) {
      if (static_cast<unsigned>(unique_path[path_index].feature_index) == split_index) break;
    }
    if (path_index != unique_depth + 1) {
      incoming_zero_fraction = unique_path[path_index].zero_fraction;
      incoming_one_fraction = unique_path[path_index].one_fraction;
      unwind_path(unique_path, unique_depth, path_index);
      unique_depth -= 1;
    }

    // divide up the condition_fraction among the recursive calls
    tfloat hot_condition_fraction = condition_fraction;
    tfloat cold_condition_fraction = condition_fraction;
    if (condition > 0 && split_index == condition_feature) {
      cold_condition_fraction = 0;
      unique_depth -= 1;
    } else if (condition < 0 && split_index == condition_feature) {
      hot_condition_fraction *= hot_zero_fraction;
      cold_condition_fraction *= cold_zero_fraction;
      unique_depth -= 1;
    }

    tree_shap_recursive(
      num_outputs, children_left, children_right, children_default, features, thresholds, values,
      node_sample_weight, x, x_missing, phi, hot_index, unique_depth + 1, unique_path,
      hot_zero_fraction * incoming_zero_fraction, incoming_one_fraction,
      split_index, condition, condition_feature, hot_condition_fraction, less_than_or_equal
    );

    tree_shap_recursive(
      num_outputs, children_left, children_right, children_default, features, thresholds, values,
      node_sample_weight, x, x_missing, phi, cold_index, unique_depth + 1, unique_path,
      cold_zero_fraction * incoming_zero_fraction, 0,
      split_index, condition, condition_feature, cold_condition_fraction, less_than_or_equal
    );
  }
}

inline int compute_expectations(TreeEnsemble &tree, int i = 0, int depth = 0) {
    unsigned max_depth = 0;
    
    if (tree.children_right[i] != -1) {
        const unsigned li = tree.children_left[i];
        const unsigned ri = tree.children_right[i];
        const unsigned depth_left = compute_expectations(tree, li, depth + 1);
        const unsigned depth_right = compute_expectations(tree, ri, depth + 1);
        const tfloat left_weight = tree.node_sample_weights[li];
        const tfloat right_weight = tree.node_sample_weights[ri];
        const unsigned li_offset = li * tree.num_outputs;
        const unsigned ri_offset = ri * tree.num_outputs;
        const unsigned i_offset = i * tree.num_outputs;
        for (unsigned j = 0; j < tree.num_outputs; ++j) {
            const tfloat v = (left_weight * tree.values[li_offset + j] + right_weight * tree.values[ri_offset + j]) / (left_weight + right_weight);
            tree.values[i_offset + j] = v;
        }
        max_depth = std::max(depth_left, depth_right) + 1;
    }
    
    if (depth == 0) tree.max_depth = max_depth;
    
    return max_depth;
}

inline void tree_shap(const TreeEnsemble& tree, const ExplanationDataset &data,
                      tfloat *out_contribs, int condition, unsigned condition_feature) {

    // update the reference value with the expected value of the tree's predictions
    if (condition == 0) {
        for (unsigned j = 0; j < tree.num_outputs; ++j) {
            out_contribs[data.M * tree.num_outputs + j] += tree.values[j];
        }
    }

    // Pre-allocate space for the unique path data
    const unsigned maxd = tree.max_depth + 2; // need a bit more space than the max depth
    PathElement *unique_path_data = new PathElement[(maxd * (maxd + 1)) / 2];

    tree_shap_recursive(
        tree.num_outputs, tree.children_left, tree.children_right, tree.children_default,
        tree.features, tree.thresholds, tree.values, tree.node_sample_weights, data.X,
        data.X_missing, out_contribs, 0, 0, unique_path_data, 1, 1, -1, condition,
        condition_feature, 1, tree.less_than_or_equal
    );

    delete[] unique_path_data;
}


unsigned build_merged_tree_recursive(TreeEnsemble &out_tree, const TreeEnsemble &trees,
                                     const tfloat *data, const bool *data_missing, int *data_inds,
                                     const unsigned num_background_data_inds, unsigned num_data_inds,
                                     unsigned M, unsigned row = 0, unsigned i = 0, unsigned pos = 0,
                                     tfloat *leaf_value = NULL) {
    tfloat new_leaf_value[trees.num_outputs];
    unsigned row_offset = row * trees.max_nodes;
  
    // we have hit a terminal leaf!!!
    if (trees.children_left[row_offset + i] == -1 && row + 1 == trees.tree_limit) {
        //std::cout << "posa " << pos << "\n";

        // create the leaf node
        const tfloat *vals = trees.values + (row * trees.max_nodes + i) * trees.num_outputs;
        if (leaf_value == NULL) {
            for (unsigned j = 0; j < trees.num_outputs; ++j) {
                out_tree.values[pos * trees.num_outputs + j] = vals[j];
            }
        } else {
            for (unsigned j = 0; j < trees.num_outputs; ++j) {
                out_tree.values[pos * trees.num_outputs + j] = leaf_value[j] + vals[j];
            }
        }
        out_tree.children_left[pos] = -1;
        out_tree.children_right[pos] = -1;
        out_tree.children_default[pos] = -1;
        out_tree.features[pos] = -1;
        out_tree.thresholds[pos] = 0;
        out_tree.node_sample_weights[pos] = num_background_data_inds;

        return pos;
    }
  
    // we hit an intermediate leaf (so just add the value to our accumulator and move to the next tree)
    if (trees.children_left[row_offset + i] == -1) {
        
        // accumulate the value of this original leaf so it will land on all eventual terminal leaves
        const tfloat *vals = trees.values + (row * trees.max_nodes + i) * trees.num_outputs;
        if (leaf_value == NULL) {
            for (unsigned j = 0; j < trees.num_outputs; ++j) {
                new_leaf_value[j] = vals[j];
            }
        } else {
            for (unsigned j = 0; j < trees.num_outputs; ++j) {
                new_leaf_value[j] = leaf_value[j] + vals[j];
            }
        }
        leaf_value = new_leaf_value;

        // move forward to the next tree
        row += 1;
        row_offset += trees.max_nodes;
        i = 0;
    }
    
    // split the data inds by this node's threshold
    const tfloat t = trees.thresholds[row_offset + i];
    const int f = trees.features[row_offset + i];
    const bool right_default = trees.children_default[row_offset + i] == trees.children_right[row_offset + i];
    int low_ptr = 0;
    int high_ptr = num_data_inds - 1;
    unsigned num_left_background_data_inds = 0;
    int low_data_ind;
    while (low_ptr <= high_ptr) {
        low_data_ind = data_inds[low_ptr];
        const int data_ind = std::abs(low_data_ind) * M + f;
        const bool is_missing = data_missing[data_ind];
        if ((!is_missing && data[data_ind] > t) || (right_default && is_missing)) {
            data_inds[low_ptr] = data_inds[high_ptr];
            data_inds[high_ptr] = low_data_ind;
            high_ptr -= 1;
        } else {
            if (low_data_ind >= 0) ++num_left_background_data_inds; // negative data_inds are not background samples
            low_ptr += 1;
        }
    }
    int *left_data_inds = data_inds;
    const unsigned num_left_data_inds = low_ptr;
    int *right_data_inds = data_inds + low_ptr;
    const unsigned num_right_data_inds = num_data_inds - num_left_data_inds;
    const unsigned num_right_background_data_inds = num_background_data_inds - num_left_background_data_inds;
  
    // all the data went right, so we skip creating this node and just recurse right
    if (num_left_data_inds == 0) {
        return build_merged_tree_recursive(
            out_tree, trees, data, data_missing, data_inds,
            num_background_data_inds, num_data_inds, M, row,
            trees.children_right[row_offset + i], pos, leaf_value
        );

    // all the data went left, so we skip creating this node and just recurse left
    } else if (num_right_data_inds == 0) {
        return build_merged_tree_recursive(
            out_tree, trees, data, data_missing, data_inds,
            num_background_data_inds, num_data_inds, M, row,
            trees.children_left[row_offset + i], pos, leaf_value
        );

    // data went both ways so we create this node and recurse down both paths
    } else {
        
        //std::cout << "pos " << pos << "\n";

        // build the left subtree
        const unsigned new_pos = build_merged_tree_recursive(
            out_tree, trees, data, data_missing, left_data_inds,
            num_left_background_data_inds, num_left_data_inds, M, row,
            trees.children_left[row_offset + i], pos + 1, leaf_value
        );

        // fill in the data for this node
        out_tree.children_left[pos] = pos + 1;
        out_tree.children_right[pos] = new_pos + 1;
        if (trees.children_left[row_offset + i] == trees.children_default[row_offset + i]) {
            out_tree.children_default[pos] = pos + 1;
        } else {
            out_tree.children_default[pos] = new_pos + 1;
        }
        
        // if (pos == 151) {
        //     int val = trees.features[row_offset + i];
        // }
        out_tree.features[pos] = trees.features[row_offset + i];
        out_tree.thresholds[pos] = trees.thresholds[row_offset + i];
        out_tree.node_sample_weights[pos] = num_background_data_inds;

        // build the right subtree
        return build_merged_tree_recursive(
            out_tree, trees, data, data_missing, right_data_inds,
            num_right_background_data_inds, num_right_data_inds, M, row,
            trees.children_right[row_offset + i], new_pos + 1, leaf_value
        );
    }
}


void build_merged_tree(TreeEnsemble &out_tree, const ExplanationDataset &data, const TreeEnsemble &trees) {
    
    // create a joint data matrix from both X and R matrices
    tfloat *joined_data = new tfloat[(data.num_X + data.num_R) * data.M];
    std::copy(data.X, data.X + data.num_X * data.M, joined_data);
    std::copy(data.R, data.R + data.num_R * data.M, joined_data + data.num_X * data.M);
    bool *joined_data_missing = new bool[(data.num_X + data.num_R) * data.M];
    std::copy(data.X_missing, data.X_missing + data.num_X * data.M, joined_data_missing);
    std::copy(data.R_missing, data.R_missing + data.num_R * data.M, joined_data_missing + data.num_X * data.M);

    // create an starting array of data indexes we will recursively sort
    int *data_inds = new int[data.num_X + data.num_R];
    for (unsigned i = 0; i < data.num_X; ++i) data_inds[i] = i;
    for (unsigned i = data.num_X; i < data.num_X + data.num_R; ++i) {
        data_inds[i] = -i; // a negative index means it won't be recorded as a background sample
    }

    build_merged_tree_recursive(
        out_tree, trees, joined_data, joined_data_missing, data_inds, data.num_R,
        data.num_X + data.num_R, data.M
    );

    out_tree.less_than_or_equal = trees.less_than_or_equal;

    delete[] joined_data;
    delete[] joined_data_missing;
    delete[] data_inds;
}



// inline void tree_shap(const unsigned M, const unsigned num_outputs, const unsigned max_depth,
//                       const int *children_left, const int *children_right,
//                       const int *children_default, const int *features,
//                       const tfloat *thresholds, const tfloat *values,
//                       const tfloat *node_sample_weight,
//                       const tfloat *x, const bool *x_missing,
//                       tfloat *out_contribs, int condition,
//                       unsigned condition_feature) {
//
//   // update the reference value with the expected value of the tree's predictions
//   if (condition == 0) {
//     for (unsigned j = 0; j < num_outputs; ++j) {
//       out_contribs[M * num_outputs + j] += values[j];
//     }
//   }
//
//   // Preallocate space for the unique path data
//   const unsigned maxd = max_depth + 2; // need a bit more space than the max depth
//   PathElement *unique_path_data = new PathElement[(maxd * (maxd + 1)) / 2];
//
//   tree_shap_recursive(
//     num_outputs, children_left, children_right, children_default, features, thresholds, values,
//     node_sample_weight, x, x_missing, out_contribs, 0, 0, unique_path_data,
//     1, 1, -1, condition, condition_feature, 1
//   );
//   delete[] unique_path_data;
// }

// Independent Tree SHAP functions below here
// ------------------------------------------
struct Node {
    short cl, cr, cd, pnode, feat, pfeat; // uint_16
    float thres, value;
    char from_flag;
};

#define FROM_NEITHER 0
#define FROM_X_NOT_R 1
#define FROM_R_NOT_X 2

// https://www.geeksforgeeks.org/space-and-time-efficient-binomial-coefficient/
inline int bin_coeff(int n, int k) { 
    int res = 1; 
    if (k > n - k)
        k = n - k; 
    for (int i = 0; i < k; ++i) { 
        res *= (n - i); 
        res /= (i + 1); 
    } 
    return res; 
} 

// inline float calc_weight(const int N, const int M) {
//   return(1.0/(N*bin_coeff(N-1,M)));
// }

// inline float calc_weight2(const int N, const int M, float *memoized_weights) {
//     if ((N < 30) && (M < 30)) {
//         return(memoized_weights[N+30*M]);
//     } else {
//         return(1.0/(N*bin_coeff(N-1,M)));
//     }
// }

inline void tree_shap_indep(const unsigned max_depth, const unsigned num_feats,
                            const unsigned num_nodes, const tfloat *x,
                            const bool *x_missing, const tfloat *r,
                            const bool *r_missing, tfloat *out_contribs,
                            float *pos_lst, float *neg_lst, signed short *feat_hist,
                            float *memoized_weights, int *node_stack, Node *mytree) {

//     const bool DEBUG = true;
//     ofstream myfile;
//     if (DEBUG) {
//       myfile.open ("/homes/gws/hughchen/shap/out.txt",fstream::app);
//       myfile << "Entering tree_shap_indep\n";
//     }
    int ns_ctr = 0;
    std::fill_n(feat_hist, num_feats, 0);
    short node = 0, feat, cl, cr, cd, pnode, pfeat = -1;
    short next_xnode = -1, next_rnode = -1;
    short next_node = -1, from_child = -1;
    float thres, pos_x = 0, neg_x = 0, pos_r = 0, neg_r = 0;
    char from_flag;
    unsigned M = 0, N = 0;
    
    Node curr_node = mytree[node];
    feat = curr_node.feat;
    thres = curr_node.thres;
    cl = curr_node.cl;
    cr = curr_node.cr;
    cd = curr_node.cd;
    
//     if (DEBUG) {
//       myfile << "\nNode: " << node << "\n";
//       myfile << "x[feat]: " << x[feat] << ", r[feat]: " << r[feat] << "\n";
//       myfile << "thres: " << thres << "\n";
//     }
    
    if (x_missing[feat]) {
        next_xnode = cd;
    } else if (x[feat] > thres) {
        next_xnode = cr;
    } else if (x[feat] <= thres) {
        next_xnode = cl;
    }
    
    if (r_missing[feat]) {
        next_rnode = cd;
    } else if (r[feat] > thres) {
        next_rnode = cr;
    } else if (r[feat] <= thres) {
        next_rnode = cl;
    }
    
    if (next_xnode != next_rnode) {
        mytree[next_xnode].from_flag = FROM_X_NOT_R;
        mytree[next_rnode].from_flag = FROM_R_NOT_X;
    } else {
        mytree[next_xnode].from_flag = FROM_NEITHER;
    }
    
    // Check if x and r go the same way
    if (next_xnode == next_rnode) {
        next_node = next_xnode;
    }
    
    // If not, go left
    if (next_node < 0) {
        next_node = cl;
        if (next_rnode == next_node) { // rpath
            N = N+1;
            feat_hist[feat] -= 1;
        } else if (next_xnode == next_node) { // xpath
            M = M+1;
            N = N+1;
            feat_hist[feat] += 1;
        }
    }
    node_stack[ns_ctr] = node;
    ns_ctr += 1;
    while (true) {
        node = next_node;
        curr_node = mytree[node];
        feat = curr_node.feat;
        thres = curr_node.thres;
        cl = curr_node.cl;
        cr = curr_node.cr;
        cd = curr_node.cd;
        pnode = curr_node.pnode;
        pfeat = curr_node.pfeat;
        from_flag = curr_node.from_flag;

        const bool x_right = x[feat] > thres;
        const bool r_right = r[feat] > thres;

        if (x_missing[feat]) {
            next_xnode = cd;
        } else if (x_right) {
            next_xnode = cr;
        } else if (!x_right) {
            next_xnode = cl;
        }
        
        if (r_missing[feat]) {
            next_rnode = cd;
        } else if (r_right) {
            next_rnode = cr;
        } else if (!r_right) {
            next_rnode = cl;
        }

        if (next_xnode >= 0) {
          if (next_xnode != next_rnode) {
              mytree[next_xnode].from_flag = FROM_X_NOT_R;
              mytree[next_rnode].from_flag = FROM_R_NOT_X;
          } else {
              mytree[next_xnode].from_flag = FROM_NEITHER;
          }
        }
        
//         if (DEBUG) {
//           myfile << "\nNode: " << node << "\n";
//           myfile << "N: " << N << ", M: " << M << "\n";
//           myfile << "from_flag==FROM_X_NOT_R: " << (from_flag==FROM_X_NOT_R) << "\n";
//           myfile << "from_flag==FROM_R_NOT_X: " << (from_flag==FROM_R_NOT_X) << "\n";
//           myfile << "from_flag==FROM_NEITHER: " << (from_flag==FROM_NEITHER) << "\n";
//           myfile << "feat_hist[feat]: " << feat_hist[feat] << "\n";
//         }
        
        // At a leaf
        if (cl < 0) {
            //      if (DEBUG) {
            //        myfile << "At a leaf\n";
            //      }

            if (M == 0) {
              out_contribs[num_feats] += mytree[node].value;
            }

            // Currently assuming a single output
            if (N != 0) {
                if (M != 0) {
                    pos_lst[node] = mytree[node].value * memoized_weights[N + max_depth * (M-1)];
                }
                if (M != N) {
                    neg_lst[node] = -mytree[node].value * memoized_weights[N + max_depth * M];
                }
            }
//             if (DEBUG) {
//               myfile << "pos_lst[node]: " << pos_lst[node] << "\n";
//               myfile << "neg_lst[node]: " << neg_lst[node] << "\n";
//             }
            // Pop from node_stack
            ns_ctr -= 1;
            next_node = node_stack[ns_ctr];
            from_child = node;
            // Unwind
            if (feat_hist[pfeat] > 0) {
                feat_hist[pfeat] -= 1;
            } else if (feat_hist[pfeat] < 0) {
                feat_hist[pfeat] += 1;
            }
            if (feat_hist[pfeat] == 0) {
                if (from_flag == FROM_X_NOT_R) {
                    N = N-1;
                    M = M-1;
                } else if (from_flag == FROM_R_NOT_X) {
                    N = N-1;
                }
            }
            continue;
        }
        
        // Arriving at node from parent
        if (from_child == -1) {
            //      if (DEBUG) {
            //        myfile << "Arriving at node from parent\n";
            //      }
            node_stack[ns_ctr] = node;
            ns_ctr += 1;
            next_node = -1;
            
            //      if (DEBUG) {
            //        myfile << "feat_hist[feat]" << feat_hist[feat] << "\n";
            //      }
            // Feature is set upstream
            if (feat_hist[feat] > 0) {
                next_node = next_xnode;
                feat_hist[feat] += 1;
            } else if (feat_hist[feat] < 0) {
                next_node = next_rnode;
                feat_hist[feat] -= 1;
            }
            
            // x and r go the same way
            if (next_node < 0) {
                if (next_xnode == next_rnode) {
                    next_node = next_xnode;
                }
            }
            
            // Go down one path
            if (next_node >= 0) {
                continue;
            }
            
            // Go down both paths, but go left first
            next_node = cl;
            if (next_rnode == next_node) {
                N = N+1;
                feat_hist[feat] -= 1;
            } else if (next_xnode == next_node) {
                M = M+1;
                N = N+1;
                feat_hist[feat] += 1;
            }
            from_child = -1;
            continue;
        }
        
        // Arriving at node from child
        if (from_child != -1) {
//             if (DEBUG) {
//               myfile << "Arriving at node from child\n";
//             }
            next_node = -1;
            // Check if we should unroll immediately
            if ((next_rnode == next_xnode) || (feat_hist[feat] != 0)) {
                next_node = pnode;
            }
            
            // Came from a single path, so unroll
            if (next_node >= 0) {
//                 if (DEBUG) {
//                   myfile << "Came from a single path, so unroll\n";
//                 }
                // At the root node
                if (node == 0) {
                    break;
                }
                // Update and unroll
                pos_lst[node] = pos_lst[from_child];
                neg_lst[node] = neg_lst[from_child];

//                 if (DEBUG) {
//                   myfile << "pos_lst[node]: " << pos_lst[node] << "\n";
//                   myfile << "neg_lst[node]: " << neg_lst[node] << "\n";
//                 }
                from_child = node;
                ns_ctr -= 1;
                
                // Unwind
                if (feat_hist[pfeat] > 0) {
                    feat_hist[pfeat] -= 1;
                } else if (feat_hist[pfeat] < 0) {
                    feat_hist[pfeat] += 1;
                }
                if (feat_hist[pfeat] == 0) {
                    if (from_flag == FROM_X_NOT_R) {
                        N = N-1;
                        M = M-1;
                    } else if (from_flag == FROM_R_NOT_X) {
                        N = N-1;
                    }
                }
                continue;
                // Go right - Arriving from the left child
            } else if (from_child == cl) {
//                 if (DEBUG) {
//                   myfile << "Go right - Arriving from the left child\n";
//                 }
                node_stack[ns_ctr] = node;
                ns_ctr += 1;
                next_node = cr;
                if (next_xnode == next_node) {
                    M = M+1;
                    N = N+1;
                    feat_hist[feat] += 1;
                } else if (next_rnode == next_node) {
                    N = N+1;
                    feat_hist[feat] -= 1;
                }
                from_child = -1;
                continue;
                // Compute stuff and unroll - Arriving from the right child
            } else if (from_child == cr) {
//                 if (DEBUG) {
//                   myfile << "Compute stuff and unroll - Arriving from the right child\n";
//                 }
                pos_x = 0;
                neg_x = 0;
                pos_r = 0;
                neg_r = 0;
                if ((next_xnode == cr) && (next_rnode == cl)) {
                    pos_x = pos_lst[cr];
                    neg_x = neg_lst[cr];
                    pos_r = pos_lst[cl];
                    neg_r = neg_lst[cl];
                } else if ((next_xnode == cl) && (next_rnode == cr)) {
                    pos_x = pos_lst[cl];
                    neg_x = neg_lst[cl];
                    pos_r = pos_lst[cr];
                    neg_r = neg_lst[cr];
                }
                // out_contribs needs to have been initialized as all zeros
                // if (pos_x + neg_r != 0) {
                //   std::cout << "val " << pos_x + neg_r << "\n";
                // }
                out_contribs[feat] += pos_x + neg_r;
                pos_lst[node] = pos_x + pos_r;
                neg_lst[node] = neg_x + neg_r;

//                 if (DEBUG) {
//                   myfile << "out_contribs[feat]: " << out_contribs[feat] << "\n";
//                   myfile << "pos_lst[node]: " << pos_lst[node] << "\n";
//                   myfile << "neg_lst[node]: " << neg_lst[node] << "\n";
//                 }
                
                // Check if at root
                if (node == 0) {
                    break;
                }
                
                // Pop
                ns_ctr -= 1;
                next_node = node_stack[ns_ctr];
                from_child = node;
                
                // Unwind
                if (feat_hist[pfeat] > 0) {
                    feat_hist[pfeat] -= 1;
                } else if (feat_hist[pfeat] < 0) {
                    feat_hist[pfeat] += 1;
                }
                if (feat_hist[pfeat] == 0) {
                    if (from_flag == FROM_X_NOT_R) {
                        N = N-1;
                        M = M-1;
                    } else if (from_flag == FROM_R_NOT_X) {
                        N = N-1;
                    }
                }
                continue;
            }
        }
    }
    //  if (DEBUG) {
    //    myfile.close();
    //  }
}




// void dense_tree_shap(const int *en_children_left, const int *en_children_right, const int *en_children_default,
//                      const int *en_features, const tfloat *en_thresholds, const tfloat *en_values,
//                      const tfloat *en_node_sample_weights, bool less_than_or_equal,
//                      const unsigned max_depth, const tfloat *X, const bool *X_missing,
//                      const tfloat *y, const tfloat *R, const bool *R_missing,
//                      const unsigned tree_limit, const tfloat base_offset, tfloat *out_contribs,
//                      const int feature_dependence,
//                      unsigned model_output, const unsigned num_X, const unsigned M, const unsigned num_R,
//                      const unsigned max_nodes, const unsigned num_outputs) {

void dense_independent(const TreeEnsemble& trees, const ExplanationDataset &data,
                       tfloat *out_contribs, unsigned model_output) {
    // this code is not ready for multi-valued trees yet
    if (trees.num_outputs > 1) {
        std::cout << "FEATURE_DEPENDENCE::independent does not support multi-output trees!\n";
        return;
    }

    // reformat the trees for faster access   
    Node *node_trees = new Node[trees.tree_limit * trees.max_nodes];
    for (unsigned i = 0; i < trees.tree_limit; ++i) {
        Node *node_tree = node_trees + i * trees.max_nodes;
        for (unsigned j = 0; j < trees.max_nodes; ++j) {
            const unsigned en_ind = i * trees.max_nodes + j;
            node_tree[j].cl = trees.children_left[en_ind];
            node_tree[j].cr = trees.children_right[en_ind];
            node_tree[j].cd = trees.children_default[en_ind];
            if (j == 0) {
                node_tree[j].pnode = 0;
            }
            if (trees.children_left[en_ind] >= 0) { // relies on all unused entires having -1 in them
                node_tree[trees.children_left[en_ind]].pnode = j;
                node_tree[trees.children_left[en_ind]].pfeat = trees.features[en_ind];
            }
            if (trees.children_right[en_ind] >= 0) { // relies on all unused entires having -1 in them
                node_tree[trees.children_right[en_ind]].pnode = j;
                node_tree[trees.children_right[en_ind]].pfeat = trees.features[en_ind];
            }

            node_tree[j].thres = trees.thresholds[en_ind];
            node_tree[j].value = trees.values[en_ind * trees.num_outputs]; // TODO: only handles num_outputs == 1 right now!!!!
            node_tree[j].feat = trees.features[en_ind];
        }
    }
    
    // preallocate arrays needed by the algorithm
    float *pos_lst = new float[trees.max_nodes];
    float *neg_lst = new float[trees.max_nodes];
    int *node_stack = new int[(unsigned) trees.max_depth];
    signed short *feat_hist = new signed short[data.M];
    tfloat *tmp_out_contribs = new tfloat[(data.M + 1) * trees.num_outputs];

    // precompute all the weight coefficients
    float *memoized_weights = new float[(trees.max_depth+1) * (trees.max_depth+1)];
    for (unsigned n = 0; n <= trees.max_depth; ++n) {
        for (unsigned m = 0; m <= trees.max_depth; ++m) {
            memoized_weights[n + trees.max_depth * m] = 1.0 / (n * bin_coeff(n-1, m));
        }
    }

    // compute the explanations for each sample
    tfloat *instance_out_contribs;
    for (unsigned i = 0; i < data.num_X; ++i) {
        instance_out_contribs = out_contribs + i * (data.M + 1) * trees.num_outputs;

        for (unsigned j = 0; j < data.num_R; ++j) {
            std::fill_n(tmp_out_contribs, (data.M + 1) * trees.num_outputs, 0);

            for (unsigned k = 0; k < trees.tree_limit; ++k) {
                tree_shap_indep(
                    trees.max_depth, data.M, trees.max_nodes, data.X + i * data.M,
                    data.X_missing + i * data.M, data.R + j * data.M, data.R_missing + j * data.M, 
                    tmp_out_contribs, pos_lst, neg_lst, feat_hist, memoized_weights, 
                    node_stack, node_trees + k * trees.max_nodes
                );
            }

            // add the effect of the current reference to our running total
            // this is where we can do per reference scaling for non-linear transformations
            for (unsigned k = 0; k < (data.M + 1) * trees.num_outputs; ++k) {
                instance_out_contribs[k] += tmp_out_contribs[k];
            }
        }

        // average the results over all the references.
        for (unsigned j = 0; j < (data.M + 1) * trees.num_outputs; ++j) {
            instance_out_contribs[j] /= data.num_R;
        }

        // apply the base offset to the bias term
        for (unsigned j = 0; j < trees.num_outputs; ++j) {
            instance_out_contribs[data.M * trees.num_outputs + j] += trees.base_offset;
        }
    }
    
    delete[] tmp_out_contribs;
    delete[] node_trees;
    delete[] pos_lst;
    delete[] neg_lst;
    delete[] node_stack;
    delete[] feat_hist;
    delete[] memoized_weights;
}


/**
 * This runs Tree SHAP with a per tree path conditional dependence assumption.
 */
void dense_tree_path_dependent(const TreeEnsemble& trees, const ExplanationDataset &data,
                               tfloat *out_contribs, unsigned model_output) {
    tfloat *instance_out_contribs;
    TreeEnsemble tree;
    ExplanationDataset instance;

    // build explanation for each sample
    for (unsigned i = 0; i < data.num_X; ++i) {
        instance_out_contribs = out_contribs + i * (data.M + 1) * trees.num_outputs;
        data.get_x_instance(instance, i);

        // aggregate the effect of explaining each tree
        // (this works because of the linearity property of Shapley values)
        for (unsigned j = 0; j < trees.tree_limit; ++j) {
            trees.get_tree(tree, j);
            tree_shap(tree, instance, instance_out_contribs, 0, 0);
        }

        // apply the base offset to the bias term
        for (unsigned j = 0; j < trees.num_outputs; ++j) {
            instance_out_contribs[data.M * trees.num_outputs + j] += trees.base_offset;
        }
    }
}

/**
 * This runs Tree SHAP with a global path conditional dependence assumption.
 * 
 * By first merging all the trees in a tree ensemble into an equivalent single tree
 * this method allows arbitrary marginal transformations and also ensures that all the
 * evaluations of the model are consistent with some training data point.
 */
void dense_global_path_dependent(const TreeEnsemble& trees, const ExplanationDataset &data,
                                 tfloat *out_contribs, unsigned model_output) {

    // allocate space for our new merged tree (we save enough room to totally split all samples if need be)
    TreeEnsemble merged_tree;
    merged_tree.allocate(1, (data.num_X + data.num_R) * 2, trees.num_outputs);
    
    // collapse the ensemble of trees into a single tree that has the same behavior
    // for all the X and R samples in the dataset
    build_merged_tree(merged_tree, data, trees);

    // compute the expected value and depth of the new merged tree
    compute_expectations(merged_tree);

    // explain each sample using our new merged tree
    ExplanationDataset instance;
    tfloat *instance_out_contribs;
    for (unsigned i = 0; i < data.num_X; ++i) {
        instance_out_contribs = out_contribs + i * (data.M + 1) * trees.num_outputs;
        data.get_x_instance(instance, i);
       
        // since we now just have a single merged tree we can just use the tree_path_dependent algorithm
        tree_shap(merged_tree, instance, instance_out_contribs, 0, 0);

        // apply the base offset to the bias term
        for (unsigned j = 0; j < trees.num_outputs; ++j) {
            instance_out_contribs[data.M * trees.num_outputs + j] += trees.base_offset;
        }
    }

    merged_tree.free();
}


/**
 * The main method for computing Tree SHAP on model using dense data.
 */
void dense_tree_shap(const TreeEnsemble& trees, const ExplanationDataset &data, tfloat *out_contribs,
                     const int feature_dependence, unsigned model_output) {
    
    std::cout << "dense_tree_shap " << feature_dependence << " " << model_output << "\n";

    switch (feature_dependence) {
        case FEATURE_DEPENDENCE::independent:
            dense_independent(trees, data, out_contribs, model_output);
            return;
        
        case FEATURE_DEPENDENCE::tree_path_dependent:
            dense_tree_path_dependent(trees, data, out_contribs, model_output);
            return;

        case FEATURE_DEPENDENCE::global_path_dependent:
            dense_global_path_dependent(trees, data, out_contribs, model_output);
            return;
    }
}
  
//   if (feature_dependence == FEATURE_DEPENDENCE::tree_path_dependent
//       && model_output == MODEL_OUTPUT::margin) {
//     std::cout << "tree_path_dependent margin\n";
//     //return;
//     tfloat *instance_out_contribs;

//     TreeEnsemble tree;
//     ExplanationDataset instance;
    
//     for (unsigned i = 0; i < data.num_X; ++i) {
//       instance_out_contribs = out_contribs + i * (M + 1) * num_outputs;
//       data.get_x_instance(i, instance);
//       for (unsigned j = 0; j < trees.tree_limit; ++j) {
//         trees.get_tree(j, tree);
//         tree_shap(tree, instance, instance_out_contribs, 0, 0);
//       }

//       // apply the base offset to the bias term
//       for (unsigned j = 0; j < num_outputs; ++j) {
//         instance_out_contribs[M * num_outputs + j] += base_offset;
//       }
//     }
  
//   } else if (feature_dependence == FEATURE_DEPENDENCE::global_path_dependent
//              && model_output == MODEL_OUTPUT::margin) {
//     std::cout << "global_path_dependent margin\n";
//     // std::cout << "data.num_X = " << std::endl;
//     // std::cout << "data.num_X = " << data.num_X << std::endl;
//     // std::cout << "num_R = " << num_R << std::endl;
    
//     if (num_R == 0) {
//       std::cout << "Error: num_R must be > 0!";
//       return;
//     }

//     // create a new joint dataset with both X and the reference datset R
//     // std::cout << "data.num_X = " << data.num_X << std::endl;
//     // std::cout << "num_R = " << num_R << std::endl;
//     // std::cout << "M = " << M << std::endl;
    
//     tfloat *joined_data = new tfloat[(data.num_X + num_R) * M];
//     std::copy(X, X + data.num_X * M, joined_data);
//     std::copy(R, R + num_R * M, joined_data + data.num_X * M);
//     bool *joined_data_missing = new bool[(data.num_X + num_R) * M];
//     std::copy(X_missing, X_missing + data.num_X * M, joined_data_missing);
//     std::copy(R_missing, R_missing + num_R * M, joined_data_missing + data.num_X * M);

//     // std::cout << "past copy " << std::endl;

//     // create an starting array of data indexes we will recursively sort
//     int *data_inds = new int[data.num_X + num_R];
//     for (unsigned i = 0; i < data.num_X; ++i) data_inds[i] = i;
//     for (unsigned i = data.num_X; i < data.num_X + num_R; ++i) {
//         data_inds[i] = -i; // a negative index means it won't be recorded as a background sample
//     }

//     // allocate space for our new merged tree (we save enough room to totally split all samples if need be)
//     int *children_left = new int[(data.num_X + num_R) * 2];
//     int *children_right = new int[(data.num_X + num_R) * 2];
//     int *children_default = new int[(data.num_X + num_R) * 2];
//     int *features = new int[(data.num_X + num_R) * 2];
//     tfloat *thresholds = new tfloat[(data.num_X + num_R) * 2];
//     tfloat *values = new tfloat[(data.num_X + num_R) * 2];
//     tfloat *node_sample_weights = new tfloat[(data.num_X + num_R) * 2];

//     // std::cout << "past allocate " << std::endl;
    
//     build_merged_tree(
//       children_left, children_right, children_default,
//       features, thresholds, values,
//       node_sample_weights, less_than_or_equal,
//       en_children_left, en_children_right, en_children_default,
//       en_features, en_thresholds, en_values,
//       trees.max_nodes, num_outputs, joined_data, joined_data_missing, data_inds,
//       num_R, trees.tree_limit, data.num_X + num_R, M
//     );

//     // std::cout << "past build_merged_tree " << std::endl;

//     unsigned merged_max_depth = compute_expectations(
//       num_outputs, children_left, children_right, node_sample_weights, values, 0, 0
//     );
//     std::cout << "past compute_expectations " << merged_max_depth << std::endl; 

//     // explain each sample using our new merged tree
//     tfloat *instance_out_contribs;
//     for (unsigned i = 0; i < data.num_X; ++i) {
//       instance_out_contribs = out_contribs + i * (M * num_outputs + 1);
//       // std::cout << "M = " << M << "\n";
//       // std::cout << "num_outputs = " << num_outputs << "\n";
//       //std::cout << "max_depth = " << max_depth << "\n";

//       // std::cout << "i = " << i << "\n";
//       // std::cout << "children_left[d] = " << children_left[d] << "\n";
      
//       tree_shap(
//         M, num_outputs, merged_max_depth, children_left, children_right, children_default,
//         features, thresholds, values,
//         node_sample_weights, X + i * M, X_missing + i * M, instance_out_contribs,
//         0, 0, less_than_or_equal
//       );

//       // tree_shap(
//       //   M, num_outputs, max_depth, children_left, children_right, children_default,
//       //   features, thresholds, values,
//       //   node_sample_weights, X + i * M, X_missing + i * M, instance_out_contribs,
//       //   0, 0, less_than_or_equal
//       // );

//       // apply the base offset to the bias term
//       for (unsigned j = 0; j < num_outputs; ++j) {
//         instance_out_contribs[M * num_outputs + j] += base_offset;
//       }
//     }

//     // std::cout << "just before delete " << std::endl;

//     delete[] data_inds;
//     delete[] joined_data;
//     delete[] joined_data_missing;
//     delete[] children_left;
//     delete[] children_right;
//     delete[] children_default;
//     delete[] features;
//     delete[] thresholds;
//     delete[] values;
//     delete[] node_sample_weights;

//   } else if (feature_dependence == FEATURE_DEPENDENCE::independent
//              && model_output == MODEL_OUTPUT::margin) {
//     std::cout << "independent margin\n";

//     // this code is not ready for multi-valued trees yet
//     if (num_outputs > 1) {
//       std::cout << "FEATURE_DEPENDENCE::independent does not support multi-output trees!\n";
//       return;
//     }

//     // Preallocating things    
//     Node *trees = new Node[trees.tree_limit * trees.max_nodes];
//     for (unsigned i = 0; i < trees.tree_limit; ++i) {
//       Node *tree = trees + i * trees.max_nodes;
//       for (unsigned j = 0; j < trees.max_nodes; ++j) {
//         const unsigned en_ind = i * trees.max_nodes + j;
//         tree[j].cl = en_children_left[en_ind];
//         tree[j].cr = en_children_right[en_ind];
//         tree[j].cd = en_children_default[en_ind];
//         if (j == 0) {
//           tree[j].pnode = 0;
//         }
//         if (en_children_left[en_ind] >= 0) { // FIXXX
//           tree[en_children_left[en_ind]].pnode = j;
//           tree[en_children_left[en_ind]].pfeat = en_features[en_ind];
//         }
//         if (en_children_right[en_ind] >= 0) {
//           tree[en_children_right[en_ind]].pnode = j;
//           tree[en_children_right[en_ind]].pfeat = en_features[en_ind];
//         }

//         tree[j].thres = en_thresholds[en_ind];
//         tree[j].value = en_values[en_ind * num_outputs]; // TODO: only handles num_outputs == 1 right now!!!!
//         tree[j].feat = en_features[en_ind];
//       }
//     }
      
//     float *pos_lst = new float[trees.max_nodes];
//     float *neg_lst = new float[trees.max_nodes];
//     int *node_stack = new int[(unsigned) max_depth];
//     signed short *feat_hist = new signed short[M];
//     float *memoized_weights = new float[(max_depth+1) * (max_depth+1)];
//     for (int n = 0; n <= max_depth; ++n) {
//       for (int m = 0; m <= max_depth; ++m) {
//         memoized_weights[n + max_depth * m] = calc_weight(n, m);
//       }
//     }

//     // std::cout << "data.num_X = " << data.num_X << std::endl;
//     // std::cout << "num_R = " << num_R << std::endl;
//     // std::cout << "M = " << M << std::endl;
//     tfloat *instance_out_contribs;
//     //int a;
//     tfloat *tmp_out_contribs = new tfloat[(M + 1) * num_outputs];
//     for (unsigned i = 0; i < data.num_X; ++i) {
//       instance_out_contribs = out_contribs + i * (M + 1) * num_outputs;

//       for (unsigned j = 0; j < num_R; ++j) {
//         std::fill_n(tmp_out_contribs, (M + 1) * num_outputs, 0);

//         for (unsigned k = 0; k < trees.tree_limit; ++k) {

//           //std::cout << "j * M = " << j * M << "\n";
//           tree_shap_indep(
//             max_depth, M, trees.max_nodes, X + i * M, X_missing + i * M, R + j * M, R_missing + j * M, 
//             tmp_out_contribs, pos_lst, neg_lst, feat_hist, memoized_weights, 
//             node_stack, trees + k * trees.max_nodes
//           );
//           for (unsigned l = 0; l < (M + 1) * num_outputs; ++l) {
//             if (!std::isfinite(tmp_out_contribs[l])) {
//               std::cout << "test " << i << " " << j << " " << k << " " << l << std::endl;
//             }
//           }
//         }

//         // add the effect of the current reference to our running total
//         // this is where we can do per reference scaling for non-linear transformations
//         for (unsigned k = 0; k < (M + 1) * num_outputs; ++k) {
//           instance_out_contribs[k] += tmp_out_contribs[k];
//         }
//       }

//       // average the results over all the references.
//       for (unsigned j = 0; j < (M + 1) * num_outputs; ++j) {
//         instance_out_contribs[j] /= num_R;
//       }

//       // apply the base offset to the bias term
//       for (unsigned j = 0; j < num_outputs; ++j) {
//         instance_out_contribs[M * num_outputs + j] += base_offset;
//       }
//     }
    
//     delete[] tmp_out_contribs;
//     delete[] trees;
//     delete[] pos_lst;
//     delete[] neg_lst;
//     delete[] node_stack;
//     delete[] feat_hist;
//     delete[] memoized_weights;
//   }
// }