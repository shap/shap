type TFloat = f64;

pub mod feature_dependence {
    pub const INDEPENDENT: u32 = 0;
    pub const TREE_PATH_DEPENDENT: u32 = 1;
    pub const GLOBAL_PATH_DEPENDENT: u32 = 2;
}

#[derive(Debug, Clone)]
pub struct TreeEnsemble {
    pub children_left: Vec<i32>,
    pub children_right: Vec<i32>,
    pub children_default: Vec<i32>,
    pub features: Vec<i32>,
    pub thresholds: Vec<TFloat>,
    pub thresholds_types: Vec<i32>,
    pub values: Vec<TFloat>,
    pub node_sample_weights: Vec<TFloat>,
    pub max_depth: u32,
    pub tree_limit: u32,
    pub base_offset: Vec<TFloat>,
    pub max_nodes: u32,
    pub num_outputs: u32,
}

pub struct TreeSlice<'a> {
    pub children_left: &'a [i32],
    pub children_right: &'a [i32],
    pub children_default: &'a [i32],
    pub features: &'a [i32],
    pub thresholds: &'a [TFloat],
    pub thresholds_types: &'a [i32],
    pub values: &'a [TFloat],
    pub node_sample_weights: &'a [TFloat],
}

impl TreeEnsemble {
    #[inline]
    pub fn is_leaf(&self, pos: usize) -> bool {
        self.children_left[pos] < 0
    }

    #[inline]
    pub fn get_tree(&self, tree_index: usize) -> TreeSlice {
        let start = self.max_nodes as usize * tree_index;
        let end = start + self.max_nodes as usize;

        let values_start = start * self.num_outputs as usize;
        let values_end = end * self.num_outputs as usize;

        TreeSlice {
            children_left: &self.children_left[start..end],
            children_right: &self.children_right[start..end],
            children_default: &self.children_default[start..end],
            features: &self.features[start..end],
            thresholds: &self.thresholds[start..end],
            thresholds_types: &self.thresholds_types[start..end],
            values: &self.values[values_start..values_end],
            node_sample_weights: &self.node_sample_weights[start..end],
        }
    }
}
