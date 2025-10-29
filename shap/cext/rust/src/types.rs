pub type TFloat = f64;

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

#[derive(Debug, Clone)]
pub struct ExplanationDataset {
    pub X: Vec<TFloat>,
    pub X_missing: Vec<bool>,
    pub y: Vec<TFloat>,
    pub R: Vec<TFloat>,
    pub R_missing: Vec<bool>,
    pub num_X: u32,
    pub M: u32,
    pub num_R: u32,
}

impl ExplanationDataset {
    #[inline]
    pub fn get_X_instance(&self, instance_index: usize) -> &[TFloat] {
        let start = instance_index * self.M as usize;
        let end = start + self.M as usize;
        &self.X[start..end]
    }
}


#[derive(Debug, Clone, Copy)]
pub struct PathElement {
    pub feature_index: i32,
    pub zero_fraction: TFloat,
    pub one_fraction: TFloat,
    pub pweight: TFloat,
}

impl PathElement {
    #[inline]
    pub fn new(feature_index: i32, zero_fraction: TFloat, one_fraction: TFloat, pweight: TFloat) -> Self {
        PathElement {
            feature_index,
            zero_fraction,
            one_fraction,
            pweight,
        }
    }
}

// Model transformation types
pub mod model_transform {
    pub const IDENTITY: u32 = 0;
    pub const LOGISTIC: u32 = 1;
    pub const LOGISTIC_NLOGLOSS: u32 = 2;
    pub const SQUARED_LOSS: u32 = 3;
}

// Transform functions
#[inline]
pub fn logistic_transform(margin: TFloat, _y: TFloat) -> TFloat {
    1.0 / (1.0 + (-margin).exp())
}

#[inline]
pub fn logistic_nlogloss_transform(margin: TFloat, y: TFloat) -> TFloat {
    (1.0 + margin.exp()).ln() - y * margin
}

#[inline]
pub fn squared_loss_transform(margin: TFloat, y: TFloat) -> TFloat {
    (margin - y) * (margin - y)
}

pub type TransformFn = fn(TFloat, TFloat) -> TFloat;

#[inline]
pub fn get_transform(model_transform: u32) -> Option<TransformFn> {
    match model_transform {
        model_transform::LOGISTIC => Some(logistic_transform),
        model_transform::LOGISTIC_NLOGLOSS => Some(logistic_nlogloss_transform),
        model_transform::SQUARED_LOSS => Some(squared_loss_transform),
        _ => None,
    }
}

// Helper function for categorical thresholds
#[inline]
pub fn category_in_threshold(threshold: TFloat, category: TFloat) -> bool {
    let category_flag = 1 << (category as i32 - 1);
    (threshold as i32 & category_flag) != 0
}

