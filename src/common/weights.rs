use std::ops::RangeBounds;

use operators::common_cpu::Blob;
use super::storage::*;
use common::Contiguous; 


pub struct Weights<'w> {
    blks: Box<[BlkStorage<Contiguous<'w, Blob>>]>,
    output_norm_bias: &'w [u8],
    output_norm_weight: &'w [u8],
    output_weight: &'w [u8],
}

impl<'w> Weights<'w> {
    pub fn new(
        model: &Storage<&'w [u8]>,
        range: impl RangeBounds<usize> + Clone,
        count: usize,
    ) -> Self {
        let Storage {
            output_norm_bias,
            output_norm_weight,
            output_weight,
            blocks,
            ..
        } = model;
        Self {
            blks: blocks
                .iter()
                .map(|blk| blk.distribute(&model.meta, range.clone(), count, Blob::new))
                .collect(),
            output_norm_bias,
            output_norm_weight,
            output_weight,
        }
    }
}