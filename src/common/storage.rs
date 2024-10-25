
use std::{ops::{DerefMut, Range, RangeBounds}, path::Path};

use common::{borrow, Contiguous};
use ext::Mmap;
use ggml_quants::digit_layout::DigitLayout;
use gguf::*;
use tensor::*;
use super::weights::Weights;
// type Worker<'w> = LlamaWorker<Operators, Weights<'w>>;

#[derive(Clone, Debug)]
pub struct Gpt2Meta {
    // gpt2config
    pub max_seq_len: usize,      // 最大序列长度  1024 // context_length
    pub vocab_size: usize,       // 词汇量  50257 
    pub padded_vocab_size: usize, // 填充词汇量  50304
    pub num_layers: usize,       // 层数  12 // block_count
    pub num_heads: usize,        // 注意力头的数量 12  // attention.head_count
    pub channels: usize,         // 通道数  768 // embeding_length
    // gguf_metaKV
    pub epsilon: f32,           // attention.layer_norm_epsilon // 1e-5 
}

#[derive(Clone)]
pub struct  Storage<T> {
    pub meta: Gpt2Meta,
    pub blocks: Box<[BlkStorage<T>]>,
    pub output_norm_bias: T,
    pub output_norm_weight: T,
    pub position_embd_weight: T,
    pub token_embd_weight: T,
    pub output_weight: T,
}
#[derive(Clone, Copy)]
pub struct BlkStorage<T> {
    pub attn_qkv_bias: T,
    pub attn_qkv_weight: T,
    pub attn_output_bias: T,
    pub attn_output_weight: T,
    pub attn_norm_bias: T,
    pub attn_norm_weight: T,

    pub ffn_up_bias: T,
    pub ffn_up_weight: T,
    pub ffn_down_bias: T,
    pub ffn_down_weight: T,
    pub ffn_norm_bias: T,
    pub ffn_norm_weight: T,
}

impl<'a> Storage<&'a [u8]> {
    pub fn from_gguf(gguf: &GGufModel<'a>) -> Self {
        let output_norm_bias = &gguf.tensors["output_norm.bias"];
        let output_norm_weight = &gguf.tensors["output_norm.weight"];
        let position_embd_weight = &gguf.tensors["position_embd.weight"];
        let token_embd_weight = &gguf.tensors["token_embd.weight"];
        let output_weight = &gguf.tensors["output.weight"];
        #[rustfmt::skip]
        let mut meta = Gpt2Meta {
            max_seq_len         : gguf.llm_context_length().unwrap(),
            vocab_size          : 50257,
            padded_vocab_size   : 50304,
            num_layers          : gguf.llm_block_count().unwrap(),
            num_heads           : gguf.llm_attention_head_count().unwrap(),
            channels            : gguf.llm_embedding_length().unwrap(),

            epsilon: 1e-5,
        };
        #[rustfmt::skip]
        let blocks = (0..meta.num_layers)
            .map(|i| BlkStorage {
                attn_qkv_bias:      gguf.tensors[&*format!("blk.{i}.attn_qkv.bias"     )].data,
                attn_qkv_weight:    gguf.tensors[&*format!("blk.{i}.attn_qkv.weight"   )].data,
                attn_output_bias:   gguf.tensors[&*format!("blk.{i}.attn_output.bias"  )].data,
                attn_output_weight: gguf.tensors[&*format!("blk.{i}.attn_output.weight")].data,
                attn_norm_bias:     gguf.tensors[&*format!("blk.{i}.attn_norm.bias"    )].data,
                attn_norm_weight:   gguf.tensors[&*format!("blk.{i}.attn_norm.weight"  )].data,

                ffn_up_bias:        gguf.tensors[&*format!("blk.{i}.ffn_up.bias"       )].data,
                ffn_up_weight:      gguf.tensors[&*format!("blk.{i}.ffn_up.weight"     )].data,
                ffn_down_bias:      gguf.tensors[&*format!("blk.{i}.ffn_down.bias"     )].data,
                ffn_down_weight:    gguf.tensors[&*format!("blk.{i}.ffn_down.weight"   )].data,
                ffn_norm_bias:      gguf.tensors[&*format!("blk.{i}.ffn_norm.bias"     )].data,
                ffn_norm_weight:    gguf.tensors[&*format!("blk.{i}.ffn_norm.weight"   )].data,
            })
            .collect();

        Self {
            meta,
            blocks,
            output_norm_bias: output_norm_bias.data,
            output_norm_weight: output_norm_weight.data,
            position_embd_weight: position_embd_weight.data,
            token_embd_weight: token_embd_weight.data,
            output_weight: output_weight.data,
    }
}
}

fn normalize(range: impl RangeBounds<usize>, count: usize) -> Range<usize> {
    use std::ops::Bound::{Excluded, Included, Unbounded};
    let start = match range.start_bound() {
        Included(&i) => i,
        Excluded(&i) => i + 1,
        Unbounded => 0,
    };
    let end = match range.end_bound() {
        Included(&i) => i + 1,
        Excluded(&i) => i,
        Unbounded => count,
    };
    assert!(start < end && end <= count);
    start..end
}


impl<'w> BlkStorage<&'w [u8]> {
    pub fn distribute<U>(
        &self,
        meta: &Gpt2Meta,
        range: impl RangeBounds<usize>,
        count: usize,
        mut f: impl FnMut(usize) -> U,
    ) -> BlkStorage<Contiguous<'w, U>>
    where
        U: DerefMut<Target = [u8]>,
    {
        let range = normalize(range, count);
        let start = range.start;
        let len = range.len();
        assert!(0 < len && range.end <= count);

        fn tensor<'t>(dt: DigitLayout, shape: &[usize], data: &'t [u8]) -> Tensor<&'t [u8]> {
            Tensor::new(dt, shape).map(|size| {
                debug_assert_eq!(size, data.len());
                data
            })
        }

        BlkStorage {
            attn_qkv_bias: borrow(&self.attn_norm_bias),
            attn_qkv_weight: borrow(&self.attn_qkv_weight),
            attn_output_bias: borrow(&self.attn_output_bias),
            attn_output_weight: borrow(&self.attn_output_weight),
            attn_norm_bias: borrow(&self.attn_norm_bias),
            attn_norm_weight: borrow(&self.attn_norm_weight),

            ffn_up_bias: borrow(&self.ffn_norm_bias),
            ffn_up_weight: borrow(&self.ffn_up_weight),
            ffn_down_bias: borrow(&self.ffn_norm_bias),
            ffn_down_weight: borrow(&self.ffn_down_weight),
            ffn_norm_bias: borrow(&self.ffn_norm_bias),
            ffn_norm_weight: borrow(&self.ffn_norm_weight),
        }
    }
}


pub fn map_gguf_files() -> Option<Box<[Mmap]>> {
    let Some(path) = std::env::var_os("TEST_MODEL") else {
        println!("TEST_MODEL not set");
        return None;
    };
    let path = Path::new(&path);
    if !path.is_file() {
        println!("{path:?} not found");
        return None;
    }
    Some(map_files(path))
}

#[test]
fn gguf_storage_test() {
    let Some(shards) = map_gguf_files() else {
        return;
    };
    let gguf = GGufModel::read(shards.iter().map(|s| &**s));
    let gpt2 = Storage::from_gguf(&gguf);
    println!("{:?}", gpt2.meta);
    // ---- common::storage::gguf_storage_test stdout ----
    // Gpt2Meta { max_seq_len: 1024, vocab_size: 50257, padded_vocab_size: 50304, num_layers: 12, num_heads: 12, channels: 768, epsilon: 1e-5 }

    let weights = Weights::new(&gpt2, .., 1);
    // let mut worker = Worker::new(&Cpu, meta.clone(), weights, true);
}



