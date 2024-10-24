
use gguf::*;
// use tensor::*;


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
            max_seq_len: gguf.llm_context_length().unwrap(),
            vocab_size: 50257,
            padded_vocab_size: 50304,
            num_layers: gguf.llm_block_count().unwrap(),
            num_heads: gguf.llm_attention_head_count().unwrap(),
            channels: gguf.llm_embedding_length().unwrap(),

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

#[test]
fn gguf_storage_test() {
    let Some(shards) = test_utils::map_gguf_files() else {
        return;
    };
    let gguf = GGufModel::read(shards.iter().map(|s| &**s));

    let model = Storage::from_gguf(&gguf);
    // assert_eq!(model.meta.distribute, 1);
    // let weights = Weights::new(&model, 0, 1);
    let Storage {
        meta,  ..
    } = model;
    println!("{meta:?}");
}