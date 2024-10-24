use gguf::*;
use tensor::*;

pub struct GPT2Config {  
    pub max_seq_len: usize,      // 最大序列长度  
    pub vocab_size: usize,       // 词汇量  
    pub padded_vocab_size: usize, // 填充词汇量  
    pub num_layers: usize,       // 层数  
    pub num_heads: usize,        // 注意力头的数量  
    pub channels: usize,         // 通道数  
}

#[test]
fn test_train() {
    let Some(shards) = test_utils::map_gguf_files() else {
        return;
    };
    let gguf = GGufModel::read(shards.iter().map(|s| &**s));

    let model = ParameterTensorsStorage::from_gguf(&gguf);
}
pub struct Gpt2Meta<'a> {
    // config
    pub config: GPT2Config,              // 模型配置
    // parameters
    pub params: ParameterTensorsStorage<&'a [u8]>,        // 模型权重参数
    pub param_sizes: [usize; 16],  // 模型参数的大小
    pub params_memory: *mut f32,         // 存储所有模型参数的内存块
    pub num_parameters: usize,           // 模型参数总数
    // para_grads
    pub grads: ParameterTensorsStorage<&'a [u8]>,         // 权重的梯度
    pub grads_memory: *mut f32,          // 存储所有梯度的内存块
    // AdamW
    pub m_memory: *mut f32,              // AdamW优化器的动量缓冲区 (m)
    pub v_memory: *mut f32,              // AdamW优化器的动量缓冲区 (v)
    // actvations
    pub acts: ActivationTensors,         // 模型激活值
    pub act_sizes: [usize; 23],   // 激活值的大小
    pub acts_memory: *mut f32,           // 存储所有激活值的内存块
    pub num_activations: usize,          // 激活值总数
    // atv_grads
    pub grads_acts: ActivationTensors,   // 激活值的梯度
    pub grads_acts_memory: *mut f32,     // 存储所有激活梯度的内存块
    // forward
    pub batch_size: usize,               // 前向传播的当前批大小
    pub seq_len: usize,                  // 前向传播的当前序列长度
    pub inputs: *mut i32,                // 当前传递的输入令牌
    pub targets: *mut i32,               // 前向传播的目标令牌
    pub mean_loss: f32,                  // 前向传播计算的平均损失值
}

pub struct Gpt2Meta_new {
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

pub struct Storage<T> {
    pub meta: Gpt2Meta_new, // Gpt2元信息
    pub wte: T,       // 令牌嵌入（V，C）。
    pub wpe: T,       // 位置嵌入（maxT，C）。
    pub blocks: Box<[BlkStorage<T>]>,
    pub lnfw: T,      // 最终层归一化权重 (C)。
    pub lnfb: T,      // 最终层归一化偏差 (C)。
}

pub struct BlkStorage<T> {
// ↓bolcks * 12 
    // ln1
    pub ln1w: T,      // 第一层的层归一化权重（L，C）。
    pub ln1b: T,      // 第一层的层归一化偏差（L，C）。
    // attn
    pub qkvw: T,      // 查询、键、值权重（L、3*C、C）。
    pub qkvb: T,      // 查询、键、值偏差 (L、3*C)。
    pub attprojw: T,  // 注意力投射权重（L、C、C）。
    pub attprojb: T,  // 注意投射偏差（L，C）。
    // ln2
    pub ln2w: T,      // 第二层的层归一化权重（L，C）。
    pub ln2b: T,      // 第二层的层归一化偏差（L，C）。
    // MLP
        // fc
    pub fcw: T,       // 全连接权重（L、4*C、C）。
    pub fcb: T,       // 全连接偏置（L、4*C）。
        // gelu()
        // fcproj
    pub fcprojw: T,   // 全连接投影权重（L、C、4*C）。
    pub fcprojb: T,   // 全连接投影偏差（L，C）。
// ↑blocks * 12
}

pub struct  Storage_new<T> {
    pub meta: Gpt2Meta_new,
    pub blocks: Box<[BlkStorage_new<T>]>,
    pub output_norm_bias: T,
    pub output_norm_weight: T,
    pub position_embd_weight: T,
    pub token_embd_weight: T,
    pub output_weight: T,
}
pub struct BlkStorage_new<T> {
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

impl<'a> Storage_new<&'a [u8]> {
    pub fn from_gguf(gguf: &GGufModel<'a>) -> Self {
        let output_norm_bias = &gguf.tensors["output_norm.bias"];
        let output_norm_weight = &gguf.tensors["output_norm.weight"];
        let position_embd_weight = &gguf.tensors["position_embd.weight"];
        let token_embd_weight = &gguf.tensors["token_embd.weight"];
        let output_weight = &gguf.tensors["output.weight"];
        #[rustfmt::skip]
        let mut meta = Gpt2Meta_new {
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
            .map(|i| BlkStorage_new {
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


pub struct ParameterTensors_orgin {
// ↓transformer
    pub wte: Tensor<f32>,       // 令牌嵌入（V，C）。
    pub wpe: Tensor<f32>,       // 位置嵌入（maxT，C）。
// ↓bolcks * 12 
    // ln1
    pub ln1w: Tensor<f32>,      // 第一层的层归一化权重（L，C）。
    pub ln1b: Tensor<f32>,      // 第一层的层归一化偏差（L，C）。
    // attn
    pub qkvw: Tensor<f32>,      // 查询、键、值权重（L、3*C、C）。
    pub qkvb: Tensor<f32>,      // 查询、键、值偏差 (L、3*C)。
    pub attprojw: Tensor<f32>,  // 注意力投射权重（L、C、C）。
    pub attprojb: Tensor<f32>,  // 注意投射偏差（L，C）。
    // ln2
    pub ln2w: Tensor<f32>,      // 第二层的层归一化权重（L，C）。
    pub ln2b: Tensor<f32>,      // 第二层的层归一化偏差（L，C）。
    // MLP
        // fc
    pub fcw: Tensor<f32>,       // 全连接权重（L、4*C、C）。
    pub fcb: Tensor<f32>,       // 全连接偏置（L、4*C）。
        // gelu()
        // fcproj
    pub fcprojw: Tensor<f32>,   // 全连接投影权重（L、C、4*C）。
    pub fcprojb: Tensor<f32>,   // 全连接投影偏差（L，C）。
// ↑blocks * 12
    pub lnfw: Tensor<f32>,      // 最终层归一化权重 (C)。
    pub lnfb: Tensor<f32>,      // 最终层归一化偏差 (C)。
// ↑transformer
// lm_head
}

pub struct ActivationTensors {  
    pub encoded: Tensor<f32>,       // 编码（B、T、C）  
    pub ln1: Tensor<f32>,           // 层归一化 1（L、B、T、C）  
    pub ln1_mean: Tensor<f32>,      // 层归一化 1 均值（L、B、T）  
    pub ln1_rstd: Tensor<f32>,      // 层归一化 1 倒数 std (L, B, T)  
    pub qkv: Tensor<f32>,           // 查询、键、值（L、B、T、3*C）  
    pub atty: Tensor<f32>,          // 注意力输出（L、B、T、C）  
    pub preatt: Tensor<f32>,        // 预注意分数（L、B、NH、T、T）  
    pub att: Tensor<f32>,           // 注意力分数（L、B、NH、T、T）  
    pub attproj: Tensor<f32>,       // 注意力投射（L、B、T、C）  
    pub residual2: Tensor<f32>,     // 第二个残差连接（L、B、T、C）  
    pub ln2: Tensor<f32>,           // 层归一化 2（L、B、T、C）  
    pub ln2_mean: Tensor<f32>,      // 层归一化 2 均值（L、B、T）  
    pub ln2_rstd: Tensor<f32>,      // 层归一化 2 倒数标准（L、B、T）  
    pub fch: Tensor<f32>,           // 全连接隐藏（L、B、T、4*C）  
    pub fch_gelu: Tensor<f32>,      // 全连接隐藏GELU激活（L、B、T、4*C）  
    pub fcproj: Tensor<f32>,        // 全连接投影（L、B、T、C）  
    pub residual3: Tensor<f32>,     // 第三个残差连接（L、B、T、C）  
    pub lnf: Tensor<f32>,           // 最终层归一化（B、T、C）  
    pub lnf_mean: Tensor<f32>,      // 最终层归一化平均值（B，T）  
    pub lnf_rstd: Tensor<f32>,      // 最终层归一化倒数 std (B, T)  
    pub logits: Tensor<f32>,        // 对数（B、T、V）  
    pub probs: Tensor<f32>,         // 概率（B、T、V）  
    pub losses: Tensor<f32>,        // 损失（B、T）  
}

pub struct ParameterTensorsStorage<T> {
    pub wte: T,       // 令牌嵌入（V，C）。
    pub wpe: T,       // 位置嵌入（maxT，C）。
    pub blocks: Box<[BlkStorage<T>]>,
    pub lnfw: T,      // 最终层归一化权重 (C)。
    pub lnfb: T,      // 最终层归一化偏差 (C)。
}



impl<'a> ParameterTensorsStorage<&'a [u8]> {
    pub fn from_gguf(gguf: &GGufModel<'a>) -> Self {
        Self {
            wte: gguf.wte,
            wpe: gguf.wpe,
            blocks: gguf.blocks.iter().map(|x| BlkStorage {
                ln1w: x.ln1w,
                ln1b: x.ln1b,
                qkvw: x.qkvw,
                qkvb: x.qkvb,
                attprojw: x.attprojw,
                attprojb: x.attprojb,
                ln2w: x.ln2w,
                ln2b: x.ln2b,
                fcw: x.fcw,
                fcb: x.fcb,
                fcprojw: x.fcprojw,
                fcprojb: x.fcprojb,
            }).collect(),
            lnfw: gguf.lnfw,
            lnfb: gguf.lnfb,
        }
    }
}

// Llama2.c
// Transformer(
//     (tok_embeddings): Embedding(32000, 768)
//     (dropout): Dropout(p=0.0, inplace=False)
//     (layers): ModuleList(
//       (0-11): 12 x TransformerBlock(
//         (attention): Attention(
//           (wq): Linear(in_features=768, out_features=768, bias=False)
//           (wk): Linear(in_features=768, out_features=768, bias=False)
//           (wv): Linear(in_features=768, out_features=768, bias=False)
//           (wo): Linear(in_features=768, out_features=768, bias=False)
//           (attn_dropout): Dropout(p=0.0, inplace=False)
//           (resid_dropout): Dropout(p=0.0, inplace=False)
//         )
//         (feed_forward): FeedForward(
//           (w1): Linear(in_features=768, out_features=2268, bias=False)
//           (w2): Linear(in_features=2268, out_features=768, bias=False)
//           (w3): Linear(in_features=768, out_features=2268, bias=False)
//           (dropout): Dropout(p=0.0, inplace=False)
//         )
//         (attention_norm): RMSNorm()
//         (ffn_norm): RMSNorm()
//       )
//     )
//     (norm): RMSNorm()
//     (output): Linear(in_features=768, out_features=32000, bias=False)
//   )
// #[derive(Clone)]
// pub struct Storage<T> {
//     pub meta: LlamaMeta,
//     pub token_embed: T,
//     pub output_norm: T,
//     pub output: T,
//     pub blocks: Box<[BlkStorage<T>]>,
// }
// #[derive(Clone, Copy)]
// pub struct BlkStorage<T> {
//     pub attn_norm: T,
//     pub attn_qkv: T,
//     pub attn_o: T,
//     pub ffn_norm: T,
//     pub ffn_gate_up: T,
//     pub ffn_down: T,
// }