use std::fs::File;
use std::io::Read;
use std::{mem, ptr};
use core::slice;
use std::alloc::{self, alloc, Layout};

use crate::passes::*;
use crate::common::tensor::Tensor_; 
use tensor::*;

pub struct ParameterTensors {
// ↓transformer
    pub wte: Tensor_<f32>,       // 令牌嵌入（V，C）。
    pub wpe: Tensor_<f32>,       // 位置嵌入（maxT，C）。
// ↓bolcks * 12 
    // ln1
    pub ln1w: Tensor_<f32>,      // 第一层的层归一化权重（L，C）。
    pub ln1b: Tensor_<f32>,      // 第一层的层归一化偏差（L，C）。
    // attn
    pub qkvw: Tensor_<f32>,      // 查询、键、值权重（L、3*C、C）。
    pub qkvb: Tensor_<f32>,      // 查询、键、值偏差 (L、3*C)。
    pub attprojw: Tensor_<f32>,  // 注意力投射权重（L、C、C）。
    pub attprojb: Tensor_<f32>,  // 注意投射偏差（L，C）。
    // ln2
    pub ln2w: Tensor_<f32>,      // 第二层的层归一化权重（L，C）。
    pub ln2b: Tensor_<f32>,      // 第二层的层归一化偏差（L，C）。
    // MLP
        // fc
    pub fcw: Tensor_<f32>,       // 全连接权重（L、4*C、C）。
    pub fcb: Tensor_<f32>,       // 全连接偏置（L、4*C）。
        // gelu()
        // fcproj
    pub fcprojw: Tensor_<f32>,   // 全连接投影权重（L、C、4*C）。
    pub fcprojb: Tensor_<f32>,   // 全连接投影偏差（L，C）。
// ↑blocks * 12
    pub lnfw: Tensor_<f32>,      // 最终层归一化权重 (C)。
    pub lnfb: Tensor_<f32>,      // 最终层归一化偏差 (C)。
// ↑transformer
// lm_head
}
// GPT(
//     (transformer): ModuleDict(
//       (wte): Embedding(50257, 768)
//       (wpe): Embedding(1024, 768)
//       (h): ModuleList(
//         (0-11): 12 x Block(
//           (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
//           (attn): CausalSelfAttention(
//             (c_attn): Linear(in_features=768, out_features=2304, bias=True)
//             (c_proj): Linear(in_features=768, out_features=768, bias=True)
//           )
//           (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
//           (mlp): MLP(
//             (c_fc): Linear(in_features=768, out_features=3072, bias=True)
//             (gelu): NewGELU()
//             (c_proj): Linear(in_features=3072, out_features=768, bias=True)
//           )
//         )
//       )
//       (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
//     )
//     (lm_head): Linear(in_features=768, out_features=50257, bias=False)
//   )
impl ParameterTensors {
    pub fn new() -> Self {
        ParameterTensors {
            wte: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            wpe: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            ln1w: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            ln1b: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            qkvw: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            qkvb: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            attprojw: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            attprojb: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            ln2w: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            ln2b: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            fcw: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            fcb: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            fcprojw: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            fcprojb: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            lnfw: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            lnfb: Tensor_::new(ptr::null_mut(), 0, 0, 0),
        }
    }

    // 为参数分配内存并将各个张量指向正确的位置
    pub unsafe fn malloc_and_point_parameters(
        &mut self,
        param_sizes: &[usize; NUM_PARAMETER_TENSORS],
    ) -> *mut f32 {
        // 计算所需的总大小
        let num_parameters: usize = param_sizes.iter().sum();
        // 一次性分配所有参数
        let layout = Layout::array::<f32>(num_parameters)
            .expect("error creating layout");
        let params_memory = alloc(layout) as *mut f32;
        // 检查是否分配成功
        if params_memory.is_null() {
            panic!("error allocating memory");
        }
        // 分配所有张量
        let mut params_memory_iterator = params_memory;
        let mut ptrs: [*mut *mut f32; NUM_PARAMETER_TENSORS] = [
            &mut self.wte.ptr, &mut self.wpe.ptr, &mut self.ln1w.ptr, &mut self.ln1b.ptr, &mut self.qkvw.ptr, &mut self.qkvb.ptr,
            &mut self.attprojw.ptr, &mut self.attprojb.ptr, &mut self.ln2w.ptr, &mut self.ln2b.ptr, &mut self.fcw.ptr, &mut self.fcb.ptr,
            &mut self.fcprojw.ptr, &mut self.fcprojb.ptr, &mut self.lnfw.ptr, &mut self.lnfb.ptr,
        ];
        // 将每个指针指向相应的内存位置
        for (i, ptr) in ptrs.iter_mut().enumerate() {
            **ptr = params_memory_iterator;
            params_memory_iterator = params_memory_iterator.add(param_sizes[i]);
        }

        params_memory
    }
}

pub struct ActivationTensors {  
    pub encoded: Tensor_<f32>,       // 编码（B、T、C）  
    pub ln1: Tensor_<f32>,           // 层归一化 1（L、B、T、C）  
    pub ln1_mean: Tensor_<f32>,      // 层归一化 1 均值（L、B、T）  
    pub ln1_rstd: Tensor_<f32>,      // 层归一化 1 倒数 std (L, B, T)  
    pub qkv: Tensor_<f32>,           // 查询、键、值（L、B、T、3*C）  
    pub atty: Tensor_<f32>,          // 注意力输出（L、B、T、C）  
    pub preatt: Tensor_<f32>,        // 预注意分数（L、B、NH、T、T）  
    pub att: Tensor_<f32>,           // 注意力分数（L、B、NH、T、T）  
    pub attproj: Tensor_<f32>,       // 注意力投射（L、B、T、C）  
    pub residual2: Tensor_<f32>,     // 第二个残差连接（L、B、T、C）  
    pub ln2: Tensor_<f32>,           // 层归一化 2（L、B、T、C）  
    pub ln2_mean: Tensor_<f32>,      // 层归一化 2 均值（L、B、T）  
    pub ln2_rstd: Tensor_<f32>,      // 层归一化 2 倒数标准（L、B、T）  
    pub fch: Tensor_<f32>,           // 全连接隐藏（L、B、T、4*C）  
    pub fch_gelu: Tensor_<f32>,      // 全连接隐藏GELU激活（L、B、T、4*C）  
    pub fcproj: Tensor_<f32>,        // 全连接投影（L、B、T、C）  
    pub residual3: Tensor_<f32>,     // 第三个残差连接（L、B、T、C）  
    pub lnf: Tensor_<f32>,           // 最终层归一化（B、T、C）  
    pub lnf_mean: Tensor_<f32>,      // 最终层归一化平均值（B，T）  
    pub lnf_rstd: Tensor_<f32>,      // 最终层归一化倒数 std (B, T)  
    pub logits: Tensor_<f32>,        // 对数（B、T、V）  
    pub probs: Tensor_<f32>,         // 概率（B、T、V）  
    pub losses: Tensor_<f32>,        // 损失（B、T）  
}

impl ActivationTensors {
    pub fn new() -> Self {
        ActivationTensors {
            encoded: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            ln1: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            ln1_mean: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            ln1_rstd: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            qkv: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            atty: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            preatt: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            att: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            attproj: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            residual2: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            ln2: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            ln2_mean: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            ln2_rstd: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            fch: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            fch_gelu: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            fcproj: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            residual3: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            lnf: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            lnf_mean: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            lnf_rstd: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            logits: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            probs: Tensor_::new(ptr::null_mut(), 0, 0, 0),
            losses: Tensor_::new(ptr::null_mut(), 0, 0, 0),
        }
    }

    // 为激活张量分配内存并将各个张量指向正确的位置
    pub unsafe fn alloc_and_point_activations(
        &mut self,
        act_sizes: &[usize; NUM_ACTIVATION_TENSORS],
    ) -> *mut f32 {
        // 计算所需的总大小
        let num_activations: usize = act_sizes.iter().sum();
        // 为所有激活分配内存
        let layout = Layout::array::<f32>(num_activations)
            .expect("error creating layout");
        let acts_memory = alloc(layout) as *mut f32;
        // 检查是否分配成功
        if acts_memory.is_null() {
            panic!("error allocating memory");
        }
        // 将张量分配给分配的内存
        let mut acts_memory_iterator = acts_memory;
        let mut ptrs: [*mut *mut f32; NUM_ACTIVATION_TENSORS] = [
            &mut self.encoded.ptr,&mut self.ln1.ptr,&mut self.ln1_mean.ptr,&mut self.ln1_rstd.ptr,&mut self.qkv.ptr,&mut self.atty.ptr,
            &mut self.preatt.ptr,&mut self.att.ptr,&mut self.attproj.ptr,&mut self.residual2.ptr, &mut self.ln2.ptr, &mut self.ln2_mean.ptr,
            &mut self.ln2_rstd.ptr,&mut self.fch.ptr,&mut self.fch_gelu.ptr,&mut self.fcproj.ptr, &mut self.residual3.ptr,&mut self.lnf.ptr,
            &mut self.lnf_mean.ptr,&mut self.lnf_rstd.ptr,&mut self.logits.ptr,&mut self.probs.ptr,&mut self.losses.ptr,
        ];
        // 将已分配内存的切片分配给每个张量
        for (i, ptr) in ptrs.iter_mut().enumerate() {
            **ptr = acts_memory_iterator;
            acts_memory_iterator = acts_memory_iterator.add(act_sizes[i]);
        }

        acts_memory
    }
}

pub struct GPT2Config {  
    pub max_seq_len: usize,      // 最大序列长度  
    pub vocab_size: usize,       // 词汇量  
    pub padded_vocab_size: usize, // 填充词汇量  
    pub num_layers: usize,       // 层数  
    pub num_heads: usize,        // 注意力头的数量  
    pub channels: usize,         // 通道数  
}
impl GPT2Config {
    fn new() -> Self {
        GPT2Config {
            max_seq_len: 0,
            vocab_size: 0,
            padded_vocab_size: 0,
            num_layers: 0,
            num_heads: 0,
            channels: 0,
        }
    }
}
pub const NUM_PARAMETER_TENSORS: usize = 16;
pub const NUM_ACTIVATION_TENSORS: usize = 23;
pub struct GPT2 {
    pub config: GPT2Config,              // 模型配置
    pub params: ParameterTensors,        // 模型权重参数
    pub param_sizes: [usize; NUM_PARAMETER_TENSORS],  // 模型参数的大小
    pub params_memory: *mut f32,         // 存储所有模型参数的内存块
    pub num_parameters: usize,           // 模型参数总数
    pub grads: ParameterTensors,         // 权重的梯度
    pub grads_memory: *mut f32,          // 存储所有梯度的内存块
    pub m_memory: *mut f32,              // AdamW优化器的动量缓冲区 (m)
    pub v_memory: *mut f32,              // AdamW优化器的动量缓冲区 (v)
    pub acts: ActivationTensors,         // 模型激活值
    pub act_sizes: [usize; NUM_ACTIVATION_TENSORS],   // 激活值的大小
    pub acts_memory: *mut f32,           // 存储所有激活值的内存块
    pub num_activations: usize,          // 激活值总数
    pub grads_acts: ActivationTensors,   // 激活值的梯度
    pub grads_acts_memory: *mut f32,     // 存储所有激活梯度的内存块
    pub batch_size: usize,               // 前向传播的当前批大小
    pub seq_len: usize,                  // 前向传播的当前序列长度
    pub inputs: *mut i32,                // 当前传递的输入令牌
    pub targets: *mut i32,               // 前向传播的目标令牌
    pub mean_loss: f32,                  // 前向传播计算的平均损失值
}

impl GPT2 {
    pub fn checkpoint(checkpoint_path: &str) -> Self {
        let mut model = GPT2 {
            config: GPT2Config::new(),
            params: ParameterTensors::new(),
            param_sizes: [0; NUM_PARAMETER_TENSORS],
            params_memory: core::ptr::null_mut(),
            num_parameters: 0,
            grads: ParameterTensors::new(),
            grads_memory: core::ptr::null_mut(),
            m_memory: core::ptr::null_mut(),
            v_memory: core::ptr::null_mut(),
            acts: ActivationTensors::new(),
            act_sizes: [0; NUM_ACTIVATION_TENSORS],
            acts_memory: core::ptr::null_mut(),
            num_activations: 0,
            grads_acts: ActivationTensors::new(),
            grads_acts_memory: core::ptr::null_mut(),
            inputs: core::ptr::null_mut(),
            targets: core::ptr::null_mut(),
            batch_size: 0,
            seq_len: 0,
            mean_loss: -1.0, // 如果返回-1.0f，则表示训练过程中没有产生有效损失。
        };

        // 从检查点文件读取模型
        let mut model_file = File::open(checkpoint_path)
            .expect("error opening model");

        // 读取模型头部
        let mut model_header = [0; 256]; 
        model_file
            .read_exact(unsafe {
                slice::from_raw_parts_mut(
                    model_header.as_mut_ptr() as *mut u8,
                    model_header.len() * mem::size_of::<i32>(),
                )
            })
            .expect("error reading model header");

        // 检查魔数和版本号
        if model_header[0] != 20240326 {
            panic!("Bad magic model file");
        }
        if model_header[1] != 3 {
            panic!("Bad version in model file\n
                    ---> HINT: try to re-run `python train_gpt2.py`");
        }

        // 读取超参数
        let maxT = model_header[2] as usize;  // 最大序列长度
        let V = model_header[3] as usize;     // 词汇量大小
        let L = model_header[4] as usize;     // 层数
        let NH = model_header[5] as usize;    // 注意力头数量
        let C = model_header[6] as usize;     // 通道数
        let Vp = model_header[7] as usize;    // 填充后的词汇量大小
        
        // 配置 GPT-2 模型的参数
        model.config = GPT2Config {
            max_seq_len: maxT,          // 设置最大序列长度
            vocab_size: V,              // 设置词汇量大小
            padded_vocab_size: Vp,      // 设置填充后的词汇量大小
            num_layers: L,              // 设置层数
            num_heads: NH,              // 设置注意力头数量
            channels: C,                // 设置通道数
        };
        
        // 输出超参数信息
        println!("[GPT-2]"); 
        println!("max_seq_len: {}", maxT);   // 输出最大序列长度
        println!("vocab_size: {}", V);       // 输出词汇量大小
        println!("padded_vocab_size: {}", Vp); // 输出填充后的词汇量大小
        println!("num_layers: {}", L);       // 输出层数
        println!("num_heads: {}", NH);       // 输出注意力头数量
        println!("channels: {}", C);         // 输出通道数
        
        // 为模型参数分配空间并读取它们
        model.param_sizes[0] = Vp * C;       // 词嵌入矩阵 (wte)
        model.param_sizes[1] = maxT * C;     // 位置嵌入矩阵 (wpe)
        model.param_sizes[2] = L * C;        // 第一层归一化权重 (ln1w)
        model.param_sizes[3] = L * C;        // 第一层归一化偏置 (ln1b)
        model.param_sizes[4] = L * (3 * C) * C; // 自注意力层 qkv 权重 (qkvw)
        model.param_sizes[5] = L * (3 * C);  // 自注意力层 qkv 偏置 (qkvb)
        model.param_sizes[6] = L * C * C;    // 注意力投影权重 (attprojw)
        model.param_sizes[7] = L * C;        // 注意力投影偏置 (attprojb)
        model.param_sizes[8] = L * C;        // 第二层归一化权重 (ln2w)
        model.param_sizes[9] = L * C;        // 第二层归一化偏置 (ln2b)
        model.param_sizes[10] = L * (4 * C) * C; // 前馈层权重 (fcw)
        model.param_sizes[11] = L * (4 * C); // 前馈层偏置 (fcb)
        model.param_sizes[12] = L * C * (4 * C); // 前馈投影权重 (fcprojw)
        model.param_sizes[13] = L * C;       // 前馈投影偏置 (fcprojb)
        model.param_sizes[14] = C;           // 最终归一化权重 (lnfw)
        model.param_sizes[15] = C;           // 最终归一化偏置 (lnfb)
        
        // 统计参数总数
        let num_parameters: usize = model.param_sizes.iter().sum(); 
        println!("num_parameters: {}", num_parameters);  // 输出总参数数
        model.num_parameters = num_parameters;           // 存储总参数数
        
        unsafe {
            // 为所有参数分配空间并读入
            model.params_memory = model.params.malloc_and_point_parameters(&model.param_sizes);
            model_file
                .read_exact(slice::from_raw_parts_mut(
                    model.params_memory as *mut u8,
                    num_parameters * mem::size_of::<f32>(),
                ))
                .expect("Failed to read parameters");
        }

        model
    }

    pub fn forward(&mut self, inputs: *mut i32, targets: *mut i32, B: usize, T: usize) {
        // target是可选的并且可以为 NULL
        // 确保模型已初始化
        if self.params_memory.is_null() {
            panic!("error initializing model");
        }

        // 方便参数（usize有助于防止int溢出）
        let V = self.config.vocab_size; // 50257
        let Vp = self.config.padded_vocab_size; // 50304
        let L = self.config.num_layers; // 12
        let NH = self.config.num_heads; // 12
        let C = self.config.channels; // 768

        // 验证输入，所有索引必须在 [0, V) 范围内
        unsafe {
            for i in 0..(B * T) {
                assert!((*inputs.add(i) >= 0 && *inputs.add(i) < V as i32));
                if !targets.is_null() {
                    assert!((*targets.add(i) >= 0 && *targets.add(i) < V as i32));
                }
            }
        }

        // 如果需要，为所有激活分配空间（此处延迟完成）
        if self.acts_memory.is_null() {
            // 并记录当前的B、T
            self.batch_size = B;
            self.seq_len = T;

            // 分配空间
            self.act_sizes[0] = B * T * C;  // 编码后的张量大小 (encoded)
            self.act_sizes[1] = L * B * T * C;  // 第一次LayerNorm输出张量大小 (ln1)
            self.act_sizes[2] = L * B * T;  // 第一次LayerNorm均值大小 (ln1_mean)
            self.act_sizes[3] = L * B * T;  // 第一次LayerNorm反向标准差大小 (ln1_rstd)
            self.act_sizes[4] = L * B * T * 3 * C;  // 查询、键、值 (qkv)
            self.act_sizes[5] = L * B * T * C;  // 注意力输出 (atty)
            self.act_sizes[6] = L * B * NH * T * T;  // 注意力预处理张量 (preatt)
            self.act_sizes[7] = L * B * NH * T * T;  // 注意力 (att)
            self.act_sizes[8] = L * B * T * C;  // 注意力投影 (attproj)
            self.act_sizes[9] = L * B * T * C;  // 第二个残差连接 (residual2)
            self.act_sizes[10] = L * B * T * C;  // 第二次LayerNorm输出张量大小 (ln2)
            self.act_sizes[11] = L * B * T;  // 第二次LayerNorm均值大小 (ln2_mean)
            self.act_sizes[12] = L * B * T;  // 第二次LayerNorm反向标准差大小 (ln2_rstd)
            self.act_sizes[13] = L * B * T * 4 * C;  // 全连接层中间层 (fch)
            self.act_sizes[14] = L * B * T * 4 * C;  // 全连接层GELU激活 (fch_gelu)
            self.act_sizes[15] = L * B * T * C;  // 全连接层投影 (fcproj)
            self.act_sizes[16] = L * B * T * C;  // 第三个残差连接 (residual3)
            self.act_sizes[17] = B * T * C;  // 最终LayerNorm输出 (lnf)
            self.act_sizes[18] = B * T;  // 最终LayerNorm均值 (lnf_mean)
            self.act_sizes[19] = B * T;  // 最终LayerNorm反向标准差 (lnf_rstd)
            self.act_sizes[20] = B * T * Vp;  // logits大小
            self.act_sizes[21] = B * T * Vp;  // 概率 (probs)
            self.act_sizes[22] = B * T;  // 损失 (losses)

            let num_activations: usize = self.act_sizes.iter().sum();
            println!("num_activations: {}", num_activations);
            self.num_activations = num_activations;

            unsafe {
                self.acts_memory = self.acts.alloc_and_point_activations(&self.act_sizes);

                // 创建用于缓存输入和目标的内存
                let input_layout = Layout::array::<i32>(B * T).expect("Failed to create layout");
                self.inputs = alloc(input_layout) as *mut i32;
                self.targets = alloc(input_layout) as *mut i32; // 如果我们从来没有目标但目标很小，则可能不会被使用
            }
        } else {
        // 验证 B,T 与我们之前分配内存的方式一致
        // 原则上我们将来可以在这里变得更聪明，目前这是最安全的
            if B != self.batch_size || T != self.seq_len {
                panic!("Model: B={} T={}, Desired: B={} T={}",self.batch_size, self.seq_len, B, T);
            }
        }

        // 缓存输入/目标
        unsafe {
            ptr::copy_nonoverlapping(inputs, self.inputs, B * T);
            if !targets.is_null() {
                ptr::copy_nonoverlapping(targets, self.targets, B * T);
            }
        }

        // 前向传播
        let params = &self.params;
        let acts = &mut self.acts;
        let mut residual: *mut f32;

        unsafe {
            encoder_forward(acts.encoded.ptr, inputs, params.wte.ptr, params.wpe.ptr, B, T, C);

            for l in 0..L {
                residual = if l == 0 {
                    acts.encoded.ptr
                } else {
                    acts.residual3.ptr.add((l - 1) * B * T * C)
                };

                // 获取该层的权重指针
                let l_ln1w = params.ln1w.ptr.add(l * C);
                let l_ln1b = params.ln1b.ptr.add(l * C);
                let l_qkvw = params.qkvw.ptr.add(l * 3 * C * C);
                let l_qkvb = params.qkvb.ptr.add(l * 3 * C);
                let l_attprojw = params.attprojw.ptr.add(l * C * C);
                let l_attprojb = params.attprojb.ptr.add(l * C);
                let l_ln2w = params.ln2w.ptr.add(l * C);
                let l_ln2b = params.ln2b.ptr.add(l * C);
                let l_fcw = params.fcw.ptr.add(l * 4 * C * C);
                let l_fcb = params.fcb.ptr.add(l * 4 * C);
                let l_fcprojw = params.fcprojw.ptr.add(l * C * 4 * C);
                let l_fcprojb = params.fcprojb.ptr.add(l * C);

                // 获取该层激活的指针
                let l_ln1 = acts.ln1.ptr.add(l * B * T * C);
                let l_ln1_mean = acts.ln1_mean.ptr.add(l * B * T);
                let l_ln1_rstd = acts.ln1_rstd.ptr.add(l * B * T);
                let l_qkv = acts.qkv.ptr.add(l * B * T * 3 * C);
                let l_atty = acts.atty.ptr.add(l * B * T * C);
                let l_preatt = acts.preatt.ptr.add(l * B * NH * T * T);
                let l_att = acts.att.ptr.add(l * B * NH * T * T);
                let l_attproj = acts.attproj.ptr.add(l * B * T * C);
                let l_residual2 = acts.residual2.ptr.add(l * B * T * C);
                let l_ln2 = acts.ln2.ptr.add(l * B * T * C);
                let l_ln2_mean = acts.ln2_mean.ptr.add(l * B * T);
                let l_ln2_rstd = acts.ln2_rstd.ptr.add(l * B * T);
                let l_fch = acts.fch.ptr.add(l * B * T * 4 * C);
                let l_fch_gelu = acts.fch_gelu.ptr.add(l * B * T * 4 * C);
                let l_fcproj = acts.fcproj.ptr.add(l * B * T * C);
                let l_residual3 = acts.residual3.ptr.add(l * B * T * C);

                // 现在进行前向传播
                layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C,);
                matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
                attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
                matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
                residual_forward(l_residual2, residual, l_attproj, B * T * C);
                layernorm_forward(l_ln2,l_ln2_mean,l_ln2_rstd,l_residual2,l_ln2w,l_ln2b,B,T,C,);
                matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
                gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
                matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
                residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
            }

            residual = acts.residual3.ptr.add((L - 1) * B * T * C); // 最后一个残差位于残差3中
            layernorm_forward(acts.lnf.ptr, acts.lnf_mean.ptr, acts.lnf_rstd.ptr, residual, params.lnfw.ptr, params.lnfb.ptr, B, T, C);
            matmul_forward(acts.logits.ptr, acts.lnf.ptr, params.wte.ptr, core::ptr::null_mut(), B, T, C, Vp);
            softmax_forward(acts.probs.ptr, acts.logits.ptr, B, T, V, Vp);

            // 如果我们有目标，还可以转发交叉熵损失函数
            if !targets.is_null() {
                crossentropy_forward(self.acts.losses.ptr, self.acts.probs.ptr, targets, B, T, Vp);
                // 为了方便起见，还评估平均损失
                let mut mean_loss = 0.0;
                for i in 0..(B * T) {
                    mean_loss += *self.acts.losses.ptr.add(i);
                }
                mean_loss /= (B * T) as f32;
                self.mean_loss = mean_loss;
            } else {
                // 如果我们没有目标，我们就没有损失
                self.mean_loss = -1.0;
            }
        }
    }
    
    // 内存归零函数
    unsafe fn zero_memory(memory_ptr: *mut f32, count: usize) {
        if !memory_ptr.is_null() {
            // 从指针创建一个切片，并将其内容设置为0
            let memory_slice = slice::from_raw_parts_mut(memory_ptr, count);
            ptr::write_bytes(memory_slice.as_mut_ptr(), 0, count);
        }
    }

    // 将模型中的所有梯度设置为零
    pub unsafe fn zero_grad(&mut self) {
        // 重置梯度内存
        Self::zero_memory(self.grads_memory, self.num_parameters);
        // 重置激活梯度内存
        Self::zero_memory(self.grads_acts_memory, self.num_activations);
    }

    // GPT2 模型的反向传播
    pub unsafe fn backward(&mut self) {
        // 仔细检查我们之前转发的目标
        if self.mean_loss == -1.0 {
            panic!("Error: must forward with targets before backward");
        }

        // 如果需要，为渐变延迟分配内存
        if self.grads_memory.is_null() {
            self.grads_memory = self.grads.malloc_and_point_parameters(&self.param_sizes);
            self.grads_acts_memory = self.grads_acts.alloc_and_point_activations(&self.act_sizes);
            self.zero_grad();
        }

        // 方便的快捷方式（usize 有助于防止 int 溢出）
        let B = self.batch_size;
        let T = self.seq_len;
        let V = self.config.vocab_size;
        let Vp = self.config.padded_vocab_size;
        let L = self.config.num_layers;
        let NH = self.config.num_heads;
        let C = self.config.channels;

        // 开始反向传播
        let params = &self.params;
        let grads = &mut self.grads;
        let acts = &self.acts;
        let grads_acts = &mut self.grads_acts;

        // 我们通过用 1.0f/(B*T) 填充 dlosses 来启动链式法则
        // 从技术上讲，这是一个小的、内联的 back() 计算过程
        // 总的最终损失作为批次中所有 (B,T) 位置的所有损失的平均值
        let dloss_mean = 1.0 / (B * T) as f32;
        for i in 0..(B * T) {
            *grads_acts.losses.ptr.add(i) = dloss_mean;
        }

        crossentropy_softmax_backward(grads_acts.logits.ptr, grads_acts.losses.ptr, acts.probs.ptr, self.targets, B, T, V, Vp); 
        matmul_backward(grads_acts.lnf.ptr, grads.wte.ptr, core::ptr::null_mut(), grads_acts.logits.ptr, acts.lnf.ptr, params.wte.ptr, B, T, C, Vp);
        let mut residual = acts.residual3.ptr.add((L - 1) * B * T * C); // 最后一层的残差
        let mut dresidual = grads_acts.residual3.ptr.add((L - 1) * B * T * C); // 写入最后一层的残差
        layernorm_backward(dresidual, grads.lnfw.ptr, grads.lnfb.ptr, grads_acts.lnf.ptr, residual, params.lnfw.ptr, acts.lnf_mean.ptr, acts.lnf_rstd.ptr, B, T, C);

        for l in (0..L).rev() {
            residual = if l == 0 {
                acts.encoded.ptr
            } else {
                acts.residual3.ptr.add((l - 1) * B * T * C)
            };

            dresidual = if l == 0 {
                grads_acts.encoded.ptr
            } else {
                grads_acts.residual3.ptr.add((l - 1) * B * T * C)
            };

            // 获取该层的权重指针
            let l_ln1w = params.ln1w.ptr.add(l * C);
            let l_qkvw = params.qkvw.ptr.add(l * 3 * C * C);
            let l_attprojw = params.attprojw.ptr.add(l * C * C);
            let l_ln2w = params.ln2w.ptr.add(l * C);
            let l_fcw = params.fcw.ptr.add(l * 4 * C * C);
            let l_fcprojw = params.fcprojw.ptr.add(l * C * 4 * C);

            // 获取该层权重梯度的指针
            let dl_ln1w = grads.ln1w.ptr.add(l * C);
            let dl_ln1b = grads.ln1b.ptr.add(l * C);
            let dl_qkvw = grads.qkvw.ptr.add(l * 3 * C * C);
            let dl_qkvb = grads.qkvb.ptr.add(l * 3 * C);
            let dl_attprojw = grads.attprojw.ptr.add(l * C * C);
            let dl_attprojb = grads.attprojb.ptr.add(l * C);
            let dl_ln2w = grads.ln2w.ptr.add(l * C);
            let dl_ln2b = grads.ln2b.ptr.add(l * C);
            let dl_fcw = grads.fcw.ptr.add(l * 4 * C * C);
            let dl_fcb = grads.fcb.ptr.add(l * 4 * C);
            let dl_fcprojw = grads.fcprojw.ptr.add(l * C * 4 * C);
            let dl_fcprojb = grads.fcprojb.ptr.add(l * C);

            // 获取该层激活的指针
            let l_ln1 = acts.ln1.ptr.add(l * B * T * C);
            let l_ln1_mean = acts.ln1_mean.ptr.add(l * B * T);
            let l_ln1_rstd = acts.ln1_rstd.ptr.add(l * B * T);
            let l_qkv = acts.qkv.ptr.add(l * B * T * 3 * C);
            let l_atty = acts.atty.ptr.add(l * B * T * C);
            let l_att = acts.att.ptr.add(l * B * NH * T * T);
            let l_residual2 = acts.residual2.ptr.add(l * B * T * C);
            let l_ln2 = acts.ln2.ptr.add(l * B * T * C);
            let l_ln2_mean = acts.ln2_mean.ptr.add(l * B * T);
            let l_ln2_rstd = acts.ln2_rstd.ptr.add(l * B * T);
            let l_fch = acts.fch.ptr.add(l * B * T * 4 * C);
            let l_fch_gelu = acts.fch_gelu.ptr.add(l * B * T * 4 * C);

            // 获取该层激活梯度的指针
            let dl_ln1 = grads_acts.ln1.ptr.add(l * B * T * C);
            let dl_qkv = grads_acts.qkv.ptr.add(l * B * T * 3 * C);
            let dl_atty = grads_acts.atty.ptr.add(l * B * T * C);
            let dl_preatt = grads_acts.preatt.ptr.add(l * B * NH * T * T);
            let dl_att = grads_acts.att.ptr.add(l * B * NH * T * T);
            let dl_attproj = grads_acts.attproj.ptr.add(l * B * T * C);
            let dl_residual2 = grads_acts.residual2.ptr.add(l * B * T * C);
            let dl_ln2 = grads_acts.ln2.ptr.add(l * B * T * C);
            let dl_fch = grads_acts.fch.ptr.add(l * B * T * 4 * C);
            let dl_fch_gelu = grads_acts.fch_gelu.ptr.add(l * B * T * 4 * C);
            let dl_fcproj = grads_acts.fcproj.ptr.add(l * B * T * C);
            let dl_residual3 = grads_acts.residual3.ptr.add(l * B * T * C);

            // 反向传播这一层
            residual_backward(dl_residual2, dl_fcproj, dl_residual3, B * T * C);
            matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4 * C, C);
            gelu_backward(dl_fch, l_fch, dl_fch_gelu, B * T * 4 * C);
            matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4 * C);
            layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
            residual_backward(dresidual, dl_attproj, dl_residual2, B * T * C);
            matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
            attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH,);
            matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3 * C);
            layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B,T, C,);
        }
        encoder_backward(grads.wte.ptr, grads.wpe.ptr, grads_acts.encoded.ptr, self.inputs, B, T, C);
    }

    // 使用 AdamW 优化更新 GPT2 模型参数。
    pub unsafe fn update(
        &mut self,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        t: usize,
    ) {
        // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        // 为 m_memory 和 v_memory 延迟分配内存
        if self.m_memory.is_null() {
            let m_layout = Layout::array::<f32>(self.num_parameters).unwrap();
            self.m_memory = alloc::alloc_zeroed(m_layout) as *mut f32;
        }
        if self.v_memory.is_null() {
            let v_layout = Layout::array::<f32>(self.num_parameters).unwrap();
            self.v_memory = alloc::alloc_zeroed(v_layout) as *mut f32;
        }

        // 使用 AdamW 迭代参数并更新
        for i in 0..self.num_parameters {
            let param = *self.params_memory.add(i);
            let grad = *self.grads_memory.add(i);

            // 更新第一时刻（动量）
            let m = beta1 * *self.m_memory.add(i) + (1.0 - beta1) * grad;
            // 更新二阶矩 (RMSprop)
            let v = beta2 * *self.v_memory.add(i) + (1.0 - beta2) * grad * grad;
            // 偏差校正两个时刻
            let m_hat = m / (1.0 - beta1.powi(t as i32));
            let v_hat = v / (1.0 - beta2.powi(t as i32));

            // 更新模型中的 m 和 v
            *self.m_memory.add(i) = m;
            *self.v_memory.add(i) = v;

            // 更新参数
            *self.params_memory.add(i) -=
                learning_rate * (m_hat / (v_hat.sqrt() + eps) + weight_decay * param);
        }
    }

    // 释放内存
    pub unsafe fn free(&mut self) {
        unsafe fn free_memory<T>(ptr: *mut T, num_elements: usize) {
            if !ptr.is_null() {
                let layout = Layout::array::<T>(num_elements)
                    .expect("error creating layout for free memory");
                alloc::dealloc(ptr as *mut u8, layout);
            }
        }

        // 释放模型参数的内存
        free_memory(self.params_memory, self.num_parameters);
        free_memory(self.grads_memory, self.num_parameters);
        free_memory(self.m_memory, self.num_parameters);
        free_memory(self.v_memory, self.num_parameters);

        // 为模型激活释放内存
        free_memory(self.acts_memory, self.num_activations);
        free_memory(self.grads_acts_memory, self.num_activations);

        // 为输入和目标释放内存
        free_memory(self.inputs, self.batch_size * self.seq_len);
        free_memory(self.targets, self.batch_size * self.seq_len);

        // 释放后将指针设置为 null
        self.params_memory = core::ptr::null_mut();
        self.grads_memory = core::ptr::null_mut();
        self.m_memory = core::ptr::null_mut();
        self.v_memory = core::ptr::null_mut();
        self.acts_memory = core::ptr::null_mut();
        self.grads_acts_memory = core::ptr::null_mut();
        self.inputs = core::ptr::null_mut();
        self.targets = core::ptr::null_mut();
    }
}