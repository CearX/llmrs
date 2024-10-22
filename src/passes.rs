#![allow(dead_code)]
#![allow(mutable_transmutes)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_assignments)]
#![allow(unused_mut)]
#[warn(unused_imports)]

use gemm::Parallelism;
use rayon::prelude::*;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicPtr, Ordering};
use crate::common::tensor;
const LOOP_UNROLL: usize = 8;

/// 执行编码器的前向传播操作，将输入的词索引与位置嵌入相加，生成输出矩阵。
/// 用来将单词的语义信息（wte）与其在序列中的位置信息（wpe）结合起来。
/// * out = wte + wpe
/// * wte 的整体形状是 (Vocab_size, C)词汇表大小 × 嵌入维度，wpe 的形状是 (Seq_length, C)最大序列长度 × 嵌入维度
///   在嵌入维度上进行累加
/// 
/// # 参数
/// * `out` (输出矩阵): 形状为 (B, T, C)，即 B 批次，T 时间步，每个时间步有 C 维嵌入向量。
/// * `inp` (输入矩阵): 形状为 (B, T)，每个元素是一个整数索引，表示某个词在词汇表中的位置。
/// * `wte` (词嵌入矩阵): 形状为 (V, C)，其中 V 是词汇表大小，每个词都有一个 C 维的嵌入向量。
/// * `wpe` (位置嵌入矩阵): 形状为 (T, C)，每个时间步都有一个 C 维的嵌入向量。
/// * `B` (批次大小): 输入数据的批次数量.
/// * `T` (时间步数): 每个批次中的时间步数量.
/// * `C` (嵌入维度): 每个嵌入向量的维度.
///
/// # 输出
/// 该函数不返回值，而是通过 `out` 指针输出形状为 (B, T, C) 的矩阵，表示经过词嵌入和位置嵌入后的结果。
pub unsafe fn encoder_forward(
    out: *mut f32,
    inp: *mut i32,
    wte: *mut f32,
    wpe: *mut f32,
    B: usize,
    T: usize,
    C: usize,
) {
    let out_atomic = AtomicPtr::new(out);
    let inp_atomic = AtomicPtr::new(inp);
    let wte_atomic = AtomicPtr::new(wte);
    let wpe_atomic = AtomicPtr::new(wpe);

    (0..B).into_par_iter().for_each(|b| { // 遍历每个批次
        (0..T).into_par_iter().for_each(|t| { // 遍历每个时间步
            let out_raw = out_atomic.load(Ordering::SeqCst);
            let inp_raw = inp_atomic.load(Ordering::SeqCst);
            let wte_raw = wte_atomic.load(Ordering::SeqCst);
            let wpe_raw = wpe_atomic.load(Ordering::SeqCst);

            let out_bt = out_raw.add(b * T * C + t * C);
            let ix = *inp_raw.add(b * T + t) as usize;
            let wte_ix = wte_raw.add(ix * C);
            let wpe_t = wpe_raw.add(t * C);

            for i in 0..C {
                *out_bt.add(i) = *wte_ix.add(i) + *wpe_t.add(i);
            }
        });
    });
}

/// 执行编码器的反向传播操作，计算词嵌入和位置嵌入的梯度。
/// * out = wte + wpe; 
/// * dwte = dout; 
/// * dwpe = dout;
/// 
/// # 参数
/// * `dwte`: 指向词嵌入梯度的指针，形状为 (V, C)。
/// * `dwpe`: 指向位置嵌入梯度的指针，形状为 (T, C)。
/// * `dout`: 指向输出梯度的指针，形状为 (B, T, C)。
/// * `inp`: 指向输入索引的指针，形状为 (B, T)。
/// * `B`: 批次大小，表示输入的样本数量。
/// * `T`: 时间步数，表示每个样本的时间步数量。
/// * `C`: 嵌入维度，表示每个词和位置的嵌入向量的维度。
pub unsafe fn encoder_backward(
    mut dwte: *mut f32,
    mut dwpe: *mut f32,
    mut dout: *mut f32,
    mut inp: *mut i32,
    B: usize,
    T: usize,
    C: usize,
) {
    let dwte_atomic = AtomicPtr::new(dwte);
    let dwpe_atomic = AtomicPtr::new(dwpe);
    let dout_atomic = AtomicPtr::new(dout);
    let inp_atomic = AtomicPtr::new(inp);
    
    // 并行遍历批次 B 和时间步 T
    (0..B).into_par_iter().for_each(|b| {
        (0..T).for_each(|t| {
            // 提取指针值，确保每个线程仅操作自己负责的内存区域
            let dout_base = dout_atomic.load(Ordering::Relaxed);
            let inp_base = inp_atomic.load(Ordering::Relaxed);
            let dwte_base = dwte_atomic.load(Ordering::Relaxed);
            let dwpe_base = dwpe_atomic.load(Ordering::Relaxed);
    
            // 计算当前线程负责的内存地址
            let dout_bt = unsafe { dout_base.add(b * T * C + t * C) };
            let inp_b_t = unsafe { inp_base.add(b * T + t) };
            let ix = unsafe { *inp_b_t };
            let dwte_ix = unsafe { dwte_base.add(ix as usize * C) };
            let dwpe_t = unsafe { dwpe_base.add(t * C) };
    
            // 并行化内层循环
            // out = wte + wpe
            // dwte = dout
            // dwpe = dout
            (0..C).for_each(|i| {
                unsafe {
                    let d = *dout_bt.add(i);
                    *dwte_ix.add(i) += d;
                    *dwpe_t.add(i) += d;
                }
            });
        });
    });
}
/// 执行 Layer Normalization 的前向传播计算，对输出数据进行归一化，并进行并行化处理。
/// 
/// # 计算公式：
/// * 均值计算： [ m = \frac{1}{C} \sum_{i=0}^{C-1} x_i ]
/// * 方差计算： [ v = \frac{1}{C} \sum_{i=0}^{C-1} (x_i - m)^2 ]
/// * 反标准差计算： [ s = \frac{1}{\sqrt{v + \epsilon}} ]
/// * 归一化： [ n = s \cdot (x_i - m) ]
/// * 应用权重和偏差： [ o = n \cdot w_i + b_i ]
/// * 这里的 ( x_i ) 是输入特征，( w_i ) 是权重，( b_i ) 是偏差，( \epsilon ) 是防止除零的微小正数。
///
/// # 参数
/// - `out`: *mut f32 - 输出数据指针，用于存储归一化后的结果。
/// - `mean`: *mut f32 - 用于存储每个样本的均值的指针。
/// - `rstd`: *mut f32 - 用于存储每个样本的反标准差（标准差的倒数）的指针。
/// - `inp`: *mut f32 - 输入数据指针，表示原始的未归一化输入数据。
/// - `weight`: *mut f32 - 权重，用于缩放归一化后的输出。
/// - `bias`: *mut f32 - 偏差，用于对归一化结果进行平移。
/// - `B`: usize - 批次大小（Batch size）。
/// - `T`: usize - 时间步长或序列长度。
/// - `C`: usize - 特征维度大小。
///
/// # 实现细节
/// - 使用 `rayon` 并行处理批次维度的循环 `(0..B)`，以提高计算效率。
/// - 对输入 `inp` 进行标准化：首先计算每个样本的均值和方差，然后根据这些值归一化输入数据。
/// - 使用给定的 `weight` 和 `bias` 对归一化后的结果进行缩放和平移。
/// - 将均值和反标准差存储到 `mean` 和 `rstd` 中，便于后续反向传播计算。
pub unsafe fn layernorm_forward(
    mut out: *mut f32,
    mut mean: *mut f32,
    mut rstd: *mut f32,
    mut inp: *mut f32,
    mut weight: *mut f32,
    mut bias: *mut f32,
    B: usize,
    T: usize,
    C: usize,
) {
    let eps: f32 = 1e-5f32; // 防止除零问题的小正则化项

    let out_atomic = AtomicPtr::new(out);
    let mean_atomic = AtomicPtr::new(mean);
    let rstd_atomic = AtomicPtr::new(rstd);
    let inp_atomic = AtomicPtr::new(inp);
    let weight_atomic = AtomicPtr::new(weight);
    let bias_atomic = AtomicPtr::new(bias);

    // 并行化批次维度的循环
    (0..B).into_par_iter().for_each(|b| {
        (0..T).for_each(|t| {
            let inp_ptr = inp_atomic.load(Ordering::Relaxed)
                .offset((b * T * C) as isize)
                .offset((t * C) as isize);
            
            // 计算均值 m
            let mut m: f32 = 0.0f32;
            for i in 0..C {
                m += *inp_ptr.offset(i as isize);
            }
            m /= C as f32;

            // 计算方差 v
            let mut v: f32 = 0.0f32;
            for i in 0..C {
                let xshift: f32 = *inp_ptr.offset(i as isize) - m;
                v += xshift * xshift;
            }
            v /= C as f32;
            
            // 计算标准差的倒数 s
            let s: f32 = 1.0f32 / (v + eps).sqrt();

            let out_ptr = out_atomic.load(Ordering::Relaxed)
                .offset((b * T * C) as isize)
                .offset((t * C) as isize);
            let weight_ptr = weight_atomic.load(Ordering::Relaxed);
            let bias_ptr = bias_atomic.load(Ordering::Relaxed);

            // 对每个特征维度归一化并应用 weight 和 bias
            for i in 0..C {
                let n: f32 = s * (*inp_ptr.offset(i as isize) - m); // 归一化
                let o: f32 = n * *weight_ptr.offset(i as isize)     // 应用 weight
                    + *bias_ptr.offset(i as isize);                // 应用 bias
                *out_ptr.offset(i as isize) = o;                    // 存储输出
            }

            // 存储均值和反标准差
            *mean_atomic.load(Ordering::Relaxed).offset((b * T + t) as isize) = m;
            *rstd_atomic.load(Ordering::Relaxed).offset((b * T + t) as isize) = s;
        });
    });
}
/// 执行层归一化的反向传播.
/// 
/// # 计算公式
/// * 以下sum均为[0, (C-1)]
/// * `dbias = sum(dout)`
/// * `dweight = sum((inp - mean) * dout)`
/// * dinp = (dnorm_i - dnorm_mean - norm_bti * dnorm_norm_mean) * rstd_bt
/// * dnorm_i = dout * weight
/// * dnorm_mean = sum(dnorm_i)
/// * dnorm_norm_mean = sum(dnorm_i * norm_bti)
/// * rstd_bt = 1 / sqrt(var + eps)
///
/// # 参数
/// - `dinp`: 输入梯度指针，计算后的输入梯度将存储在此。
/// - `dweight`: 权重梯度指针，计算后的权重梯度将存储在此。
/// - `dbias`: 偏差梯度指针，计算后的偏差梯度将存储在此。
/// - `dout`: 输出梯度指针，从前向传播的输出梯度获取。
/// - `inp`: 输入特征指针。
/// - `weight`: 权重指针。
/// - `mean`: 输入特征的均值指针。
/// - `rstd`: 输入特征的反标准差指针。
/// - `B`: 批量大小。
/// - `T`: 时间步数。
/// - `C`: 特征维度.
///
/// # 计算过程
/// - 在第一步中，计算两个中间值：`dnorm_mean` 和 `dnorm_norm_mean`，用于后续的梯度计算。
/// - 在第二步中，基于这两个中间值计算偏差、权重和输入的梯度，并累加到对应的张量中。
pub unsafe fn layernorm_backward(
    dinp: *mut f32,
    dweight: *mut f32,
    dbias: *mut f32,
    dout: *mut f32,
    inp: *mut f32,
    weight: *mut f32,
    mean: *mut f32,
    rstd: *mut f32,
    B: usize,
    T: usize,
    C: usize,
) {
    let dinp_atomic = AtomicPtr::new(dinp);
    let dweight_atomic = AtomicPtr::new(dweight);
    let dbias_atomic = AtomicPtr::new(dbias);
    let dout_atomic = AtomicPtr::new(dout);
    let inp_atomic = AtomicPtr::new(inp);
    let weight_atomic = AtomicPtr::new(weight);
    let mean_atomic = AtomicPtr::new(mean);
    let rstd_atomic = AtomicPtr::new(rstd);

    (0..B).into_par_iter().for_each(|b| {
        (0..T).into_par_iter().for_each(|t| {
            let dinp_raw = dinp_atomic.load(Ordering::SeqCst);
            let dweight_raw = dweight_atomic.load(Ordering::SeqCst);
            let dbias_raw = dbias_atomic.load(Ordering::SeqCst);
            let dout_raw = dout_atomic.load(Ordering::SeqCst);
            let inp_raw = inp_atomic.load(Ordering::SeqCst);
            let weight_raw = weight_atomic.load(Ordering::SeqCst);
            let mean_raw = mean_atomic.load(Ordering::SeqCst);
            let rstd_raw = rstd_atomic.load(Ordering::SeqCst);

            // 计算基地址
            let dout_bt = dout_raw.add(b * T * C + t * C);
            let inp_bt = inp_raw.add(b * T * C + t * C);
            let dinp_bt = dinp_raw.add(b * T * C + t * C);
            let mean_bt = *mean_raw.add(b * T + t);
            let rstd_bt = *rstd_raw.add(b * T + t);

            // 第一：两次reduce操作
            let mut dnorm_mean: f32 = 0.0;
            let mut dnorm_norm_mean: f32 = 0.0;
            for i in 0..C {
                let norm_bti = (*inp_bt.add(i) - mean_bt) * rstd_bt;
                let dnorm_i = *weight_raw.add(i) * *dout_bt.add(i);
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean /= C as f32;
            dnorm_norm_mean /= C as f32;

            // 现在再次迭代并累积所有梯度
            for i in 0..C {
                let norm_bti = (*inp_bt.add(i) - mean_bt) * rstd_bt;
                let dnorm_i = *weight_raw.add(i) * *dout_bt.add(i);

                // 梯度对偏差的贡献
                // * `dbias = sum(dout)`
                *dbias_raw.add(i) += *dout_bt.add(i);

                // * `dweight = sum(inp - mean) * dout`
                // 梯度对权重的贡献
                *dweight_raw.add(i) += norm_bti * *dout_bt.add(i);

                // 梯度对输入的贡献
                // * dinp = (dnorm_i - dnorm_mean - norm_bti * dnorm_norm_mean) * rstd_bt
                // dnorm_i = dout * weight
                // dnorm_mean = sum(dnorm_i)
                // dnorm_norm_mean = sum(dnorm_i * norm_bti)
                // rstd_bt = 1 / sqrt(var + eps)
                let mut dval: f32 = 0.0;
                dval += dnorm_i; 
                dval -= dnorm_mean; 
                dval -= norm_bti * dnorm_norm_mean; 
                dval *= rstd_bt; 
                *dinp_bt.add(i) += dval;
            }
        });
    });
}

/// 计算矩阵乘前向传播
/// 
/// # 矩阵形状
/// - 输入矩阵 `inp`: 形状为 [B, T, C] 
/// - 权重矩阵 `weight`: 形状为 [OC, C]
/// - 偏置矩阵 `bias`: 形状为 [OC]
/// - 输出矩阵 `out`: 形状为 [B, T, OC]
///
/// 计算耗时主要在matmul_forward和matmul_backward
pub unsafe fn matmul_forward(
    mut out: *mut f32,
    mut inp: *mut f32,
    mut weight: *mut f32,
    mut bias: *mut f32,
    mut B: usize,
    mut T: usize,
    mut C: usize,
    mut OC: usize,
) {
    // 初始化 out，将 bias 添加到 out
    if !bias.is_null() {
        for obt in 0..(B * T) {
            for o in 0..OC {
                *out.add(obt * OC + o) = *bias.add(o);
            }
        }
    } else {
        // 如果没有 bias，初始化 out 为 0
        for obt in 0..(B * T) {
            for o in 0..OC {
                *out.add(obt * OC + o) = 0.0;
            }
        }
    }
    (0..B).into_iter().for_each(|b| {
        // 调用 gemm 来进行矩阵乘法, dst := alpha×dst + beta×lhs×rhs
        gemm::gemm( 
            T,  // m
            OC, // n
            C,  // k
            out.add(b*T*OC),    // dst // [B * T, OC]
            1 as isize,  // dst_cs: 输出矩阵的列步长 //1
            OC as isize,      // dst_rs: 输出矩阵的行步长 //oc
            true,   // read_dst: 是否读取dst的初始值（这里为true，因为我们已将bias赋值给out）
            inp.add(b*T*C),    // lhs: 左乘矩阵 // [B * T, C]
            1 as isize,  // lhs_cs: 左乘矩阵的列步长 //1
            C as isize,      // lhs_rs: 左乘矩阵的行步长 //c
            weight, // rhs: 右乘矩阵 // [OC, C]
            C as isize,      // rhs_cs: 右乘矩阵的列步长 //c
            1 as isize,  // rhs_rs: 右乘矩阵的行步长 //1
            1.0,    // alpha
            1.0,    // beta
            false,  // conj_dst: 是否对 dst 做共轭转置
            false,  // conj_lhs: 是否对 lhs 做共轭转置
            false,  // conj_rhs: 是否对 rhs 做共轭转置
            Parallelism::Rayon(12),  // 并行化控制
        );
    });
}

/// 计算矩阵乘反向传播
/// * 计算耗时主要在matmul_forward和matmul_backward
/// # 参数
/// * dout：(B, T, OC)
/// * dinp：(B, T, C)
/// * weight：(OC, C)
/// * dweight：(OC, C)
/// * dbias：(OC)
/// * inp：(B, T, C) 
/// 
/// # 计算公式：
/// * dinp = dout * weight.T
/// * dw = dout.T * inp
/// * dbias = sum(dout, axis=0)
pub unsafe fn matmul_backward(
    mut dinp: *mut f32,
    mut dweight: *mut f32,
    mut dbias: *mut f32,
    mut dout: *mut f32,
    mut inp: *mut f32,
    mut weight: *mut f32,
    B: usize,
    T: usize,
    C: usize,
    OC: usize,
) {
        // dinp = dout * weight.T
        (0..B).into_iter().for_each(|b|{
            // 调用 gemm 来进行矩阵乘法, dst := alpha×dst + beta×lhs×rhs
            gemm::gemm( 
                T,  // m
                C, // n
                OC,  // k
                dinp.add(b*T*C),    // dst // dinp：(B, T, C)
                1 as isize,  // dst_cs: 输出矩阵的列步长 //1
                C as isize,      // dst_rs: 输出矩阵的行步长 //c
                false,   // read_dst: 是否读取dst的初始值
                dout.add(b*T*OC),    // lhs: 左乘矩阵 // /// dout：(B, T, OC)
                1 as isize,  // lhs_cs: 左乘矩阵的列步长 //1
                OC as isize,      // lhs_rs: 左乘矩阵的行步长 //oc
                weight, // rhs: 右乘矩阵 /// dweight：(OC, C)
                1 as isize,      // rhs_cs: 右乘矩阵的列步长 //
                C as isize,  // rhs_rs: 右乘矩阵的行步长 //
                0.0,    // alpha
                1.0,    // beta
                false,  // conj_dst: 是否对 dst 做共轭转置
                false,  // conj_lhs: 是否对 lhs 做共轭转置
                false,  // conj_rhs: 是否对 rhs 做共轭转置
                Parallelism::Rayon(12),  // 并行化控制
            );
        });

    // dweight：(OC, C)
    // inp：(B, T, C)
    // dout：(B, T, OC)
    
    // dw: [OC, C]
    // [OC, C] = [OC, B * T] * [B * T, C]
    // dw = dout.T * inp
        // 调用 gemm 来进行矩阵乘法, dst := alpha×dst + beta×lhs×rhs
        gemm::gemm( 
            OC,  // m
            C, // n
            B * T,  // k
            dweight,    // dst            // dw: [OC, C]
            1 as isize,  // dst_cs: 输出矩阵的列步长 
            C as isize,      // dst_rs: 输出矩阵的行步长 
            false,   // read_dst: 是否读取dst的初始值
            dout,    // lhs: 左乘矩阵            dout.T: [OC, B * T]
            OC as isize,  // lhs_cs: 左乘矩阵的列步长 
            1 as isize,      // lhs_rs: 左乘矩阵的行步长 
            inp, // rhs: 右乘矩阵                // inp：[B * T, C]
            1 as isize,      // rhs_cs: 右乘矩阵的列步长 
            C as isize,  // rhs_rs: 右乘矩阵的行步长 
            0.0,    // alpha
            1.0,    // beta
            false,  // conj_dst: 是否对 dst 做共轭转置
            false,  // conj_lhs: 是否对 lhs 做共轭转置
            false,  // conj_rhs: 是否对 rhs 做共轭转置
            Parallelism::Rayon(12),  // 并行化控制
        );
    // dbias = sum(dout, axis=0)
    // [1, OC] = [1, B * T] * [B * T, OC]
    //gemm: dw有微小误差，db有segmentation fault
    // // 定义 lhs 为全1的向量
    // let lhs: Vec<f32> = vec![1.0; B * T]; // 大小为 B * T
    // gemm::gemm( 
    //     1,  // m
    //     OC, // n
    //     B * T,  // k
    //     dbias,    // dst    [1, OC]
    //     1 as isize,  // dst_cs: 输出矩阵的列步长 
    //     OC as isize,      // dst_rs: 输出矩阵的行步长 
    //     false,   // read_dst: 是否读取dst的初始值
    //     lhs.as_ptr(),    // lhs: 左乘矩阵  // lhs 是全1向量，大小为 B*T    lhs[1, B * T]
    //     1 as isize,  // lhs_cs: 左乘矩阵的列步长 
    //     (B * T) as isize,      // lhs_rs: 左乘矩阵的行步长 
    //     dout, // rhs: 右乘矩阵                // dout[B * T, OC]
    //     1 as isize,      // rhs_cs: 右乘矩阵的列步长 
    //     OC as isize,  // rhs_rs: 右乘矩阵的行步长 
    //     0.0,    // alpha
    //     1.0,    // beta
    //     false,  // conj_dst: 是否对 dst 做共轭转置
    //     false,  // conj_lhs: 是否对 lhs 做共轭转置
    //     false,  // conj_rhs: 是否对 rhs 做共轭转置
    //     Parallelism::None,  // 并行化控制
    // );

    let dout_atomic = AtomicPtr::new(dout);
    let dbias_atomic = AtomicPtr::new(dbias);
    (0..OC).into_par_iter().for_each(|o| {
        let mut sum = 0.0;
        for b in 0..B {
            for t in 0..T {
                let dout_raw = dout_atomic.load(Ordering::SeqCst);
                let dout_bt = dout_raw.add(b * T * OC + t * OC);
                sum += *dout_bt.add(o);
            }
        }
        if !dbias_atomic.load(Ordering::SeqCst).is_null() {
            *dbias_atomic.load(Ordering::SeqCst).add(o) += sum;
        }
    });

}

/// 计算注意力机制的前向传播.
///
/// # 参数
/// - `out`: 输出的指针，形状为 (B, T, C)。
/// - `preatt`: 预注意力的指针，形状为 (B, NH, T)。
/// - `att`: 注意力权重的指针，形状为 (B, NH, T)。
/// - `inp`: 输入的指针，形状为 (B, T, C)。
/// - `B`: 批大小.
/// - `T`: 序列长度.
/// - `C`: 特征维度.
/// - `NH`: 注意力头数.
///
/// # 计算过程
/// 1. **预注意力计算**：
///    对于每个 `b` 和 `h`，计算：
///    ```math
///    \text{val} = \sum_{i=0}^{hs-1} \text{query}_t[i] \cdot \text{key}_{t2}[i]
///    ```
///    并记录最大值 `maxval`。
///
/// 2. **注意力权重计算**：
///    对于每个 `t2`，计算：
///    ```math
///    \text{att}_b^{h}[t2] = \exp(\text{preatt}_b^{h}[t2] - \text{maxval})
///    ```
///    归一化：
///    ```math
///    \text{att}_b^{h}[t2] = \frac{\text{att}_b^{h}[t2]}{\sum_{j=0}^{t} \text{att}_b^{h}[j]}
///    ```
///    
/// 3. **输出更新**：
///    对于每个 `t2`，计算：
///    ```math
///    \text{out}_b[t][h] += \text{att}_b^{h}[t2] \cdot \text{value}_{t2}
///    ```
///    其中，`value_{t2}` 是从输入中提取的特征。
/// 
/// # 代码实现细节：
/// * 预注意力计算：通过点积计算查询和键之间的相似度。
/// * Softmax 归一化：通过指数化和归一化，将预注意力值转换为概率分布，这些概率分布表示模型应该关注输入序列的哪些部分。
/// * 输出计算：使用注意力权重对值进行加权求和，以生成最终的输出。

pub unsafe fn attention_forward(
    mut out: *mut f32,
    mut preatt: *mut f32,
    mut att: *mut f32,
    mut inp: *mut f32,
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    // 使用 AtomicPtr 包装指针以保证线程安全
    let out_atomic = AtomicPtr::new(out);
    let preatt_atomic = AtomicPtr::new(preatt);
    let att_atomic = AtomicPtr::new(att);
    let inp_atomic = AtomicPtr::new(inp);
    
    let C3: usize = (C * 3) as usize;
    let hs: usize = (C / NH) as usize;
    let scale: f32 = (1.0 / (hs as f32).sqrt()) as f32;

    // 将 B 的循环并行化
    (0..B as usize).into_par_iter().for_each(|b| {
        // 获取 out, preatt, att 和 inp 的 AtomicPtr 值
        let out_ptr = out_atomic.load(Ordering::SeqCst);
        let preatt_ptr = preatt_atomic.load(Ordering::SeqCst);
        let att_ptr = att_atomic.load(Ordering::SeqCst);
        let inp_ptr = inp_atomic.load(Ordering::SeqCst);

        // 并行化 t 循环
        (0..T as usize).for_each(|t| {
            for h in 0..NH as usize {
                // 计算 query_t，preatt_bth，att_bth 的偏移量
                let query_t = inp_ptr
                    .offset((b * T as usize * C3) as isize)
                    .offset((t * C3) as isize)
                    .offset((h * hs) as isize);
                let preatt_bth = preatt_ptr
                    .offset((b * NH as usize * T as usize * T as usize) as isize)
                    .offset((h * T as usize * T as usize) as isize)
                    .offset((t * T as usize) as isize);
                let att_bth = att_ptr
                    .offset((b * NH as usize * T as usize * T as usize) as isize)
                    .offset((h * T as usize * T as usize) as isize)
                    .offset((t * T as usize) as isize);

                let mut maxval = -10000.0f32;
                
                // 计算 pre-attention 和最大值
                for t2 in 0..=t {
                    let key_t2 = inp_ptr
                        .offset((b * T as usize * C3) as isize)
                        .offset((t2 * C3) as isize)
                        .offset((h * hs) as isize)
                        .offset(C as isize);
                    let mut val = 0.0f32;
                    for i in 0..hs {
                        val += *query_t.offset(i as isize) * *key_t2.offset(i as isize);
                    }
                    val *= scale;
                    if val > maxval {
                        maxval = val;
                    }
                    *preatt_bth.offset(t2 as isize) = val;
                }

                // 计算注意力权重
                let mut expsum = 0.0f32;
                for t2 in 0..=t {
                    let expv = (*preatt_bth.offset(t2 as isize) - maxval).exp();
                    expsum += expv;
                    *att_bth.offset(t2 as isize) = expv;
                }
                let expsum_inv = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };
                
                // 归一化注意力权重
                for t2 in 0..T as usize {
                    if t2 <= t {
                        *att_bth.offset(t2 as isize) *= expsum_inv;
                    } else {
                        *att_bth.offset(t2 as isize) = 0.0f32;
                    }
                }

                // 更新输出
                let out_bth = out_ptr
                    .offset((b * T as usize * C as usize) as isize)
                    .offset((t * C as usize) as isize)
                    .offset((h * hs) as isize);
                for i in 0..hs {
                    *out_bth.offset(i as isize) = 0.0;
                }
                for t2 in 0..=t {
                    let value_t2 = inp_ptr
                        .offset((b * T as usize * C3) as isize)
                        .offset((t2 * C3) as isize)
                        .offset((h * hs) as isize)
                        .offset((C * 2) as isize);
                    let att_btht2 = *att_bth.offset(t2 as isize);
                    for i in 0..hs {
                        *out_bth.offset(i as isize) += att_btht2 * *value_t2.offset(i as isize);
                    }
                }
            }
        });
    });
}

/// 注意力机制的反向传播计算（naive实现）
///
/// # 参数
/// - `dinp`: 指向输入梯度的指针，形状为 (B, T, C)。
/// - `dpreatt`: 指向预注意力梯度的指针，形状为 (B, NH, T)。
/// - `datt`: 指向注意力权重梯度的指针，形状为 (B, NH, T)。
/// - `dout`: 指向输出梯度的指针，形状为 (B, T, C)。
/// - `inp`: 指向输入数据的指针，形状为 (B, T, C)。
/// - `att`: 指向注意力权重的指针，形状为 (B, NH, T)。
/// - `B`: 批大小（batch size）。
/// - `T`: 序列长度（time steps）。
/// - `C`: 特征维度（features）。
/// - `NH`: 注意力头数（number of attention heads）。
///
/// # 计算过程
/// 1. **梯度传播4: 通过 value 进行反向传播**  
///    对于每个 `t2` 和每个 `i`，计算：  
///    ```math
///    \frac{\partial \text{att}_b^{h}[t2]}{\partial \text{value}_{t2}} = \text{datt}_b^{h}[t2] + \text{value}_{t2} \cdot \text{dout}_b[t][h]
///    ```  
///    并更新 `datt_bth` 和 `dvalue_t2`。
///
/// 2. **梯度传播2 & 3: softmax 的反向传播**  
///    对于每个 `t2` 和 `t3`，计算 softmax 的梯度：  
///    ```math
///    \frac{\partial \text{softmax}}{\partial \text{preatt}} = \text{att}_b^{h}[t2] \cdot (\delta_{t2t3} - \text{att}_b^{h}[t3])
///    ```  
///    更新 `dpreatt_bth`。
///
/// 3. **梯度传播1: query 和 key 的矩阵乘法反向传播**  
///    对于每个 `t2` 和每个 `i`，计算：  
///    ```math
///    \frac{\partial \text{query}_t[i]}{\partial \text{key}_{t2}[i]} = \text{query}_t[i] \cdot \text{dpreatt}_b^{h}[t2] \cdot \text{scale}
///    ```  
///    同时更新 `dquery_t` 和 `dkey_t2`。
///
pub unsafe fn attention_backward_naive(
    dinp: *mut f32,
    dpreatt: *mut f32,
    datt: *mut f32,
    dout: *mut f32,
    inp: *mut f32,
    att: *mut f32,
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    let C3 = C * 3; // 特征维度缩放3
    let hs = C / NH; // 每个头的尺寸 (head size)
    let scale = 1.0 / (hs as f32).sqrt(); // 缩放因子，1/√hs

    // 用 AtomicPtr 包装指针以实现线程安全
    let dinp_atomic = AtomicPtr::new(dinp);
    let dpreatt_atomic = AtomicPtr::new(dpreatt);
    let datt_atomic = AtomicPtr::new(datt);
    let dout_atomic = AtomicPtr::new(dout);
    let inp_atomic = AtomicPtr::new(inp);
    let att_atomic = AtomicPtr::new(att);

    // 并行化批量计算
    (0..B).into_par_iter().for_each(|b| {
        // 并行化时间步长计算
        (0..T).into_par_iter().for_each(|t| {
            (0..NH).into_par_iter().for_each(|h| {
                let dinp_raw = dinp_atomic.load(Ordering::SeqCst);
                let dpreatt_raw = dpreatt_atomic.load(Ordering::SeqCst);
                let datt_raw = datt_atomic.load(Ordering::SeqCst);
                let dout_raw = dout_atomic.load(Ordering::SeqCst);
                let inp_raw = inp_atomic.load(Ordering::SeqCst);
                let att_raw = att_atomic.load(Ordering::SeqCst);

                let att_bth = att_raw.add(b * NH * T * T + h * T * T + t * T);
                let datt_bth = datt_raw.add(b * NH * T * T + h * T * T + t * T);
                let dpreatt_bth = dpreatt_raw.add(b * NH * T * T + h * T * T + t * T);
                let dquery_t = dinp_raw.add(b * T * C3 + t * C3 + h * hs);
                let query_t = inp_raw.add(b * T * C3 + t * C3 + h * hs);

                // 向后传递4：通过 value 反向传播
                let dout_bth = dout_raw.add(b * T * C + t * C + h * hs);
                for t2 in 0..=t {
                    let value_t2 = inp_raw.add(b * T * C3 + t2 * C3 + h * hs + 2 * C); // +C*2 因为它是 value
                    let dvalue_t2 = dinp_raw.add(b * T * C3 + t2 * C3 + h * hs + 2 * C); // +C*2 因为它是 value
                    for i in 0..hs {
                        *datt_bth.add(t2) += *value_t2.add(i) * *dout_bth.add(i);
                        *dvalue_t2.add(i) += *att_bth.add(t2) * *dout_bth.add(i);
                    }
                }

                // 向后传递2 & 3：softmax 的反向传播
                for t2 in 0..=t {
                    for t3 in 0..=t {
                        let indicator = if t2 == t3 { 1.0 } else { 0.0 };
                        let local_derivative = *att_bth.add(t2) * (indicator - *att_bth.add(t3));
                        *dpreatt_bth.add(t3) += local_derivative * *datt_bth.add(t2);
                    }
                }

                // 向后传递1：查询与 key 的矩阵乘法反向传播
                for t2 in 0..=t {
                    let key_t2 = inp_raw.add(b * T * C3 + t2 * C3 + h * hs + C); // +C 因为它是 key
                    let dkey_t2 = dinp_raw.add(b * T * C3 + t2 * C3 + h * hs + C); // +C 因为它是 key
                    for i in 0..hs {
                        *dquery_t.add(i) += *key_t2.add(i) * *dpreatt_bth.add(t2) * scale;
                        *dkey_t2.add(i) += *query_t.add(i) * *dpreatt_bth.add(t2) * scale;
                    }
                }
            });
        });
    });
}

/// 执行注意力机制的反向传播.
///
/// # 参数
/// - `dinp`: 输入的梯度指针，形状为 (B, T, C)。
/// - `dpreatt`: 预注意力的梯度指针，形状为 (B, NH, T, T)。
/// - `datt`: 注意力权重的梯度指针，形状为 (B, NH, T, T)。
/// - `dout`: 输出的梯度指针，形状为 (B, T, C)。
/// - `inp`: 输入的指针，形状为 (B, T, C)。
/// - `att`: 注意力权重的指针，形状为 (B, NH, T, T)。
/// - `B`: 批大小。
/// - `T`: 序列长度。
/// - `C`: 特征维度。
/// - `NH`: 注意力头数.
///
/// # 计算过程
/// 1. **反向传递通过价值积累**：
///    对于每个 `b` 和 `h`，更新 `datt` 和 `dvalue`：
///    ```math
///    \frac{\partial \text{att}_{b}^{h}[t2]}{\partial \text{dout}_{b}[t]} = \sum_{i=0}^{hs-1} \text{value}_{t2} \cdot \text{dout}_{b}[i]
///    ```
///    
/// 2. **Softmax 的反向传播**：
///    计算局部导数，并更新 `dpreatt`：
///    ```math
///    \frac{\partial \text{dpreatt}_{b}^{h}[t3]}{\partial \text{att}_{b}^{h}[t2]} = \text{att}_{b}^{h}[t2] \cdot (\delta_{t2, t3} - \text{att}_{b}^{h}[t3])
///    ```
///    
/// 3. **查询和键的矩阵乘法反向传播**：
///    更新 `dquery` 和 `dkey`：
///    ```math
///    \frac{\partial \text{dquery}_{b}[t]}{\partial \text{dpreatt}_{b}^{h}[t2]} = \sum_{i=0}^{hs-1} \text{key}_{t2} \cdot \text{dpreatt}_{b}^{h}[t2] \cdot \text{scale}
///    ```

pub unsafe fn attention_backward(
    mut dinp: *mut f32,
    mut dpreatt: *mut f32,
    mut datt: *mut f32,
    mut dout: *mut f32,
    mut inp: *mut f32,
    mut att: *mut f32,
    B: usize,
    T: usize,
    C: usize,
    NH: usize,
) {
    let C3 = C * 3; // 特征尺寸缩放3
    let hs = C / NH; // head size
    let scale = 1.0 / (hs as f32).sqrt(); // 求点积

    let dinp_atomic = AtomicPtr::new(dinp);
    let dpreatt_atomic = AtomicPtr::new(dpreatt);
    let datt_atomic = AtomicPtr::new(datt);
    let dout_atomic = AtomicPtr::new(dout);
    let inp_atomic = AtomicPtr::new(inp);
    let att_atomic = AtomicPtr::new(att);

    // 如果 B * T 不是 LOOP_UNROLL 的倍数，则回退到朴素实现
    if (B * T) % LOOP_UNROLL != 0 {
        attention_backward_naive(dinp, dpreatt, datt, dout, inp, att, B, T, C, NH);
        return;
    }

    (0..B * T)
        .into_par_iter()
        .step_by(LOOP_UNROLL)
        .for_each(|obt| {
            let dinp_raw = dinp_atomic.load(Ordering::SeqCst);
            let dpreatt_raw = dpreatt_atomic.load(Ordering::SeqCst);
            let datt_raw = datt_atomic.load(Ordering::SeqCst);
            let dout_raw = dout_atomic.load(Ordering::SeqCst);
            let inp_raw = inp_atomic.load(Ordering::SeqCst);
            let att_raw = att_atomic.load(Ordering::SeqCst);

            for ibt in 0..LOOP_UNROLL {
                let bt = obt + ibt;
                let b = bt / T;
                let t = bt % T;

                for h in 0..NH {
                    let att_bth = att_raw.add(b * NH * T * T + h * T * T + t * T);
                    let datt_bth = datt_raw.add(b * NH * T * T + h * T * T + t * T);
                    let dpreatt_bth = dpreatt_raw.add(b * NH * T * T + h * T * T + t * T);
                    let dquery_t = dinp_raw.add(b * T * C3 + t * C3 + h * hs);
                    let query_t = inp_raw.add(b * T * C3 + t * C3 + h * hs);

                    // 向后传递4：通过价值积累
                    let dout_bth = dout_raw.add(b * T * C + t * C + h * hs);
                    for t2 in 0..=t {
                        let value_t2 = inp_raw.add(b * T * C3 + t2 * C3 + h * hs + 2 * C); // +C*2因为它是值
                        let dvalue_t2 = dinp_raw.add(b * T * C3 + t2 * C3 + h * hs + 2 * C); // +C*2因为它是值
                        for i in 0..hs {
                            *datt_bth.add(t2) += *value_t2.add(i) * *dout_bth.add(i);
                            *dvalue_t2.add(i) += *att_bth.add(t2) * *dout_bth.add(i);
                        }
                    }

                    // 向后传递2和3：softmax
                    for t2 in 0..=t {
                        for t3 in 0..=t {
                            let indicator = if t2 == t3 { 1.0 } else { 0.0 };
                            let local_derivative =
                                *att_bth.add(t2) * (indicator - *att_bth.add(t3));
                            *dpreatt_bth.add(t3) += local_derivative * *datt_bth.add(t2);
                        }
                    }

                    // 反向传递1：查询 @ key matmul
                    for t2 in 0..=t {
                        let key_t2 = inp_raw.add(b * T * C3 + t2 * C3 + h * hs + C); // +C因为它是关键
                        let dkey_t2 = dinp_raw.add(b * T * C3 + t2 * C3 + h * hs + C); // +C因为它是关键
                        for i in 0..hs {
                            *dquery_t.add(i) += *key_t2.add(i) * *dpreatt_bth.add(t2) * scale;
                            *dkey_t2.add(i) += *query_t.add(i) * *dpreatt_bth.add(t2) * scale;
                        }
                    }
                }
            }
        });
}

/// 计算输入张量的高斯误差线性单元（GeLU）前向传播。
///
/// # 参数
/// - `out`: 指向输出张量的指针，形状为 (N)。
/// - `inp`: 指向输入张量的指针，形状为 (N)。
/// - `N`: 输入和输出张量的元素数量。
///
/// # 计算过程
/// 对于每个输入元素 `x`，计算其 GeLU 值：
///
/// ```math
/// \text{GeLU}(x) = 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)\right)\right)
/// ```
///
/// 该公式通过以下步骤计算：
/// 1. 计算立方项：
///    ```math
///    \text{cube} = 0.044715 \cdot x^3
///    ```
/// 2. 应用 tanh 函数：
///    ```math
///    \text{tanh\_input} = \sqrt{\frac{2}{\pi}} \cdot (x + \text{cube})
///    ```
/// 3. 最终 GeLU 输出：
///    ```math
///    \text{out} = 0.5 \cdot x \cdot (1 + \text{tanh\_input})
///    ```
///
pub unsafe fn gelu_forward(out: *mut f32, inp: *mut f32, N: usize) {
    let out_atomic = AtomicPtr::new(out);
    let inp_atomic = AtomicPtr::new(inp);

    (0..N).into_par_iter().for_each(|i| {
        let out_raw = out_atomic.load(Ordering::SeqCst);
        let inp_raw = inp_atomic.load(Ordering::SeqCst);

        // 加载输入值
        let x = *inp_raw.add(i);
        // 计算立方项
        let cube = 0.044715 * x * x * x;
        // 应用 GeLU 函数
        *out_raw.add(i) = 0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + cube)).tanh());
    });
}

/// 计算输入张量的高斯误差线性单元（GeLU）反向传播。
///
/// # 参数
/// - `dinp`: 指向输入梯度的指针，形状为 (N)。
/// - `inp`: 指向输入张量的指针，形状为 (N)。
/// - `dout`: 指向输出梯度的指针，形状为 (N)。
/// - `N`: 输入、输出张量的元素数量。
///
/// # 计算过程
/// 对于每个输入元素 `x`，计算其 GeLU 反向传播的梯度：
///
/// 1. **计算三次项**：  
///    ```math
///    \text{cube} = 0.044715 \cdot x^3
///    ```
///  
/// 2. **计算 tanh 函数的参数和输出**：  
///    ```math
///    \text{tanh\_arg} = \sqrt{\frac{2}{\pi}} \cdot (x + \text{cube})  
///    \text{tanh\_out} = \tanh(\text{tanh\_arg})
///    ```
///  
/// 3. **计算双曲余弦和 sech**：  
///    ```math
///    \text{coshf\_out} = \cosh(\text{tanh\_arg})  
///    \text{sech\_out} = \frac{1}{\text{coshf\_out}^2}
///    ```
///  
/// 4. **计算局部梯度**：  
///    ```math
///    \text{local\_grad} = 0.5 \cdot (1 + \text{tanh\_out}) + x \cdot 0.5 \cdot \text{sech\_out} \cdot \sqrt{\frac{2}{\pi}} \cdot (1 + 3 \cdot 0.044715 \cdot x^2)
///    ```
///  
/// 5. **累加梯度到 dinp**：  
///    ```math
///    \text{dinp}[i] += \text{local\_grad} \cdot \text{dout}[i]
///    ```
///
pub unsafe fn gelu_backward(dinp: *mut f32, inp: *mut f32, dout: *mut f32, N: usize) {
    let gelu_scaling_factor = (2.0 / PI).sqrt();

    let dinp_atomic = AtomicPtr::new(dinp);
    let inp_atomic = AtomicPtr::new(inp);
    let dout_atomic = AtomicPtr::new(dout);

    (0..N).into_par_iter().for_each(|i| {
        let dinp_raw = dinp_atomic.load(Ordering::SeqCst);
        let inp_raw = inp_atomic.load(Ordering::SeqCst);
        let dout_raw = dout_atomic.load(Ordering::SeqCst);

        // 加载输入值
        let x = *inp_raw.add(i);
        let dout_val = *dout_raw.add(i);

        // 计算三次项
        let cube = 0.044715 * x * x * x;

        // 计算 tanh 函数的参数和输出
        let tanh_arg = gelu_scaling_factor * (x + cube);
        let tanh_out = tanh_arg.tanh();

        // 计算双曲余弦和 sech（双曲正割）
        let coshf_out = tanh_arg.cosh();
        let sech_out = 1.0 / (coshf_out * coshf_out);

        // 计算局部梯度
        let local_grad = 0.5 * (1.0 + tanh_out)
            + x * 0.5 * sech_out * gelu_scaling_factor * (1.0 + 3.0 * 0.044715 * x * x);

        // 将梯度累加到 dinp 中
        *dinp_raw.add(i) += local_grad * dout_val;
    });
}

/// 计算残差连接的前向传播，将两个输入张量逐元素相加。
///
/// # 参数
/// - `out`: 指向输出张量的指针，形状为 (N)。
/// - `inp1`: 指向第一个输入张量的指针，形状为 (N)。
/// - `inp2`: 指向第二个输入张量的指针，形状为 (N)。
/// - `N`: 输入和输出张量的元素数量。
///
/// # 计算过程
/// 对于每个输入元素，计算其残差值：
///
/// ```math
/// \text{out}[i] = \text{inp1}[i] + \text{inp2}[i]
/// ```
///
/// 其中 `i` 为元素的索引，表示在数组中的位置。
///
pub unsafe fn residual_forward(out: *mut f32, inp1: *mut f32, inp2: *mut f32, N: usize) {
    let out_atomic = AtomicPtr::new(out);
    let inp1_atomic = AtomicPtr::new(inp1);
    let inp2_atomic = AtomicPtr::new(inp2);

    (0..N).into_par_iter().for_each(|i| {
        let out_raw = out_atomic.load(Ordering::SeqCst);
        let inp1_raw = inp1_atomic.load(Ordering::SeqCst);
        let inp2_raw = inp2_atomic.load(Ordering::SeqCst);

        // 执行逐元素加法
        *out_raw.add(i) = *inp1_raw.add(i) + *inp2_raw.add(i);
    });
}

/// 计算残差连接的反向传播，更新输入梯度。
///
/// # 参数
/// - `dinp1`: 指向第一个输入梯度的指针，形状为 (N)。
/// - `dinp2`: 指向第二个输入梯度的指针，形状为 (N)。
/// - `dout`: 指向输出梯度的指针，形状为 (N)。
/// - `N`: 输入和输出张量的元素数量.
///
/// # 计算过程
/// 对于每个输入元素，计算其梯度更新：
///
/// ```math
/// \text{dinp1}[i] += \text{dout}[i]
/// \text{dinp2}[i] += \text{dout}[i]
/// ```
///
/// 其中 `i` 为元素的索引，表示在数组中的位置。
///
pub unsafe fn residual_backward(dinp1: *mut f32, dinp2: *mut f32, dout: *mut f32, N: usize) {
    let dinp1_atomic = AtomicPtr::new(dinp1);
    let dinp2_atomic = AtomicPtr::new(dinp2);
    let dout_atomic = AtomicPtr::new(dout);

    (0..N).into_par_iter().for_each(|i| {
        let dinp1_raw = dinp1_atomic.load(Ordering::SeqCst);
        let dinp2_raw = dinp2_atomic.load(Ordering::SeqCst);
        let dout_raw = dout_atomic.load(Ordering::SeqCst);

        // 更新输入的梯度
        *dinp1_raw.add(i) += *dout_raw.add(i);
        *dinp2_raw.add(i) += *dout_raw.add(i);
    });
}

/// 计算输入张量的 Softmax 前向传播，输出归一化的概率。
///
/// # 参数
/// - `probs`: 指向输出概率张量的指针，形状为 (B, T, Vp)，其中 B 为批次大小，T 为序列长度，V 为词汇表大小，Vp 为填充后的词汇表大小。
/// - `logits`: 指向输入 logits 张量的指针，形状为 (B, T, Vp)。
/// - `B`: 批次大小。
/// - `T`: 序列长度。
/// - `V`: 词汇表大小。
/// - `Vp`: 填充后的词汇表大小（≥ V）。
///
/// # 计算过程
/// 对于每个批次 `b` 和每个时间步 `t`，计算 softmax：
///
/// 1. **计算数值稳定性的最大值**：
///    ```math
///    \text{maxval} = \max_{i \in [0, V)} \text{logits}[b, t, i]
///    ```
///
/// 2. **计算 softmax 分子和分母（和）**：
///    ```math
///    \text{exp\_val}[i] = \exp(\text{logits}[b, t, i] - \text{maxval})
///    ```
///    ```math
///    \text{sum} = \sum_{i=0}^{V} \text{exp\_val}[i]
///    ```
///
/// 3. **计算归一化的概率**：
///    ```math
///    \text{probs}[b, t, i] = \frac{\text{exp\_val}[i]}{\text{sum}}
///    ```
///
/// 4. **对填充部分进行处理**：对于 `i ∈ [V, Vp)`，将对应的概率设为 0：
///    ```math
///    \text{probs}[b, t, i] = 0, \quad \text{for } i \geq V
///    ```
pub unsafe fn softmax_forward(
    probs: *mut f32,
    logits: *mut f32,
    B: usize,
    T: usize,
    V: usize,
    Vp: usize,
) {
    let probs_atomic = AtomicPtr::new(probs);
    let logits_atomic = AtomicPtr::new(logits);

    (0..B).into_par_iter().for_each(|b| {
        (0..T).into_par_iter().for_each(|t| {
            // 将 AtomicPtr 值加载到当前范围的原始指针中
            let probs_raw = probs_atomic.load(Ordering::SeqCst);
            let logits_raw = logits_atomic.load(Ordering::SeqCst);

            // 计算基地址
            let logits_bt = logits_raw.add(b * T * Vp + t * Vp);
            let probs_bt = probs_raw.add(b * T * Vp + t * Vp);

            // 计算数值稳定性的 maxval
            let mut maxval = f32::NEG_INFINITY;
            for i in 0..V {
                let logit = *logits_bt.add(i);
                if logit > maxval {
                    maxval = logit;
                }
            }

            // 计算softmax分子和分母（和）
            let mut sum = 0.0;
            for i in 0..V {
                let exp_val = (logits_bt.add(i).read() - maxval).exp();
                probs_bt.add(i).write(exp_val);
                sum += exp_val;
            }

            // 概率标准化
            for i in 0..V {
                probs_bt.add(i).write(probs_bt.add(i).read() / sum);
            }

            // 将填充尺寸设置为零
            for i in V..Vp {
                probs_bt.add(i).write(0.0);
            }
        });
    });
}

/// 计算交叉熵损失的前向传递
///
/// # 参数
/// - `losses`: 指向存储损失值的指针，大小为 `B * T`
/// - `probs`: 指向存储预测概率的指针，大小为 `B * T * Vp`
/// - `targets`: 指向存储目标标签的指针，大小为 `B * T`
/// - `B`: 批次大小 (batch size)
/// - `T`: 时间步长 (time steps)
/// - `Vp`: 每个时间步的类别数量 (vocabulary size)
///
/// # 计算公式
/// 对于每个批次 `b` 和每个时间步 `t`，交叉熵损失的计算公式为：
///
/// \[ \text{loss}(b, t) = -\log(p_{bt}(y_{bt})) \]
///
/// 其中：
///
/// - `p_{bt}(y_{bt})` 是预测概率 `probs` 中的目标类别 `y_{bt}` 对应的概率值
/// - `y_{bt}` 是目标标签 `targets` 中批次 `b` 和时间步 `t` 对应的类别索引
///
/// 损失值存储在 `losses` 数组中。
pub unsafe fn crossentropy_forward(
    losses: *mut f32,
    probs: *mut f32,
    targets: *const i32,
    B: usize,
    T: usize,
    Vp: usize,
) {
    let losses_atomic = AtomicPtr::new(losses);
    let probs_atomic = AtomicPtr::new(probs);
    let targets_atomic = AtomicPtr::new(targets as *mut i32);

    (0..B).into_par_iter().for_each(|b| {
        (0..T).into_par_iter().for_each(|t| {
            let losses_raw = losses_atomic.load(Ordering::SeqCst);
            let probs_raw = probs_atomic.load(Ordering::SeqCst);
            let targets_raw = targets_atomic.load(Ordering::SeqCst);

            // 计算probs的基地址
            let probs_bt = probs_raw.add(b * T * Vp + t * Vp);

            // 获取目标索引
            let ix = *targets_raw.add(b * T + t) as usize;

            // 计算交叉熵损失并存储它
            *losses_raw.add(b * T + t) = -probs_bt.add(ix).read().ln();
        });
    });
}
/// 计算交叉熵损失的反向传播
///
/// # 参数
/// - `dlogits`: 指向存储梯度的指针，大小为 `B * T * Vp`
/// - `dlosses`: 指向存储损失梯度的指针，大小为 `B * T`
/// - `probs`: 指向存储预测概率的指针，大小为 `B * T * Vp`
/// - `targets`: 指向存储目标标签的指针，大小为 `B * T`
/// - `B`: 批次大小 (batch size)
/// - `T`: 时间步长 (time steps)
/// - `V`: 类别数量 (number of classes)
/// - `Vp`: 每个时间步的类别数量 (vocabulary size)
///
/// # 计算公式
/// 对于每个批次 `b` 和每个时间步 `t`，反向传播的计算公式为：
///
/// \[ \frac{\partial L}{\partial z_{bt}} = p_{bt} - \delta(y_{bt}) \]
///
/// 其中：
/// - `p_{bt}` 是预测概率 `probs` 中的第 `b` 批次和第 `t` 时间步对应的概率值
/// - \(\delta(y_{bt})\) 是目标标签的指示函数，
///   \[ \delta(y_{bt}) = \begin{cases} 
///   1 & \text{if } i = y_{bt} \\
///   0 & \text{otherwise} 
///   \end{cases} \]
///
/// 梯度存储在 `dlogits` 数组中。
pub unsafe fn crossentropy_softmax_backward(
    dlogits: *mut f32,
    dlosses: *mut f32,
    probs: *mut f32,
    targets: *const i32,
    B: usize,
    T: usize,
    V: usize,
    Vp: usize,
) {
    let dlogits_atomic = AtomicPtr::new(dlogits);
    let dlosses_atomic = AtomicPtr::new(dlosses);
    let probs_atomic = AtomicPtr::new(probs);
    let targets_atomic = AtomicPtr::new(targets as *mut i32);

    (0..B).into_par_iter().for_each(|b| {
        (0..T).into_par_iter().for_each(|t| {
            let dlogits_raw = dlogits_atomic.load(Ordering::SeqCst);
            let dlosses_raw = dlosses_atomic.load(Ordering::SeqCst);
            let probs_raw = probs_atomic.load(Ordering::SeqCst);
            let targets_raw = targets_atomic.load(Ordering::SeqCst);

            // 计算基地址
            let dlogits_bt = dlogits_raw.add(b * T * Vp + t * Vp);
            let probs_bt = probs_raw.add(b * T * Vp + t * Vp);
            let dloss = *dlosses_raw.add(b * T + t);
            let ix = *targets_raw.add(b * T + t) as usize;

            // 仅循环到 V，保持填充尺寸不变
            for i in 0..V {
                let p = *probs_bt.add(i);
                let indicator = if i == ix { 1.0 } else { 0.0 };
                *dlogits_bt.add(i) += (p - indicator) * dloss;
            }
        });
    });
}