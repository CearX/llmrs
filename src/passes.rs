#![allow(dead_code)]
#![allow(mutable_transmutes)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_assignments)]
#![allow(unused_mut)]


use gemm::Parallelism;
use rayon::prelude::*;
use std::f32::consts::PI;
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
    
    // 并行遍历 B 和 T
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
    
// 并行化内层循环，确保对各个 C 元素的操作是独立的
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
    let eps: f32 = 1e-5f32;

    let out_atomic = AtomicPtr::new(out);
    let mean_atomic = AtomicPtr::new(mean);
    let rstd_atomic = AtomicPtr::new(rstd);
    let inp_atomic = AtomicPtr::new(inp);
    let weight_atomic = AtomicPtr::new(weight);
    let bias_atomic = AtomicPtr::new(bias);

// 使用 rayon 并行化外层循环
    (0..B).into_par_iter().for_each(|b| {
        (0..T).for_each(|t| {
            let inp_ptr = inp_atomic.load(Ordering::Relaxed)
                .offset((b * T * C) as isize)
                .offset((t * C) as isize);
            let mut m: f32 = 0.0f32;
            for i in 0..C {
                m += *inp_ptr.offset(i as isize);
            }
            m /= C as f32;

            let mut v: f32 = 0.0f32;
            for i in 0..C {
                let xshift: f32 = *inp_ptr.offset(i as isize) - m;
                v += xshift * xshift;
            }
            v /= C as f32;
            let s: f32 = 1.0f32 / (v + eps).sqrt();

            let out_ptr = out_atomic.load(Ordering::Relaxed)
                .offset((b * T * C) as isize)
                .offset((t * C) as isize);
            let weight_ptr = weight_atomic.load(Ordering::Relaxed);
            let bias_ptr = bias_atomic.load(Ordering::Relaxed);

            for i in 0..C {
                let n: f32 = s * (*inp_ptr.offset(i as isize) - m);
                let o: f32 = n * *weight_ptr.offset(i as isize)
                    + *bias_ptr.offset(i as isize);
                *out_ptr.offset(i as isize) = o;
            }

            *mean_atomic.load(Ordering::Relaxed).offset((b * T + t) as isize) = m;
            *rstd_atomic.load(Ordering::Relaxed).offset((b * T + t) as isize) = s;
        });
    });
}
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
                *dbias_raw.add(i) += *dout_bt.add(i);

                // 梯度对权重的贡献
                *dweight_raw.add(i) += norm_bti * *dout_bt.add(i);

                // 输入的梯度贡献
                let mut dval: f32 = 0.0;
                dval += dnorm_i; // 第 1 学期
                dval -= dnorm_mean; // 第 2 学期
                dval -= norm_bti * dnorm_norm_mean; // 第三学期
                dval *= rstd_bt; // 最终规模
                *dinp_bt.add(i) += dval;
            }
        });
    });
}


use std::sync::atomic::{AtomicPtr, Ordering};
const LOOP_UNROLL: usize = 8;

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
    (0..B).into_iter().for_each(|b|{
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

///计算耗时主要在matmul_forward和matmul_backward
/// dout：(B, T, OC)
/// dinp：(B, T, C)
/// weight：(OC, C)
/// dweight：(OC, C)
/// dbias：(OC)
/// inp：(B, T, C) 
/// 计算公式：
/// dinp = dout * weight.T
/// dweight = inp.T * dout
/// dbias = sum(dout, axis=0)
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
    // 计算 dbias = sum(dout, axis=0)
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
//计算 query_t，preatt_bth，att_bth 的偏移量
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
    let C3 = C * 3;//特征尺寸缩放3
    let hs = C / NH;// head size
    let scale = 1.0 / (hs as f32).sqrt();//求点积

    let dinp_atomic = AtomicPtr::new(dinp);
    let dpreatt_atomic = AtomicPtr::new(dpreatt);
    let datt_atomic = AtomicPtr::new(datt);
    let dout_atomic = AtomicPtr::new(dout);
    let inp_atomic = AtomicPtr::new(inp);
    let att_atomic = AtomicPtr::new(att);

    (0..B).into_par_iter().for_each(|b| {
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

//向后传递4：通过价值积累
                let dout_bth = dout_raw.add(b * T * C + t * C + h * hs);
                for t2 in 0..=t {
                    let value_t2 = inp_raw.add(b * T * C3 + t2 * C3 + h * hs + 2 * C);// +C*2因为它是值
                    let dvalue_t2 = dinp_raw.add(b * T * C3 + t2 * C3 + h * hs + 2 * C);// +C*2因为它是值
                    for i in 0..hs {
                        *datt_bth.add(t2) += *value_t2.add(i) * *dout_bth.add(i);
                        *dvalue_t2.add(i) += *att_bth.add(t2) * *dout_bth.add(i);
                    }
                }

//向后传递2和3：softmax
                for t2 in 0..=t {
                    for t3 in 0..=t {
                        let indicator = if t2 == t3 { 1.0 } else { 0.0 };
                        let local_derivative = *att_bth.add(t2) * (indicator - *att_bth.add(t3));
                        *dpreatt_bth.add(t3) += local_derivative * *datt_bth.add(t2);
                    }
                }

//反向传递1：查询@ key matmul
                for t2 in 0..=t {
                    let key_t2 = inp_raw.add(b * T * C3 + t2 * C3 + h * hs + C);// +C因为它是关键
                    let dkey_t2 = dinp_raw.add(b * T * C3 + t2 * C3 + h * hs + C);// +C因为它是关键
                    for i in 0..hs {
                        *dquery_t.add(i) += *key_t2.add(i) * *dpreatt_bth.add(t2) * scale;
                        *dkey_t2.add(i) += *query_t.add(i) * *dpreatt_bth.add(t2) * scale;
                    }
                }
            });
        });
    });
}

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
    let C3 = C * 3;//特征尺寸缩放3
    let hs = C / NH;// head size
    let scale = 1.0 / (hs as f32).sqrt();//求点积

    let dinp_atomic = AtomicPtr::new(dinp);
    let dpreatt_atomic = AtomicPtr::new(dpreatt);
    let datt_atomic = AtomicPtr::new(datt);
    let dout_atomic = AtomicPtr::new(dout);
    let inp_atomic = AtomicPtr::new(inp);
    let att_atomic = AtomicPtr::new(att);

//如果B * T不是LOOP_UNROLL的倍数，则回退到朴素实现
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

//向后传递4：通过价值积累
                    let dout_bth = dout_raw.add(b * T * C + t * C + h * hs);
                    for t2 in 0..=t {
                        let value_t2 = inp_raw.add(b * T * C3 + t2 * C3 + h * hs + 2 * C);// +C*2因为它是值
                        let dvalue_t2 = dinp_raw.add(b * T * C3 + t2 * C3 + h * hs + 2 * C);// +C*2因为它是值
                        for i in 0..hs {
                            *datt_bth.add(t2) += *value_t2.add(i) * *dout_bth.add(i);
                            *dvalue_t2.add(i) += *att_bth.add(t2) * *dout_bth.add(i);
                        }
                    }

//向后传递2和3：softmax
                    for t2 in 0..=t {
                        for t3 in 0..=t {
                            let indicator = if t2 == t3 { 1.0 } else { 0.0 };
                            let local_derivative =
                                *att_bth.add(t2) * (indicator - *att_bth.add(t3));
                            *dpreatt_bth.add(t3) += local_derivative * *datt_bth.add(t2);
                        }
                    }

//反向传递1：查询@ key matmul
                    for t2 in 0..=t {
                        let key_t2 = inp_raw.add(b * T * C3 + t2 * C3 + h * hs + C);// +C因为它是关键
                        let dkey_t2 = dinp_raw.add(b * T * C3 + t2 * C3 + h * hs + C);// +C因为它是关键
                        for i in 0..hs {
                            *dquery_t.add(i) += *key_t2.add(i) * *dpreatt_bth.add(t2) * scale;
                            *dkey_t2.add(i) += *query_t.add(i) * *dpreatt_bth.add(t2) * scale;
                        }
                    }
                }
            }
        });
}
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

        // 将梯度累加到dinp中
        *dinp_raw.add(i) += local_grad * dout_val;
    });
}

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