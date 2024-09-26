// use std::sync::atomic::AtomicPtr;

// use operators::common_cpu::Cpu;
// use operators::{mat_mul::common_cpu::Operator as RefOp, Operator};


// #[cfg(test)]
// mod test {
//     use super::*;
//     use operators::mat_mul::Args;
//     use operators::common::TensorLayout;
//     use operators::mat_mul::DigitLayout;
//     use operators::mat_mul::Hardware;
//     const ALPHA: f32 = 0.5;
//     const BETA: f32 = 1.;

//     fn args<H: Hardware>(
//         dt: DigitLayout,
//         batch: usize,
//         m: usize,
//         n: usize,
//         k: usize,
//         c_base: *mut H::Byte,
//         a_base: *const H::Byte,
//         b_base: *const H::Byte,
//     ) -> Args<H> {
//         Args {
//             c_layout: TensorLayout::new_contiguous(dt, &[batch, m, n]),
//             c_base,
//             beta: BETA,
//             a_layout: TensorLayout::new_contiguous(dt, &[batch, m, k]),
//             a_base,
//             b_layout: TensorLayout::new_contiguous(dt, &[batch, k, n]),
//             b_base,
//             alpha: ALPHA,
//         }
//     }
//         #[test]
//         fn test_matmul() {
//             matmul_forward_h();
//         }
// }


// pub fn matmul_forward_h(
//     // mut out: *mut f32,
//     // mut inp: *const f32,
//     // mut weight: *const f32,
//     // mut bias: *const f32,
//     // B: usize,
//     // T: usize,
//     // C: usize,
//     // OC: usize,
// ) { 
//     const B:usize = 2;
//     const T:usize= 2;
//     const C:usize = 3;
//     const OC:usize = 2;

//     let batch = 2;
//     let k = 2;
//     let n = 3;
//     let m = 2;
//         // 创建输入矩阵 inp, weight 和 bias
//         let mut inp: Vec<f32> = vec![1., 2., 3.,
//                                      4., 5., 6.,
//                                      1., 2., 3.,
//                                      4., 5., 6.]; // [B * T, C]
//         let mut weight: Vec<f32> = vec![1., 2., 3.,
//                                         4., 5., 6.]; // [OC, C]
//         let mut bias: Vec<f32> = vec![0.1, 0.2]; // [OC]        
//         // 创建输出矩阵
//         let mut out: Vec<f32> = vec![0.0f32; (B * T * OC) as usize]; // [B * T, OC]

//         let a = inp;
//         let b = weight;
//         let c = out;

//     let cpu_op = RefOp::new(&Cpu);
//     let mut c_ref = c;
//     cpu_op
//         .launch(
//             &args(
//                 F64,
//                 batch,
//                 m,
//                 n,
//                 k,
//                 c_ref.as_mut_ptr().cast(),
//                 a.as_ptr().cast(),
//                 b.as_ptr().cast(),
//             ),
//             &mut [],
//             &ThisThread,
//         )
//         .unwrap();


// }