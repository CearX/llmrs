#![allow(non_snake_case)]

use std::io::{self, Write};
use std::alloc::{self, alloc, Layout};
use std::time::Instant;

pub mod dataloader;
pub mod gpt2;
pub mod tokenizer;
pub mod passes;
pub mod common;
pub mod random_sample;

use dataloader::DataLoaderNaive;
use gpt2::*;
use tokenizer::*;
use random_sample::*;


// ----------------------------------------------------------------------------
// 主训练循环

pub fn main() {
    // Build GPT-2 models from checkpoints
    let mut gpt2 = GPT2::checkpoint("bin/gpt2_124M.bin");

    // 从token文件中构建data loaders
    let train_set = "bin/tiny_shakespeare_train.bin";
    let val_set = "bin/tiny_shakespeare_val.bin";

    let B: usize = 4; // 批量大小 4（即将训练 4 个独立的令牌序列）
    let T: usize = 64; // 序列长度 64（即每个序列有 64 个令牌长）。必须 <= maxT，对于 GPT-2 为 1024

    unsafe {
        let mut train_loader = DataLoaderNaive::init(train_set, B, T);
        let mut val_loader = DataLoaderNaive::init(val_set, B, T);
        println!("train dataset num_batches: {}", train_loader.num_batches);
        println!("val dataset num_batches: {}", val_loader.num_batches);
        let val_num_batches = 5;

        // 构建分词器
        let mut tokenizer = Tokenizer::init("bin/gpt2_tokenizer.bin");

        // 随机种子
        let mut rng_state: u64 = 1337;
        // 分配内存，用于从模型生成样本
        let gen_tokens_layout = Layout::array::<i32>(B * T)
            .expect("error creating layout");
        let gen_tokens = alloc(gen_tokens_layout) as *mut i32;
        let genT = 64; // 我们将进行的推理步骤

        // 训练循环
        for step in 0..=40 {
            // 每10步估计验证损失
            if step % 10 == 0 {  // 每当step能被10整除时执行以下代码
                let mut val_loss = 0.0;  // 初始化val_loss变量，用于存储验证集上的损失值
                val_loader.reset();  // 重置验证集加载器，准备加载数据
                for _ in 0..val_num_batches {  // 循环处理每个批次的数据
                    val_loader.next_batch();  // 获取下一个批次的数据
                    gpt2.forward(val_loader.inputs, val_loader.targets, B, T);  // 使用gpt2模型进行前向传播并计算损失
                    val_loss += gpt2.mean_loss;  // 累加当前批次的损失值
                }
                val_loss /= val_num_batches as f32;  // 计算验证集的平均损失
                println!("val loss {}", val_loss);  // 打印验证集的平均损失值
            }

            // 每20步进行模型推理，打印生成的文本
            if step > 0 && step % 20 == 0 {
                // 用 GPT2_EOT 填充 gen_tokens，这将开始生成
                for i in 0..B * T {
                    *gen_tokens.add(i) = 50256;
                }
                // 现在从模型中进行自回归采样
                println!("generating:\n---");
                for t in 1..genT {
                    // 请注意，这里的推理非常浪费，因为对于每个标记
                    // 我们从头开始重新计算所有 (B,T) 位置的前向传播
                    // 但无论如何，这里的推论只是为了进行合理性检查
                    // 我们也许可以稍后通过仔细的测试进行更多优化
                    gpt2.forward(gen_tokens, core::ptr::null_mut(), B, T);
                    // 此外，下面我们仅使用所有 B 行中的 b=0 （即第一行）
                    // 原则上我们在这里并行运行 B 个“推理流”
                    // 但仅使用位置 0
                    // 得到Vp维向量 probs[0, t-1, :]
                    let probs = gpt2
                        .acts
                        .probs.ptr // 模型在当前时间步 t 的输出概率分布，包含对整个词汇表的预测概率
                        .add((t - 1) * gpt2.config.padded_vocab_size); //模型输出的激活结果，经过 softmax 得到的概率分布
                        // 这里获取了 t-1 时间步上的概率分布
                    let coin = random_f32(&mut rng_state); // 生成一个介于 0 和 1 之间的随机浮点数
                    // 请注意，我们仅从前 V 元素中采样，忽略填充。（无论如何，填充区域中的概率应该为零）
                    // 使用 coin 从概率分布 probs 中对 vocab_size 个候选 token 进行采样，返回采样得到的下一个 token ID (next_token)
                    let next_token = sample_mult(probs, gpt2.config.vocab_size, coin) as u32;
                    // 将采样得到的 next_token 添加到 gen_tokens 数组的第 t 个位置，准备用于下一步的推理
                    *gen_tokens.add(t) = next_token as i32;
                    // 将 next_token 解码为对应的文本，并通过 safe_print(token_str) 打印
                    if tokenizer.init_ok {
                        let token_str = tokenizer.decode(next_token);
                        safe_print(token_str);
                    } else {
                        // 如果 tokenizer 初始化失败，则直接打印 token 的 ID
                        print!("{} ", next_token);
                    }
                    // 刷新标准输出缓冲区，确保所有已经写入的数据被立即输出到屏幕或目标设备
                    io::stdout().flush().unwrap();
                }
                println!("\n---");
            }

             // 进行训练步骤
            let start = Instant::now();  // 记录开始时间
            train_loader.next_batch();  // 获取下一个训练批次
            gpt2.forward(train_loader.inputs, train_loader.targets, B, T);  // 使用gpt2模型进行前向传播
            gpt2.zero_grad();  // 清零梯度
            gpt2.backward();  // 进行反向传播计算梯度
            gpt2.update(1e-4, 0.9, 0.999, 1e-8, 0.0, step + 1);  // 更新模型参数
            let duration = start.elapsed();  // 计算训练步骤所用时间
            println!("step {}: train loss {:.6} (took {:.2} ms)", step, gpt2.mean_loss, duration.as_secs_f64() * 1000.0); // 打印当前步骤、训练损失及耗时 
        }
        // 释放内存
        train_loader.free();
        val_loader.free();
        tokenizer.free();
        gpt2.free();
        alloc::dealloc(gen_tokens as *mut u8, gen_tokens_layout);
    }
}