
pub fn random_u32(state: &mut u64) -> u32 {
    // xorshift算法: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    (x.wrapping_mul(0x2545F4914F6CDD1D) >> 32)
        .try_into()
        .expect("error random")
}
pub fn random_f32(state: &mut u64) -> f32 {
    // random float32 in [0,1)
    (random_u32(state) >> 8) as f32 / 16777216.0
}
pub fn sample_mult(probabilities: *const f32, n: usize, coin: f32) -> usize {
    // 概率的样本索引（它们的总和必须为 1！）
    // coin 是 [0, 1) 中的随机数，通常来自 random_f32()
    let mut cdf = 0.0; // 累积分布函数（CDF）
    for i in 0..n {
        let prob = unsafe { 
            *probabilities.add(i) };
        cdf += prob;
        if coin < cdf {
            return i;
        }
    }
    n - 1 // 如果出现舍入错误
}