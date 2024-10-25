

// pub struct LlamaWorker<Ops: Operators, W> {
//     meta: LlamaMeta,
//     weights: WeightDecorator<W>,
//     rms_norm: Ops::RmsNorm,
//     mat_mul: Ops::MatMul,
//     rope: Ops::Rope,
//     attn_kv_cached: Ops::AttnKVCached,
//     mlp: Ops::Mlp,
//     rearrange: Ops::Rearrange,
//     all_reduce: Ops::AllReduce,
//     residual: bool,
//     pub debug: bool,
// }