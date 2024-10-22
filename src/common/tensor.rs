#[warn(dead_code)]

// 定义一个 Tensor_ 结构体，包装裸指针，并添加参数
pub struct Tensor_<T> {
    pub ptr: *mut T,
    b: usize, // batch
    r: usize, // row
    c: usize, // column
}

// 实现 Send 和 Sync，使 Tensor_ 可以在线程中安全并行
unsafe impl<T> Send for Tensor_<T> {}
unsafe impl<T> Sync for Tensor_<T> {}

impl<T> Tensor_<T> {
    // 构造函数，用于创建 Tensor_
    pub fn new(ptr: *mut T, b: usize, r: usize, c: usize) -> Self {
        Tensor_ {
            ptr,
            b,
            r,
            c,
        }
    }
}
