use std::alloc::Layout;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::slice;

pub struct DataLoaderNaive {
    // 超参数    
    pub B: usize,  // 批大小
    pub T: usize,  // 序列长度
    //输入处理及其状态
    pub tokens_file: Option<File>,  // 令牌文件
    pub file_size: u64,  // 文件大小
    pub current_position: u64,  // 当前文件位置
    //输出内存
    pub batch: *mut i32,  // 批次内存指针
    pub inputs: *mut i32,  // 输入令牌指针
    pub targets: *mut i32,  // 目标令牌指针
    //方便变量
    pub num_batches: usize,  // 批次数量
}

impl DataLoaderNaive {
    pub fn init(filename: &str, B: usize, T: usize) -> Self {
        let mut loader = DataLoaderNaive {
            B,
            T,
            tokens_file: None,
            file_size: 0,
            current_position: 0,
            batch: core::ptr::null_mut(),
            inputs: core::ptr::null_mut(),
            targets: core::ptr::null_mut(),
            num_batches: 0,
        };

        loader.tokens_file = File::open(filename).map(Some) // 打开 tokens 文件
            .expect("Error opening tokens file"); 
        
        if let Some(file) = &loader.tokens_file {
            let metadata = file.metadata() // 获取文件的元数据
                .expect("Error getting file metadata");
            loader.file_size = metadata.len(); // 获取文件大小
        }
        
        let required_size = (B * T + 1) * std::mem::size_of::<i32>(); // 计算所需大小
        if loader.file_size < required_size as u64 { // 检查文件大小是否足够
            panic!("Error! file size is too small"); 
        }
        loader.current_position = 0; // 从头开始

        // 为存储输入和目标分配 B*T + 1 个整数的空间
        unsafe {
            let layout =
                Layout::array::<i32>(required_size)
                    .expect("error creating layout");
            loader.batch = std::alloc::alloc(layout) as *mut i32;
            loader.inputs = loader.batch;
            loader.targets = loader.batch.add(1); // 目标移动一格->下一个令牌预测
            loader.num_batches = (loader.file_size as usize) / (B * T * std::mem::size_of::<i32>());
        }

        loader
    }

    // 将 DataLoaderNaive 重置为从文件开头开始。
    pub fn reset(&mut self) {
        self.current_position = 0;
    }

    // 将下一批数据加载到 DataLoaderNaive 的内存中。
    pub fn next_batch(&mut self) {
        let B = self.B;
        let T = self.T;
        let len = (B * T + 1) * std::mem::size_of::<i32>();

        // 如果我们位于文件末尾，则循环回到开头
        if self.current_position + len as u64 > self.file_size {
            self.current_position = 0;
        }
        
        // 将文件中的 B*T+1 整数读取到批处理中
        if let Some(tokens_file) = &mut self.tokens_file {
            // 查找文件中的当前位置
            tokens_file
                .seek(SeekFrom::Start(self.current_position))
                .expect("error seeking in tokens file");
        
            // 直接将 B*T+1 个整数从文件读取到批处理中
            unsafe {
                let batch_slice = std::slice::from_raw_parts_mut(self.batch, B * T + 1);
                tokens_file.read_exact(slice::from_raw_parts_mut(
                    batch_slice.as_mut_ptr() as *mut u8,
                    (B * T + 1) * std::mem::size_of::<i32>(),
                )).expect("error reading tokens file");
            }
        
            // 将当前位置前进 B*T 整数
            self.current_position += (B * T) as u64 * std::mem::size_of::<i32>() as u64;
        } else {
            panic!("Error! file is not open");
        }
    }

    // 释放 DataLoaderNaive 分配的内存。
    pub fn free(&mut self) {
        if let Some(file) = self.tokens_file.take() {
            drop(file); // 关闭文件
        }
        if !self.batch.is_null() { // 释放内存
            let layout =
                std::alloc::Layout::array::<i32>(self.B * self.T + 1)
                    .expect("error creating layout");
                unsafe {
                    std::alloc::dealloc(self.batch as *mut u8, layout);
                }
        }
        self.batch = core::ptr::null_mut();
        self.inputs = core::ptr::null_mut();
        self.targets = core::ptr::null_mut();
    }
}

pub struct DataLoaderDistributed {
    // 与分布式训练相关的变量
    pub process_rank: i32, // 当前进程的排名
    pub num_processes: i32, // 进程总数
    
    // 批次和令牌信息
    pub B: usize, // 批次大小
    pub T: usize, // 序列长度
    pub num_tokens: usize, // 令牌总数
    pub shard_num_samples: usize, // 当前分片中的样本总数
    
    // 分片和当前读取位置
    pub glob_result: Vec<String>, // 存储glob操作结果的向量
    pub current_shard_idx: usize, // 当前正在读取的分片索引
    pub current_sample_idx: usize, // 当前正在读取的样本索引
    
    // 文件句柄
    pub tokens_file: Option<File>, // 令牌文件的句柄
    
    // 数据缓冲区
    pub buffer: Vec<u16>, // 从文件读取数据的缓冲区
    pub inputs: Vec<i32>, // 输入令牌
    pub targets: Vec<i32>, // 目标令牌
    
    // 随机打乱相关变量
    pub shuffle_rng: Mt19937State, // 用于打乱的随机数生成器
    pub should_shuffle: i32, // 是否打乱数据
    pub shard_indices: Vec<i32>, // 分片索引
    pub intra_shard_indices: Vec<i32>, // 内部分片索引
    
    // 字节大小
    pub total_batch_size_bytes: usize, // 所有进程的总批次大小
    pub local_batch_offset_bytes: usize, // 当前进程在批次内的偏移
    pub header_bytes: usize, // 头部大小（字节）
    pub file_size_bytes: i64, // 文件大小（字节）
}

impl DataLoaderDistributed {
    // 创建新的数据加载器实例
    pub fn init(process_rank: i32, num_processes: i32, B: usize, T: usize, should_shuffle: i32) -> Self {
        Self {
            process_rank,
            num_processes,
            B,
            T,
            num_tokens: 0,
            shard_num_samples: 0,
            glob_result: Vec::new(),
            current_shard_idx: 0,
            current_sample_idx: 0,
            tokens_file: None,
            buffer: Vec::new(),
            inputs: Vec::new(),
            targets: Vec::new(),
            shuffle_rng: Mt19937State::new(64), // 初始化随机数生成器
            should_shuffle,
            shard_indices: Vec::new(),
            intra_shard_indices: Vec::new(),
            total_batch_size_bytes: 0,
            local_batch_offset_bytes: 0,
            header_bytes: 0,
            file_size_bytes: 0,
        }
    }

    // TODO
}

// 定义 MERSENNE_STATE_N 的常量
const MERSENNE_STATE_N: usize = 624;

// mt19937 随机数生成器的状态结构体
#[derive(Debug)]
pub struct Mt19937State {
    pub seed_: u64, // 生成器的种子
    pub left_: i32, // 剩余随机数的数量
    pub next_: u32, // 下一个随机数的索引
    pub state_: [u32; MERSENNE_STATE_N], // 存储状态的数组
    pub matrix_a: [u32; 2], // 用于矩阵乘法的常量数组
}

impl Mt19937State {
    // 创建新的 mt19937 状态结构体
    pub fn new(seed: u64) -> Self {
        Self {
            seed_: seed,
            left_: 0,
            next_: 0,
            state_: [0; MERSENNE_STATE_N], // 初始化 state 数组为 0
            matrix_a: [0x0, 0x9908b0df], // 初始化矩阵乘法常量
        }
    }

    // TODO
}