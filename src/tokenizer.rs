use std::fs::File;
use std::io::Read;

pub struct Tokenizer {
    vocab_size: u32,
    token_table: Vec<String>,
    pub init_ok: bool,
}

pub unsafe fn safe_print(piece: &str) {
    // 这些令牌是原始字节，我们只想打印可打印的那些。  
    // 许多字节可以是各种控制代码、退格等。
    if piece.is_empty() {
        return;
    }
    // 处理单个字节令牌
    let bytes = piece.as_bytes();
    if bytes.len() == 1 && !(bytes[0].is_ascii_graphic() || bytes[0].is_ascii_whitespace()) {
        return; // 奇怪的字节，不要打印它
    }
    // 如果字符串有效则打印该字符串
    print!("{}", piece);
}

impl Tokenizer {
    pub fn init(filename: &str) -> Self {
        let mut tokenizer = Tokenizer {
            vocab_size: 0,
            token_table: Vec::new(),
            init_ok: false,
        };
        // 打开token文件
        let mut file = File::open(filename)
            .expect("error opening tokenizer file");
        // 读取文件头
        let mut header = [0; 256];
        file.read_exact(unsafe {
            std::slice::from_raw_parts_mut(
                header.as_mut_ptr() as *mut u8,
                header.len() * std::mem::size_of::<u32>(),
            )
        })
        .expect("error reading tokenizer header");
        // 检查魔数和版本
        if header[0] != 20240328 {
            panic!("Bad magic tokenizer file");
        }
        if header[1] != 2 {
            panic!("Bad version in tokenizer file")
        }
        // 读取词表大小
        tokenizer.vocab_size = header[2];
        // 读取词表
        for _ in 0..tokenizer.vocab_size { // 遍历词表的大小
            let mut length = [0]; // 初始化一个数组，用于存储标记长度
            file.read_exact(&mut length) // 从文件中读取标记长度
                .expect("error reading token length"); // 读取失败时抛出错误
        
            assert!(length[0] > 0); // 确保每个标记至少有一个字符
            let mut token_bytes = vec![0u8; length[0] as usize]; // 根据标记长度创建字节数组
            file.read_exact(&mut token_bytes) // 从文件中读取标记的字节
                .expect("error reading token bytes"); // 读取失败时抛出错误
        
            let token = match String::from_utf8(token_bytes) { // 尝试将字节数组转换为字符串
                Ok(token) => token, // 转换成功，保存标记
                Err(_) => String::new(), // 转换失败，返回空字符串
            };
            tokenizer.token_table.push(token); // 将标记添加到词表中
        }
        // 初始化成功
        tokenizer.init_ok = true;
        // 返回tokenizer
        tokenizer
    }

    // 将令牌 ID 解码为其相应的字符串。
    pub fn decode(&mut self, token_id: u32) -> &str {
        if !self.init_ok || token_id >= self.vocab_size {
            if token_id >= self.vocab_size {
                println!("invalid token id {}!", token_id);
            }
            return ""; // 直接返回空字符串
        }
        &self.token_table[token_id as usize] // 返回对应的标记
    }

    // 释放资源
    pub fn free(&mut self) {
        if self.init_ok != false {
            self.token_table.clear();
            self.init_ok = false;
        }
    }
}
