# Model IR Parser 设计文档

## 1. 项目概述

Model IR Parser 是一个独立的模块，专门用于将九歌（Jiuge）模型的配置文件解析并转换为统一的中间表示（Intermediate Representation, IR）。该 IR 结构设计参考了 APEX+ 的 Transformer IR，但作为独立模块实现，不依赖于任何外部框架。

### 1.1 核心目标
- **独立性**：模块完全独立，不依赖 APEX+ 或其他外部框架
- **兼容性**：生成的 IR 结构与 APEX+ Transformer IR 保持一致
- **简洁性**：专注于九歌模型，避免过度设计
- **可集成性**：方便与 InfiniLM 等推理引擎集成

### 1.2 支持的模型类型
- 九歌系列模型（FM9G）
- 支持的配置格式：HuggingFace config.json

## 2. 系统架构

### 2.1 模块结构（简化版）
```
model_ir_parser/
├── __init__.py
├── ir/                         # IR数据结构定义
│   ├── __init__.py
│   ├── transformer.py         # Transformer顶层结构
│   ├── block.py              # Block层定义
│   ├── cell.py               # Cell基类和具体实现
│   └── task.py               # Task层定义
├── parser.py                  # 九歌模型解析器
├── utils/                      # 工具函数
│   └── dtype.py              # 数据类型定义（参考APEX）
└── examples/                   # 使用示例
    └── parse_jiuge.py
```

## 3. IR 数据结构设计

### 3.1 层次结构（基于最新分析修正）
```
Transformer (顶层)
    ├── embedding: Embedding        # 词嵌入层
    ├── num_encoder_blocks: int     # 编码器块数量
    ├── encoder_block: Optional[Block] # 编码器块（可选，单个Block实例）
    ├── num_decoder_blocks: int     # 解码器块数量
    ├── decoder_block: Block        # 解码器块（单个Block实例，所有层共享）
    ├── sampler: Sampler           # 输出采样层
    ├── hidden_size: int           # 隐藏层维度
    └── metadata: ModelMetadata    # 模型元信息

Block (块层)
    └── cells: List[Cell]          # 单元列表（通常包含注意力和FFN两个Cell）

Cell (单元层) - 抽象基类
    ├── MHA                        # 标准多头注意力（num_heads == num_kv_heads）
    ├── MQA                        # 多查询注意力（包括GQA情况，num_kv_heads < num_heads）
    ├── MLP/GLU/SwiGLU             # 前馈网络
    ├── MoE/SwiMoE                 # 混合专家
    └── ParallelAttentionMLP       # 并行注意力和MLP

Task (任务层)
    ├── AttentionHead              # 注意力头
    └── FFNFilter                  # 前馈过滤器
```

### 3.2 核心类定义

```python
# transformer.py
@dataclass
class Transformer:
    """Transformer模型的顶层IR表示（与APEX保持一致）"""
    embedding: Embedding
    num_encoder_blocks: int
    encoder_block: Optional[Block]  # 注意：是单个Block，不是List
    num_decoder_blocks: int
    decoder_block: Block            # 注意：是单个Block，不是List
    sampler: Sampler
    hidden_size: int
    metadata: ModelMetadata

    @classmethod
    def from_blocks(cls, vocab_size: int, hidden_size: int,
                   num_encoder_blocks: int, encoder_block: Optional[Block],
                   num_decoder_blocks: int, decoder_block: Block) -> 'Transformer':
        """工厂方法，与APEX的接口保持一致"""
        embedding = Embedding(vocab_size, hidden_size)
        sampler = Sampler(vocab_size, hidden_size)
        return cls(
            embedding, num_encoder_blocks, encoder_block,
            num_decoder_blocks, decoder_block, sampler, hidden_size,
            metadata=None
        )

# cell.py - 修正后的Cell定义
@dataclass
class MHA(Cell):
    """标准多头注意力（所有头共享KV）"""
    num_heads: int
    hidden_size: int
    head_size: int

    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0
        if self.head_size != self.hidden_size // self.num_heads:
            raise ValueError("head_size must equal hidden_size // num_heads")

@dataclass
class MQA(Cell):
    """多查询注意力（支持GQA：num_kv_heads < num_query_heads）"""
    num_query_heads: int    # 查询头数
    num_kv_heads: int       # KV头数（GQA时小于查询头数）
    head_size: int
    hidden_size: int

    def __post_init__(self):
        assert self.num_query_heads % self.num_kv_heads == 0
        self.num_query_per_kv = self.num_query_heads // self.num_kv_heads
```

## 4. 九歌模型解析器

### 4.1 解析器实现

```python
# parser.py
class JiugeParser:
    """九歌模型配置解析器"""

    def parse_file(self, config_path: str) -> Transformer:
        """从配置文件解析"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return self.parse(config)

    def parse(self, config: Dict) -> Transformer:
        """解析配置生成IR"""
        # 验证是否为九歌模型
        if config.get("model_type") not in ["fm9g", "fm9g7b"]:
            raise ValueError(f"Unsupported model type: {config.get('model_type')}")

        # 提取元信息
        metadata = self.extract_metadata(config)

        # 解析九歌特有的scale参数
        scale_emb = config.get("scale_emb", 1.0)
        scale_depth = config.get("scale_depth", 1.0)
        dim_model_base = config.get("dim_model_base", 256)

        # 构建注意力Cell（支持GQA）
        attention = self._build_attention(config)

        # 构建FFN Cell
        ffn = self._build_ffn(config)

        # 组装Block
        block = Block([attention, ffn])

        # 创建Transformer
        return Transformer(
            embedding=Embedding(config["vocab_size"], config["hidden_size"]),
            num_encoder_blocks=0,
            encoder_blocks=None,
            num_decoder_blocks=config["num_hidden_layers"],
            decoder_blocks=[block] * config["num_hidden_layers"],
            sampler=Sampler(config["vocab_size"], config["hidden_size"]),
            hidden_size=config["hidden_size"],
            metadata=metadata
        )

    def _build_attention(self, config: Dict) -> Cell:
        """构建注意力层"""
        num_heads = config["num_attention_heads"]
        num_kv_heads = config.get("num_key_value_heads", num_heads)
        hidden_size = config["hidden_size"]
        head_size = hidden_size // num_heads

        if num_kv_heads < num_heads:
            # 使用MQA（九歌模型的GQA实际上是MQA的一种）
            return MQA(num_heads, num_kv_heads, head_size, hidden_size)
        else:
            # 使用标准MHA
            return MHA(num_heads, hidden_size, head_size)

    def _build_ffn(self, config: Dict) -> Cell:
        """构建前馈网络层"""
        activation = config.get("hidden_act", "silu")
        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]

        if activation == "silu":
            return SwiGLU(hidden_size, intermediate_size)
        elif activation == "gelu":
            return GLU(hidden_size, intermediate_size)
        else:
            return MLP(hidden_size, intermediate_size)

    def extract_metadata(self, config: Dict) -> ModelMetadata:
        """提取模型元信息"""
        return ModelMetadata(
            model_type=config.get("model_type"),
            vocab_size=config.get("vocab_size"),
            max_position=config.get("max_position_embeddings"),
            rope_theta=config.get("rope_theta"),
            rope_scaling=config.get("rope_scaling"),
            dtype=config.get("torch_dtype", "float16"),
            # 九歌特有参数
            scale_emb=config.get("scale_emb", 1.0),
            scale_depth=config.get("scale_depth", 1.0),
            dim_model_base=config.get("dim_model_base", 256)
        )
```

## 5. 工具函数

### 5.1 数据类型定义（参考APEX实现）

```python
# utils/dtype.py
import enum

class DTYPE(enum.Enum):
    """数据类型枚举"""
    FLOAT32 = 0
    FLOAT16 = 1
    FLOAT8 = 2
    BFLOAT16 = 3

# 设置每种类型的字节大小
DTYPE.FLOAT32.size = 4
DTYPE.FLOAT16.size = 2
DTYPE.FLOAT8.size = 1
DTYPE.BFLOAT16.size = 2

def get_dtype(dtype_str: str) -> DTYPE:
    """从字符串获取DTYPE枚举"""
    dtype_map = {
        "float32": DTYPE.FLOAT32,
        "float": DTYPE.FLOAT32,
        "float16": DTYPE.FLOAT16,
        "half": DTYPE.FLOAT16,
        "float8": DTYPE.FLOAT8,
        "bfloat16": DTYPE.BFLOAT16,
    }
    if dtype_str.lower() not in dtype_map:
        raise ValueError(f"Unknown dtype {dtype_str}")
    return dtype_map[dtype_str.lower()]
```

## 6. 使用接口

### 6.1 基本使用

```python
from model_ir_parser import JiugeParser

# 创建解析器
parser = JiugeParser()

# 从配置文件解析
ir = parser.parse_file("config.json")

# 从字典解析
config_dict = {"model_type": "fm9g", ...}
ir = parser.parse(config_dict)

# 访问IR结构
print(f"模型类型: {ir.metadata.model_type}")
print(f"层数: {ir.num_decoder_blocks}")
print(f"隐藏层维度: {ir.hidden_size}")

# 遍历所有Block
for i, block in enumerate(ir.decoder_blocks):
    print(f"Block {i}: {len(block.cells)} cells")
    for cell in block.cells:
        print(f"  - {cell.__class__.__name__}")
```

### 6.2 与InfiniLM集成示例

```python
# 与InfiniLM推理引擎集成
from model_ir_parser import JiugeParser

def load_model_from_config(config_path: str):
    """从配置文件加载模型结构"""
    parser = JiugeParser()
    ir = parser.parse_file(config_path)

    # 根据IR创建InfiniLM的JiugeMeta结构
    meta = create_jiuge_meta_from_ir(ir)

    return meta

def create_jiuge_meta_from_ir(ir: Transformer):
    """将IR转换为JiugeMeta"""
    return JiugeMeta(
        nlayer=ir.num_decoder_blocks,
        d=ir.hidden_size,
        nh=ir.decoder_blocks[0].cells[0].num_heads,
        nkvh=ir.decoder_blocks[0].cells[0].num_kv_heads,
        # ... 其他参数
    )
```

## 7. 九歌模型配置示例（基于InfiniLM实现分析）

### 7.1 配置文件与InfiniLM参数映射

基于jiuge.cpp的分析，InfiniLM在运行时需要以下关键参数：

| jiuge.cpp使用的参数 | config.json字段 | 说明 |
|-------------------|----------------|------|
| nlayer | num_hidden_layers | 层数（62层） |
| d | hidden_size | 隐藏维度（2560） |
| nh | num_attention_heads | 注意力头数（40） |
| nkvh | num_key_value_heads | KV头数（4，实现GQA） |
| dh | hidden_size // num_attention_heads | 每头维度（64） |
| di | intermediate_size | FFN中间层维度（6400） |
| dvoc | vocab_size | 词表大小（73448） |
| dctx | max_position_embeddings | 最大位置（32768） |
| epsilon | rms_norm_eps | RMS归一化参数 |
| theta | rope_theta | RoPE参数（1000000.0） |

### 7.2 InfiniLM推理流程中的模型结构

从jiuge.cpp的实现可以看到：

1. **注意力计算流程**（第195-234行）：
   - RMSNorm → QKV投影 → RoPE → 自注意力 → O投影
   - 使用`w_attn_qkv`权重矩阵（合并的QKV）
   - 支持可选的QKV bias（`b_attn_qkv`）

2. **FFN计算流程**（第245-257行）：
   - RMSNorm → Gate+Up投影（合并） → SwiGLU激活 → Down投影
   - 使用`w_ffn_gate_up`（合并的gate和up）
   - 使用SwiGLU激活函数（符合config中的"silu"）

3. **分布式推理支持**：
   - 支持多GPU并行（通过`ndev`参数）
   - 使用AllReduce进行梯度同步
   - KV cache按设备分片

### 7.3 解析后生成的IR结构

```json
{
  "model_type": "fm9g",
  "num_hidden_layers": 62,      // 62层decoder
  "hidden_size": 2560,           // 隐藏层维度
  "num_attention_heads": 40,     // 注意力头数
  "num_key_value_heads": 4,      // KV头数（GQA特性，10:1压缩比）
  "intermediate_size": 6400,     // FFN中间层维度
  "hidden_act": "silu",          // SwiGLU激活
  "vocab_size": 73448,           // 词表大小
  "max_position_embeddings": 32768,  // 最大位置
  "rope_theta": 1000000.0,       // RoPE参数

  // 九歌特有参数（用于权重缩放）
  "scale_emb": 12,               // 嵌入缩放
  "scale_depth": 1.4,            // 深度缩放
  "dim_model_base": 256          // 基础模型维度
}
```

解析后生成的IR结构：
- **Transformer层**：62个decoder blocks，无encoder blocks
- **Block层**：每个block包含2个cells
  - MQA(num_query_heads=40, num_kv_heads=4, head_size=64)
  - SwiGLU(hidden_size=2560, intermediate_size=6400)
- **特殊处理**：
  - 保留scale参数供权重加载时使用
  - QKV权重是合并的（需要在权重加载时处理）
  - Gate和Up权重是合并的（需要在权重加载时处理）

## 8. 实现要点（基于深度分析更新）

### 8.1 关键设计修正

基于对APEX和InfiniLM的深度分析，以下是必要的修正：

1. **使用MQA而非GQA类名**
   - APEX中没有GQA类，而是用MQA处理所有多查询注意力情况
   - 当`num_kv_heads < num_query_heads`时，MQA自动实现GQA行为

2. **decoder_block是单个实例**
   - 不是`List[Block]`，而是单个`Block`实例
   - 所有层共享相同的Block结构定义

3. **权重格式兼容性**
   - InfiniLM使用合并的QKV权重（`w_attn_qkv`）
   - InfiniLM使用合并的Gate+Up权重（`w_ffn_gate_up`）
   - IR结构不需要关心权重格式，但metadata中应记录

### 8.2 为什么不需要复杂的注册机制
- 目前只支持九歌模型，不需要动态注册
- APEX也是用简单字典，没有复杂的装饰器
- 保持代码简洁，避免过度设计

### 8.2 必要的工具函数
- **dtype.py**：处理float16/bfloat16等数据类型（与APEX保持一致）
- 其他工具函数根据实际需要添加

### 8.3 与APEX的兼容性
- IR结构完全对齐APEX的定义
- 使用相同的类名和属性名
- 方便未来直接使用APEX的优化器