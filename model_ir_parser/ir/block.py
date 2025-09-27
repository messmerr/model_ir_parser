"""Block层定义 - 包含多个Cell的块"""

from dataclasses import dataclass
from typing import List, Optional
from .cell import Cell


@dataclass
class Block:
    """Transformer块，包含多个Cell"""
    cells: List[Cell]
    block_id: Optional[int] = None

    def get_attention_cells(self) -> List[Cell]:
        """获取所有注意力相关的Cell"""
        return [cell for cell in self.cells if cell.is_attn()]

    def get_ffn_cells(self) -> List[Cell]:
        """获取所有前馈网络相关的Cell"""
        return [cell for cell in self.cells if not cell.is_attn()]

    def __repr__(self):
        cell_names = [cell.get_name() for cell in self.cells]
        return f"Block(cells={cell_names}, id={self.block_id})"