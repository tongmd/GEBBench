# One Structure, Six Voices：V1 数据集设计

## 1. 数据集总体原则

每个样本都从统一的 latent schema 生成，而不是从 story 开始逐步翻译。

$$
z \rightarrow \{ S, M, P, D, I, I_{\mathrm{style}} \}
$$

其中：

* $S$：Story
* $M$：Mathematics
* $P$：Program
* $D$：Drawing Program
* $I$：Canonical Image
* $I_{\mathrm{style}}$：Structure-preserving Stylized Image

每个 latent schema 都应至少包含：

```json
{
  "objects": [],
  "relations": [],
  "operators": [],
  "initial_state": {},
  "goal": {},
  "structural_properties": {},
  "render_constraints": {}
}
```

V1 只做三个核心 family：

1. **Graph Worlds**
2. **Recursive Gardens**
3. **Symmetry Labs**

这三个 family 共同覆盖：

* 拓扑结构
* 层级结构
* 几何结构

并且都能承受明显的风格变化。

---

# 2. Family A：Graph Worlds

## 2.1 核心抽象

Graph Worlds 表达：

$$
G=(V,E,w,\ell)
$$

其中：

* $V$：节点
* $E$：边
* $w$：可选权重
* $\ell$：节点或边类型

主要结构性质：

* connectivity
* reachability
* shortest path
* cycle
* DAG
* strongly connected component
* source/sink
* bottleneck
* flow
* dependency
* topological order

---

## 2.2 推荐子任务

V1 建议包含六类：

### A1. Reachability

判断节点 $u$ 是否能够到达 $v$。

### A2. Shortest Path

寻找从 $u$ 到 $v$ 的最短路径。

### A3. Dependency Order

在 DAG 中寻找合法执行顺序。

### A4. Cycle Detection

判断依赖关系中是否存在循环。

### A5. Bottleneck

寻找所有路径必须经过的关键节点。

### A6. Flow Conservation

网络中的流量满足节点守恒。

---

## 2.3 Latent schema

```json
{
  "family": "graph_world",
  "task_type": "reachability",
  "directed": true,
  "nodes": [
    {"id": "v0", "role": "start"},
    {"id": "v1", "role": "ordinary"},
    {"id": "v2", "role": "ordinary"},
    {"id": "v3", "role": "goal"}
  ],
  "edges": [
    ["v0", "v1"],
    ["v1", "v2"],
    ["v2", "v3"]
  ],
  "query": {
    "source": "v0",
    "target": "v3"
  },
  "answer": {
    "reachable": true,
    "witness_path": ["v0", "v1", "v2", "v3"]
  },
  "structural_properties": {
    "num_nodes": 4,
    "num_edges": 3,
    "has_cycle": false,
    "shortest_path_length": 3
  }
}
```

---

## 2.4 Story renderer

同一个 graph 可以渲染成不同故事领域。

### 城市道路

> 四座城市通过单向公路连接。A 可以到达 B，B 可以到达 C，C 可以到达 D。旅行者能否从 A 到达 D？

### 软件依赖

> 模块 A 依赖模块 B，模块 B 依赖模块 C，模块 C 依赖模块 D。构建 A 是否需要先构建 D？

### 任务工作流

> 完成任务 A 后可以执行 B，完成 B 后可以执行 C，完成 C 后可以执行 D。是否存在从 A 推进到 D 的合法工作流？

### 信息传播

> 用户 A 可以向 B 发送消息，B 可以转发给 C，C 可以转发给 D。信息能否从 A 传播到 D？

Story 中的名字、领域和叙事都可以变化，但 graph 结构不变。

---

## 2.5 Mathematics renderer

### 邻接关系

$$
E=\{(v_0,v_1),(v_1,v_2),(v_2,v_3)\}.
$$

判断：

$$
v_0\leadsto v_3.
$$

### 邻接矩阵

$$
A=
\begin{bmatrix}
0&1&0&0\\
0&0&1&0\\
0&0&0&1\\
0&0&0&0
\end{bmatrix}.
$$

判断是否存在 $k$，使：

$$
(A^k)_{0,3}>0.
$$

### 图论命题

$$
\exists p=(v_0,\ldots,v_3)
\quad
\text{s.t. }
(v_i,v_{i+1})\in E.
$$

---

## 2.6 Program renderer

```python
from collections import deque

def reachable(graph: dict[str, list[str]], start: str, target: str) -> bool:
    queue = deque([start])
    visited = {start}

    while queue:
        node = queue.popleft()
        if node == target:
            return True

        for nxt in graph.get(node, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)

    return False
```

不同 program voice 可以生成：

* BFS
* DFS
* adjacency matrix multiplication
* recursive search
* topological algorithm

程序语义一致，但实现形式不同。

---

## 2.7 Drawing program DSL

建议使用 SVG 或 Graphviz 风格的安全 DSL：

```json
{
  "canvas": {"width": 768, "height": 512},
  "nodes": [
    {"id": "v0", "x": 100, "y": 250, "shape": "circle"},
    {"id": "v1", "x": 280, "y": 160, "shape": "circle"},
    {"id": "v2", "x": 460, "y": 160, "shape": "circle"},
    {"id": "v3", "x": 640, "y": 250, "shape": "circle"}
  ],
  "edges": [
    {"source": "v0", "target": "v1", "directed": true},
    {"source": "v1", "target": "v2", "directed": true},
    {"source": "v2", "target": "v3", "directed": true}
  ]
}
```

Drawing renderer 输出 SVG。

---

## 2.8 Canonical image

Canonical image 必须满足：

* 纯色背景
* 所有节点完整可见
* 所有边不遮挡
* 箭头方向明确
* 无复杂纹理
* 节点标签可选
* 图布局尽量减少交叉

建议同时保存：

* raster PNG
* SVG
* binary node mask
* binary edge mask
* node coordinate metadata

---

## 2.9 Structure signature

每张图对应：

```json
{
  "node_count": 4,
  "edge_count": 3,
  "directed_edges": [
    ["v0", "v1"],
    ["v1", "v2"],
    ["v2", "v3"]
  ],
  "degree_sequence": {
    "in": [0, 1, 1, 1],
    "out": [1, 1, 1, 0]
  },
  "has_cycle": false,
  "connected_components": 1,
  "shortest_path": {
    "v0_to_v3": 3
  }
}
```

---

## 2.10 Style-safe transformations

### 安全

* blueprint
* chalkboard
* pencil sketch
* ink drawing
* subway map
* hand-drawn diagram
* technical illustration
* pixel-art diagram
* paper cut-out
* monochrome engraving

### 应严格限制

* watercolor with blurred edges
* oil painting
* dense texture
* surreal background
* heavy shadows
* photorealistic scene conversion

---

## 2.11 Style 后验证

验证器检查：

1. 节点数是否保持
2. 边数是否保持
3. 每条边的 source-target 是否保持
4. 箭头方向是否保持
5. 连通性是否保持
6. 是否产生虚假边
7. 是否丢失节点

严格条件：

$$
\hat E_{\mathrm{style}}=E_{\mathrm{canonical}}.
$$

宽松条件：

$$
F1_{\mathrm{edge}}\geq 0.98.
$$

---

## 2.12 Hard negatives

### 结构错、表面像

* 风格完全相同，但偷偷删除一条边
* 反转一条箭头
* 加入一条造成 cycle 的边
* 把 goal 节点连接到错误节点

### 结构对、表面不同

* Graphviz 风格
* 地铁图风格
* 城市道路图风格
* 软件模块图风格

---

# 3. Family B：Recursive Gardens

## 3.1 核心抽象

Recursive Gardens 以递归生成系统为核心：

$$
X_{n+1}=R(X_n).
$$

可用形式：

* recursive tree
* L-system
* substitution grammar
* recursive geometric construction
* nested composition
* fractal curve

主要属性：

* recursion depth
* branching factor
* rewrite rule
* self-similarity
* termination
* scale ratio
* orientation transformation
* node count by depth

---

## 3.2 推荐子任务

### B1. Binary Recursive Tree

每个节点产生两个子节点。

### B2. Variable Branching Tree

每个节点产生 $b$ 个子节点。

### B3. L-System Growth

符号字符串反复重写并绘制。

### B4. Recursive Polygon

每个多边形中生成缩小子多边形。

### B5. Recursive Path

Koch-like、Dragon-like 路径。

### B6. Mutual Recursion

两个对象类型交替生成。

---

## 3.3 Latent schema

```json
{
  "family": "recursive_garden",
  "task_type": "binary_tree",
  "axiom": "F",
  "rewrite_rules": {
    "F": "F[+F][-F]"
  },
  "depth": 4,
  "drawing_semantics": {
    "F": "move_forward",
    "+": "rotate_left",
    "-": "rotate_right",
    "[": "push_state",
    "]": "pop_state"
  },
  "parameters": {
    "angle": 28,
    "scale_ratio": 0.72,
    "initial_length": 140
  },
  "structural_properties": {
    "branching_factor": 2,
    "recursion_depth": 4,
    "leaf_count": 16,
    "self_similar": true
  }
}
```

---

## 3.4 Story renderer

### 植物生长

> 一株植物从一根主枝开始。每一轮，每个末端枝条都会长出左右两条更短的新枝。经过四轮后，植物形成四层分叉结构。

### 家族故事

> 每个人都有两个后代，每一代重复相同规则。四代后家族树有多少末代成员？

### 任务分解

> 一个任务会被拆分为两个规模更小的子任务，每个子任务继续以相同方式拆分，直到达到指定深度。

### 城市道路生成

> 每条道路末端都会向左右分出两条更短的道路，形成递归道路网络。

---

## 3.5 Mathematics renderer

### 叶节点数量

$$
L_d=2^d.
$$

### 总节点数

$$
N_d=1+2+\cdots+2^d=2^{d+1}-1.
$$

### 长度递推

若每层缩放比例为 $r$：

$$
\ell_{d+1}=r\ell_d.
$$

### 递归定义

$$
T_d=\operatorname{Node}(T_{d-1},T_{d-1}),
\qquad
T_0=\operatorname{Leaf}.
$$

---

## 3.6 Program renderer

```python
from dataclasses import dataclass

@dataclass
class Node:
    left: "Node | None" = None
    right: "Node | None" = None

def build_tree(depth: int) -> Node:
    if depth == 0:
        return Node()

    return Node(
        left=build_tree(depth - 1),
        right=build_tree(depth - 1),
    )
```

L-system 版本：

```python
def rewrite(sequence: str, rules: dict[str, str], depth: int) -> str:
    for _ in range(depth):
        sequence = "".join(rules.get(symbol, symbol) for symbol in sequence)
    return sequence
```

---

## 3.7 Drawing program DSL

```json
{
  "primitive": "branch",
  "root": {
    "x": 384,
    "y": 480,
    "angle": -90,
    "length": 150
  },
  "recursion": {
    "depth": 4,
    "children": [
      {"angle_delta": -28, "scale": 0.72},
      {"angle_delta": 28, "scale": 0.72}
    ]
  }
}
```

renderer 必须同时输出：

* SVG
* branch skeleton
* node coordinates
* parent-child edges
* recursion-level mask

---

## 3.8 Canonical image

Canonical image 应：

* 分支线清晰
* 每层颜色可区分
* 无叶片或复杂纹理
* 保持标准方向
* 背景干净
* 所有节点不重叠

可以有两种 canonical view：

### Structural tree

明确显示节点和边。

### Geometric tree

只显示分支曲线，但保留 skeleton。

---

## 3.9 Structure signature

```json
{
  "depth": 4,
  "branching_factor": 2,
  "node_count": 31,
  "leaf_count": 16,
  "parent_child_edges": [],
  "angle_sequence": [-28, 28],
  "scale_ratio": 0.72,
  "bilateral_symmetry": true
}
```

---

## 3.10 Style-safe transformations

### 安全

* botanical ink
* architectural blueprint
* woodcut
* pencil tree
* neon circuit
* stained glass
* low-poly tree
* paper cut
* mechanical branch structure
* monochrome etching

### 高风险

* dense foliage generation
* photorealistic forest
* painterly tree with overlapping leaves
* abstract expressionism
* foggy watercolor

原因是这些风格可能遮蔽真实分支数量和 parent-child 关系。

---

## 3.11 Style 后验证

验证：

* skeleton topology
* root count
* branch count
* leaf count
* recursion depth
* bilateral symmetry
* approximate branch angle
* parent-child connectivity

结构条件：

$$
\operatorname{TreeEditDistance}
\left(
T_{\mathrm{canonical}},
T_{\mathrm{style}}
\right)=0.
$$

允许几何误差：

$$
|\hat\theta-\theta|\leq 8^\circ.
$$

---

## 3.12 Hard negatives

* 同样是树，但深度少一层
* branching factor 从 2 改成 3
* 一个子树没有继续递归
* 左右分支不再对称
* 简单重复纹理冒充自相似结构
* 只保留“像植物”，但删除真实递归骨架

---

# 4. Family C：Symmetry Labs

## 4.1 核心抽象

Symmetry Labs 表达一个对象在变换群下的行为：

$$
g\cdot x,
\qquad
g\in G.
$$

主要变换：

* translation
* rotation
* reflection
* glide reflection
* inversion
* permutation
* scaling
* repeated group action

主要性质：

* identity
* inverse
* order
* orbit
* stabilizer
* chirality
* commutativity
* equivariance
* symmetry breaking

---

## 4.2 推荐子任务

### C1. Reflection

判断一个变换是否为镜像。

### C2. Rotation Order

判断图形具有几阶旋转对称。

### C3. Composition of Transforms

计算先旋转再反射的结果。

### C4. Inverse Transform

找出撤销某操作的变换。

### C5. Chirality

判断变换是否改变手性。

### C6. Symmetry Breaking

找出破坏整体对称的局部元素。

---

## 4.3 Latent schema

```json
{
  "family": "symmetry_lab",
  "task_type": "rotation",
  "base_shape": {
    "type": "asymmetric_arrow",
    "points": [
      [0, 0],
      [2, 0],
      [2, -1],
      [4, 1],
      [2, 3],
      [2, 2],
      [0, 2]
    ]
  },
  "group_action": {
    "type": "rotation",
    "angle_degrees": 90,
    "center": [0, 0]
  },
  "repetitions": 4,
  "structural_properties": {
    "rotation_order": 4,
    "reflection_symmetric": false,
    "chirality_preserved": true
  }
}
```

---

## 4.4 Story renderer

### 机械机关

> 一个机械箭头每次顺时针转动 90 度。连续操作四次后，它回到原始方向。

### 舞蹈动作

> 舞者每轮绕舞台中心旋转四分之一圈。完成四轮后，位置和朝向恢复。

### 房间布局

> 四个相同房间围绕中心旋转排列，每个房间相对前一个旋转 90 度。

### 密码操作

> 一个图形密码经过四次相同的旋转操作后恢复原状。

---

## 4.5 Mathematics renderer

旋转矩阵：

$$
R_{\theta}=
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}.
$$

当：

$$
\theta=\frac{\pi}{2},
$$

有：

$$
R_{\pi/2}^4=I.
$$

对于反射：

$$
F^2=I.
$$

对于组合：

$$
R_{\theta}F\neq FR_{\theta}
$$

可以用来生成非交换 hard negatives。

---

## 4.6 Program renderer

```python
import math

def rotate(points: list[tuple[float, float]], angle_deg: float):
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    return [
        (
            cos_a * x - sin_a * y,
            sin_a * x + cos_a * y,
        )
        for x, y in points
    ]
```

组合变换：

```python
def compose(*transforms):
    def combined(points):
        result = points
        for transform in reversed(transforms):
            result = transform(result)
        return result
    return combined
```

---

## 4.7 Drawing program DSL

```json
{
  "base_primitive": {
    "type": "polygon",
    "points": []
  },
  "transform_set": [
    {"type": "rotate", "angle": 0},
    {"type": "rotate", "angle": 90},
    {"type": "rotate", "angle": 180},
    {"type": "rotate", "angle": 270}
  ],
  "composition": "union",
  "canvas_center": [384, 384]
}
```

renderer 输出：

* canonical SVG
* individual orbit masks
* symmetry-axis metadata
* rotation-center metadata
* base-shape mask

---

## 4.8 Canonical image

使用明显不对称的 base primitive，避免旋转和反射无法区分。

推荐 base shapes：

* curved arrow
* asymmetric L-shape
* chiral spiral
* irregular polygon
* one-sided key shape
* asymmetric animal-like icon

Canonical image 应清楚显示：

* 旋转中心
* 每个 orbit member
* 对称轴或角度
* 原始 primitive

---

## 4.9 Structure signature

```json
{
  "group_type": "cyclic",
  "group_order": 4,
  "transformations": [
    "rotate_0",
    "rotate_90",
    "rotate_180",
    "rotate_270"
  ],
  "rotation_center": [384, 384],
  "reflection_axes": [],
  "chirality": "preserved",
  "orbit_size": 4
}
```

---

## 4.10 Style-safe transformations

### 安全

* ceramic tile
* stained glass
* woodcut
* geometric poster
* textile pattern
* blueprint
* neon sign
* paper cut
* ink ornament
* metal engraving

### 高风险

* freehand surrealism
* object replacement
* photorealistic conversion
* heavy perspective
* 3D scene transformation
* asymmetrical lighting that hides shape

---

## 4.11 Style 后验证

验证：

* base primitive 数量
* orbit size
* rotation center
* rotation angle
* reflection axis
* chirality
* symmetry order
* whether transformations compose correctly

旋转对称误差：

$$
E_{\mathrm{rot}}(k)=
\frac{
|I-R_{2\pi/k}(I)|
}{
|I|
}.
$$

要求：

$$
E_{\mathrm{rot}}(k)<\epsilon.
$$

同时测试错误阶数：

$$
E_{\mathrm{rot}}(k)
<
E_{\mathrm{rot}}(j),
\quad
j\neq k.
$$

---

## 4.12 Hard negatives

* 旋转 90° 被替换成反射
* 四阶旋转中缺少一个元素
* 一个元素角度错误
* 手性被翻转
* 整体近似对称，但有一个局部异常
* 图像风格相同，但 group action 不同

---

# 5. 三个 Family 的统一 DSL

建议底层统一成：

```json
{
  "entities": [],
  "relations": [],
  "transformations": [],
  "composition_rule": [],
  "global_properties": [],
  "query": {},
  "proof": {},
  "visual_grammar": {},
  "style_constraints": {}
}
```

三个 family 对应：

| Family            | Entities        | Relations      | Transformations | Global property       |
| ----------------- | --------------- | -------------- | --------------- | --------------------- |
| Graph Worlds      | nodes           | edges          | traversal/edit  | connectivity/path     |
| Recursive Gardens | recursive units | parent-child   | rewrite/expand  | depth/self-similarity |
| Symmetry Labs     | shape instances | orbit relation | group action    | symmetry/equivariance |

---

# 6. Style transfer 的安全协议

每张 canonical image 最多生成 8 个 stylized candidates。

流程：

$$
I_{\mathrm{canonical}}
\rightarrow
T_{\mathrm{style}}
\rightarrow
\{I_1,\ldots,I_8\}
\rightarrow
V_{\mathrm{structure}}
\rightarrow
I_{\mathrm{accepted}}.
$$

## 第一层：基础图像质量

检查：

* 尺寸
* 非空
* 对比度
* 无大面积遮挡
* 主体完整

## 第二层：结构验证

使用 family-specific verifier。

## 第三层：style difference

确保：

$$
d_{\mathrm{style}}
(I_{\mathrm{canonical}},I_{\mathrm{style}})
> \delta.
$$

避免所谓 stylized image 只是轻微颜色变化。

## 第四层：结构保持

确保：

$$
d_{\mathrm{struct}}
(I_{\mathrm{canonical}},I_{\mathrm{style}})
<\epsilon.
$$

## 第五层：人工抽检

每个 style family 抽检 100–200 张，确认自动 verifier 没有系统性漏洞。

---

# 7. 推荐的 V1 数据量

## Graph Worlds

* 8 graph templates
* 每个 template 1,500 instances
* 12,000 latent schemas
* 每个 schema 3 stories
* 每个 schema 2 program implementations
* 每个 schema 1 canonical image
* 每个 schema 3 accepted styles

合计约：

* 36K stories
* 24K programs
* 12K canonical images
* 36K stylized images

## Recursive Gardens

* 8 recursion templates
* 每个 template 1,000 instances
* 8,000 latent schemas
* 每个 schema 3 styles

约：

* 8K canonical images
* 24K stylized images

## Symmetry Labs

* 8 transform templates
* 每个 template 1,000 instances
* 8,000 latent schemas
* 每个 schema 3 styles

约：

* 8K canonical images
* 24K stylized images

## V1 总规模

$$
28{,}000\text{ latent schemas}
$$

约：

* 84K stories
* 56K executable programs
* 28K drawing programs
* 28K canonical images
* 84K stylized images

这是一个足够有规模、但仍可严格验证的第一版。

---

# 8. 数据切分

不能按 instance 随机切。

## Template OOD

测试集使用未在训练出现的结构模板。

## Size OOD

### Graph

训练：

$$
|V|\leq 8
$$

测试：

$$
9\leq |V|\leq 16.
$$

### Recursive

训练：

$$
d\leq 4
$$

测试：

$$
5\leq d\leq 7.
$$

### Symmetry

训练只见：

$$
C_2,C_4
$$

测试：

$$
C_3,C_6,D_3,D_4.
$$

## Style OOD

训练使用：

* blueprint
* pencil
* technical

测试使用：

* woodcut
* stained glass
* paper cut

## Story-domain OOD

训练使用：

* 城市
* 软件
* 植物

测试使用：

* 社会关系
* 生物系统
* 机械装置

## Composition OOD

训练只见单独属性：

* graph + reachability
* tree + recursion
* shape + rotation

测试加入：

* recursive graph
* symmetric tree
* graph with group-action structure

---

# 9. 每个样本的最终格式

```json
{
  "sample_id": "gw_reach_000001",
  "family": "graph_world",
  "latent_schema": {},
  "voices": {
    "story": [
      {"domain": "city", "text": "..."},
      {"domain": "software", "text": "..."}
    ],
    "math": {
      "notation": "...",
      "canonical_form": "...",
      "proof_or_solution": "..."
    },
    "program": [
      {
        "language": "python",
        "implementation": "bfs",
        "code": "...",
        "tests": []
      }
    ],
    "drawing_program": {
      "language": "svg_dsl",
      "spec": {}
    },
    "canonical_image": {
      "path": "...",
      "masks": {},
      "structure_signature": {}
    },
    "stylized_images": [
      {
        "style": "blueprint",
        "path": "...",
        "verification_score": 1.0
      }
    ]
  },
  "hard_negatives": [
    {
      "type": "edge_reversal",
      "modified_schema": {},
      "modified_image": "..."
    }
  ],
  "split_metadata": {
    "template_id": "...",
    "size_bucket": "...",
    "style_family": "...",
    "story_domain": "..."
  }
}
```

---

# 10. 最优开发顺序

## 第一步

先做 Graph Worlds。

原因：

* validator 最容易
* drawing program 最成熟
* style transfer 后最容易恢复拓扑
* 能最快验证整个 pipeline

## 第二步

加入 Recursive Gardens。

原因：

* 视觉效果强
* GEB story 最明显
* 能测试递归和层级

## 第三步

加入 Symmetry Labs。

原因：

* 数学抽象最强
* 最适合 style robustness
* 能制造高质量 hard negative

## 第四步

最后才做跨 family composition：

* symmetric recursive tree
* recursive graph rewriting
* graph orbit under transformations

---

# 11. V1 的核心成功标准

一个 family 可以进入数据集，必须同时满足：

$$
\text{Executable}
\land
\text{Mathematically canonical}
\land
\text{Visually structural}
\land
\text{Story-natural}
\land
\text{Style-verifiable}.
$$

一张 stylized image 可以入库，必须满足：

$$
\operatorname{StructureScore}\geq 0.98
$$

并且：

$$
\operatorname{StyleDistance}\geq \delta.
$$

V1 不追求艺术风格数量，而追求：

> **风格明显不同，但图的拓扑、树的层级或图形的对称群绝对不变。**

---

# 12. 数据集名称

推荐三个候选：

### One Structure, Six Voices

最符合整体思想。

### StructureVerse

更适合数据集品牌。

### GEB-6

最直接，但容易显得借书名营销。

最推荐正式名称：

> **StructureVerse: One Structure across Stories, Mathematics, Programs, Drawing Code, and Stylized Images**

V1 子集名称：

* StructureVerse-Graphs
* StructureVerse-Recursion
* StructureVerse-Symmetry
