# Text2VectorCluster-Web-ZH

基于纯浏览器的中文短文本聚类工具，无需后端服务器，即开即用。

## 项目特点

- 🚀 **纯浏览器实现**：完全在浏览器端运行，无需后端服务器
- 💪 **GPU加速支持**：自动检测并使用WebGPU进行向量计算加速
- 🔍 **高质量文本向量**：使用BGE-small-zh模型进行文本向量化
- 📊 **智能聚类算法**：采用DBSCAN聚类算法，自动发现文本簇
- 📈 **结果可视化**：直观展示聚类结果，支持Excel导出
- ⚡ **高性能计算**：基于transformers.js实现高效的向量计算

## 使用方法

1. 访问在线演示页面
2. 在左侧输入框中输入要聚类的文本（每行一句）
3. 调整DBSCAN参数（可选）：
   - Epsilon：控制聚类的紧密程度
   - MinPts：最小簇大小
4. 点击"开始聚类"按钮
5. 在右侧查看聚类结果
6. 点击"下载结果"导出Excel文件

## 技术实现

- 前端框架：Next.js 13
- UI组件：Tailwind CSS
- 向量计算：transformers.js
- 文本向量模型：BGE-small-zh
- 聚类算法：DBSCAN

## 开发部署

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 构建生产版本
npm run build
```

## 致谢

- [Transformers.js](https://huggingface.co/docs/transformers.js/index) - 在浏览器中运行Transformer模型
- [BGE-small-zh-v1.5](https://huggingface.co/Xenova/bge-small-zh-v1.5) - 高质量的中文文本向量模型

## License

MIT License
