import { pipeline, env } from "@huggingface/transformers";
import clustering from 'density-clustering';

// 使用Singleton模式实现向量计算pipeline的懒加载
class PipelineSingleton {
    static task = 'feature-extraction';
    static model = 'Xenova/bge-small-zh-v1.5';
    static instance = null;
    static device = null;

    static async getInstance(progress_callback = null) {
        if (!this.instance) {
            self.postMessage({ status: 'initiate' });

            // 检测WebGPU支持
            try {
                if (await env.backends.webgpu.isSupported()) {
                    this.device = 'webgpu';
                } else {
                    this.device = 'wasm';
                }
            } catch (e) {
                this.device = 'wasm';
            }

            // 创建pipeline
            this.instance = await pipeline(this.task, this.model, {
                progress_callback,
                device: this.device
            });

            // 通知前端当前使用的设备
            self.postMessage({ 
                status: 'ready',
                device: this.device
            });
        }
        return this.instance;
    }
}

// 向量计算缓存
const vectorCache = new Map();
let lastTexts = null;
let lastEmbeddings = null;

// 启动时就开始加载模型
PipelineSingleton.getInstance(x => {
    self.postMessage({
        status: 'loading',
        progress: x
    });
});

// 计算文本向量，使用缓存优化
async function computeEmbeddings(texts, extractor) {
    const startTime = performance.now();
    const embeddings = [];
    const textToIndexMap = new Map();
    
    // 创建唯一文本集合并建立映射
    const uniqueTexts = [...new Set(texts)];
    texts.forEach((text, index) => {
        if (!textToIndexMap.has(text)) {
            textToIndexMap.set(text, []);
        }
        textToIndexMap.get(text).push(index);
    });

    // 计算未缓存文本的向量
    let processedCount = 0;
    for (let i = 0; i < uniqueTexts.length; i++) {
        const text = uniqueTexts[i];
        const indices = textToIndexMap.get(text);
        
        // 更新进度
        processedCount += indices.length;
        const currentTime = performance.now();
        const elapsedSeconds = ((currentTime - startTime) / 1000).toFixed(1);
        const speed = (processedCount / (currentTime - startTime) * 1000).toFixed(1);

        self.postMessage({
            status: 'computing',
            progress: {
                current: processedCount,
                total: texts.length,
                elapsedSeconds,
                speed
            }
        });

        // 如果缓存中存在，直接使用缓存
        if (vectorCache.has(text)) {
            const cachedVector = vectorCache.get(text);
            indices.forEach(index => {
                embeddings[index] = cachedVector;
            });
            continue;
        }

        // 计算新向量
        const output = await extractor(text, {
            pooling: 'mean',
            normalize: true,
        });
        const vector = Array.from(output.data);
        
        // 保存到缓存
        vectorCache.set(text, vector);
        
        // 填充所有相同文本的位置
        indices.forEach(index => {
            embeddings[index] = vector;
        });
    }

    return {
        embeddings,
        vectorizationTime: performance.now() - startTime
    };
}

// 执行DBSCAN聚类
function performClustering(embeddings, epsilon, minPts) {
    const startTime = performance.now();
    const dbscan = new clustering.DBSCAN();
    const clusters = dbscan.run(embeddings, epsilon, minPts);
    return {
        clusters,
        noise: dbscan.noise,
        clusteringTime: performance.now() - startTime
    };
}

// 监听主线程消息
self.addEventListener('message', async (event) => {
    const { type, data } = event.data;

    if (type === 'compute_embeddings') {
        try {
            const texts = data.texts.filter(text => text.trim() !== '');
            const startTime = performance.now();
            
            // 检查是否只需要重新聚类
            let embeddings;
            let vectorizationTime;
            
            if (lastTexts && 
                lastEmbeddings && 
                texts.length === lastTexts.length && 
                texts.every((text, i) => text === lastTexts[i])) {
                // 文本没有变化，直接使用上次的向量结果
                embeddings = lastEmbeddings;
                vectorizationTime = 0;
                
                // 发送跳过计算的通知
                self.postMessage({
                    status: 'computing',
                    progress: {
                        current: texts.length,
                        total: texts.length,
                        elapsedSeconds: '0.0',
                        speed: '∞'
                    }
                });
            } else {
                // 文本有变化，需要计算向量
                const extractor = await PipelineSingleton.getInstance();
                const result = await computeEmbeddings(texts, extractor);
                embeddings = result.embeddings;
                vectorizationTime = result.vectorizationTime;
                
                // 保存本次结果
                lastTexts = [...texts];
                lastEmbeddings = [...embeddings];
            }

            // 发送开始聚类的状态
            self.postMessage({
                status: 'clustering'
            });

            // 执行聚类
            const { clusters, noise, clusteringTime } = performClustering(embeddings, data.epsilon, data.minPts);

            // 将聚类结果与原文本对应
            const results = clusters.map(cluster => ({
                size: cluster.length,
                texts: cluster.map(index => texts[index])
            }));

            // 按cluster大小降序排序
            results.sort((a, b) => b.size - a.size);

            // 获取噪声点
            const noiseTexts = noise.map(index => texts[index]);

            // 计算总耗时
            const totalTime = vectorizationTime + clusteringTime;

            // 发送结果回主线程
            self.postMessage({
                status: 'complete',
                results,
                noise: noiseTexts,
                performance: {
                    vectorizationTime: (vectorizationTime / 1000).toFixed(1),
                    clusteringTime: (clusteringTime / 1000).toFixed(1),
                    totalTime: (totalTime / 1000).toFixed(1),
                    averageSpeed: vectorizationTime === 0 ? '∞' : (texts.length / (vectorizationTime / 1000)).toFixed(1)
                }
            });
        } catch (error) {
            self.postMessage({
                status: 'error',
                error: error.message
            });
        }
    }
});
