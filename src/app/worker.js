import { pipeline } from "@huggingface/transformers";
import { dbscan } from './dbscan';

// 使用Singleton模式实现向量计算pipeline的懒加载
class PipelineSingleton {
    static task = 'feature-extraction';
    static model = 'Xenova/bge-small-zh-v1.5';
    static instance = null;
    static device = 'wasm';  // 默认使用 wasm
    static useGPU = false;

    static async getInstance(progress_callback = null) {
        if (!this.instance) {
            self.postMessage({ status: 'initiate' });

            // 检测WebGPU支持
            try {
                if ('gpu' in navigator) {
                    const adapter = await navigator.gpu.requestAdapter();
                    if (adapter) {
                        const device = await adapter.requestDevice();
                        if (device) {
                            this.device = 'webgpu';
                            this.useGPU = true;
                            console.log('WebGPU is supported and will be used for computations');
                        }
                    }
                }
            } catch (e) {
                console.log('WebGPU initialization failed:', e);
                this.device = 'wasm';
                this.useGPU = false;
            }

            try {
                // 配置选项
                const options = {
                    progress_callback,
                    device: this.device,
                    quantized: false,  // 禁用量化以使用fp32
                    session_options: {
                        logSeverityLevel: 3  // 设置日志级别为ERROR (0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal)
                    }
                };

                // 创建pipeline
                this.instance = await pipeline(this.task, this.model, options);

                // 通知前端当前使用的设备
                self.postMessage({ 
                    status: 'ready',
                    device: this.device,
                    useGPU: this.useGPU
                });
            } catch (error) {
                // 处理各种可能的错误
                let errorMessage = '初始化失败: ';
                
                if (error.message.includes('Failed to fetch')) {
                    errorMessage += '无法下载模型，请检查网络连接';
                } else if (error.name === 'TypeError') {
                    errorMessage += '浏览器版本过低，请使用最新版Chrome等现代浏览器';
                } else {
                    errorMessage += error.message;
                }
                
                throw new Error(errorMessage);
            }
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
}).catch(error => {
    self.postMessage({
        status: 'error',
        error: error.message
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

        try {
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
        } catch (error) {
            throw new Error(`处理文本时出错: ${error.message}`);
        }
    }

    return {
        embeddings,
        vectorizationTime: performance.now() - startTime
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

            // 将距离阈值转换为相似度阈值
            // epsilon 原来是距离阈值 [0,1]，现在需要转换为相似度阈值 [-1,1]
            // 例如：如果原来的距离阈值是 0.3，那么相似度阈值应该是 0.7
            const similarityThreshold = 1 - data.epsilon;

            // 执行聚类，使用 WebGPU（如果支持）
            const { clusters, noise } = await dbscan(embeddings, similarityThreshold, data.minPts, PipelineSingleton.useGPU);
            const clusteringTime = performance.now() - startTime - vectorizationTime;

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
