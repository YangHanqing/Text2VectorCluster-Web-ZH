'use client'

import { useState, useEffect, useRef } from 'react'
import * as XLSX from 'xlsx'

export default function Home() {
  const [texts, setTexts] = useState('');
  const [modelStatus, setModelStatus] = useState('loading'); // loading, ready
  const [device, setDevice] = useState(null); // webgpu, wasm
  const [clusterStatus, setClusterStatus] = useState(null); // null, computing, clustering, complete
  const [computeProgress, setComputeProgress] = useState(null); // { current, total, elapsedSeconds, speed }
  const [performance, setPerformance] = useState(null);
  const [results, setResults] = useState(null);
  const [epsilon, setEpsilon] = useState(0.15);
  const [minPts, setMinPts] = useState(2);
  
  const worker = useRef(null);

  useEffect(() => {
    if (!worker.current) {
      worker.current = new Worker(new URL('./worker.js', import.meta.url), {
        type: 'module'
      });
    }

    const onMessageReceived = (e) => {
      switch (e.data.status) {
        case 'initiate':
          setModelStatus('loading');
          break;
        case 'ready':
          setModelStatus('ready');
          setDevice(e.data.device);
          setClusterStatus(null);
          setComputeProgress(null);
          setPerformance(null);
          break;
        case 'computing':
          setClusterStatus('computing');
          setComputeProgress(e.data.progress);
          break;
        case 'clustering':
          setClusterStatus('clustering');
          setComputeProgress(null);
          break;
        case 'complete':
          setClusterStatus('complete');
          setComputeProgress(null);
          setResults(e.data);
          setPerformance(e.data.performance);
          break;
        case 'error':
          setClusterStatus('error');
          setComputeProgress(null);
          console.error(e.data.error);
          break;
      }
    };

    worker.current.addEventListener('message', onMessageReceived);
    return () => worker.current.removeEventListener('message', onMessageReceived);
  }, []);

  const handleCluster = () => {
    if (!texts.trim()) return;
    
    setClusterStatus('computing');
    const textArray = texts.split('\n').filter(text => text.trim() !== '');
    
    worker.current.postMessage({
      type: 'compute_embeddings',
      data: {
        texts: textArray,
        epsilon,
        minPts
      }
    });
  };

  const handleDownload = () => {
    if (!results) return;

    // 准备Excel数据
    const excelData = [];
    
    // 添加表头
    excelData.push(['query', 'cluster_id']);
    
    // 添加正常聚类结果
    results.results.forEach((cluster, clusterIdx) => {
      cluster.texts.forEach(text => {
        excelData.push([text, clusterIdx]);
      });
    });
    
    // 添加噪声点
    results.noise.forEach(text => {
      excelData.push([text, -1]);
    });

    // 创建工作簿
    const wb = XLSX.utils.book_new();
    const ws = XLSX.utils.aoa_to_sheet(excelData);
    XLSX.utils.book_append_sheet(wb, ws, 'Clustering Results');

    // 生成文件名：clustered_results_YYYYMMDD_HHMMSS.xlsx
    const now = new Date();
    const timestamp = now.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    }).replace(/[/:\s]/g, '').replace(',', '_');
    
    // 下载文件
    XLSX.writeFile(wb, `clustered_results_${timestamp}.xlsx`);
  };

  const loadTestData = async () => {
    try {
      const response = await fetch('/test-data.txt');
      const data = await response.text();
      setTexts(data);
    } catch (error) {
      console.error('加载测试数据失败:', error);
    }
  };

  return (
    <main className="flex min-h-screen bg-gray-50">
      {/* 左侧面板 */}
      <div className="w-1/2 p-6 bg-white shadow-lg">
        <h1 className="text-3xl font-bold mb-2">文本聚类分析</h1>
        <h2 className="text-lg text-gray-600 mb-6">BGE-small-zh + DBSCAN聚类算法</h2>
        
        <div className="mb-4">
          <div className="text-sm font-medium mb-1">模型状态</div>
          <div className="flex items-center gap-2">
            <span className={`text-sm ${modelStatus === 'ready' ? 'text-green-600' : 'text-orange-500'}`}>
              {modelStatus === 'ready' ? '✓ 就绪' : '⟳ 加载中...'}
            </span>
            {device && (
              <span className={`
                text-xs px-2 py-0.5 rounded-full
                ${device === 'webgpu' ? 'bg-green-100 text-green-800' : 'bg-blue-100 text-blue-800'}
              `}>
                {device === 'webgpu' ? 'GPU' : 'WASM'}
              </span>
            )}
          </div>
        </div>

        <div className="mb-4">
          <div className="flex justify-between items-center mb-2">
            <label className="block text-sm font-medium">输入文本（每行一句）</label>
            <button
              onClick={loadTestData}
              className="text-xs px-2 py-1 text-gray-600 hover:text-blue-600 hover:bg-blue-50 rounded transition-colors"
            >
              使用测试数据
            </button>
          </div>
          <textarea
            className="w-full h-48 p-3 border rounded-lg shadow-inner bg-gray-50 focus:bg-white focus:ring-2 focus:ring-blue-200 focus:outline-none"
            value={texts}
            onChange={(e) => setTexts(e.target.value)}
            placeholder="请输入要聚类的文本，每行一句..."
          />
        </div>

        <div className="mb-6">
          <label className="block text-sm font-medium mb-2">DBSCAN参数</label>
          <div className="flex gap-4">
            <div>
              <label className="block text-xs text-gray-600 mb-1">Epsilon</label>
              <input
                type="number"
                className="w-32 p-2 border rounded-lg shadow-sm focus:ring-2 focus:ring-blue-200 focus:outline-none"
                value={epsilon}
                onChange={(e) => setEpsilon(parseFloat(e.target.value))}
                step="0.1"
                min="0"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-600 mb-1">MinPts</label>
              <input
                type="number"
                className="w-32 p-2 border rounded-lg shadow-sm focus:ring-2 focus:ring-blue-200 focus:outline-none"
                value={minPts}
                onChange={(e) => setMinPts(parseInt(e.target.value))}
                min="1"
              />
            </div>
          </div>
        </div>

        <button
          className="w-full py-3 px-4 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed shadow-sm transition-colors"
          onClick={handleCluster}
          disabled={modelStatus !== 'ready' || clusterStatus === 'computing' || clusterStatus === 'clustering'}
        >
          {clusterStatus === 'computing' ? '计算向量中...' : 
           clusterStatus === 'clustering' ? '聚类中...' : 
           '开始聚类'}
        </button>

        {clusterStatus && (
          <div className="mt-4 space-y-2">
            <div className="text-sm">
              <div className="font-medium mb-1">处理状态</div>
              <div className={`
                ${clusterStatus === 'complete' ? 'text-green-600' : 
                  clusterStatus === 'error' ? 'text-red-600' : 
                  'text-blue-500'}
              `}>
                {clusterStatus === 'computing' && computeProgress && (
                  <div className="flex items-center gap-2">
                    <div className="flex items-center">
                      <span>⟳ 计算向量中 (</span>
                      <span className="font-mono w-16 text-right">{computeProgress.current}</span>
                      <span>/</span>
                      <span className="font-mono w-16 text-right">{computeProgress.total}</span>
                      <span>)</span>
                    </div>
                    <span className="text-xs text-gray-500 whitespace-nowrap">
                      <span className="font-mono w-8 inline-block text-right">{computeProgress.elapsedSeconds}</span>秒 · 
                      <span className="font-mono w-16 inline-block text-right">{computeProgress.speed}</span>条/秒
                    </span>
                  </div>
                )}
                {clusterStatus === 'clustering' && '⟳ 聚类中...'}
                {clusterStatus === 'complete' && '✓ 聚类完成'}
                {clusterStatus === 'error' && '× 处理出错'}
              </div>
            </div>

            {performance && (
              <div className="text-sm bg-gray-50 p-3 rounded-lg">
                <div className="font-medium mb-2">性能统计</div>
                <div className="space-y-1 text-gray-600">
                  <div>向量计算: {performance.vectorizationTime}秒 ({performance.averageSpeed}条/秒)</div>
                  <div>聚类计算: {performance.clusteringTime}秒</div>
                  <div>总耗时: {performance.totalTime}秒</div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* 右侧面板 */}
      <div className="w-1/2 p-6 relative">
        <div className="absolute inset-0 p-6 overflow-y-auto">
          <div className="sticky top-0 bg-gray-50 py-2">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">聚类结果</h2>
              {results && (
                <button
                  onClick={handleDownload}
                  className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
                >
                  下载结果
                </button>
              )}
            </div>
          </div>
          
          {results && (
            <div className="space-y-4">
              {results.results.map((cluster, idx) => (
                <div key={idx} className="border rounded-lg p-4 bg-white shadow-sm hover:shadow-md transition-shadow">
                  <h3 className="font-medium mb-2 text-blue-600">
                    簇 #{idx + 1} - {cluster.size} 条文本
                  </h3>
                  <ul className="list-disc pl-5 space-y-1">
                    {cluster.texts.map((text, textIdx) => (
                      <li key={textIdx} className="text-sm text-gray-700">{text}</li>
                    ))}
                  </ul>
                </div>
              ))}

              {results.noise.length > 0 && (
                <div className="border rounded-lg p-4 bg-gray-100 shadow-sm">
                  <h3 className="font-medium mb-2 text-gray-600">
                    噪声点 - {results.noise.length} 条文本
                  </h3>
                  <ul className="list-disc pl-5 space-y-1">
                    {results.noise.map((text, idx) => (
                      <li key={idx} className="text-sm text-gray-600">{text}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </main>
  )
}
