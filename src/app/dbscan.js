// 计算两个向量之间的相似度（点积，因为向量已经归一化）
function vectorSimilarity(point1, point2) {
    let dotProduct = 0;
    for (let i = 0; i < point1.length; i++) {
        dotProduct += point1[i] * point2[i];
    }
    return dotProduct; // 返回点积（即相似度）
}

// 如果支持 WebGPU，使用 GPU 计算相似度矩阵
async function computeDistanceMatrix(points, useGPU = false) {
    if (useGPU && 'gpu' in navigator) {
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) throw new Error('No GPU adapter found');
            const device = await adapter.requestDevice();

            const numPoints = points.length;
            const vectorSize = points[0].length;
            const totalElements = numPoints * numPoints;

            const pointsBuffer = device.createBuffer({
                size: numPoints * vectorSize * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            });

            const resultBuffer = device.createBuffer({
                size: totalElements * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            });

            const computePipeline = device.createComputePipeline({
                layout: 'auto',
                compute: {
                    module: device.createShaderModule({
                        code: `
                            @group(0) @binding(0) var<storage, read> points: array<f32>;
                            @group(0) @binding(1) var<storage, read_write> similarities: array<f32>;

                            fn computeSimilarity(i: u32, j: u32, vectorSize: u32) -> f32 {
                                var dotProduct: f32 = 0.0;
                                for (var k: u32 = 0u; k < vectorSize; k = k + 1u) {
                                    dotProduct = dotProduct + points[i * vectorSize + k] * points[j * vectorSize + k];
                                }
                                return dotProduct;
                            }

                            @compute @workgroup_size(256)
                            fn main(@builtin(global_invocation_id) global_id: vec3u) {
                                let idx = global_id.x;
                                if (idx >= ${totalElements}u) {
                                    return;
                                }
                                let i = idx / ${numPoints}u;
                                let j = idx % ${numPoints}u;
                                similarities[idx] = computeSimilarity(i, j, ${vectorSize}u);
                            }
                        `
                    }),
                    entryPoint: 'main'
                }
            });

            device.queue.writeBuffer(pointsBuffer, 0, new Float32Array(points.flat()));

            const bindGroup = device.createBindGroup({
                layout: computePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: pointsBuffer } },
                    { binding: 1, resource: { buffer: resultBuffer } }
                ]
            });

            const commandEncoder = device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(computePipeline);
            passEncoder.setBindGroup(0, bindGroup);

            const workgroupSize = 256;
            const numWorkgroups = Math.ceil(totalElements / workgroupSize);
            passEncoder.dispatchWorkgroups(numWorkgroups);
            passEncoder.end();

            const readbackBuffer = device.createBuffer({
                size: totalElements * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });

            commandEncoder.copyBufferToBuffer(resultBuffer, 0, readbackBuffer, 0, totalElements * 4);
            device.queue.submit([commandEncoder.finish()]);

            await readbackBuffer.mapAsync(GPUMapMode.READ);
            const similarities = new Float32Array(readbackBuffer.getMappedRange());

            const similarityMatrix = Array.from({ length: numPoints }, (_, i) =>
                similarities.slice(i * numPoints, (i + 1) * numPoints)
            );

            readbackBuffer.unmap();

            return similarityMatrix;
        } catch (error) {
            console.log('GPU computation failed, falling back to CPU:', error);
            return computeDistanceMatrixCPU(points);
        }
    }
    return computeDistanceMatrixCPU(points);
}

// CPU 版本的相似度矩阵计算
function computeDistanceMatrixCPU(points) {
    const similarityMatrix = Array.from({ length: points.length }, () => Array(points.length).fill(0));
    for (let i = 0; i < points.length; i++) {
        for (let j = 0; j < points.length; j++) {
            similarityMatrix[i][j] = vectorSimilarity(points[i], points[j]);
        }
    }
    return similarityMatrix;
}

// DBSCAN 主算法
export async function dbscan(points, epsilon, minPts, useGPU = false) {
    const visited = new Set();
    const noise = new Set();
    const clusters = [];
    const assignments = new Array(points.length).fill(null);
    const similarityMatrix = await computeDistanceMatrix(points, useGPU);

    function getNeighborsFromMatrix(pointIndex) {
        const neighbors = [];
        for (let i = 0; i < points.length; i++) {
            if (similarityMatrix[pointIndex][i] >= epsilon) {
                neighbors.push(i);
            }
        }
        return neighbors;
    }

    function expandCluster(pointIndex, neighbors, clusterId) {
        assignments[pointIndex] = clusterId;

        const seeds = neighbors.slice();
        while (seeds.length > 0) {
            const currentPoint = seeds.shift();

            if (!visited.has(currentPoint)) {
                visited.add(currentPoint);
                const resultNeighbors = getNeighborsFromMatrix(currentPoint);

                if (resultNeighbors.length >= minPts) {
                    for (const n of resultNeighbors) {
                        if (!visited.has(n)) {
                            seeds.push(n);
                        }
                    }
                }
            }

            if (assignments[currentPoint] === null) {
                assignments[currentPoint] = clusterId;
            }
        }
    }

    let clusterId = 0;
    for (let i = 0; i < points.length; i++) {
        if (visited.has(i)) continue;

        visited.add(i);
        const neighbors = getNeighborsFromMatrix(i);

        if (neighbors.length < minPts) {
            noise.add(i);
            assignments[i] = -1;
        } else {
            expandCluster(i, neighbors, clusterId);
            clusterId++;
        }
    }

    for (let i = 0; i < clusterId; i++) {
        clusters[i] = [];
    }

    for (let i = 0; i < assignments.length; i++) {
        if (assignments[i] >= 0) {
            clusters[assignments[i]].push(i);
        }
    }

    return {
        clusters,
        noise: Array.from(noise)
    };
}
