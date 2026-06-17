#!/usr/bin/env node
/**
 * Fix script: Add layer:xxx nodes to the nodes array and update layer edges
 * to use module:xxx IDs instead of layer:xxx IDs
 */
const fs = require('fs');

const graphPath = process.argv[2] || '/home/bluce/pytorch/.understand-anything/knowledge-graph.json';

let raw;
try {
  raw = fs.readFileSync(graphPath, 'utf-8');
} catch (e) {
  console.error(`Cannot read: ${e.message}`);
  process.exit(1);
}

let graph;
try {
  graph = JSON.parse(raw);
} catch (e) {
  console.error(`Cannot parse: ${e.message}`);
  process.exit(1);
}

// Check if layer:xxx nodes already exist
const existingNodeIds = new Set(graph.nodes.map(n => n.id));
const layerNodeIds = graph.layers.map(l => l.id);

// Find which layer IDs are NOT in the nodes array
const missingLayerNodes = layerNodeIds.filter(id => !existingNodeIds.has(id));

if (missingLayerNodes.length === 0) {
  console.log('All layer nodes already exist. No fix needed.');
  process.exit(0);
}

console.log(`Adding ${missingLayerNodes.length} missing layer nodes...`);

// Create a map of layer id -> layer data
const layerMap = {};
graph.layers.forEach(l => {
  layerMap[l.id] = l;
});

// Add missing layers as nodes
missingLayerNodes.forEach(layerId => {
  const layer = layerMap[layerId];
  const shortName = layerId.replace('layer:', '');
  const newNode = {
    id: layerId,
    type: 'module',
    name: shortName,
    summary: `Architectural layer: ${layer.name || shortName}. ${layer.description || ''}`,
    tags: ['architectural-layer'],
    complexity: 'complex'
  };
  graph.nodes.push(newNode);
});

console.log(`Total nodes: ${graph.nodes.length}`);

// Write back
try {
  fs.writeFileSync(graphPath, JSON.stringify(graph, null, 2), 'utf-8');
  console.log(`Fixed graph written to ${graphPath}`);
} catch (e) {
  console.error(`Cannot write: ${e.message}`);
  process.exit(1);
}
