#!/usr/bin/env node
/**
 * Knowledge Graph Validation Script
 * Usage: node ua-graph-validate.js <graph-file-path> <output-file-path>
 */
const fs = require('fs');
const path = require('path');

// ── Configuration ────────────────────────────────────────────────────────────

const VALID_NODE_TYPES = [
  'file', 'function', 'class', 'module', 'concept', 'config', 'document',
  'service', 'table', 'endpoint', 'pipeline', 'schema', 'resource',
  'domain', 'flow', 'step'
];

const VALID_EDGE_TYPES = [
  'imports', 'exports', 'contains', 'inherits', 'implements', 'calls',
  'subscribes', 'publishes', 'middleware', 'reads_from', 'writes_to',
  'transforms', 'validates', 'depends_on', 'tested_by', 'configures',
  'related', 'similar_to', 'deploys', 'serves', 'migrates', 'documents',
  'provisions', 'routes', 'defines_schema', 'triggers', 'contains_flow',
  'flow_step', 'cross_domain'
];

const VALID_DIRECTIONS = ['forward', 'backward', 'bidirectional'];
const VALID_COMPLEXITIES = ['simple', 'moderate', 'complex'];

// File-level node types (must appear in a layer)
const FILE_LEVEL_TYPES = ['file', 'config', 'document', 'service', 'pipeline', 'table', 'schema', 'resource', 'endpoint'];

// Domain-level node types (for domain graph detection)
const DOMAIN_NODE_TYPES = ['domain', 'flow', 'step'];

// ── Main ─────────────────────────────────────────────────────────────────────

function main() {
  const graphPath = process.argv[2];
  const outputPath = process.argv[3];

  if (!graphPath || !outputPath) {
    console.error('Usage: node ua-graph-validate.js <graph-file-path> <output-file-path>');
    process.exit(1);
  }

  // Read graph
  let raw;
  try {
    raw = fs.readFileSync(graphPath, 'utf-8');
  } catch (e) {
    console.error(`Cannot read graph file: ${e.message}`);
    process.exit(1);
  }

  let graph;
  try {
    graph = JSON.parse(raw);
  } catch (e) {
    console.error(`Cannot parse graph JSON: ${e.message}`);
    process.exit(1);
  }

  // Read tour from separate file if available
  const graphDir = path.dirname(graphPath);
  const intermediateDir = path.join(path.dirname(path.dirname(graphDir)), 'intermediate');
  const tourPath = path.join(intermediateDir, 'tour.json');
  let tour = null;
  try {
    if (fs.existsSync(tourPath)) {
      const tourData = JSON.parse(fs.readFileSync(tourPath, 'utf-8'));
      if (Array.isArray(tourData)) {
        tour = tourData;
      } else if (tourData && Array.isArray(tourData.steps)) {
        tour = tourData.steps;
      }
    }
  } catch (e) {
    // Tour file may not exist
  }

  const issues = [];
  const warnings = [];

  // Collect stats
  const nodeTypeCounts = {};
  const edgeTypeCounts = {};

  // ── Check 5: Uniqueness ──────────────────────────────────────────────────
  const nodeIdSet = new Set();
  const duplicateIds = [];
  const nodeIdIndex = {}; // id -> index in nodes array

  (graph.nodes || []).forEach((node, idx) => {
    const id = node && node.id;
    if (id !== undefined && id !== null) {
      if (nodeIdSet.has(id)) {
        duplicateIds.push({ id, index: idx });
      } else {
        nodeIdSet.add(id);
      }
      nodeIdIndex[id] = idx;
    }
  });

  if (duplicateIds.length > 0) {
    duplicateIds.forEach(d => {
      issues.push(`Duplicate node ID '${d.id}' found at node index ${d.index}`);
    });
  }

  // Rebuild the node set (use the last occurrence for each ID)
  // Actually, let's use a proper set of unique IDs
  const allNodeIds = new Set();
  (graph.nodes || []).forEach(node => {
    if (node && node.id) allNodeIds.add(node.id);
  });

  // ── Check 1: Schema Validation - Nodes ───────────────────────────────────
  const nodeIdPrefixMap = {};

  (graph.nodes || []).forEach((node, idx) => {
    const id = node.id || '';
    const type = node.type || '';
    const name = node.name || '';
    const summary = node.summary || '';
    const tags = node.tags || [];
    const complexity = node.complexity || '';

    // Missing required fields
    if (!id) {
      issues.push(`Node at index ${idx} has missing or empty 'id'`);
    }
    if (!type) {
      issues.push(`Node ${id || idx} has missing or empty 'type'`);
    } else if (!VALID_NODE_TYPES.includes(type)) {
      issues.push(`Node '${id}' has invalid type '${type}'. Valid types: ${VALID_NODE_TYPES.join(', ')}`);
    }
    if (!name) {
      issues.push(`Node '${id}' has missing or empty 'name'`);
    }
    if (!summary) {
      issues.push(`Node '${id}' has missing or empty 'summary'`);
    }
    if (!tags || !Array.isArray(tags)) {
      issues.push(`Node '${id}' has missing or invalid 'tags' (must be array)`);
    } else if (tags.length === 0) {
      issues.push(`Node '${id}' has empty 'tags' array (must have at least 1 tag)`);
    } else {
      // Check tags are lowercase and hyphenated
      tags.forEach((tag, ti) => {
        if (typeof tag === 'string' && tag !== tag.toLowerCase()) {
          issues.push(`Node '${id}' tag at index ${ti} ('${tag}') is not lowercase`);
        }
        if (typeof tag === 'string' && /[^a-z0-9-]/.test(tag)) {
          warnings.push(`Node '${id}' tag at index ${ti} ('${tag}') is not properly hyphenated`);
        }
      });
    }
    if (!complexity) {
      issues.push(`Node '${id}' has missing or empty 'complexity'`);
    } else if (!VALID_COMPLEXITIES.includes(complexity)) {
      issues.push(`Node '${id}' has invalid complexity '${complexity}'`);
    }

    // Track node type for stats
    if (type) {
      nodeTypeCounts[type] = (nodeTypeCounts[type] || 0) + 1;
    }

    // Check ID prefix consistency (Check 9)
    if (id && type) {
      const expectedPrefix = `${type}:`;
      if (!id.startsWith(expectedPrefix)) {
        warnings.push(`Node '${id}' has type '${type}' but ID does not start with '${expectedPrefix}'`);
      }
      nodeIdPrefixMap[id] = type;
    }
  });

  // ── Check 1: Schema Validation - Edges ───────────────────────────────────
  (graph.edges || []).forEach((edge, idx) => {
    const source = edge.source || '';
    const target = edge.target || '';
    const type = edge.type || '';
    const direction = edge.direction || '';
    const weight = edge.weight;

    // Missing required fields
    if (!source) {
      issues.push(`Edge at index ${idx} has missing or empty 'source'`);
    }
    if (!target) {
      issues.push(`Edge at index ${idx} has missing or empty 'target'`);
    }
    if (!type) {
      issues.push(`Edge at index ${idx} (${source} -> ${target}) has missing or empty 'type'`);
    } else if (!VALID_EDGE_TYPES.includes(type)) {
      issues.push(`Edge at index ${idx} (${source} -> ${target}) has invalid type '${type}'`);
    }
    if (!direction) {
      issues.push(`Edge at index ${idx} (${source} -> ${target}) has missing or empty 'direction'`);
    } else if (!VALID_DIRECTIONS.includes(direction)) {
      issues.push(`Edge at index ${idx} has invalid direction '${direction}'`);
    }
    if (weight === undefined || weight === null || typeof weight !== 'number') {
      issues.push(`Edge at index ${idx} (${source} -> ${target}) has missing or non-numeric weight`);
    } else if (weight < 0 || weight > 1) {
      issues.push(`Edge at index ${idx} (${source} -> ${target}) has weight ${weight} outside [0, 1]`);
    }

    // Track edge type for stats
    if (type) {
      edgeTypeCounts[type] = (edgeTypeCounts[type] || 0) + 1;
    }
  });

  // ── Check 2: Referential Integrity ───────────────────────────────────────
  (graph.edges || []).forEach((edge, idx) => {
    const source = edge.source;
    const target = edge.target;

    // Skip layer:xxx references for referential integrity (they are inter-layer edges)
    const isLayerRef = (id) => id && id.startsWith('layer:');

    if (source && !isLayerRef(source) && !allNodeIds.has(source)) {
      issues.push(`Edge at index ${idx} references non-existent source node '${source}'`);
    }
    if (target && !isLayerRef(target) && !allNodeIds.has(target)) {
      issues.push(`Edge at index ${idx} references non-existent target node '${target}'`);
    }
  });

  // Check nodeIds in layers
  (graph.layers || []).forEach((layer, li) => {
    const layerId = layer.id || `layer_${li}`;
    const nodeIds = layer.nodeIds || [];
    nodeIds.forEach((nid, ni) => {
      if (!allNodeIds.has(nid)) {
        issues.push(`Layer '${layerId}' nodeIds entry at index ${ni} references non-existent node '${nid}'`);
      }
    });
  });

  // Check nodeIds in tour
  const tourSteps = graph.tour || (tour || []);
  if (Array.isArray(tourSteps)) {
    tourSteps.forEach((step, si) => {
      const stepId = step.id || `step_${si}`;
      const nodeIds = step.nodeIds || [];
      nodeIds.forEach((nid, ni) => {
        if (!allNodeIds.has(nid)) {
          issues.push(`Tour step '${stepId}' nodeIds entry at index ${ni} references non-existent node '${nid}'`);
        }
      });
    });
  }

  // ── Check 3: Completeness ────────────────────────────────────────────────
  const hasDomainNodes = (graph.nodes || []).some(n =>
    DOMAIN_NODE_TYPES.includes(n.type)
  );

  if ((graph.nodes || []).length === 0) {
    issues.push('Graph has zero nodes');
  }
  if ((graph.edges || []).length === 0) {
    issues.push('Graph has zero edges');
  }

  const layerCount = (graph.layers || []).length;
  const tourStepCount = Array.isArray(tourSteps) ? tourSteps.length : 0;

  if (layerCount === 0) {
    if (hasDomainNodes) {
      warnings.push('Graph has zero layers (domain graph)');
    } else {
      issues.push('Graph has zero layers');
    }
  }
  if (tourStepCount === 0) {
    if (hasDomainNodes) {
      warnings.push('Graph has zero tour steps (domain graph)');
    } else {
      issues.push('Graph has zero tour steps');
    }
  }

  // ── Check 4: Layer Coverage ──────────────────────────────────────────────
  if (!hasDomainNodes || layerCount > 0) {
    // Build a map: nodeId -> list of layer indices it appears in
    const nodeLayerCounts = {};

    (graph.layers || []).forEach((layer, li) => {
      const nodeIds = layer.nodeIds || [];
      if (nodeIds.length === 0) {
        issues.push(`Layer '${layer.id || li}' has empty nodeIds array`);
      }
      nodeIds.forEach(nid => {
        if (!nodeLayerCounts[nid]) nodeLayerCounts[nid] = [];
        nodeLayerCounts[nid].push(li);
      });
    });

    // Check file-level nodes are in exactly one layer
    (graph.nodes || []).forEach((node, idx) => {
      if (node && FILE_LEVEL_TYPES.includes(node.type)) {
        const coverage = nodeLayerCounts[node.id] || [];
        if (coverage.length === 0) {
          issues.push(`File-level node '${node.id}' (type: ${node.type}) is not assigned to any layer`);
        } else if (coverage.length > 1) {
          issues.push(`File-level node '${node.id}' appears in multiple layers: ${coverage.join(', ')}`);
        }
      }
    });
  }

  // ── Check 6: Tour Validation ─────────────────────────────────────────────
  if (Array.isArray(tourSteps) && tourSteps.length > 0) {
    // Check sequential order starting from 1
    const orders = tourSteps.map((s, i) => ({ order: s.order, index: i })).filter(o => o.order !== undefined);
    if (orders.length > 0) {
      const sortedOrders = [...orders].sort((a, b) => a.order - b.order);
      if (sortedOrders[0].order !== 1) {
        warnings.push(`Tour steps do not start at order 1 (first order is ${sortedOrders[0].order})`);
      }
      // Check for gaps/duplicates
      const orderSet = new Set();
      orders.forEach(({ order, index }) => {
        if (orderSet.has(order)) {
          warnings.push(`Duplicate tour order value ${order} at step index ${index}`);
        }
        orderSet.add(order);
      });
      for (let i = 1; i <= sortedOrders.length; i++) {
        if (!orderSet.has(i)) {
          warnings.push(`Tour is missing order value ${i}`);
        }
      }
    }
    // Each step has at least 1 nodeIds entry
    tourSteps.forEach((step, si) => {
      const nodeIds = step.nodeIds || [];
      if (nodeIds.length === 0) {
        warnings.push(`Tour step '${step.id || si}' has empty nodeIds array`);
      }
    });
    // Check tour step count (5-15)
    if (tourStepCount < 5) {
      warnings.push(`Tour has only ${tourStepCount} steps (recommended: 5-15)`);
    } else if (tourStepCount > 15) {
      warnings.push(`Tour has ${tourStepCount} steps (recommended: 5-15)`);
    }
  }

  // ── Check 7: Quality Checks ──────────────────────────────────────────────
  // Self-referencing edges
  (graph.edges || []).forEach((edge, idx) => {
    if (edge.source && edge.target && edge.source === edge.target) {
      warnings.push(`Edge at index ${idx} is self-referencing (${edge.source} -> ${edge.target})`);
    }
  });

  // Short/generic summaries
  const summaryIssues = [];
  (graph.nodes || []).forEach((node) => {
    if (!node || !node.summary) return;
    const name = node.name || '';
    const id = node.id || '';
    const summary = node.summary;
    const fileName = name || id.split(':')[1] || '';

    // Check if summary just restates the filename
    // Pattern: "filename — ..." or just "filename"
    const summaryWithoutPrefix = summary.replace(/^[^—]+—\s*/, '').trim();
    if (!summaryWithoutPrefix || summaryWithoutPrefix.length < 10) {
      warnings.push(`Node '${id}' has a generic or short summary: "${summary.substring(0, 80)}"`);
      return;
    }
    // Check if summary just says "...中的函数" or "XXX文件" with minimal info
    if (/的[^。]*$/.test(summary) && summary.length < 30) {
      warnings.push(`Node '${id}' has a minimal summary: "${summary}"`);
    }
  });

  // Orphan nodes (no edges connecting to or from them)
  const nodeEdgeCounts = {};
  (graph.nodes || []).forEach(n => {
    if (n && n.id) nodeEdgeCounts[n.id] = 0;
  });
  (graph.edges || []).forEach(edge => {
    if (edge.source && nodeEdgeCounts[edge.source] !== undefined) {
      nodeEdgeCounts[edge.source]++;
    }
    if (edge.target && nodeEdgeCounts[edge.target] !== undefined) {
      nodeEdgeCounts[edge.target]++;
    }
  });

  // Collect orphan stats per type
  const orphanCounts = {};
  (graph.nodes || []).forEach(node => {
    if (node && node.id && nodeEdgeCounts[node.id] === 0) {
      const type = node.type || 'unknown';
      orphanCounts[type] = (orphanCounts[type] || 0) + 1;
    }
  });

  Object.entries(orphanCounts).forEach(([type, count]) => {
    warnings.push(`${count} node(s) of type '${type}' have no edges connecting to or from them`);
  });

  // ── Check 8: Non-Code Node Quality Checks ────────────────────────────────
  const nodeEdgeMap = {};
  (graph.nodes || []).forEach(n => {
    if (n && n.id) {
      nodeEdgeMap[n.id] = { incoming: [], outgoing: [] };
    }
  });
  (graph.edges || []).forEach(edge => {
    if (edge.source && nodeEdgeMap[edge.source]) {
      nodeEdgeMap[edge.source].outgoing.push(edge);
    }
    if (edge.target && nodeEdgeMap[edge.target]) {
      nodeEdgeMap[edge.target].incoming.push(edge);
    }
  });

  (graph.nodes || []).forEach(node => {
    if (!node || !node.id) return;
    const id = node.id;
    const type = node.type;
    const edges = nodeEdgeMap[id];
    if (!edges) return;
    const allEdges = [...edges.incoming, ...edges.outgoing];
    const edgeTypes = allEdges.map(e => e.type);

    if (type === 'document' && !edgeTypes.includes('documents')) {
      // Only warn if node looks like a README/doc file
      const name = node.name || '';
      if (name.toLowerCase().startsWith('readme') || name.endsWith('.md') || name.endsWith('.rst')) {
        warnings.push(`Document node '${id}' has no 'documents' edges`);
      }
    }
    if (type === 'service' && !edgeTypes.some(t => t === 'deploys' || t === 'depends_on')) {
      warnings.push(`Service node '${id}' has no 'deploys' or 'depends_on' edges`);
    }
    if (type === 'pipeline' && !edgeTypes.includes('triggers')) {
      warnings.push(`Pipeline node '${id}' has no 'triggers' edges`);
    }
    if (type === 'table' && !edgeTypes.some(t => t === 'migrates' || t === 'defines_schema')) {
      warnings.push(`Table node '${id}' has no 'migrates' or 'defines_schema' edges`);
    }
    if (type === 'schema' && !edgeTypes.includes('defines_schema')) {
      warnings.push(`Schema node '${id}' has no 'defines_schema' edges`);
    }
    if (type === 'domain' && !edgeTypes.includes('contains_flow')) {
      warnings.push(`Domain node '${id}' has no 'contains_flow' edges`);
    }
    if (type === 'flow' && !edgeTypes.includes('flow_step')) {
      warnings.push(`Flow node '${id}' has no 'flow_step' edges`);
    }
  });

  // ── Check 9: Node Type / ID Prefix Consistency ───────────────────────────
  // Already checked inline in Check 1

  // ── Assemble Stats ───────────────────────────────────────────────────────
  const stats = {
    totalNodes: (graph.nodes || []).length,
    totalEdges: (graph.edges || []).length,
    totalLayers: (graph.layers || []).length,
    tourSteps: tourStepCount,
    nodeTypes: nodeTypeCounts,
    edgeTypes: edgeTypeCounts
  };

  // ── Output ────────────────────────────────────────────────────────────────
  const result = {
    scriptCompleted: true,
    issues: issues,
    warnings: warnings,
    stats: stats
  };

  try {
    const outDir = path.dirname(outputPath);
    if (!fs.existsSync(outDir)) {
      fs.mkdirSync(outDir, { recursive: true });
    }
    fs.writeFileSync(outputPath, JSON.stringify(result, null, 2), 'utf-8');
    console.log(`Validation results written to ${outputPath}`);
    console.log(`Issues: ${issues.length}, Warnings: ${warnings.length}`);
  } catch (e) {
    console.error(`Cannot write output file: ${e.message}`);
    process.exit(1);
  }
}

main();
