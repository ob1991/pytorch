const fs = require('fs');

// Read input
const inputPath = process.argv[2];
const outputPath = process.argv[3];
const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
const { fileNodes, importEdges, allEdges } = input;

// A. Directory Grouping - group by first directory segment
// Find common prefix
const paths = fileNodes.map(n => n.filePath || n.id.replace(/^(file|config|document|service|pipeline|table|schema|resource|endpoint):/, ''));
const firstDirs = paths.map(p => p.split('/')[0]);
const commonPrefix = firstDirs.every(d => d === firstDirs[0]) ? firstDirs[0] + '/' : '';

function getGroupKey(filePath) {
  const p = filePath.replace(/^(file|config|document|service|pipeline|table|schema|resource|endpoint):/, '');
  if (commonPrefix && p.startsWith(commonPrefix)) {
    const rest = p.slice(commonPrefix.length);
    const parts = rest.split('/');
    return parts.length > 1 ? parts[0] : 'root';
  }
  const parts = p.split('/');
  return parts.length > 1 ? parts[0] : 'root';
}

const directoryGroups = {};
const nodeToGroup = {};
for (const n of fileNodes) {
  const key = getGroupKey(n.filePath || n.id);
  if (!directoryGroups[key]) directoryGroups[key] = [];
  directoryGroups[key].push(n.id);
  nodeToGroup[n.id] = key;
}

// B. Node Type Grouping
const nodeTypeGroups = {};
for (const n of fileNodes) {
  if (!nodeTypeGroups[n.type]) nodeTypeGroups[n.type] = [];
  nodeTypeGroups[n.type].push(n.id);
}

// C. Import Adjacency Matrix
const fanIn = {}, fanOut = {};
for (const n of fileNodes) {
  fanIn[n.id] = 0;
  fanOut[n.id] = 0;
}
for (const e of importEdges) {
  fanOut[e.source] = (fanOut[e.source] || 0) + 1;
  fanIn[e.target] = (fanIn[e.target] || 0) + 1;
}

// D. Cross-Category Dependency Analysis
const crossCategoryEdges = {};
for (const e of allEdges) {
  const srcNode = fileNodes.find(n => n.id === e.source);
  const tgtNode = fileNodes.find(n => n.id === e.target);
  if (!srcNode || !tgtNode) continue;
  const key = `${srcNode.type}->${tgtNode.type} (${e.type})`;
  crossCategoryEdges[key] = (crossCategoryEdges[key] || 0) + 1;
}

const crossCategoryList = Object.entries(crossCategoryEdges)
  .map(([k, count]) => {
    const m = k.match(/(\w+)->(\w+) \((\w+)\)/);
    return m ? { fromType: m[1], toType: m[2], edgeType: m[3], count } : null;
  })
  .filter(Boolean)
  .sort((a, b) => b.count - a.count);

// E. Inter-Group Import Frequency
const interGroupImports = {};
for (const e of importEdges) {
  const srcGroup = nodeToGroup[e.source];
  const tgtGroup = nodeToGroup[e.target];
  if (!srcGroup || !tgtGroup || srcGroup === tgtGroup) continue;
  const key = `${srcGroup}->${tgtGroup}`;
  interGroupImports[key] = (interGroupImports[key] || 0) + 1;
}

const interGroupList = Object.entries(interGroupImports)
  .map(([k, count]) => {
    const [from, to] = k.split('->');
    return { from, to, count };
  })
  .sort((a, b) => b.count - a.count);

// F. Intra-Group Import Density
const intraGroupDensity = {};
for (const [group, nodes] of Object.entries(directoryGroups)) {
  let internal = 0, total = 0;
  const nodeSet = new Set(nodes);
  for (const e of importEdges) {
    if (nodeSet.has(e.source)) {
      total++;
      if (nodeSet.has(e.target)) internal++;
    }
  }
  intraGroupDensity[group] = {
    internalEdges: internal,
    totalEdges: total,
    density: total > 0 ? parseFloat((internal / total).toFixed(3)) : 0
  };
}

// G. Directory Pattern Matching
const patternMap = {
  'nn': 'module', 'modules': 'module', 'layers': 'module',
  'optim': 'optimizer',  'optimizers': 'optimizer',
  'cuda': 'gpu-backend', 'mps': 'gpu-backend', 'xpu': 'gpu-backend',
  'accelerator': 'gpu-backend',
  'distributed': 'distributed',
  'dynamo': 'compiler', '_dynamo': 'compiler',
  '_inductor': 'compiler', 'inductor': 'compiler',
  'fx': 'transformation',
  'jit': 'compiler',
  'onnx': 'export', 'export': 'export',
  'quantization': 'optimization', 'ao': 'optimization',
  'profiler': 'profiler',
  'autograd': 'autograd',
  'utils': 'utility', '_utils': 'utility',
  'testing': 'test',
  'gen': 'codegen', 'torchgen': 'codegen',
  'sparse': 'tensor', 'linalg': 'tensor', 'fft': 'tensor', 'special': 'tensor',
  'distributions': 'probability',
  'overrides': 'utility',
  'backends': 'gpu-backend',
  'package': 'serialization',
  'monitor': 'monitoring',
  'signal': 'utility',
  'decomp': 'compiler', '_decomp': 'compiler',
  '_refs': 'compiler', '_prims': 'compiler', '_prims_common': 'compiler',
  '_functorch': 'transformation',
  '_subclasses': 'utility',
  '_higher_order_ops': 'compiler',
  '_library': 'core', '_custom_op': 'core',
  '_numpy': 'compatibility',
  '_lazy': 'compiler',
  '_logging': 'utility',
  '_strobelight': 'profiler',
  'amp': 'optimization',
  '_awaits': 'utility',
  'root': 'root',
};

const patternMatches = {};
for (const group of Object.keys(directoryGroups)) {
  const key = group.toLowerCase().replace(/^torch[\/_]/, '');
  patternMatches[group] = patternMap[key] || patternMap[group] || 'other';
}

// H. Deployment Topology
const deploymentTopology = {
  hasDockerfile: false,
  hasCompose: false,
  hasK8s: false,
  hasTerraform: false,
  hasCI: false,
  infraFiles: []
};

for (const n of fileNodes) {
  const p = n.filePath || n.id;
  if (p.includes('Dockerfile')) { deploymentTopology.hasDockerfile = true; deploymentTopology.infraFiles.push(p); }
  if (p.includes('docker-compose')) { deploymentTopology.hasCompose = true; }
  if (p.includes('.github/workflows')) { deploymentTopology.hasCI = true; deploymentTopology.infraFiles.push(p); }
}

// I. Data Pipeline
const dataPipeline = { schemaFiles: [], migrationFiles: [], dataModelFiles: [], apiHandlerFiles: [] };

// J. Documentation Coverage
const groupsWithDocs = new Set();
const allGroups = Object.keys(directoryGroups);
for (const n of fileNodes) {
  if (n.type === 'document') {
    const g = nodeToGroup[n.id];
    if (g) groupsWithDocs.add(g);
  }
}

const docCoverage = {
  groupsWithDocs: groupsWithDocs.size,
  totalGroups: allGroups.length,
  coverageRatio: allGroups.length > 0 ? parseFloat((groupsWithDocs.size / allGroups.length).toFixed(2)) : 0,
  undocumentedGroups: allGroups.filter(g => !groupsWithDocs.has(g))
};

// K. Dependency Direction
const depCounts = {};
for (const e of importEdges) {
  const sg = nodeToGroup[e.source];
  const tg = nodeToGroup[e.target];
  if (!sg || !tg || sg === tg) continue;
  const fwd = `${sg}->${tg}`;
  depCounts[fwd] = (depCounts[fwd] || 0) + 1;
}

const depDirection = [];
const handledPairs = new Set();
for (const e of importEdges) {
  const sg = nodeToGroup[e.source];
  const tg = nodeToGroup[e.target];
  if (!sg || !tg || sg === tg) continue;
  const pair = [sg, tg].sort().join('|||');
  if (handledPairs.has(pair)) continue;
  handledPairs.add(pair);
  const fwd = depCounts[`${sg}->${tg}`] || 0;
  const rev = depCounts[`${tg}->${sg}`] || 0;
  if (fwd > rev) depDirection.push({ dependent: sg, dependsOn: tg });
  else if (rev > fwd) depDirection.push({ dependent: tg, dependsOn: sg });
}

// File stats
const fileStats = {
  totalFileNodes: fileNodes.length,
  filesPerGroup: Object.fromEntries(Object.entries(directoryGroups).map(([k, v]) => [k, v.length])),
  nodeTypeCounts: Object.fromEntries(Object.entries(nodeTypeGroups).map(([k, v]) => [k, v.length]))
};

// Fan-in and fan-out top 20
const topFanIn = Object.entries(fanIn).sort((a, b) => b[1] - a[1]).slice(0, 20)
  .map(([id, count]) => ({ id, count }));
const topFanOut = Object.entries(fanOut).sort((a, b) => b[1] - a[1]).slice(0, 20)
  .map(([id, count]) => ({ id, count }));

const result = {
  scriptCompleted: true,
  directoryGroups,
  nodeTypeGroups,
  crossCategoryEdges: crossCategoryList,
  interGroupImports: interGroupList,
  intraGroupDensity,
  patternMatches,
  deploymentTopology,
  dataPipeline,
  docCoverage,
  dependencyDirection: depDirection,
  fileStats,
  fileFanIn: topFanIn,
  fileFanOut: topFanOut
};

fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
console.error(`Wrote structural analysis to ${outputPath}`);
console.error(`Groups: ${Object.keys(directoryGroups).length}, types: ${Object.keys(nodeTypeGroups).length}`);
console.error(`Inter-group imports: ${interGroupList.length}, dep directions: ${depDirection.length}`);
