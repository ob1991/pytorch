const fs = require('fs');
const data = require('/home/bluce/pytorch/.understand-anything/intermediate/scan-result.json');

const BATCH_SIZE = 25;
const files = data.files;
const importMap = data.importMap;

// Sort: Python files first (grouped by directory), then non-code, then C++
const pyFiles = files.filter(f => f.language === 'python');
const nonCodeFiles = files.filter(f => f.fileCategory !== 'code');
const cppFiles = files.filter(f => f.language === 'cpp' || f.language === 'c' || f.language === 'cuda');
const otherCode = files.filter(f => f.fileCategory === 'code' && f.language !== 'python' && f.language !== 'cpp' && f.language !== 'c' && f.language !== 'cuda');

// Group Python files by their top-level module directory
const pyByDir = {};
for (const f of pyFiles) {
  const parts = f.path.split('/');
  const topDir = parts.slice(0, Math.min(3, parts.length-1)).join('/');
  if (!pyByDir[topDir]) pyByDir[topDir] = [];
  pyByDir[topDir].push(f);
}

// Flatten: interleave Python directory groups with non-code and other files
let ordered = [];
const allPyDirectories = Object.keys(pyByDir).sort();
const nonCodeChunks = [];
for (let i = 0; i < nonCodeFiles.length; i += 5) {
  nonCodeChunks.push(nonCodeFiles.slice(i, i + 5));
}

let pyIdx = 0;
let ncIdx = 0;
let otherIdx = 0;

while (pyIdx < allPyDirectories.length || ncIdx < nonCodeChunks.length || otherIdx < otherCode.length) {
  if (pyIdx < allPyDirectories.length) {
    const dir = allPyDirectories[pyIdx++];
    ordered.push(...pyByDir[dir]);
  }
  if (ncIdx < nonCodeChunks.length) {
    ordered.push(...nonCodeChunks[ncIdx++]);
  }
}

// For remaining - add other code files to fill up
ordered.push(...cppFiles);
ordered.push(...otherCode);

// Verify we have all files
const allPaths = new Set(ordered.map(f => f.path));
const originalPaths = new Set(files.map(f => f.path));
const missing = [...originalPaths].filter(p => !allPaths.has(p));
if (missing.length > 0) {
  console.error(`MISSING ${missing.length} files, adding`);
  for (const p of missing) {
    const f = files.find(x => x.path === p);
    if (f) ordered.push(f);
  }
}
console.error(`Ordered ${ordered.length} files (expected ${files.length})`);

// Create batches
const batches = [];
for (let i = 0; i < ordered.length; i += BATCH_SIZE) {
  const batchFiles = ordered.slice(i, i + BATCH_SIZE);
  const batchImportData = {};
  for (const f of batchFiles) {
    batchImportData[f.path] = importMap[f.path] || [];
  }
  batches.push({
    index: batches.length,
    files: batchFiles,
    importData: batchImportData
  });
}

console.error(`Created ${batches.length} batches`);

// Write batch metadata (one JSON per batch)
const metaDir = '/home/bluce/pytorch/.understand-anything/tmp/batches';
fs.mkdirSync(metaDir, { recursive: true });

// Write a summary of each batch
const batchSummary = batches.map(b => ({
  index: b.index,
  fileCount: b.files.length,
  languages: [...new Set(b.files.map(f => f.language))],
  categories: [...new Set(b.files.map(f => f.fileCategory))],
  files: b.files.map(f => f.path)
}));

fs.writeFileSync(`${metaDir}/_summary.json`, JSON.stringify(batchSummary, null, 2));
console.error(`Batch summary written to ${metaDir}/_summary.json`);

// Also write the full batch list index
fs.writeFileSync(`${metaDir}/_index.json`, JSON.stringify({
  totalBatches: batches.length,
  totalFiles: ordered.length,
  batchSize: BATCH_SIZE
}));

console.log(`BATCHES=${batches.length}`);
