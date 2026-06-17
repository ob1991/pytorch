const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const PROJECT_ROOT = '/home/bluce/pytorch';
const stdout = execSync('git ls-files -- torch/ torchgen/', { cwd: PROJECT_ROOT, encoding: 'utf-8' });
const allFiles = stdout.trim().split('\n').filter(Boolean);

const defaultExcludes = [
  /\/node_modules\//, /\.git\//, /\/__pycache__\//, /\/dist\//, /\/build\//,
  /\.lock$/, /\.png$/, /\.jpg$/, /\.jpeg$/, /\.gif$/, /\.svg$/, /\.ico$/,
  /\.woff$/, /\.woff2$/, /\.ttf$/, /\.eot$/, /\.mp3$/, /\.mp4$/, /\.pdf$/,
  /\.zip$/, /\.tar$/, /\.gz$/, /\.min\.js$/, /\.min\.css$/, /\.map$/,
  /LICENSE$/, /\.gitignore$/, /\.editorconfig$/, /\.prettierrc/,
  /\.o$/, /\.obj$/, /\.so$/, /\.dylib$/, /\.dll$/, /\.a$/, /\.lib$/,
  /\.pyc$/, /\.pyd$/, /\.ninja/,
  /torch\/include\//, /torch\/share\//, /torch\/lib\//, /torch\/bin\//,
  /torch\/test\//, /torch\/\.\w/,
];

const filteredFiles = allFiles.filter(f => !defaultExcludes.some(re => re.test(f)));

const langMap = {
  '.ts': 'typescript', '.tsx': 'typescript', '.js': 'javascript', '.jsx': 'javascript',
  '.py': 'python', '.pyi': 'python',
  '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp', '.h': 'cpp', '.hpp': 'cpp',
  '.c': 'c', '.cs': 'csharp', '.java': 'java',
  '.sh': 'shell', '.bash': 'shell', '.ps1': 'powershell', '.bat': 'batch',
  '.md': 'markdown', '.rst': 'markdown',
  '.yaml': 'yaml', '.yml': 'yaml', '.json': 'json', '.jsonc': 'jsonc', '.toml': 'toml',
  '.sql': 'sql', '.html': 'html', '.htm': 'html',
  '.css': 'css', '.scss': 'css', '.sass': 'css', '.less': 'css',
  '.xml': 'xml', '.cfg': 'config', '.ini': 'ini',
  '.cu': 'cuda', '.cuh': 'cuda', '.hip': 'hip', '.metal': 'metal',
  '.glsl': 'glsl', '.mm': 'objc',
  '.cmake': 'cmake', '.bzl': 'bazel',
  '.ipynb': 'jupyter', '.jinja': 'jinja', '.thrift': 'thrift',
  '.fbs': 'flatbuffers', '.ld': 'linker',
};

function detectLang(filename) {
  if (filename.endsWith('CMakeLists.txt')) return 'cmake';
  if (filename.endsWith('Dockerfile')) return 'dockerfile';
  if (/\.(cli|txt)$/.test(path.extname(filename))) {
    const base = path.basename(filename);
    if (base === 'requirements.txt' || base === 'requirements-build.txt') return 'python';
  }
  const ext = path.extname(filename);
  const base = path.basename(filename);
  if (filename.endsWith('.pyi.in')) return 'python';
  if (base === 'build.bzl' || base === 'defs.bzl') return 'bazel';
  if (/\.bzl$/.test(filename)) return 'bazel';
  return langMap[ext] || (ext ? ext.replace('.','') : 'unknown');
}

function detectCategory(filename) {
  const base = path.basename(filename);
  const ext = path.extname(filename);
  if (/\.(md|rst)$/.test(ext) || (ext === '.txt' && base !== 'LICENSE')) return 'docs';
  if (/\.(yaml|yml|json|jsonc|toml|xml|cfg|ini)$/.test(ext) ||
      base === 'tsconfig.json' || base === 'package.json' || base === 'pyproject.toml') return 'config';
  if (base === 'Dockerfile' || /\.tf$/.test(ext) || base === 'Makefile' || base === 'Jenkinsfile') return 'infra';
  if (/\.(sql|graphql|gql|proto)$/.test(ext)) return 'data';
  if (/\.(sh|bash|ps1|bat)$/.test(ext)) return 'script';
  if (/\.(html|htm|css|scss|sass|less)$/.test(ext)) return 'markup';
  return 'code';
}

function countLines(files) {
  const results = {};
  const batchSize = 200;
  for (let i = 0; i < files.length; i += batchSize) {
    const batch = files.slice(i, i + batchSize);
    try {
      const out = execSync('wc -l ' + batch.map(f => '"' + f + '"').join(' ') + ' 2>/dev/null', { cwd: PROJECT_ROOT, encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 });
      const lines = out.trim().split('\n');
      for (const line of lines) {
        const m = line.match(/^\s*(\d+)\s+(.+)$/);
        if (m) results[m[2].trim()] = parseInt(m[1]);
      }
    } catch(e) {}
  }
  return results;
}

console.error(`Files: ${filteredFiles.length} after filter`);
const lineCounts = countLines(filteredFiles);

const files = filteredFiles.map(f => ({
  path: f,
  language: detectLang(f),
  sizeLines: lineCounts[f] || 0,
  fileCategory: detectCategory(f)
})).sort((a, b) => a.path.localeCompare(b.path));

// Python import resolution
const allFilesSet = new Set(filteredFiles);

function resolvePythonImports(content, filePath) {
  const imports = new Set();
  const lines = content.split('\n');
  
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('#') || trimmed.startsWith('"""') || trimmed.startsWith("'''")) continue;
    
    // from X import Y (absolute)
    let m = trimmed.match(/^from\s+([\w.]+)\s+import\s+(.+)/);
    if (m) {
      const mod = m[1].trim();
      if (mod === 'torch' || mod === 'torchgen') continue; // skip self-import
      
      const pyPath = mod.replace(/\./g, '/') + '.py';
      const initPath = mod.replace(/\./g, '/') + '/__init__.py';
      
      if (allFilesSet.has(pyPath)) imports.add(pyPath);
      else if (allFilesSet.has(initPath)) {
        imports.add(initPath);
        // Probe submodule names
        const names = m[2].split(',').map(s => s.trim().split(' as ')[0].split('//')[0].strip()).filter(Boolean);
        for (const name of names) {
          const subPy = mod.replace(/\./g, '/') + '/' + name + '.py';
          const subInit = mod.replace(/\./g, '/') + '/' + name + '/__init__.py';
          if (allFilesSet.has(subPy)) imports.add(subPy);
          else if (allFilesSet.has(subInit)) imports.add(subInit);
        }
      }
      continue;
    }
    
    // import X (absolute, multi-module)
    m = trimmed.match(/^import\s+(.+)/);
    if (m) {
      const modules = m[1].split(',').map(s => s.trim().split(' as ')[0].trim());
      for (const mod of modules) {
        if (mod === 'torch' || mod === 'torchgen') continue;
        const pyPath = mod.replace(/\./g, '/') + '.py';
        const initPath = mod.replace(/\./g, '/') + '/__init__.py';
        if (allFilesSet.has(pyPath)) imports.add(pyPath);
        else if (allFilesSet.has(initPath)) imports.add(initPath);
      }
    }
  }
  return [...imports];
}

function stripComments(s) { return s.split('#')[0].trim(); }

console.error('Building import map...');
const importMap = {};
let imported = 0;
for (const f of filteredFiles) {
  if (detectCategory(f) !== 'code') {
    importMap[f] = [];
    continue;
  }
  try {
    const content = fs.readFileSync(path.join(PROJECT_ROOT, f), 'utf-8');
    if (f.endsWith('.py')) {
      importMap[f] = resolvePythonImports(content, f);
      if (importMap[f].length > 0) imported++;
    } else {
      importMap[f] = [];
    }
  } catch(e) {
    importMap[f] = [];
  }
}
console.error(`Files with imports: ${imported}`);

// Frameworks
const frameworks = ['PyTorch', 'CMake', 'setuptools', 'pytest', 'pybind11', 'TorchDynamo', 'TorchInductor'];
if (files.some(f => f.path.startsWith('torch/distributed/'))) frameworks.push('NCCL', 'Gloo', 'MPI');
if (files.some(f => f.path.startsWith('torch/cuda/'))) frameworks.push('CUDA');
if (files.some(f => f.path.startsWith('torch/xpu/'))) frameworks.push('XPU');
if (files.some(f => f.path.startsWith('torch/mps/'))) frameworks.push('MPS');
if (files.some(f => f.path.startsWith('torch/onnx/'))) frameworks.push('ONNX');
if (files.some(f => f.path.startsWith('torch/quantization/'))) frameworks.push('Quantization');
if (files.some(f => f.path.startsWith('torch/fx/'))) frameworks.push('FX');
if (files.some(f => f.path.startsWith('torch/export/'))) frameworks.push('Export');

const langCounts = {};
for (const f of files) {
  langCounts[f.language] = (langCounts[f.language] || 0) + 1;
}
const languages = Object.keys(langCounts).sort();
const complexity = files.length > 500 ? 'very-large' : files.length > 150 ? 'large' : 'moderate';

const result = {
  name: 'PyTorch',
  description: 'Tensors and Dynamic neural networks in Python with strong GPU acceleration. 开源深度学习框架，提供张量计算和基于自动微分的深度神经网络。',
  languages,
  frameworks: [...new Set(frameworks)],
  files,
  totalFiles: files.length,
  filteredByIgnore: 0,
  estimatedComplexity: complexity,
  importMap
};

fs.writeFileSync('/home/bluce/pytorch/.understand-anything/intermediate/scan-result.json', JSON.stringify(result, null, 2));
console.error(`Done: ${files.length} files, ${Object.keys(langCounts).length} languages, ${imported} files with resolved imports`);
