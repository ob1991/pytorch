#!/usr/bin/env python3
"""Structural analysis script for PyTorch architectural layer detection."""
import json
import sys
import os
from collections import defaultdict, Counter

def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)

def get_path(node_id):
    colon_idx = node_id.index(':')
    return node_id[colon_idx + 1:]

input_path = sys.argv[1]
output_path = sys.argv[2]

if not input_path or not output_path:
    die('Usage: ua-arch-analyze.py <input.json> <output.json>')

try:
    with open(input_path, 'r') as f:
        input_data = json.load(f)
except Exception as e:
    die(f"Failed to read input: {e}")

file_nodes = input_data.get('fileNodes', [])
import_edges = input_data.get('importEdges', [])
all_edges = input_data.get('allEdges', [])

total_file_nodes = len(file_nodes)

node_map = {}
for n in file_nodes:
    node_map[n['id']] = n

# Node type groups
node_type_groups = defaultdict(list)
for n in file_nodes:
    t = n.get('type', 'file')
    node_type_groups[t].append(n['id'])

# ─── A. Directory Grouping ───
def compute_directory_groups(nodes):
    paths = [get_path(n['id']) for n in nodes]

    # Find common prefix
    common_prefix = ''
    if paths:
        parts = [p.split('/') for p in paths]
        min_len = min(len(p) for p in parts)
        for i in range(min_len):
            first = parts[0][i]
            if all(p[i] == first for p in parts):
                common_prefix += ('/' if common_prefix else '') + first
            else:
                break

    groups = {}
    file_to_group = {}

    # Check if flat
    relative_dirs = []
    for p in paths:
        if common_prefix:
            offset = len(common_prefix) + (1 if p[len(common_prefix):].startswith('/') else 0)
            rel = p[offset:]
        else:
            rel = p
        relative_dirs.append(rel)

    all_flat = all('/' not in p for p in relative_dirs)

    if all_flat:
        ext_group_map = {
            '.test.js': 'test', '.test.ts': 'test', '.test.jsx': 'test', '.test.tsx': 'test',
            '.spec.js': 'test', '.spec.ts': 'test',
            '.config.js': 'config', '.config.ts': 'config', '.config.yaml': 'config', '.config.yml': 'config',
            '.md': 'docs', '.sql': 'sql', '.json': 'json', '.yaml': 'yaml', '.yml': 'yaml',
            '.toml': 'toml',
        }
        for n in nodes:
            p = get_path(n['id'])
            ext = os.path.splitext(p)[1]
            name = os.path.basename(p)
            group = 'root'
            if name.startswith('test_') or name.startswith('_test') or '_test.' in name:
                group = 'test'
            elif ext in ext_group_map:
                group = ext_group_map[ext]
            else:
                group = f'ext_{ext[1:]}' if ext else 'root'
            groups.setdefault(group, []).append(n['id'])
            file_to_group[n['id']] = group
    else:
        for n in nodes:
            p = get_path(n['id'])
            if common_prefix:
                offset = len(common_prefix)
                rel = p[offset:]
                if rel.startswith('/'):
                    rel = rel[1:]
            else:
                rel = p
            first_seg = rel.split('/')[0] if '/' in rel else 'root'
            groups.setdefault(first_seg, []).append(n['id'])
            file_to_group[n['id']] = first_seg

    return groups, file_to_group

all_dir_groups, all_file_to_group = compute_directory_groups(file_nodes)

# ─── B. Import Adjacency ───
import_adj = defaultdict(set)
reverse_adj = defaultdict(set)

for e in import_edges:
    if e.get('type') == 'imports':
        import_adj[e['source']].add(e['target'])
        reverse_adj[e['target']].add(e['source'])

file_fan_in = {}
file_fan_out = {}
for n in file_nodes:
    nid = n['id']
    file_fan_out[nid] = len(import_adj.get(nid, set()))
    file_fan_in[nid] = len(reverse_adj.get(nid, set()))

# Sort and take top 50
sorted_fan_in = sorted([(k, v) for k, v in file_fan_in.items() if v > 0], key=lambda x: -x[1])[:50]
sorted_fan_out = sorted([(k, v) for k, v in file_fan_out.items() if v > 0], key=lambda x: -x[1])[:50]

# ─── C. Cross-Category Dependency Analysis ───
cat_edge_count = {}
for e in all_edges:
    from_node = node_map.get(e['source'])
    to_node = node_map.get(e['target'])
    if from_node and to_node:
        from_type = from_node.get('type', 'file')
        to_type = to_node.get('type', 'file')
        edge_type = e.get('type', 'unknown')
        key = f"{from_type}|{to_type}|{edge_type}"
        if key not in cat_edge_count:
            cat_edge_count[key] = {'fromType': from_type, 'toType': to_type, 'edgeType': edge_type, 'count': 0}
        cat_edge_count[key]['count'] += 1

cross_category_edges = list(cat_edge_count.values())

# ─── D. Inter-Group Import Frequency ───
file_only_nodes = [n for n in file_nodes if n.get('type', 'file') == 'file']
group_pair_count = {}
for e in import_edges:
    if e.get('type') == 'imports':
        src_group = all_file_to_group.get(e['source'])
        tgt_group = all_file_to_group.get(e['target'])
        if src_group and tgt_group and src_group != tgt_group:
            key = f"{src_group}||{tgt_group}"
            if key not in group_pair_count:
                group_pair_count[key] = {'from': src_group, 'to': tgt_group, 'count': 0}
            group_pair_count[key]['count'] += 1

inter_group_imports = sorted(group_pair_count.values(), key=lambda x: -x['count'])

# ─── E. Intra-Group Import Density ───
intra_group_density = {}
for group, ids in all_dir_groups.items():
    id_set = set(ids)
    internal_edges = 0
    total_edges = 0
    for e in import_edges:
        if e.get('type') == 'imports':
            if e['source'] in id_set:
                total_edges += 1
                if e['target'] in id_set:
                    internal_edges += 1
    density = round(internal_edges / total_edges, 3) if total_edges > 0 else 0
    intra_group_density[group] = {'internalEdges': internal_edges, 'totalEdges': total_edges, 'density': density}

# ─── F. Directory Pattern Matching ───
dir_pattern_map = {
    'routes': 'api', 'api': 'api', 'controllers': 'api', 'endpoints': 'api', 'handlers': 'api',
    'services': 'service', 'core': 'service', 'domain': 'service', 'logic': 'service',
    'models': 'data', 'db': 'data', 'data': 'data', 'persistence': 'data', 'repository': 'data', 'entities': 'data',
    'migrations': 'data', 'entity': 'data', 'sql': 'data', 'database': 'data', 'schema': 'data',
    'components': 'ui', 'views': 'ui', 'pages': 'ui', 'ui': 'ui', 'layouts': 'ui', 'screens': 'ui',
    'middleware': 'middleware', 'plugins': 'middleware', 'interceptors': 'middleware', 'guards': 'middleware',
    'utils': 'utility', 'helpers': 'utility', 'common': 'utility', 'shared': 'utility', 'tools': 'utility',
    'pkg': 'utility', 'templatetags': 'utility',
    'config': 'config', 'constants': 'config', 'env': 'config', 'settings': 'config',
    '__tests__': 'test', 'tests': 'test', 'spec': 'test', 'specs': 'test', 'test': 'test',
    'types': 'types', 'interfaces': 'types', 'schemas': 'types', 'contracts': 'types', 'dtos': 'types',
    'hooks': 'hooks',
    'store': 'state', 'state': 'state', 'reducers': 'state', 'actions': 'state', 'slices': 'state',
    'assets': 'assets', 'static': 'assets', 'public': 'assets',
    'docs': 'documentation', 'documentation': 'documentation', 'wiki': 'documentation',
    'deploy': 'infrastructure', 'deployment': 'infrastructure', 'infra': 'infrastructure', 'infrastructure': 'infrastructure',
    'k8s': 'infrastructure', 'kubernetes': 'infrastructure', 'helm': 'infrastructure', 'charts': 'infrastructure',
    'terraform': 'infrastructure', 'tf': 'infrastructure', 'docker': 'infrastructure',
    'bin': 'entry', 'cmd': 'entry',
    'internal': 'service',
    'dto': 'types', 'request': 'types', 'response': 'types',
    'controller': 'api', 'routers': 'api',
    'composables': 'service', 'blueprints': 'api',
    'signals': 'service', 'serializers': 'api', 'management': 'config', 'commands': 'config',
    'mailers': 'service', 'jobs': 'service', 'channels': 'service',
    # PyTorch-specific directory patterns
    '_dynamo': 'compilation',
    '_inductor': 'compilation',
    'fx': 'compilation',
    'autograd': 'core', 'nn': 'neural-network', 'optim': 'neural-network',
    'distributed': 'distributed',
    'cuda': 'gpu-backend', 'mps': 'gpu-backend', 'xpu': 'gpu-backend',
    'onnx': 'export', 'export': 'export', 'jit': 'export',
    'quantization': 'optimization', 'sparse': 'core', 'linalg': 'core', 'fft': 'core',
    'profiler': 'tooling', 'testing': 'tooling',
    'distributions': 'neural-network',
    'torchgen': 'codegen', 'aten': 'core',
    'package': 'tooling',
    'signal': 'core',
    'monitor': 'tooling',
    'masked': 'core',
    'backends': 'gpu-backend',
    'future': 'core',
    'overrides': 'core',
    'special': 'core',
    '_subclasses': 'core',
    '_decomp': 'compilation',
    '_custom_op': 'core',
    '_awaits': 'core',
    '_strobelight': 'tooling',
    '_functorch': 'compilation',
    'func': 'compilation',
    'library': 'core',
    '_higher_order_ops': 'core',
    '_library': 'core',
    '_lazy': 'codegen',
    'compiler': 'compilation',
    'accelerator': 'gpu-backend',
    '_refs': 'core',
    '_prims': 'core',
    '_prims_common': 'core',
    '_logging': 'tooling',
    '_numpy': 'core',
    'amp': 'optimization',
}

def match_file_pattern(file_path):
    name = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1]

    # Test files
    if any(p in name for p in ['.test.', '.spec.', 'test_', '_test.']) or \
       name.endswith('Test.java') or name.endswith('_spec.rb') or name.endswith('Tests.cs'):
        return 'test'

    if ext == '.d.ts':
        return 'types'

    if name in ['index.ts', 'index.js', '__init__.py']:
        return 'entry'
    if name == 'manage.py':
        return 'entry'
    if name in ['wsgi.py', 'asgi.py']:
        return 'config'
    if name in ['Application.java', 'Program.cs']:
        return 'entry'
    if name == 'config.ru':
        return 'entry'
    if name in ['main.rs', 'lib.rs']:
        return 'entry'

    if name in ['Cargo.toml', 'go.mod', 'Gemfile', 'composer.json', 'pom.xml', 'build.gradle']:
        return 'config'

    if name == 'Dockerfile' or name.startswith('docker-compose'):
        return 'infrastructure'
    if ext in ('.tf', '.tfvars'):
        return 'infrastructure'
    if name == 'Jenkinsfile':
        return 'ci-cd'
    if name == '.gitlab-ci.yml':
        return 'ci-cd'
    if '.github/workflows/' in file_path:
        return 'ci-cd'
    if name == 'Makefile':
        return 'infrastructure'

    if ext == '.sql':
        return 'data'
    if ext in ('.graphql', '.gql', '.proto'):
        return 'types'

    if ext in ('.md', '.rst'):
        return 'documentation'

    return None

pattern_matches = {}
for group in all_dir_groups:
    # Direct directory pattern match
    if group in dir_pattern_map:
        pattern_matches[group] = dir_pattern_map[group]
        continue

    # Try file-level patterns
    ids = all_dir_groups[group]
    best_match = None
    for nid in ids:
        fp = get_path(nid)
        match = match_file_pattern(fp)
        if match:
            best_match = match
            break

    if not best_match:
        lower = group.lower()
        if any(x in lower for x in ['test', 'spec']):
            best_match = 'test'
        elif any(x in lower for x in ['util', 'helper', 'common']):
            best_match = 'utility'
        elif any(x in lower for x in ['config', 'setting']):
            best_match = 'config'
        elif any(x in lower for x in ['doc', 'readme']):
            best_match = 'documentation'
        elif any(x in lower for x in ['model', 'data', 'db']):
            best_match = 'data'
        elif any(x in lower for x in ['api', 'route', 'endpoint']):
            best_match = 'api'
        elif any(x in lower for x in ['infra', 'deploy', 'docker']):
            best_match = 'infrastructure'
        else:
            best_match = 'service'

    pattern_matches[group] = best_match

# ─── G. Deployment Topology ───
all_paths = [get_path(n['id']) for n in file_nodes]
infra_files = [p for p in all_paths if any(x in p for x in ['Dockerfile', 'docker-compose', '.github/workflows/', '.gitlab-ci.yml', 'Jenkinsfile', '.tf'])]

deployment_topology = {
    'hasDockerfile': any(p.endswith('Dockerfile') for p in all_paths),
    'hasCompose': any(p.startswith('docker-compose') for p in [os.path.basename(p) for p in all_paths]),
    'hasK8s': any(p.endswith('.yaml') and ('k8s' in p or 'kubernetes' in p or 'helm' in p) for p in all_paths),
    'hasTerraform': any(p.endswith(('.tf', '.tfvars')) for p in all_paths),
    'hasCI': any('.github/workflows/' in p or p.endswith('.gitlab-ci.yml') or p.endswith('Jenkinsfile') for p in all_paths),
    'infraFiles': infra_files,
}

# ─── H. Data Pipeline ───
schema_files = [p for p in all_paths if p.endswith(('.sql', '.graphql', '.proto'))]
migration_files = [p for p in all_paths if 'migration' in p or 'migrate' in p]
data_model_files = [p for p in all_paths if ('/models/' in p or '/entities/' in p or '/orm/' in p) and not p.endswith('.md')]
api_handler_files = [p for p in all_paths if ('/routes/' in p or '/api/' in p or '/controllers/' in p or '/handlers/' in p) and not p.endswith('.md')]

data_pipeline = {
    'schemaFiles': schema_files,
    'migrationFiles': migration_files,
    'dataModelFiles': data_model_files,
    'apiHandlerFiles': api_handler_files,
}

# ─── I. Documentation Coverage ───
doc_nodes = [n for n in file_nodes if get_path(n['id']).endswith(('.md', '.rst')) or n.get('type') == 'document']
doc_group_set = set()
for n in doc_nodes:
    g = all_file_to_group.get(n['id'])
    if g:
        doc_group_set.add(g)

all_groups = list(all_dir_groups.keys())
groups_with_docs = [g for g in doc_group_set if g in all_groups]
undocumented_groups = [g for g in all_groups if g not in doc_group_set]

doc_coverage = {
    'groupsWithDocs': len(groups_with_docs),
    'totalGroups': len(all_groups),
    'coverageRatio': round(len(groups_with_docs) / len(all_groups), 2) if all_groups else 0,
    'undocumentedGroups': undocumented_groups,
}

# ─── J. Dependency Direction ───
dep_dir_count = {}
for e in import_edges:
    if e.get('type') == 'imports':
        src_group = all_file_to_group.get(e['source'])
        tgt_group = all_file_to_group.get(e['target'])
        if src_group and tgt_group and src_group != tgt_group:
            key_fwd = f"{src_group}||{tgt_group}"
            key_rev = f"{tgt_group}||{src_group}"
            if key_fwd not in dep_dir_count:
                dep_dir_count[key_fwd] = {'fwd': 0, 'from': src_group, 'to': tgt_group}
            dep_dir_count[key_fwd]['fwd'] += 1
            if key_rev not in dep_dir_count:
                dep_dir_count[key_rev] = {'fwd': 0, 'from': tgt_group, 'to': src_group}

dependency_direction = []
seen_pairs = set()
for d in sorted(dep_dir_count.values(), key=lambda x: -x['fwd']):
    pair_key = tuple(sorted([d['from'], d['to']]))
    if pair_key in seen_pairs:
        continue
    seen_pairs.add(pair_key)

    rev_key = f"{d['to']}||{d['from']}"
    rev = dep_dir_count.get(rev_key)
    net_fwd = d['fwd']
    net_rev = rev['fwd'] if rev else 0

    if net_fwd > net_rev:
        dependency_direction.append({'dependent': d['from'], 'dependsOn': d['to'], 'weight': net_fwd - net_rev})
    elif net_rev > net_fwd:
        dependency_direction.append({'dependent': d['to'], 'dependsOn': d['from'], 'weight': net_rev - net_fwd})

dependency_direction.sort(key=lambda x: -x['weight'])

# ─── K. File Stats ───
files_per_group = {}
for g, ids in all_dir_groups.items():
    files_per_group[g] = len(ids)

node_type_counts = {}
for t, ids in node_type_groups.items():
    node_type_counts[t] = len(ids)

result = {
    'scriptCompleted': True,
    'directoryGroups': dict(all_dir_groups),
    'nodeTypeGroups': dict(node_type_groups),
    'crossCategoryEdges': cross_category_edges,
    'interGroupImports': inter_group_imports,
    'intraGroupDensity': intra_group_density,
    'patternMatches': pattern_matches,
    'deploymentTopology': deployment_topology,
    'dataPipeline': data_pipeline,
    'docCoverage': doc_coverage,
    'dependencyDirection': dependency_direction,
    'fileStats': {
        'totalFileNodes': total_file_nodes,
        'filesPerGroup': files_per_group,
        'nodeTypeCounts': node_type_counts,
    },
    'fileFanIn': {k: v for k, v in sorted_fan_in},
    'fileFanOut': {k: v for k, v in sorted_fan_out},
}

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Analysis written to {output_path}")
sys.exit(0)
