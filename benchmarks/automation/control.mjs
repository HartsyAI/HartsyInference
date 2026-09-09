// Runs from the trusted base/default branch. Pull request content is always treated as bounded data.
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {submissionIdFor, matchesMergedHead} from './policy.mjs';
const repository = 'HartsyAI/HartsyInference';
const temporary = process.env.RUNNER_TEMP || '/tmp';
const runner = path.resolve('benchmarks/HartsyInference.BenchmarkRunner/bin/Release/net10.0/hartsy-bench.dll');
function execute(program, args, options = {}) {
  const result = spawnSync(program, args, {encoding: 'utf8', maxBuffer: 4 * 1024 * 1024, ...options});
  if (result.status !== 0) throw new Error(`${program} failed: ${(result.stderr || result.error || '').toString().slice(0, 1000)}`);
  return result.stdout;
}
const gh = (...args) => execute('gh', args);
const api = endpoint => JSON.parse(gh('api', endpoint));
const bench = (...args) => execute('dotnet', [runner, ...args]);
const hashPattern = /^[0-9a-f]{64}$/;
function rootFor(id) { if (!hashPattern.test(id)) throw new Error('Invalid submission id'); return path.join(temporary, 'benchmark-review', id); }
async function pullData(number) {
  if (!/^[0-9]+$/.test(number)) throw new Error('Invalid PR number');
  const pr = api(`repos/${repository}/pulls/${number}`);
  const files = JSON.parse(gh('api', `repos/${repository}/pulls/${number}/files`, '--paginate', '--slurp')).flat();
  const id = submissionIdFor(files);
  const root = rootFor(id); fs.mkdirSync(root, {recursive: true});
  let bytes = 0;
  for (const file of files) {
    const blob = api(`repos/${pr.head.repo.full_name}/git/blobs/${file.sha}`);
    if (blob.encoding !== 'base64' || blob.size > 2 * 1024 * 1024 || (bytes += blob.size) > 2 * 1024 * 1024) throw new Error('Oversized PR data');
    const data = Buffer.from(blob.content, 'base64');
    if (data.length !== blob.size) throw new Error('Blob length mismatch');
    fs.writeFileSync(path.join(root, path.basename(file.filename)), data, {flag: 'wx'});
  }
  bench('metadata', '--input', root);
  return {pr, id, root};
}
function validateEvidence(root, id, source) {
  const zip = path.join(temporary, `${id}.zip`);
  bench('download', '--input', root, '--bundle', zip, '--source', source);
  const extracted = path.join(temporary, 'benchmark-extracted', id);
  bench('extract', '--bundle', zip, '--output', extracted);
  const report = JSON.parse(bench('validate', '--input', extracted));
  if (!report.valid) throw new Error('Evidence validation failed');
  const recorded = fs.readFileSync(path.join(root, 'campaign.json'));
  if (!recorded.equals(fs.readFileSync(path.join(extracted, 'campaign.json')))) throw new Error('Bundle campaign differs from PR');
  return {zip, extracted, report};
}
function archiveAsset(file) {
  // An existing asset may be reused only when byte-identical. Never use --clobber.
  let release;
  try { release = api(`repos/${repository}/releases/tags/benchmark-evidence`); }
  catch (error) {
    if (!error.message.includes('404')) throw error;
    gh('release', 'create', 'benchmark-evidence', '--repo', repository, '--target', 'main',
      '--title', 'Community benchmark evidence', '--notes', 'Immutable evidence bundles and append-only maintainer review receipts.');
    release = api(`repos/${repository}/releases/tags/benchmark-evidence`);
  }
  const name = path.basename(file), existing = release.assets.find(a => a.name === name);
  if (existing) {
    const old = path.join(temporary, 'existing-' + name);
    execute('gh', ['api', `repos/${repository}/releases/assets/${existing.id}`, '-H', 'Accept: application/octet-stream'],
      {encoding: null, stdio: ['ignore', fs.openSync(old, 'w'), 'pipe']});
    if (!fs.readFileSync(old).equals(fs.readFileSync(file))) throw new Error('Immutable asset name collision');
  } else gh('release', 'upload', 'benchmark-evidence', file, '--repo', repository);
}
const mode = process.argv[2];
if (mode === 'validate' || mode === 'promote') {
  const {pr, id, root} = await pullData(process.env.PR_NUMBER || '');
  const {zip, report} = validateEvidence(root, id, 'staging');
  if (mode === 'promote') {
    if (pr.state !== 'open') throw new Error('Promotion requires an open PR');
    if (process.env.OUTPUTS_REVIEWED !== 'true') throw new Error('Maintainer must explicitly confirm output review');
    const revision = JSON.parse(fs.readFileSync(path.join(root, 'campaign.json'))).environment.engineRevision;
    if (!/^[0-9a-f]{40}$/.test(revision)) throw new Error('Unversioned build cannot be promoted');
    // Require the engine revision to be reachable from the engine main branch.
    const comparison = api(`repos/${repository}/compare/${revision}...main`);
    if (!['ahead', 'identical'].includes(comparison.status)) throw new Error('Engine revision is not an ancestor of main');
    archiveAsset(zip);
    const receipt = path.join(temporary, `${id}.${process.env.GITHUB_RUN_ID}${process.env.GITHUB_RUN_ATTEMPT || 1}.review.json`);
    bench('review', '--input', root, '--output', receipt, '--status', 'accepted', '--reviewer', process.env.GITHUB_ACTOR,
      '--pr', pr.html_url, '--head', pr.head.sha, '--reason', 'Maintainer inspected saved outputs; trusted validator and archive hash checks passed.');
    // Re-read head immediately before publishing an acceptance check. A later push has a different check identity.
    if (api(`repos/${repository}/pulls/${pr.number}`).head.sha !== pr.head.sha) throw new Error('PR changed during review');
    archiveAsset(receipt);
    const check = {name: 'benchmark-evidence-reviewed', head_sha: pr.head.sha, status: 'completed', conclusion: 'success',
      output: {title: 'Evidence mirrored and outputs reviewed', summary: `Campaign ${id}; headline eligible: ${report.headlineEligible}.`}};
    execute('gh', ['api', `repos/${repository}/check-runs`, '--method', 'POST', '--input', '-'], {input: JSON.stringify(check)});
  }
  console.log(JSON.stringify({id, valid: report.valid, headlineEligible: report.headlineEligible}));
} else if (mode === 'publish') {
  const submissions = path.resolve('benchmarks/submissions');
  const receipts = path.join(temporary, 'benchmark-receipts'); fs.mkdirSync(receipts, {recursive: true});
  let release;
  try { release = api(`repos/${repository}/releases/tags/benchmark-evidence`); }
  catch (error) {
    if (fs.readdirSync(submissions).some(n => hashPattern.test(n))) throw error;
    release = {assets: []};
  }
  for (const asset of release.assets.filter(a => /^[0-9a-f]{64}\.[0-9]+\.review\.json$/.test(a.name))) {
    if (asset.size > 16384) throw new Error('Oversized review receipt');
    const raw = gh('api', `repos/${repository}/releases/assets/${asset.id}`, '-H', 'Accept: application/octet-stream');
    const review = JSON.parse(raw);
    if (!hashPattern.test(review.submissionId) || !/^https:\/\/github\.com\/HartsyAI\/HartsyInference\/pull\/[0-9]+$/.test(review.pullRequest)) throw new Error('Malformed review');
    const pr = api(`repos/${repository}/pulls/${review.pullRequest.split('/').at(-1)}`);
    if (!matchesMergedHead(pr, review)) continue;
    fs.writeFileSync(path.join(receipts, asset.name), raw);
  }
  for (const id of fs.readdirSync(submissions).filter(n => hashPattern.test(n))) {
    try { validateEvidence(path.join(submissions, id), id, 'archive'); }
    catch (error) {
      fs.rmSync(path.join(temporary, 'benchmark-extracted', id), {recursive: true, force: true});
      console.error(`Excluded ${id}: archive unavailable or invalid`);
    }
  }
  bench('publish', '--input', submissions, '--reviews', receipts, '--evidence', path.join(temporary, 'benchmark-extracted'), '--output', 'benchmarks/generated');
} else if (mode === 'withdraw') {
  const id = process.env.SUBMISSION_ID || ''; if (!hashPattern.test(id)) throw new Error('Invalid submission');
  if (!/^https:\/\/github\.com\/HartsyAI\/HartsyInference\/pull\/[0-9]+$/.test(process.env.REVIEW_PR || '')) throw new Error('Invalid review PR');
  const pr = api(`repos/${repository}/pulls/${process.env.REVIEW_PR.split('/').at(-1)}`);
  if (!matchesMergedHead(pr, {verifiedHead: process.env.REVIEW_HEAD})) throw new Error('Withdrawal must identify the merged reviewed head');
  const files = JSON.parse(gh('api', `repos/${repository}/pulls/${pr.number}/files`, '--paginate', '--slurp')).flat();
  if (submissionIdFor(files) !== id) throw new Error('Withdrawal PR does not contain this submission');
  const root = path.resolve('benchmarks/submissions', id), receipt = path.join(temporary, `${id}.${process.env.GITHUB_RUN_ID}${process.env.GITHUB_RUN_ATTEMPT || 1}.review.json`);
  bench('review', '--input', root, '--output', receipt, '--status', 'withdrawn', '--reviewer', process.env.GITHUB_ACTOR,
    '--pr', process.env.REVIEW_PR, '--head', process.env.REVIEW_HEAD, '--reason', process.env.REVIEW_REASON);
  archiveAsset(receipt);
} else throw new Error('Unknown automation mode');
