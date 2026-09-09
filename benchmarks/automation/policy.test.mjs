import {test} from 'node:test';
import assert from 'node:assert/strict';
import {submissionIdFor, matchesMergedHead, hasSubmissionChanges} from './policy.mjs';
const id = 'a'.repeat(64);
const files = ['campaign', 'submission'].map(name => ({status: 'added', filename: `benchmarks/submissions/${id}/${name}.json`}));
test('accepts one data-only campaign', () => assert.equal(submissionIdFor(files), id));
test('rejects executable code, mixed changes and deletion', () => {
  assert.throws(() => submissionIdFor([...files, {status:'added',filename:'payload.sh'}]));
  assert.throws(() => submissionIdFor([{...files[0],status:'modified'}, files[1]]));
  assert.throws(() => submissionIdFor([{...files[0],status:'removed'}, files[1]]));
  assert.throws(() => submissionIdFor([{...files[0],filename:files[0].filename.replace(id,'../escape')},files[1]]));
});
test('rejects duplicate and multiple campaigns', () => {
  assert.throws(() => submissionIdFor([files[0],files[0]]));
  assert.throws(() => submissionIdFor([files[0],{...files[1],filename:files[1].filename.replace(id,'b'.repeat(64))}]));
});
test('approval is valid only for the exact merged PR head', () => {
  const review = {verifiedHead:'a'.repeat(40)};
  assert.equal(matchesMergedHead({merged:true,head:{sha:review.verifiedHead}},review),true);
  assert.equal(matchesMergedHead({merged:false,head:{sha:review.verifiedHead}},review),false);
  assert.equal(matchesMergedHead({merged:true,head:{sha:'b'.repeat(40)}},review),false);
});

test('merge gate distinguishes code-only PRs from any result mutation', () => {
  assert.equal(hasSubmissionChanges([{filename:'src/engine.cs'}]),false);
  assert.equal(hasSubmissionChanges(files),true);
  assert.equal(hasSubmissionChanges([{filename:files[0].filename,status:'removed'}]),true);
});
