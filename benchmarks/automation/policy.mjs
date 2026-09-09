// Pure policy checks shared by trusted automation and offline adversarial tests.
export function submissionIdFor(files) {
  if (files.length !== 2 || files.some(file => file.status !== 'added'
    || !/^benchmarks\/submissions\/[0-9a-f]{64}\/(campaign|submission)\.json$/.test(file.filename)))
    throw new Error('A results PR must add exactly campaign.json and submission.json for one new campaign.');
  const id = files[0].filename.split('/')[2];
  if (files.some(file => file.filename.split('/')[2] !== id)
    || new Set(files.map(file => file.filename)).size !== 2) throw new Error('Multiple or duplicate campaigns in PR');
  return id;
}
export function matchesMergedHead(pr, review) {
  return pr.merged === true && /^[0-9a-f]{40}$/.test(review.verifiedHead) && pr.head.sha === review.verifiedHead;
}

export function hasSubmissionChanges(files) {
  return files.some(file => file.filename.startsWith('benchmarks/submissions/'));
}
