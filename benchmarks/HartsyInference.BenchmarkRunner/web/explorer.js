'use strict';
const node = (tag, text) => { const element = document.createElement(tag); if (text !== undefined) element.textContent = text; return element; };
async function main() {
  const response = await fetch('summary.json'); if (!response.ok) throw new Error('Dataset unavailable');
  const rows = await response.json();
  const fields = {case: 'caseId', revision: 'engineRevision', backend: 'backend'};
  for (const [id, field] of Object.entries(fields)) {
    const select = document.getElementById(id); const all = node('option', 'All'); all.value = ''; select.append(all);
    for (const value of [...new Set(rows.map(r => r[field]))].sort()) { const option = node('option', value); option.value = value; select.append(option); }
    select.addEventListener('change', render);
  }
  function render() {
    const selected = rows.filter(row => Object.entries(fields).every(([id, field]) => !document.getElementById(id).value || row[field] === document.getElementById(id).value));
    document.getElementById('empty').hidden = selected.length > 0;
    const root = document.getElementById('results'); root.replaceChildren();
    const table = node('table'); const header = node('tr');
    for (const text of ['GPU / workload', 'Backend / engine / driver', 'Warm median', 'TTFT / decode', '95% interval', 'Coverage / evidence']) header.append(node('th', text));
    table.append(header);
    for (const row of selected) {
      const tr = node('tr'); tr.append(node('td', row.gpu + ' / ' + row.caseId));
      tr.append(node('td', row.backend + ' / ' + row.engineRevision.slice(0, 7) + ' / ' + row.driver));
      const latency = node('td', row.medianMs.toFixed(1) + ' ms');
      const bar = node('div');
      const maximum = Math.max(...selected.filter(r => r.caseId === row.caseId && r.engineRevision === row.engineRevision).map(r => r.medianMs));
      bar.style.width = Math.max(1, row.medianMs / maximum * 150) + 'px';
      bar.style.height = '6px'; bar.style.background = '#7dd3fc'; latency.append(bar); tr.append(latency);
      tr.append(node('td', (row.medianFirstTokenMs?.toFixed(1) ?? '—') + ' ms / ' + (row.medianDecodeTokensPerSecond?.toFixed(1) ?? '—') + ' tok/s'));
      tr.title = row.operatingSystem + ' | ' + row.runtime + ' | ' + (row.deviceMemoryBytes / 1073741824).toFixed(1) + ' GiB | config ' + row.configuration + ' | native ' + row.nativeConfiguration;
      tr.append(node('td', row.machines > 1 ? row.intervalLowMs.toFixed(1) + '–' + row.intervalHighMs.toFixed(1) + ' ms' : 'Single machine'));
      const evidence = node('td', row.machines + ' machines, ' + row.sessions + ' sessions ');
      for (const [index, url] of row.evidence.entries()) {
        if (!/^https:\/\/github\.com\/HartsyAI\/HartsyInference\/releases\/download\/benchmark-evidence\/[0-9a-f]{64}\.zip$/.test(url)) continue;
        const link = node('a', '[' + (index + 1) + '] '); link.href = url; evidence.append(link);
      }
      tr.append(evidence); table.append(tr);
    }
    root.append(table);
  }
  render();
}
main().catch(error => { document.getElementById('empty').hidden = false; document.getElementById('empty').textContent = error.message; });
