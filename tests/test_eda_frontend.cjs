const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const context = {fetch: () => new Promise(() => {}), document: {getElementById: () => ({})},
  Plotly: {}, console};
vm.createContext(context);
vm.runInContext(fs.readFileSync('eda_frontend/app.js', 'utf8'), context);

const base = {cohort: 'AFML freeze', data_id: 'data-a', split_settings_id: 'split-a',
  horizon: 'best', cost_id: 'fee=4'};
assert.equal(context.compatibleKey(base), 'AFML freeze|data-a|split-a|best|fee=4');
assert.notEqual(context.compatibleKey(base), context.compatibleKey({...base, cohort: 'V2 early'}));
assert.notEqual(context.compatibleKey(base), context.compatibleKey({...base, split_settings_id: 'split-b'}));
assert.notEqual(context.compatibleKey(base), context.compatibleKey({...base, cost_id: 'fee=8'}));
assert.equal(context.compatibleKey({...base, split_settings_id: null}), null);

const rows = [{backtest: {validation: {trades: 1}, test: {trades: null}}, forecast: []},
  {backtest: {validation: {trades: null}, test: {trades: 2}}, forecast: []}];
assert.equal(context.rowsForSplit(rows, 'validation').length, 1);
assert.equal(context.rowsForSplit(rows, 'test').length, 1);
console.log('frontend compatibility and split tests passed');
