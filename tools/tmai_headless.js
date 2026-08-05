// Headless runner for lvandeve/tmai's ai_lode, used as an EXTERNAL baseline.
//
// Why this exists: every strength claim in this repo is otherwise measured
// against a greedy heuristic we wrote ourselves, which is self-grading.
// tmai is an independent Terra Mystica implementation with its own
// hand-tuned AI, so its final scores are a yardstick nobody here
// calibrated. Comparing absolute VP (ours ~65, Div 1-3 humans ~136) against
// ai_lode's tells us whether our agents are weak, or whether ~65 is simply
// what non-human Terra Mystica looks like.
//
// tmai is a browser app, but its game logic is DOM-free -- only util.js and
// two setTimeout calls touch the browser -- so shims plus the engine's own
// synchronous `gameLoopBlocking` are enough to run whole games.
//
// Usage: node tools/tmai_headless.js <tmai_dir> <num_games> [seed]
// Output: one JSON line per game, then a JSON summary line.

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const tmaiDir = process.argv[2] || '/Users/keerthikmuruganandam/code/tmai';
const numGames = parseInt(process.argv[3] || '5', 10);
const seed = parseInt(process.argv[4] || '1', 10);

// --- browser shims ---------------------------------------------------------
const makeEl = () => ({
  style: {}, appendChild() {}, removeChild() {}, setAttribute() {},
  addEventListener() {}, removeEventListener() {}, focus() {}, click() {},
  getContext: () => ({
    fillRect() {}, clearRect() {}, drawImage() {}, beginPath() {}, arc() {},
    fill() {}, stroke() {}, moveTo() {}, lineTo() {}, fillText() {},
    save() {}, restore() {}, translate() {}, scale() {}, closePath() {},
    measureText: () => ({ width: 0 }),
  }),
  innerHTML: '', childNodes: [], children: [], classList: { add() {}, remove() {} },
  parentNode: null, offsetWidth: 0, offsetHeight: 0, value: '',
});

const documentShim = {
  createElement: makeEl, createTextNode: makeEl,
  getElementById: makeEl, getElementsByTagName: () => [],
  body: makeEl(), documentElement: { scrollLeft: 0, scrollTop: 0 },
  addEventListener() {}, removeEventListener() {}, onkeydown: null,
};

const sandbox = {
  document: documentShim,
  window: { setTimeout: (fn) => fn(), addEventListener() {}, location: { href: '' } },
  setTimeout: (fn) => fn(),
  clearTimeout: () => {},
  setInterval: () => 0,
  clearInterval: () => {},
  console,
  Image: function () { return makeEl(); },
  alert: () => {},
};
sandbox.global = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);

const files = [
  'util.js', 'enums.js', 'game.js', 'world.js', 'faction.js', 'player.js',
  'action.js', 'rules.js', 'render.js', 'menu.js', 'strategy.js', 'actor.js',
  'ai_lode.js', 'ai_lou.js', 'ai_random.js', 'human.js', 'state.js',
  'snellman.js', 'save.js',
  // Fire & Ice defines faction constants (F_RIVERWALKERS et al) that the
  // AIs reference at runtime even with the expansion switched off.
  // unittest.js is deliberately skipped -- it is not needed to play.
  'fireice.js',
];
for (const f of files) {
  vm.runInContext(fs.readFileSync(path.join(tmaiDir, f), 'utf8'), sandbox, { filename: f });
}

// Rendering and logging are pure output; stub them so no DOM is touched
// during play. Game logic never reads them back.
vm.runInContext(`
  drawHud = function() {};
  drawMap = function() {};
  drawHud2 = function() {};
  displayLog = function() {};
  initialGameRender = function() {};
  drawSaveLoadUI = function() {};
  popupElement = { innerHTML: '' };
`, sandbox);

// Deterministic RNG so runs are reproducible (tmai uses Math.random).
vm.runInContext(`
  var __s = ${seed} >>> 0;
  Math.random = function() {
    __s = (__s * 1664525 + 1013904223) >>> 0;
    return __s / 4294967296;
  };
`, sandbox);

function makeParams() {
  return vm.runInContext(`(function() {
    return {
      numplayers: 4,
      startplayer: -1,
      finalscoring: 0,
      worldGenerator: initStandardWorld,
      worldMap: 'standard',
      presetfaction: ['random','random','random','random'],
      presetround: [T_NONE,T_NONE,T_NONE,T_NONE,T_NONE,T_NONE],
      presetbonus: {},
      newcultistsrule: true,
      towntilepromo2013: false,
      bonustilepromo2013: false,
      fireice: false,
      fireiceerrata: false,
      roundtilepromo2015: false,
      turnorder: false,
      aiAlgorithm: 0,
      allai: true
    };
  })()`, sandbox);
}

const results = [];
for (let g = 0; g < numGames; g++) {
  try {
    sandbox.__params = makeParams();
    const scores = vm.runInContext(`(function() {
      game = new Game();
      state = new State();
      startQuickGameButtonFun(__params);
      // startQuickGameButtonFun ends by kicking off the non-blocking loop;
      // with setTimeout shimmed to run inline that already advances play.
      // Drive it synchronously to completion regardless.
      var guard = 0;
      while (state.type != S_GAME_OVER && guard++ < 20000) {
        gameLoopBlocking(function() { return state.type == S_GAME_OVER; });
      }
      var out = [];
      for (var i = 0; i < game.players.length; i++) {
        out.push({ faction: game.players[i].faction ? game.players[i].faction.name : '?',
                   vp: game.players[i].vp });
      }
      return JSON.stringify({ over: state.type == S_GAME_OVER, round: state.round, players: out });
    })()`, sandbox);
    const parsed = JSON.parse(scores);
    results.push(parsed);
    console.log(JSON.stringify({ game: g, ...parsed }));
  } catch (err) {
    console.log(JSON.stringify({ game: g, error: String(err.message).slice(0, 200) }));
  }
}

const finished = results.filter((r) => r.over);
if (finished.length) {
  const all = finished.flatMap((r) => r.players.map((p) => p.vp));
  const winners = finished.map((r) => Math.max(...r.players.map((p) => p.vp)));
  const totals = finished.map((r) => r.players.reduce((a, p) => a + p.vp, 0));
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  console.log(JSON.stringify({
    summary: true,
    games_finished: finished.length,
    games_attempted: numGames,
    vp_per_player_mean: +mean(all).toFixed(1),
    winner_vp_mean: +mean(winners).toFixed(1),
    table_total_mean: +mean(totals).toFixed(1),
  }));
} else {
  console.log(JSON.stringify({ summary: true, games_finished: 0, games_attempted: numGames }));
}
