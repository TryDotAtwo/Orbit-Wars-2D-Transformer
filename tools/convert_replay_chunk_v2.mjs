import fs from 'node:fs';

const FORMAT_V2 = 'compact_replay_v2';
const FORMAT_V1 = 'compact_replay_v1';
const ARG_PATH_INDEX = 2;
const GENERATION_INDEX = 0;
const MODEL_ID_INDEX = 1;
const GAME_INDEX_INDEX = 2;
const OPPONENT_ID_INDEX = 3;
const REWARD_INDEX = 4;
const FRAMES_INDEX = 5;
const V2_PARTICIPANTS_INDEX = 2;
const V2_FRAMES_INDEX = 3;

const path = process.argv[ARG_PATH_INDEX];
if (!path) {
  throw new Error('usage: node tools/convert_replay_chunk_v2.mjs <replay_chunk.json>');
}

const artifact = JSON.parse(fs.readFileSync(path, 'utf8'));
if (artifact.format === FORMAT_V2) {
  const stats = replayChunkStats(artifact);
  console.log(JSON.stringify({ path, converted: false, ...stats }));
  process.exit(0);
}
if (artifact.format !== FORMAT_V1) {
  throw new Error(`unsupported_replay_chunk_format=${artifact.format}`);
}

const actualGames = new Map();
for (const game of artifact.games ?? []) {
  const key = `${game[GENERATION_INDEX]}:${game[GAME_INDEX_INDEX]}`;
  let actualGame = actualGames.get(key);
  if (!actualGame) {
    actualGame = [
      game[GENERATION_INDEX],
      game[GAME_INDEX_INDEX],
      [],
      game[FRAMES_INDEX],
    ];
    actualGames.set(key, actualGame);
  }
  actualGame[V2_PARTICIPANTS_INDEX].push([
    game[MODEL_ID_INDEX],
    game[OPPONENT_ID_INDEX],
    game[REWARD_INDEX],
  ]);
}

const converted = {
  generation: artifact.generation,
  format: FORMAT_V2,
  games: Array.from(actualGames.values()),
};
fs.writeFileSync(path, JSON.stringify(converted));
const stats = replayChunkStats(converted);
console.log(JSON.stringify({ path, converted: true, ...stats }));

function replayChunkStats(artifact) {
  const games = artifact.games ?? [];
  const participantViews = games.reduce(
    (sum, game) => sum + (game[V2_PARTICIPANTS_INDEX]?.length ?? 0),
    0,
  );
  return {
    format: artifact.format,
    actualGames: games.length,
    participantViews,
    firstGameFrames: games[0]?.[V2_FRAMES_INDEX]?.length ?? 0,
    bytes: fs.statSync(path).size,
  };
}
