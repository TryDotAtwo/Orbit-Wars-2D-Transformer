import type { DashboardTelemetry, Fleet, Planet, ReplayCometGroup, ReplayFrame, ReplayGame, ReplayGameIndexEntry } from './types';

type CompactReplayGame = [number, string, number, string, number, CompactReplayFrame[]];
type CompactReplayActualGame = [number, number, CompactReplayParticipant[], CompactReplayFrame[]];
type CompactReplayParticipant = [string, string, number];
type CompactReplayFrame = [number, CompactPlanet[], CompactFleet[], CompactCometGroup[]];
type CompactPlanet = [number, number, number, number, number, number, number];
type CompactFleet = [number, number, number, number, number, number];
type CompactCometGroup = [number[], Array<Array<[number, number]>>, number];
type IndexedReplayGameManifest = {
  generation: number;
  gameIndex: number;
  path: string;
  participants: Array<{ modelId: string; opponentId: string; reward: number }>;
  frameCount: number;
};

type LoadTelemetryOptions = {
  includeLiveReplay?: boolean;
};

const LIVE_REPLAY_FORMAT = 'orbit_live_replay_v1';
const LIVE_REPLAY_SLOT_FORMAT = 'orbit_live_replay_slots_v1';
const LIVE_REPLAY_MAGIC = 'OWLIVE1\n';
const LIVE_REPLAY_SLOT_MAGIC = 'OWSLOT1\n';
const LIVE_REPLAY_RECORD_HEADER = 1;
const LIVE_REPLAY_RECORD_FRAME = 2;
const LIVE_REPLAY_RECORD_RESULT = 3;
const LIVE_REPLAY_SLOT_LENGTH_BYTES = 4;
const BINARY_U8_BYTES = 1;
const BINARY_U32_BYTES = 4;
const BINARY_I32_BYTES = 4;
const BINARY_F32_BYTES = 4;
const MODEL_ID_PAD_WIDTH = 3;

export const sampleTelemetry: DashboardTelemetry = {
  source: 'sample',
  sourceMessage: 'embedded sample telemetry; trainer artifact feed not connected',
  runId: 'sample',
  activeGeneration: 18,
  runProfile: 'sample',
  trainingMode: 'sample',
  strictGameRules: false,
  episodeSteps: 500,
  expectedFramesPerGame: 501,
  populationSize: 128,
  eliteCount: 12,
  gamesPerModel: 8,
  replaysPerModel: 10,
  generationReplayGameCount: 4,
  playersPerGame: 4,
  simultaneousGames: 1024,
  inferenceBatchCalls: 64000,
  maxInferenceBatchSize: 32,
  mapSource: 'embedded_sample',
  turnLoop: 'sample',
  fullReplay: false,
  replayFrameStride: 8,
  storedReplayGames: 0,
  gamesPerSecond: 42136,
  turnsPerSecond: 110000,
  gpuUtilization: 84,
  cpuWorkers: 12,
  evalQueue: 384,
  submissionGate: 'timing submit 1 pending',
  validationErrors: ['golden trace gate not completed', 'kaggle timing gate not submitted'],
  metrics: [
    metric(1, 0.48, 9100, 47000, 0.42, 34),
    metric(4, 0.52, 17200, 65000, 0.38, 49),
    metric(8, 0.57, 28600, 84000, 0.31, 66),
    metric(12, 0.61, 35400, 101000, 0.27, 76),
    metric(16, 0.64, 40900, 109000, 0.24, 82),
    metric(18, 0.66, 42136, 110000, 0.23, 84),
  ],
  generationWinRates: [
    generationWinRate(12, 1, 0.42, 4, 19, 6, 19),
    generationWinRate(12, 4, 0.48, 4, 21, 7, 16),
    generationWinRate(12, 8, 0.55, 4, 26, 5, 13),
    generationWinRate(12, 12, 0.61, 4, 29, 4, 11),
    generationWinRate(18, 4, 0.46, 4, 22, 5, 21),
    generationWinRate(18, 8, 0.53, 4, 25, 6, 17),
    generationWinRate(18, 12, 0.59, 4, 28, 5, 15),
    generationWinRate(18, 16, 0.64, 4, 31, 4, 13),
    generationWinRate(18, 18, 0.68, 4, 34, 3, 11),
  ],
  models: [
    modelRow('M-017', 'M-004', 681, 332, 23, 211, true),
    modelRow('M-081', 'M-017', 672, 318, 19, 219, true),
    modelRow('M-044', 'M-012', 664, 301, 18, 226, true),
    modelRow('M-103', 'M-081', 657, 289, 16, 230, false),
    modelRow('M-006', 'seed', 641, 264, 14, 246, false),
  ],
  frames: [
    {
      step: 137,
      planets: [
        { id: 1, owner: 0, x: 20, y: 24, radius: 2.6, ships: 38, production: 4 },
        { id: 2, owner: 1, x: 80, y: 76, radius: 2.2, ships: 28, production: 3 },
        { id: 3, owner: -1, x: 37, y: 61, radius: 1.7, ships: 16, production: 2 },
        { id: 4, owner: 0, x: 57, y: 34, radius: 2.6, ships: 54, production: 4 },
        { id: 5, owner: 1, x: 73, y: 29, radius: 1.0, ships: 9, production: 1 },
        { id: 6, owner: -1, x: 25, y: 80, radius: 1.0, ships: 11, production: 1 },
      ],
      fleets: [
        { id: 101, owner: 0, x: 42, y: 47, angle: 0.29, ships: 19 },
        { id: 102, owner: 1, x: 67, y: 58, angle: -2.37, ships: 14 },
        { id: 103, owner: 0, x: 54, y: 29, angle: 1.18, ships: 22 },
      ],
      comets: [
        {
          planetIds: [5],
          paths: [
            [
              [5, 19],
              [24, 28],
              [48, 44],
              [72, 70],
              [95, 88],
            ],
          ],
          pathIndex: 2,
        },
      ],
    },
  ],
  replayChunks: [],
  replayGames: [],
  warnings: ['sample_data_not_training_artifact'],
};

export async function loadTelemetry(options: LoadTelemetryOptions = {}): Promise<DashboardTelemetry> {
  try {
    const response = await fetch('/telemetry/latest.json', { cache: 'no-store' });
    if (!response.ok) {
      return sampleTelemetry;
    }
    const artifact = await response.json();
    const manifestReplayGames = (artifact.replayGames ?? []).map(hydrateReplayGame);
    const liveReplayWarnings: string[] = [];
    const liveReplayGames = options.includeLiveReplay
      ? await loadLiveReplayGames(artifact).catch((error) => {
          liveReplayWarnings.push(`live_replay_load_failed=${String(error)}`);
          return [] as ReplayGame[];
        })
      : [];
    const replayGames = liveReplayGames.length > 0 ? liveReplayGames : manifestReplayGames;
    const artifactFrames = (artifact.frames ?? []).map(hydrateFrame);
    const frames = liveReplayGames[0]?.frames ?? artifactFrames;
    return {
      ...sampleTelemetry,
      ...artifact,
      replayGames,
      frames,
      metrics: completedMetrics(artifact.metrics ?? [], artifact.activeGeneration).map(hydrateMetric),
      generationWinRates: dedupGenerationWinRates(artifact.generationWinRates ?? []),
      models: (artifact.models ?? []).map(hydrateModel),
      validationErrors: artifact.validationErrors ?? [],
      replayChunks: dedupByGeneration(artifact.replayChunks ?? []),
      runId: artifact.runId ?? 'artifact',
      trainingMode: artifact.trainingMode ?? 'unknown',
      fullReplay: artifact.fullReplay ?? false,
      replayFrameStride: artifact.replayFrameStride ?? 0,
      storedReplayGames: artifact.storedReplayGames ?? replayGames.length,
      generationReplayGameCount: artifact.generationReplayGameCount ?? sampleTelemetry.generationReplayGameCount,
      playersPerGame: artifact.playersPerGame ?? sampleTelemetry.playersPerGame,
      simultaneousGames: artifact.simultaneousGames ?? sampleTelemetry.simultaneousGames,
      inferenceBatchCalls: artifact.inferenceBatchCalls ?? sampleTelemetry.inferenceBatchCalls,
      maxInferenceBatchSize: artifact.maxInferenceBatchSize ?? sampleTelemetry.maxInferenceBatchSize,
      liveReplayFormat: artifact.liveReplayFormat,
      liveReplayPath: artifact.liveReplayPath,
      liveReplayFrameCount: artifact.liveReplayFrameCount ?? liveReplayGames[0]?.frames.length,
      warnings: [...(artifact.warnings ?? []), ...liveReplayWarnings],
      source: 'artifact',
      sourceMessage: artifact.sourceMessage ?? 'loaded from /telemetry/latest.json',
    };
  } catch {
    return sampleTelemetry;
  }
}

function dedupByGeneration<T extends { generation: number }>(items: T[]): T[] {
  const byGeneration = new Map<number, T>();
  for (const item of items) {
    byGeneration.set(Number(item.generation), item);
  }
  return Array.from(byGeneration.values()).sort(
    (left, right) => Number(left.generation) - Number(right.generation),
  );
}

function completedMetrics<T extends { generation: number }>(items: T[], activeGeneration: number): T[] {
  return dedupByGeneration(items).filter((item) => Number(item.generation) < Number(activeGeneration));
}

function dedupGenerationWinRates(
  items: Array<Partial<DashboardTelemetry['generationWinRates'][number]>>,
): DashboardTelemetry['generationWinRates'] {
  const byKey = new Map<string, DashboardTelemetry['generationWinRates'][number]>();
  for (const item of items) {
    const point = hydrateGenerationWinRate(item);
    byKey.set(generationWinRateKey(point.validationGeneration, point.evaluatedGeneration), point);
  }
  return Array.from(byKey.values()).sort(
    (left, right) =>
      Number(left.validationGeneration) - Number(right.validationGeneration) ||
      Number(left.evaluatedGeneration) - Number(right.evaluatedGeneration),
  );
}

function generationWinRateKey(validationGeneration: number, evaluatedGeneration: number) {
  return `${Number(validationGeneration)}:${Number(evaluatedGeneration)}`;
}

function hydrateFrame(frame: any) {
  return {
    ...frame,
    comets: frame.comets ?? [],
  };
}

function hydrateReplayGame(game: ReplayGame): ReplayGame {
  return {
    ...game,
    frames: game.frames.map(hydrateFrame),
  };
}

type LiveReplayParticipant = {
  modelId: string;
  opponentId: string;
  reward: number;
};

type LiveReplayGameMeta = {
  generation: number;
  gameIndex: number;
  participants: LiveReplayParticipant[];
  frames: ReplayFrame[];
};

async function loadLiveReplayGames(artifact: any): Promise<ReplayGame[]> {
  if (
    (artifact.liveReplayFormat !== LIVE_REPLAY_FORMAT && artifact.liveReplayFormat !== LIVE_REPLAY_SLOT_FORMAT) ||
    !artifact.liveReplayPath
  ) {
    return [];
  }
  if (artifact.liveReplayFormat === LIVE_REPLAY_SLOT_FORMAT) {
    return [];
  }
  const response = await fetch(artifact.liveReplayPath, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`live_replay_fetch_failed=${artifact.liveReplayPath}`);
  }
  const buffer = await response.arrayBuffer();
  return decodeLiveReplay(buffer);
}

function decodeLiveReplay(buffer: ArrayBuffer): ReplayGame[] {
  const reader = new LiveReplayBinaryReader(buffer);
  if (!reader.readMagic()) {
    return [];
  }
  const gamesByIndex = new Map<number, LiveReplayGameMeta>();
  while (!reader.done()) {
    try {
      const recordType = reader.readU8();
      if (recordType === LIVE_REPLAY_RECORD_HEADER) {
        decodeLiveReplayHeader(reader, gamesByIndex);
      } else if (recordType === LIVE_REPLAY_RECORD_FRAME) {
        decodeLiveReplayFrameRecord(reader, gamesByIndex);
      } else if (recordType === LIVE_REPLAY_RECORD_RESULT) {
        decodeLiveReplayResultRecord(reader, gamesByIndex);
      } else {
        break;
      }
    } catch {
      break;
    }
  }
  return Array.from(gamesByIndex.values()).flatMap((game) =>
    game.participants.map((participant) => ({
      generation: game.generation,
      modelId: participant.modelId,
      gameIndex: game.gameIndex,
      opponentId: participant.opponentId,
      reward: participant.reward,
      frames: game.frames,
    })),
  );
}

function decodeSlotReplay(buffer: ArrayBuffer): ReplayGame[] {
  const reader = new LiveReplayBinaryReader(buffer);
  if (!reader.readMagic(LIVE_REPLAY_SLOT_MAGIC)) {
    return [];
  }
  const generation = reader.readU32();
  reader.readU32();
  const gameCount = reader.readU32();
  const frameSlotsPerGame = reader.readU32();
  const frameSlotBytes = reader.readU32();
  const frameBaseOffset = reader.readU64();
  const games: LiveReplayGameMeta[] = [];
  for (let gameOffset = 0; gameOffset < gameCount; gameOffset += 1) {
    const gameIndex = reader.readU32();
    const participantCount = reader.readU32();
    const participants: LiveReplayParticipant[] = [];
    for (let participantOffset = 0; participantOffset < participantCount; participantOffset += 1) {
      participants.push({
        modelId: modelIdFromIndex(reader.readU32()),
        opponentId: reader.readString(),
        reward: 0,
      });
    }
    games.push({ generation, gameIndex, participants, frames: [] });
  }
  const resultBaseOffset =
    frameBaseOffset + gameCount * frameSlotsPerGame * (LIVE_REPLAY_SLOT_LENGTH_BYTES + frameSlotBytes);
  for (let gameOffset = 0; gameOffset < gameCount; gameOffset += 1) {
    const game = games[gameOffset];
    for (let frameSlot = 0; frameSlot < frameSlotsPerGame; frameSlot += 1) {
      const slotOffset =
        frameBaseOffset +
        (gameOffset * frameSlotsPerGame + frameSlot) * (LIVE_REPLAY_SLOT_LENGTH_BYTES + frameSlotBytes);
      const frameLength = reader.readU32At(slotOffset);
      if (frameLength === 0 || frameLength > frameSlotBytes) {
        continue;
      }
      const frameReader = new LiveReplayBinaryReader(
        buffer.slice(slotOffset + LIVE_REPLAY_SLOT_LENGTH_BYTES, slotOffset + LIVE_REPLAY_SLOT_LENGTH_BYTES + frameLength),
      );
      game.frames.push(decodeLiveReplayFrame(frameReader));
    }
    const resultOffset = resultBaseOffset + gameOffset * (LIVE_REPLAY_SLOT_LENGTH_BYTES + game.participants.length * 4);
    const rewardCount = reader.readU32At(resultOffset);
    if (rewardCount > 0 && rewardCount <= game.participants.length) {
      const rewardReader = new LiveReplayBinaryReader(
        buffer.slice(
          resultOffset + LIVE_REPLAY_SLOT_LENGTH_BYTES,
          resultOffset + LIVE_REPLAY_SLOT_LENGTH_BYTES + rewardCount * 4,
        ),
      );
      for (let rewardIndex = 0; rewardIndex < rewardCount; rewardIndex += 1) {
        game.participants[rewardIndex].reward = rewardReader.readI32();
      }
    }
  }
  return games.flatMap((game) =>
    game.participants.map((participant) => ({
      generation: game.generation,
      modelId: participant.modelId,
      gameIndex: game.gameIndex,
      opponentId: participant.opponentId,
      reward: participant.reward,
      frames: game.frames,
    })),
  );
}

function decodeLiveReplayHeader(reader: LiveReplayBinaryReader, gamesByIndex: Map<number, LiveReplayGameMeta>) {
  const generation = reader.readU32();
  reader.readU32();
  const gameCount = reader.readU32();
  for (let gameOffset = 0; gameOffset < gameCount; gameOffset += 1) {
    const gameIndex = reader.readU32();
    const participantCount = reader.readU32();
    const participants: LiveReplayParticipant[] = [];
    for (let participantOffset = 0; participantOffset < participantCount; participantOffset += 1) {
      participants.push({
        modelId: modelIdFromIndex(reader.readU32()),
        opponentId: reader.readString(),
        reward: 0,
      });
    }
    gamesByIndex.set(gameIndex, {
      generation,
      gameIndex,
      participants,
      frames: [],
    });
  }
}

function decodeLiveReplayFrameRecord(reader: LiveReplayBinaryReader, gamesByIndex: Map<number, LiveReplayGameMeta>) {
  const gameIndex = reader.readU32();
  const frame = decodeLiveReplayFrame(reader);
  gamesByIndex.get(gameIndex)?.frames.push(frame);
}

function decodeLiveReplayResultRecord(reader: LiveReplayBinaryReader, gamesByIndex: Map<number, LiveReplayGameMeta>) {
  const gameIndex = reader.readU32();
  const rewardCount = reader.readU32();
  const game = gamesByIndex.get(gameIndex);
  for (let rewardIndex = 0; rewardIndex < rewardCount; rewardIndex += 1) {
    const reward = reader.readI32();
    if (game?.participants[rewardIndex]) {
      game.participants[rewardIndex].reward = reward;
    }
  }
}

function decodeLiveReplayFrame(reader: LiveReplayBinaryReader): ReplayFrame {
  const step = reader.readU32();
  const planetCount = reader.readU32();
  const planets: Planet[] = [];
  for (let planetIndex = 0; planetIndex < planetCount; planetIndex += 1) {
    planets.push({
      id: reader.readI32(),
      owner: reader.readI32(),
      x: reader.readF32(),
      y: reader.readF32(),
      radius: reader.readF32(),
      ships: reader.readF32(),
      production: reader.readF32(),
    });
  }
  const fleetCount = reader.readU32();
  const fleets: Fleet[] = [];
  for (let fleetIndex = 0; fleetIndex < fleetCount; fleetIndex += 1) {
    fleets.push({
      id: reader.readI32(),
      owner: reader.readI32(),
      x: reader.readF32(),
      y: reader.readF32(),
      angle: reader.readF32(),
      ships: reader.readF32(),
    });
  }
  const cometCount = reader.readU32();
  const comets: ReplayCometGroup[] = [];
  for (let cometIndex = 0; cometIndex < cometCount; cometIndex += 1) {
    comets.push(decodeLiveReplayComet(reader));
  }
  return { step, planets, fleets, comets };
}

function decodeLiveReplayComet(reader: LiveReplayBinaryReader): ReplayCometGroup {
  const planetIdCount = reader.readU32();
  const planetIds: number[] = [];
  for (let planetIdIndex = 0; planetIdIndex < planetIdCount; planetIdIndex += 1) {
    planetIds.push(reader.readI32());
  }
  const pathCount = reader.readU32();
  const paths: Array<Array<[number, number]>> = [];
  for (let pathIndex = 0; pathIndex < pathCount; pathIndex += 1) {
    const pointCount = reader.readU32();
    const path: Array<[number, number]> = [];
    for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
      path.push([reader.readF32(), reader.readF32()]);
    }
    paths.push(path);
  }
  return {
    planetIds,
    paths,
    pathIndex: reader.readI32(),
  };
}

function modelIdFromIndex(modelIndex: number) {
  return `M-${modelIndex.toString().padStart(MODEL_ID_PAD_WIDTH, '0')}`;
}

class LiveReplayBinaryReader {
  private readonly view: DataView;
  private readonly decoder = new TextDecoder();
  private offset = 0;

  constructor(buffer: ArrayBuffer) {
    this.view = new DataView(buffer);
  }

  done() {
    return this.offset >= this.view.byteLength;
  }

  readMagic(magic = LIVE_REPLAY_MAGIC) {
    if (this.view.byteLength < magic.length) {
      return false;
    }
    for (let index = 0; index < magic.length; index += 1) {
      if (this.view.getUint8(index) !== magic.charCodeAt(index)) {
        return false;
      }
    }
    this.offset = magic.length;
    return true;
  }

  readU8() {
    this.ensure(BINARY_U8_BYTES);
    const value = this.view.getUint8(this.offset);
    this.offset += BINARY_U8_BYTES;
    return value;
  }

  readU32() {
    this.ensure(BINARY_U32_BYTES);
    const value = this.view.getUint32(this.offset, true);
    this.offset += BINARY_U32_BYTES;
    return value;
  }

  readU32At(offset: number) {
    if (offset < 0 || offset + BINARY_U32_BYTES > this.view.byteLength) {
      return 0;
    }
    return this.view.getUint32(offset, true);
  }

  readU64() {
    this.ensure(BINARY_U32_BYTES * 2);
    const low = this.view.getUint32(this.offset, true);
    const high = this.view.getUint32(this.offset + BINARY_U32_BYTES, true);
    this.offset += BINARY_U32_BYTES * 2;
    return high * 0x100000000 + low;
  }

  readI32() {
    this.ensure(BINARY_I32_BYTES);
    const value = this.view.getInt32(this.offset, true);
    this.offset += BINARY_I32_BYTES;
    return value;
  }

  readF32() {
    this.ensure(BINARY_F32_BYTES);
    const value = this.view.getFloat32(this.offset, true);
    this.offset += BINARY_F32_BYTES;
    return value;
  }

  readString() {
    const byteLength = this.readU32();
    this.ensure(byteLength);
    const value = this.decoder.decode(new Uint8Array(this.view.buffer, this.offset, byteLength));
    this.offset += byteLength;
    return value;
  }

  private ensure(byteCount: number) {
    if (this.offset + byteCount > this.view.byteLength) {
      throw new Error('live_replay_partial_record');
    }
  }
}

function modelRow(
  id: string,
  parent: string,
  rating: number,
  wins: number,
  draws: number,
  losses: number,
  selected: boolean,
) {
  const games = wins + draws + losses;
  return {
    id,
    parent,
    rating,
    wins,
    draws,
    losses,
    games,
    captures: Math.round(wins * 1.7),
    fleetHits: Math.round(games * 4.2),
    hitShips: Math.round(games * 35),
    sunDestroyedFleets: Math.round(losses * 0.6),
    sunDestroyedShips: Math.round(losses * 8),
    outOfBoundsFleets: Math.round(losses * 0.3),
    outOfBoundsShips: Math.round(losses * 4),
    launchActions: Math.round(games * 6.1),
    launchedShips: Math.round(games * 42),
    avgFleetSize: 13.4,
    mutation: selected ? 'self-play' : 'candidate',
    selected,
  };
}

function hydrateMetric(point: Partial<DashboardTelemetry['metrics'][number]>) {
  return {
    ...metric(0, 0, 0, 0, 0, 0),
    ...point,
  };
}

function hydrateGenerationWinRate(point: Partial<DashboardTelemetry['generationWinRates'][number]>) {
  return {
    ...generationWinRate(0, 0, 0, 0, 0, 0, 0),
    ...point,
  };
}

function hydrateModel(model: Partial<DashboardTelemetry['models'][number]>) {
  return {
    ...modelRow('M-000', 'unknown', 0, 0, 0, 0, false),
    ...model,
  };
}

function metric(
  generation: number,
  winRate: number,
  gamesPerSecond: number,
  turnsPerSecond: number,
  p95LatencyMs: number,
  gpuUtilization: number,
) {
  return {
    generation,
    winRate,
    gamesPerSecond,
    turnsPerSecond,
    p95LatencyMs,
    gpuUtilization,
    evaluatedGames: 1024,
    sampledReplayGames: 128,
    modelActionCalls: 65536,
    launchActions: 24000,
    launchedShips: 180000,
    captures: 820,
    fleetHits: 12400,
    hitShips: 420000,
    sunDestroyedFleets: 640,
    sunDestroyedShips: 9100,
    avgFleetSize: 13.4,
    avgLaunchActionsPerTurn: 3.7,
    avgLaunchedShipsPerTurn: 27.4,
    avgModelActionMs: 0.03,
    inferenceBatchCalls: 64000,
    maxInferenceBatchSize: 32,
    simultaneousGames: 1024,
    modelActionSeconds: 1.2,
    simulationStepSeconds: 0.8,
    evaluationSeconds: 3.4,
    replaySeconds: 2.2,
    replayWriteSeconds: 0.1,
    generationValidationGames: 128,
    generationValidationSeconds: 1.6,
    backpropSamples: 5000,
    backpropModels: 12,
    backpropSeconds: 0.9,
    reproductionSeconds: 0.05,
    generationSeconds: 5.8,
  };
}

function generationWinRate(
  validationGeneration: number,
  evaluatedGeneration: number,
  winRate: number,
  modelCount: number,
  wins: number,
  draws: number,
  losses: number,
) {
  return {
    validationGeneration,
    evaluatedGeneration,
    modelCount,
    games: wins + draws + losses,
    wins,
    draws,
    losses,
    winRate,
  };
}

export async function loadReplayChunk(path: string): Promise<ReplayGame[]> {
  if (path.endsWith('.owlive')) {
    const response = await fetch(path, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`replay_chunk_fetch_failed=${path}`);
    }
    return decodeLiveReplay(await response.arrayBuffer());
  }
  if (path.endsWith('.owslot')) {
    const response = await fetch(path, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`replay_chunk_fetch_failed=${path}`);
    }
    return decodeSlotReplay(await response.arrayBuffer());
  }
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`replay_chunk_fetch_failed=${path}`);
  }
  const artifact = await response.json();
  if (artifact.format === 'compact_replay_v1') {
    return (artifact.games ?? []).map(decodeCompactReplayGame);
  }
  if (artifact.format === 'compact_replay_v2') {
    return (artifact.games ?? []).flatMap(decodeCompactReplayActualGame);
  }
  if (artifact.format === 'indexed_replay_v1') {
    return (artifact.games ?? []).flatMap(decodeIndexedReplayGameManifest);
  }
  if (artifact.format === 'binary_replay_alias_v1') {
    if (artifact.binaryPath) {
      const binaryResponse = await fetch(artifact.binaryPath, { cache: 'no-store' });
      if (!binaryResponse.ok) {
        throw new Error(`replay_binary_alias_fetch_failed=${artifact.binaryPath}`);
      }
      const buffer = await binaryResponse.arrayBuffer();
      return artifact.binaryPath.endsWith('.owslot') ? decodeSlotReplay(buffer) : decodeLiveReplay(buffer);
    }
    return artifact.replayGames ?? [];
  }
  return artifact.replayGames ?? [];
}

export async function loadReplayGame(path: string): Promise<ReplayGame[]> {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`replay_game_fetch_failed=${path}`);
  }
  const artifact = await response.json();
  if (artifact.format === 'compact_replay_game_v1') {
    return decodeCompactReplayActualGame(artifact.game);
  }
  if (artifact.format === 'compact_replay_v2') {
    return (artifact.games ?? []).flatMap(decodeCompactReplayActualGame);
  }
  return artifact.replayGames ?? [];
}

function decodeIndexedReplayGameManifest(game: IndexedReplayGameManifest): ReplayGameIndexEntry[] {
  return game.participants.map((participant) => ({
    generation: game.generation,
    gameIndex: game.gameIndex,
    modelId: participant.modelId,
    opponentId: participant.opponentId,
    reward: participant.reward,
    frames: [],
    gamePath: game.path,
    indexed: true,
  }));
}

function decodeCompactReplayGame(game: CompactReplayGame): ReplayGame {
  return {
    generation: game[0],
    modelId: game[1],
    gameIndex: game[2],
    opponentId: game[3],
    reward: game[4],
    frames: game[5].map(decodeCompactReplayFrame),
  };
}

function decodeCompactReplayActualGame(game: CompactReplayActualGame): ReplayGame[] {
  const frames = game[3].map(decodeCompactReplayFrame);
  return game[2].map((participant) => ({
    generation: game[0],
    gameIndex: game[1],
    modelId: participant[0],
    opponentId: participant[1],
    reward: participant[2],
    frames,
  }));
}

function decodeCompactReplayFrame(frame: CompactReplayFrame) {
  return {
    step: frame[0],
    planets: frame[1].map(decodeCompactPlanet),
    fleets: frame[2].map(decodeCompactFleet),
    comets: frame[3].map(decodeCompactCometGroup),
  };
}

function decodeCompactCometGroup(group: CompactCometGroup) {
  return {
    planetIds: group[0],
    paths: group[1],
    pathIndex: group[2],
  };
}

function decodeCompactPlanet(planet: CompactPlanet) {
  return {
    id: planet[0],
    owner: planet[1],
    x: planet[2],
    y: planet[3],
    radius: planet[4],
    ships: planet[5],
    production: planet[6],
  };
}

function decodeCompactFleet(fleet: CompactFleet) {
  return {
    id: fleet[0],
    owner: fleet[1],
    x: fleet[2],
    y: fleet[3],
    angle: fleet[4],
    ships: fleet[5],
  };
}
